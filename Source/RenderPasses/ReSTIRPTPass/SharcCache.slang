import Scene.ShadingData;
import Scene.Material.MaterialData;
import Rendering.Materials.Microfacet;
#include "Utils/Math/MathConstants.slangh"

// The update and query passes must agree on the exact voxel and key layout.
// The SDK's lock-buffer path supports 64-bit spatial keys on shader model 6.5.
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 0
#define SHARC_UPDATE 1
#define SHARC_QUERY 1
#define SHARC_MATERIAL_DEMODULATION 1
#define SHARC_SEPARATE_EMISSIVE 1
#define SHARC_SAMPLE_NUM_THRESHOLD 4
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_CACHE_RESAMPLING 0
// The sparse training path opens one cache entry and propagates its entire suffix.
#define SHARC_PROPAGATION_DEPTH 1
#include "ThirdParty/SHARC/SharcCommon.h"

#ifndef SHARC_STATS
#define SHARC_STATS 0
#endif

SharcHitData makeSharcHit(ShadingData sd, float3 emission)
{
    SharcHitData hit;
    hit.positionWorld = sd.posW;
    hit.normalWorld = sd.frontFacing ? sd.faceN : -sd.faceN;
    hit.emissive = emission;

    float3 diffuse = (1.f - sd.diffuseTransmission) * (1.f - sd.specularTransmission) * sd.diffuse;
    float3 specular = approxSpecularIntegralGGX((1.f - sd.specularTransmission) * sd.specular,
        sd.linearRoughness * sd.linearRoughness, saturate(dot(sd.N, sd.V)));
    float3 factor = max(float3(0.01f), saturate(diffuse) + saturate(specular));
    hit.materialDemodulation = all(isfinite(factor)) ? factor : float3(1.f);
    return hit;
}

struct SharcQueryData
{
    SharcParameters cache;
    float minRoughness;
#if SHARC_STATS
    RWStructuredBuffer<uint> stats;
#endif

    void invalidateAt(uint index, StructuredBuffer<float4> bounds, uint count)
    {
        if (index >= cache.hashGridData.capacity) return;
        const HashGridKey key = cache.hashGridData.hashEntriesBuffer[index];
        if (key == HASH_GRID_INVALID_HASH_KEY) return;

        const uint level = uint((key >> HASH_GRID_LEVEL_BIT_OFFSET) & HASH_GRID_LEVEL_BIT_MASK);
        const float halfSize = 0.5f * HashGridGetVoxelSize(level, cache.hashGridParameters);
        const float3 center = HashGridGetPositionFromKey(key, cache.hashGridParameters);
        const float3 voxelMin = center - halfSize;
        const float3 voxelMax = center + halfSize;
        for (uint i = 0; i < count; ++i)
        {
            if (all(voxelMax >= bounds[2u * i].xyz) && all(voxelMin <= bounds[2u * i + 1u].xyz))
            {
                // Keep the key so training can refill the same slot this frame.
                cache.accumulationBuffer[index] = SharcZeroAccumulationData();
                cache.resolvedBuffer[index] = SharcZeroPackedData();
                break;
            }
        }
    }

    bool allowsSurface(ShadingData sd, float3 previousPosition, bool previousDelta, bool previousDiffuse, float previousRoughness)
    {
        if (previousDelta || !sd.frontFacing || sd.getMaterialType() != MaterialType::Standard ||
            sd.linearRoughness < max(minRoughness, ROUGHNESS_THRESHOLD) || any(sd.emissive > 0.f) ||
            sd.diffuseTransmission > 0.f || sd.specularTransmission > 0.f) return false;

        uint level = HashGridGetLevel(sd.posW, cache.hashGridParameters);
        float voxelSize = HashGridGetVoxelSize(level, cache.hashGridParameters);
        float segmentLength = length(sd.posW - previousPosition);
        if (!(segmentLength >= voxelSize)) return false;
        if (!previousDiffuse)
        {
            float alpha = previousRoughness * previousRoughness;
            float alpha2 = alpha * alpha;
            float footprint = 2.f * segmentLength * sqrt(0.5f * alpha2 / max(1e-6f, 1.f - alpha2));
            if (!(footprint >= voxelSize)) return false;
        }

        return true;
    }

    bool lookup(ShadingData sd, float3 previousPosition, bool previousDelta, bool previousDiffuse,
        float previousRoughness, out float3 radiance)
    {
        radiance = 0.f;
        if (!allowsSurface(sd, previousPosition, previousDelta, previousDiffuse, previousRoughness)) return false;
#if SHARC_STATS
        InterlockedAdd(stats[0], 1u);
#endif
        float3 value;
        if (!SharcGetCachedRadiance(cache, makeSharcHit(sd, 0.f), value, false) ||
            !all(isfinite(value)) || any(value < 0.f)) return false;
#if SHARC_STATS
        InterlockedAdd(stats[1], 1u);
#endif
        radiance = value;
        return true;
    }
};
