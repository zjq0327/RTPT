/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#include "Scene/Material/MaterialDefines.slangh"

import Rendering.Materials.MaterialShading;
import LoadShadingData;
import Utils.Debug.PixelDebug;
import Utils.Math.HashUtils;
import Params;
import PathReservoir;
import Rendering.Utils.PixelStats;
import Scene.RaytracingInline;
import Rendering.Materials.MaterialShading;
import Utils.Sampling.TinyUniformSampleGenerator;

static const int kRcWiCount = 1;

/// Builds a Path and a PathPrefix.
struct PathBuilder
{
    TriMeshHitInfo rcVertexHit;
    float3 rcVertexWi[kRcWiCount];
    uint cachedRandomSeed;
    TinyUniformSampleGenerator sg;
    int rcVertexLength = kMaximumPathLength;
    ReSTIRPathFlags pathFlags;
    float3 cachedJacobian = 1.f;

#if GROUP_GENERATE_SUFFIX
    float suffixRootSurvivalPdf = 1.f;
    float suffixWeightSum = 0.f;
    float suffixBestRank;
    float suffixSelectedLuminance;
    float3 suffixSelectedF;
    float3 suffixSelectedIncident;
    float suffixSelectedLightPdf;
    ReSTIRPathFlags suffixSelectedFlags;

    [mutating]
    void offerSuffixEvent(uint depth, float3 F, float3 postfix, float survivalPdf,
        bool nee, float misWeight, float lightPdf, uint lightType)
    {
        // Rank events by suffix luminance so camera throughput cannot change the ranking.
        float luminance = PathReservoir::toScalar(postfix);
        float weight = luminance * suffixRootSurvivalPdf / survivalPdf;
        if (!(weight > 0.f) || !isfinite(weight)) return;

        // Match the reference sampler's event indices and use the post-reconnection seed.
        uint eventIndex = depth * 3u + (nee ? 1u : 0u);
        uint bits = jenkinsHash(cachedRandomSeed ^ jenkinsHash(eventIndex + 1u));
        float u = (float(bits >> 9u) + 0.5f) * (1.f / 8388608.f);
        float rank = -log(u) / weight;
        bool selected = suffixWeightSum == 0.f || rank < suffixBestRank;
        suffixWeightSum += weight;
        if (!selected) return;

        suffixBestRank = rank;
        suffixSelectedLuminance = luminance;
        suffixSelectedF = F;
        suffixSelectedIncident = depth == rcVertexLength ? postfix / misWeight : postfix;
        suffixSelectedLightPdf = lightPdf;
        suffixSelectedFlags = pathFlags;
        suffixSelectedFlags.insertPathLength(depth);
        suffixSelectedFlags.insertRcVertexLength(rcVertexLength);
        suffixSelectedFlags.insertLastVertexNEE(nee);
        suffixSelectedFlags.insertLightType(lightType);
    }

    [mutating]
    void finalizeSuffix(inout PathReservoir reservoir)
    {
        if (!(suffixWeightSum > 0.f)) return;

        float innerWeight = suffixWeightSum / suffixSelectedLuminance;
        if (!reservoir.add(suffixSelectedF * innerWeight, suffixRootSurvivalPdf, sg)) return;

        reservoir.pathFlags = suffixSelectedFlags;
        reservoir.rcVertexIrradiance[0] = suffixSelectedIncident * innerWeight;
        reservoir.rcVertexWi[0] = rcVertexWi[0];
        reservoir.rcVertexHit = rcVertexHit;
        reservoir.cachedJacobian = cachedJacobian;
        reservoir.rcRandomSeed = cachedRandomSeed;
        reservoir.lightPdf = suffixSelectedLightPdf;
        reservoir.suffixRISWeight = innerWeight;
    }
#endif

    [mutating]
    void init<S : ISampleGenerator>(inout S sg)
    {
        this.sg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));
        rcVertexWi[0] = float3(0.f);
        rcVertexHit.instanceID = 0xffffffff;
    }

    [mutating] void finalize(inout PathReservoir pathReservoir)
    {
#if GROUP_GENERATE_SUFFIX
        finalizeSuffix(pathReservoir);
#endif
        // All endpoints came from one sampled path.
        pathReservoir.M = 1;
    }

    void storeEndpoint(uint depth, bool nee, uint lightType, float lightPdf, float3 F,
        float3 incident, float3 wi, inout PathReservoir reservoir)
    {
        reservoir.pathFlags = pathFlags;
        reservoir.pathFlags.insertPathLength(depth);
        reservoir.pathFlags.insertRcVertexLength(rcVertexLength);
        reservoir.pathFlags.insertLastVertexNEE(nee);
        reservoir.pathFlags.insertLightType(lightType);
        reservoir.F = F;
        reservoir.rcVertexIrradiance[0] = incident;
        reservoir.rcVertexWi[0] = wi;
        reservoir.rcVertexHit = rcVertexHit;
        reservoir.rcRandomSeed = cachedRandomSeed;
        reservoir.cachedJacobian = cachedJacobian;
        reservoir.lightPdf = lightPdf;
    }

    [mutating]
    bool addEscapeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)
    {
        if (pathLength < 1) return false;
#if GROUP_GENERATE_SUFFIX
        // At the reconnection depth an escape is reached by its outgoing BSDF ray.
        if (!forceAdd && rcVertexLength < kMaximumPathLength && pathLength >= rcVertexLength)
        {
            offerSuffixEvent(pathLength, pathWeight, postfixWeight, russianroulettePDF, false, misWeight, lightPdf, lightType);
            return false;
        }
#endif
        if (!forceAdd && !pathReservoir.add(pathWeight, russianroulettePDF, sg)) return false;
        const bool atRc = pathLength == rcVertexLength;
        // A shift re-evaluates the outgoing edge's MIS weight at the reconnection vertex.
        storeEndpoint(pathLength, false, lightType, lightPdf, pathWeight, atRc ? postfixWeight / misWeight : postfixWeight,
            atRc ? wi : rcVertexWi[0], pathReservoir);
        return true;
    }

    [mutating]
    bool addNeeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)
    {
        if (pathLength < 1) return false;
#if GROUP_GENERATE_SUFFIX
        if (!forceAdd && rcVertexLength < kMaximumPathLength && pathLength > rcVertexLength)
        {
            offerSuffixEvent(pathLength, pathWeight, postfixWeight, russianroulettePDF, true, misWeight, lightPdf, lightType);
            return false;
        }
#endif
        if (!forceAdd && !pathReservoir.add(pathWeight, russianroulettePDF, sg)) return false;
        const bool atRc = pathLength == rcVertexLength;
        storeEndpoint(pathLength, true, lightType, lightPdf, pathWeight, atRc ? postfixWeight * lightPdf / misWeight : postfixWeight,
            atRc ? wi : rcVertexWi[0], pathReservoir);
        return true;
    }
};

