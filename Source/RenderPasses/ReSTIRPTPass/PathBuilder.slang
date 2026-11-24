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

#if BPR
static const int kRcWiCount = 2;
#else
static const int kRcWiCount = 1;
#endif

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

#if SHARC_ENABLE && !BPR
    // The outer candidate represents one seed after the RC bounce. Choose one
    // explicit/cache endpoint within it and carry its inner RIS normalization.
    float suffixWeightSum = 0.f;
    float suffixRootSurvivalPdf = 1.f;
    float suffixBestRank;
    float suffixSelectedLuminance;
    float3 suffixSelectedF;
    float3 suffixSelectedPostfix;
    float suffixSelectedLightPdf;
    ReSTIRPathFlags suffixSelectedFlags;

    [mutating] void offerSuffixEvent(uint depth, float3 F, float3 postfix, float survivalPdf,
        bool nee, bool cache, float misWeight, float lightPdf, uint lightType)
    {
        // Select in the suffix domain. Camera throughput must not change the
        // winner for the same suffix replayed from another pixel.
        float luma = PathReservoir::toScalar(postfix);
        float weight = luma * suffixRootSurvivalPdf / survivalPdf;
        if (!(weight > 0.f) || !isfinite(weight)) return;
        uint event = depth * 3u + (cache ? 2u : nee ? 1u : 0u);
        uint bits = jenkinsHash(cachedRandomSeed ^ jenkinsHash(event + 1u));
        float u = (float(bits >> 9u) + 0.5f) * (1.f / 8388608.f);
        float rank = -log(u) / weight;
        bool selected = suffixWeightSum == 0.f || rank < suffixBestRank;
        suffixWeightSum += weight;
        if (!selected) return;
        suffixBestRank = rank;
        suffixSelectedLuminance = luma;
        suffixSelectedF = F;
        suffixSelectedPostfix = depth == rcVertexLength ? postfix / misWeight : postfix;
        suffixSelectedLightPdf = lightPdf;
        suffixSelectedFlags = pathFlags;
        suffixSelectedFlags.insertPathLength(depth);
        suffixSelectedFlags.insertRcVertexLength(rcVertexLength);
        suffixSelectedFlags.insertLastVertexNEE(nee);
        suffixSelectedFlags.insertLightType(lightType);
        suffixSelectedFlags.setSuffixEvent(true, cache);
    }

    [mutating] void finalizeSuffix(inout PathReservoir reservoir)
    {
        if (!(suffixWeightSum > 0.f)) return;
        float innerWeight = suffixWeightSum / suffixSelectedLuminance;
        if (!reservoir.add(suffixSelectedF * innerWeight, suffixRootSurvivalPdf, sg)) return;
        reservoir.pathFlags = suffixSelectedFlags;
        reservoir.rcVertexIrradiance[0] = suffixSelectedPostfix * innerWeight;
        reservoir.rcVertexWi[0] = rcVertexWi[0];
        reservoir.rcVertexHit = rcVertexHit;
        reservoir.cachedJacobian = cachedJacobian;
        reservoir.rcRandomSeed = cachedRandomSeed;
        reservoir.lightPdf = suffixSelectedLightPdf;
    }

#endif

    [mutating]
    void init<S : ISampleGenerator>(inout S sg)
    {
        this.sg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));
        rcVertexWi[0] = float3(0.f);
        rcVertexHit.instanceID = 0xffffffff;
#if BPR
        rcVertexWi[1] = float3(0.f);
#endif
    }

    [mutating] void finalize(inout PathReservoir pathReservoir)
    {
#if SHARC_ENABLE && !BPR
        finalizeSuffix(pathReservoir);
#endif
#if BPR
        pathReservoir.weight = PathReservoir::toScalar(pathReservoir.F);

        // when kNEESamples == 1, we can use cachedJacobian slot to store NEE contribution so that we don't need to recompute in reuse
        if (any(pathReservoir.rcVertexIrradiance[1] > 0.f)) pathReservoir.F -= pathReservoir.cachedJacobian;
        else pathReservoir.cachedJacobian = 0.f;
#endif
        pathReservoir.M = 1;
    }

    [mutating]
    bool addEscapeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight, bool useHybridShift,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)
    {
        bool selected = false;

        if (pathLength >= 1)
        {
            bool is_rcVertex = pathLength == rcVertexLength;

#if BPR
            pathReservoir.F += pathWeight;

            if (is_rcVertex)
            {
                pathReservoir.rcVertexBSDFLightSamplingIrradiance = postfixWeight / misWeight;
                pathReservoir.lightPdf = lightPdf;
                pathReservoir.rcVertexWi[0] = wi;
            }
            else
            {
                pathReservoir.rcVertexWi[0] = rcVertexWi[0];
                pathReservoir.rcVertexIrradiance[0] = postfixWeight;
            }
            pathReservoir.rcVertexHit = rcVertexHit;

            pathReservoir.pathFlags = pathFlags;
            pathReservoir.pathFlags.insertPathLength(pathLength); // excluding the NEE vertex
            pathReservoir.pathFlags.insertRcVertexLength(rcVertexLength);
#else
            // An escape at pathLength == RC is reached by the outgoing RC ray.
#if SHARC_ENABLE
            bool grouped = useHybridShift && !forceAdd && rcVertexLength < kMaximumPathLength && pathLength >= rcVertexLength;
            if (grouped)
            {
                offerSuffixEvent(pathLength, pathWeight, postfixWeight, russianroulettePDF, false, false, misWeight, lightPdf, lightType);
                return false;
            }
#endif
            if (forceAdd || pathReservoir.add(pathWeight, russianroulettePDF, sg))
            {
                selected = true;
                pathReservoir.pathFlags = pathFlags;
                pathReservoir.pathFlags.insertPathLength(pathLength);
                pathReservoir.pathFlags.insertRcVertexLength(rcVertexLength);
                pathReservoir.pathFlags.insertLastVertexNEE(false);
#if SHARC_ENABLE
                pathReservoir.pathFlags.setSuffixEvent(grouped, false);
#endif
                pathReservoir.rcVertexIrradiance[0] = postfixWeight;
                pathReservoir.F = pathWeight;
                pathReservoir.rcVertexWi[0] = rcVertexWi[0];
                pathReservoir.rcVertexHit = rcVertexHit;

                pathReservoir.cachedJacobian = cachedJacobian;

                pathReservoir.rcRandomSeed = cachedRandomSeed;

                if (is_rcVertex) // at rcVertex
                {
                    pathReservoir.rcVertexWi[0] = wi;
                    pathReservoir.rcVertexIrradiance[0] /= misWeight; // exclude last vertex dependent mis weight
                }
                pathReservoir.lightPdf = lightPdf;
                pathReservoir.pathFlags.insertLightType(lightType);
            }
#endif
        }

        return selected;
    }

    [mutating]
    bool addNeeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight, bool useHybridShift,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)
    {
        bool selected = false;

        if (pathLength >= 1)
        {
            bool is_rcVertex = pathLength == rcVertexLength;

#if BPR
            pathReservoir.F += pathWeight;

            if (is_rcVertex)
            {
                pathReservoir.cachedJacobian = pathWeight;
                pathReservoir.rcVertexIrradiance[1] = postfixWeight * lightPdf / misWeight;
                pathReservoir.insertRcLightType(lightType);
                pathReservoir.rcLightPdf = lightPdf;
                pathReservoir.rcVertexWi[1] = wi;

            }
            else
            {
                pathReservoir.rcVertexWi[0] = rcVertexWi[0];
                pathReservoir.rcVertexIrradiance[0] = postfixWeight;
            }
            pathReservoir.rcVertexHit = rcVertexHit;

            pathReservoir.pathFlags = pathFlags;
            pathReservoir.pathFlags.insertPathLength(pathLength); // excluding the NEE vertex
            pathReservoir.pathFlags.insertRcVertexLength(rcVertexLength);

            selected = true;
#else
#if SHARC_ENABLE
            bool grouped = useHybridShift && !forceAdd && rcVertexLength < kMaximumPathLength && pathLength > rcVertexLength;
            if (grouped)
            {
                offerSuffixEvent(pathLength, pathWeight, postfixWeight, russianroulettePDF, true, false, misWeight, lightPdf, lightType);
                return false;
            }
#endif
            if (forceAdd || pathReservoir.add(pathWeight, russianroulettePDF, sg))
            {
                selected = true;
                pathReservoir.pathFlags = pathFlags;
                pathReservoir.pathFlags.insertPathLength(pathLength); // excluding the NEE vertex
                pathReservoir.pathFlags.insertRcVertexLength(rcVertexLength);
                pathReservoir.pathFlags.insertLastVertexNEE(true);
#if SHARC_ENABLE
                pathReservoir.pathFlags.setSuffixEvent(grouped, false);
#endif
                pathReservoir.rcVertexIrradiance[0] = postfixWeight;
                pathReservoir.F = pathWeight;
                pathReservoir.rcVertexWi[0] = rcVertexWi[0];
                pathReservoir.rcVertexHit = rcVertexHit;

                pathReservoir.cachedJacobian = cachedJacobian;

                pathReservoir.rcRandomSeed = cachedRandomSeed;

                if (is_rcVertex) // at rcVertex
                {
                    pathReservoir.rcVertexIrradiance[0] = pathReservoir.rcVertexIrradiance[0] * lightPdf / misWeight;
                    pathReservoir.rcVertexWi[0] = wi;
                }
                pathReservoir.lightPdf = lightPdf;
                pathReservoir.pathFlags.insertLightType(lightType);
            }
#endif
        }

        return selected;
    }

    [mutating] void markEscapeVertexAsRcVertex(RestirPathTracerParams params, uint pathLength, inout PathReservoir pathReservoir, HitInfo hit, bool isDelta,
        bool isTransmission, bool isSpecularBounce, float lightPdf, uint lightType, float3 rcVertexIrradiance, float3 rcVertexWi, float prevScatterPdf, float geometryFactor)
    {
        pathReservoir.pathFlags.insertRcVertexLength(pathLength);

        pathReservoir.rcVertexHit.initFromHitInfo(hit);
        pathReservoir.pathFlags.insertIsDeltaEvent(isDelta, true);
        pathReservoir.pathFlags.insertIsTransmissionEvent(isTransmission, true);
        pathReservoir.pathFlags.insertIsSpecularBounce(isSpecularBounce, true);
        pathReservoir.lightPdf = lightPdf;
        pathReservoir.cachedJacobian.x = prevScatterPdf;
        pathReservoir.pathFlags.insertLightType(lightType);
        pathReservoir.rcVertexIrradiance[0] = rcVertexIrradiance;
        pathReservoir.rcVertexWi[0] = rcVertexWi;
        pathReservoir.cachedJacobian.z = geometryFactor;
    }
};

