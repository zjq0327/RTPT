/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/

#include "Utils/Math/MathConstants.slangh"

import PathReservoir;
import Utils.Sampling.SampleGenerator;
import LoadShadingData;
import Params;
import Utils.Debug.PixelDebug;
import Rendering.Materials.MaterialShading;
import Rendering.Utils.PixelStats;
import Scene.RaytracingInline;
import Scene.Scene;
import PathTracer;
import PGDiffuse;
import SharcQueryParams;
import RenderPasses.Shared.Denoising.NRDData;
#ifndef PG_TEMPORAL_STATS
#define PG_TEMPORAL_STATS 0
#endif
#ifndef PG_SPATIAL_STATS
#define PG_SPATIAL_STATS 0
#endif
#if PG_TEMPORAL_STATS
RWStructuredBuffer<uint> pgTemporalCounts;
#define PG_SHIFT_COUNTS pgTemporalCounts
#elif PG_SPATIAL_STATS
RWStructuredBuffer<uint> pgSpatialCounts;
#define PG_SHIFT_COUNTS pgSpatialCounts
#endif

ParameterBlock<PathTracer> gPathTracer;

bool isJacobianInvalid(float Jacobian)
{
    return Jacobian <= 0.f || isnan(Jacobian) || isinf(Jacobian);
}

// Only the diffuse-reflection term changes. A sampled S component needs the
// proposal table only if the caller also consumes the full NEE competitor.
float4 lookupReuseScatterMass<let IsTemporalReuse : int>(inout float4 cachedMass, bool centerPrimary, ShadingData sd, uint lobes, bool needMarginal,
    bool reverseEvaluation, uint scatterDepth = 0u)
{
    if (pgDiffuseFraction == 0.f || (!needMarginal && (lobes & uint(SampledBSDFFlags::DiffuseReflection)) == 0u))
        return 0.f;
    float4 mass;
    // A negative first component is unread. Zero is a cached miss.
    if (IsTemporalReuse == 0 && scatterDepth == 0u && cachedMass.x >= 0.f) mass = cachedMass;
    else
    {
#if PG_SPATIAL_STATS
        bool requestedGrid;
        mass = lookupDiffusePG(sd, requestedGrid);
        if (scatterDepth == 0u && requestedGrid) InterlockedAdd(pgSpatialCounts[centerPrimary ? 8u : 9u], 1u);
#else
        mass = lookupDiffusePG(sd);
#endif
        if (IsTemporalReuse == 0 && scatterDepth == 0u) cachedMass = mass;
    }
#if PG_TEMPORAL_STATS || PG_SPATIAL_STATS
    if (any(mass > 0.f)) InterlockedAdd(PG_SHIFT_COUNTS[reverseEvaluation ? 3u : 2u], 1u);
#endif
    return mass;
}

float evalReuseScatterPdf<let IsTemporalReuse : int>(inout float4 cachedMass, bool centerPrimary, ShadingData sd, float3 wi,
    uint lobes, bool needMarginal,
    bool reverseEvaluation, out float pdfAll, uint scatterDepth = 0u)
{
    float4 mass = lookupReuseScatterMass<IsTemporalReuse>(cachedMass, centerPrimary, sd, lobes, needMarginal, reverseEvaluation, scatterDepth);
    return evalPdfWithDiffusePG(sd, wi, mass, pdfAll, lobes);
}

float evalReuseScatterPdf<let IsTemporalReuse : int>(ShadingData sd, float3 wi, uint lobes, bool needMarginal,
    bool reverseEvaluation, out float pdfAll, uint scatterDepth = 0u)
{
    float4 unread = -1.f;
    return evalReuseScatterPdf<IsTemporalReuse>(unread, false, sd, wi, lobes, needMarginal, reverseEvaluation, pdfAll, scatterDepth);
}

// dstPdf * dstJacobian transforms pdf in dst space to src space
// srcPdf / dstJacobian transforms pdf in src space to dst space
float3 computeShiftedIntegrandReconnection<let IsTemporalReuse : int>(inout float4 dstMass, inout float4 srcMass,
                                const RestirPathTracerParams params, inout float dstJacobian, ShadingData dstPrimarySd,
                                ShadingData srcPrimarySd, inout PathReservoir srcReservoir, bool evalVisibility, bool reverseEvaluation)
{
    dstJacobian = 0.f;

    const int rcVertexLength = 1;

    HitInfo rcVertexHit = srcReservoir.rcVertexHit.getHitInfo();
    float3 rcVertexIrradiance = srcReservoir.rcVertexIrradiance[0];
    float3 rcVertexWi = srcReservoir.rcVertexWi[0];
    bool rcVertexHitExists = rcVertexHit.isValid();

    bool isTransmission = srcReservoir.pathFlags.decodeIsTransmissionEvent(true);
    uint allowedSampledTypes1 = getAllowedBSDFFlags(srcReservoir.pathFlags.decodeIsSpecularBounce(true));

    // The grid key uses the original surface position, as in Generate.
    float3 srcOrigin = srcPrimarySd.computeNewRayOrigin(!isTransmission);
    float3 dstOrigin = dstPrimarySd.computeNewRayOrigin(!isTransmission);

    if (!rcVertexHitExists) 
    {
        float3 dstIntegrand = 0.f;
        // are we having a infinite light as rcVertex?
        if (kUseMIS && srcReservoir.pathFlags.lightType() == (uint)PathTracer::LightSampleType::EnvMap && srcReservoir.pathFlags.pathLength() + 1 == rcVertexLength && !srcReservoir.pathFlags.lastVertexNEE())
        {
            float3 wi = rcVertexWi;
            bool isVisible = evalSegmentVisibility(dstOrigin, wi, true); // test along a direction
            if (isVisible)
            {
                float dstPDF1All;
                float srcPDF1;
                float dstPDF1;
                if (pgEnabled)
                {
                    float unused;
                    srcPDF1 = pgDiffuseFraction > 0.f
                        ? evalReuseScatterPdf<IsTemporalReuse>(srcMass, reverseEvaluation, srcPrimarySd, wi,
                            allowedSampledTypes1, false, reverseEvaluation, unused)
                        : evalPdfBSDF(srcPrimarySd, wi);
                    dstPDF1 = evalReuseScatterPdf<IsTemporalReuse>(dstMass, !reverseEvaluation, dstPrimarySd, wi,
                        allowedSampledTypes1, true, reverseEvaluation, dstPDF1All);
                }
                else
                {
                    srcPDF1 = evalPdfBSDF(srcPrimarySd, wi);
                    dstPDF1 = evalPdfBSDF(dstPrimarySd, wi, dstPDF1All, allowedSampledTypes1);
                }
                float3 dstF1 = evalBSDFCosine(dstPrimarySd, wi, allowedSampledTypes1);
                float misWeight = PathTracer::evalMIS(1, dstPDF1All, 1, srcReservoir.lightPdf);//   dstPDF1 / (dstPDF1 + srcReservoir.lightPdf);
                dstIntegrand = dstF1 / dstPDF1 * misWeight * rcVertexIrradiance;
                dstJacobian = dstPDF1 / srcPDF1;
            }
        }

        // fill in rcVertex0 information
        if (isJacobianInvalid(dstJacobian)) dstJacobian = 0.f;
        if (any(isnan(dstIntegrand) || isinf(dstIntegrand))) return 0.f;

        return dstIntegrand;
    }

    bool isRcVertexFinal = srcReservoir.pathFlags.pathLength() == rcVertexLength;
    bool isRcVertexEscapedVertex = srcReservoir.pathFlags.pathLength() + 1 == rcVertexLength && !srcReservoir.pathFlags.lastVertexNEE();
    bool isRcVertexNEE = isRcVertexFinal && srcReservoir.pathFlags.lastVertexNEE();

    bool isDelta1 = srcReservoir.pathFlags.decodeIsDeltaEvent(true);
    bool isDelta2 = srcReservoir.pathFlags.decodeIsDeltaEvent(false);

    // delta bounce before/after rcVertex (if isRcVertexNEE, deltaAfterRc won't be set)
    if (isDelta1 || isDelta2) return 0.f;

    float rcLod = pgEnabled && pgDiffuseFraction > 0.f ? params.lodBias : 0.f;
    ShadingData rcVertexSd = loadShadingDataWithPrevVertexPosition(rcVertexHit, dstOrigin, false, rcLod);

    // need to evaluate source PDF of BSDF sampling
    float3 dstConnectionV = -rcVertexSd.V; // direction point from dst primary hit point to reconnection vertex
    float3 srcConnectionV = normalize(rcVertexSd.posW - srcOrigin);

    float3 shiftedDisp = rcVertexSd.posW - dstOrigin;
    float shifted_dist2 = dot(shiftedDisp, shiftedDisp);
    float shifted_cosine = abs(dot(rcVertexSd.faceN, -dstConnectionV));

    float geometry = shifted_cosine / shifted_dist2;
    float3 originalDisp = rcVertexSd.posW - srcOrigin;
    float original_dist2 = dot(originalDisp, originalDisp);
    float original_cosine = abs(dot(rcVertexSd.faceN, -srcConnectionV));
    float Jacobian = geometry * original_dist2 / original_cosine;
    if (isJacobianInvalid(Jacobian)) return 0.f;

    // assuming BSDF sampling
    assert(kUseBSDFSampling);

    float3 dstWoLocal = dstPrimarySd.toLocal(dstPrimarySd.V);
    float3 dstWiLocal = dstPrimarySd.toLocal(dstConnectionV);
    FalcorBSDF dstBSDF;
    dstBSDF.setup(dstPrimarySd);

    float dstPDF1All = 0.f;
    float dstPDF1;
    float srcPDF1;
    if (pgEnabled)
    {
        float4 dstGuideMass = lookupReuseScatterMass<IsTemporalReuse>(dstMass, !reverseEvaluation, dstPrimarySd,
            allowedSampledTypes1, isRcVertexEscapedVertex, reverseEvaluation);
        dstPDF1 = evalPdfWithDiffusePG(dstBSDF, dstPrimarySd, dstConnectionV, dstWoLocal, dstWiLocal, dstGuideMass,
            dstPDF1All, allowedSampledTypes1);
        float unused;
        srcPDF1 = evalReuseScatterPdf<IsTemporalReuse>(srcMass, reverseEvaluation, srcPrimarySd, srcConnectionV,
            allowedSampledTypes1, false, reverseEvaluation, unused);
    }
    else
    {
        dstPDF1 = dstBSDF.evalPdfAll(dstWoLocal, dstWiLocal, dstPDF1All, allowedSampledTypes1);
        srcPDF1 = evalPdfBSDF(srcPrimarySd, srcConnectionV, allowedSampledTypes1);
    }

    Jacobian *= dstPDF1 / srcPDF1;

    if (isJacobianInvalid(Jacobian)) return 0.f;

    float3 dstF1 = dstBSDF.eval(dstWoLocal, dstWiLocal, allowedSampledTypes1);

    float dstRcVertexScatterPdfAll = 0.f;
    float dstPDF2 = 1.f;
    float dstRcVertexScatterPdf = 1.f; 
    float srcRcVertexScatterPdf = 1.f;  

    uint allowedSampledTypes2 = isRcVertexNEE ? -1 : getAllowedBSDFFlags(srcReservoir.pathFlags.decodeIsSpecularBounce(false));

    if (!isRcVertexEscapedVertex)
    {
        if (pgEnabled)
        {
            bool needsLightMIS = isRcVertexFinal && kUseMIS &&
                srcReservoir.pathFlags.lightType() != uint(PathTracer::LightSampleType::Analytic);
            dstRcVertexScatterPdf = evalReuseScatterPdf<IsTemporalReuse>(rcVertexSd, rcVertexWi, allowedSampledTypes2,
                needsLightMIS, reverseEvaluation, dstRcVertexScatterPdfAll, rcVertexLength);
            if (!isRcVertexNEE)
            {
                ShadingData srcRcSd = rcVertexSd;
                srcRcSd.V = -srcConnectionV;
                srcRcSd.NdotV = dot(srcRcSd.N, srcRcSd.V);
                // Opposite-side and view-adjusted normals require a source-side frame.
                bool sourceFrontFacing = dot(srcRcSd.faceN, srcRcSd.V) >= 0.f;
                if (pgDiffuseFraction > 0.f && (kAdjustShadingNormals || sourceFrontFacing != srcRcSd.frontFacing))
                    srcRcSd = loadShadingDataWithPrevVertexPosition(rcVertexHit, srcOrigin, false, rcLod);
                float unused;
                srcRcVertexScatterPdf = evalReuseScatterPdf<IsTemporalReuse>(srcRcSd, rcVertexWi, allowedSampledTypes2,
                    false, reverseEvaluation, unused, rcVertexLength);
            }
        }
        else
        {
            dstRcVertexScatterPdf = evalPdfBSDF(rcVertexSd, rcVertexWi, dstRcVertexScatterPdfAll, allowedSampledTypes2);
            srcRcVertexScatterPdf = evalPdfBSDFWithV(rcVertexSd, -srcConnectionV, rcVertexWi, allowedSampledTypes2);
        }

        if (!isRcVertexNEE) dstPDF2 = dstRcVertexScatterPdf;
        else dstPDF2 = srcReservoir.lightPdf;
    }

    float3 dstF2 = 1.f;

    if (!isRcVertexEscapedVertex)
        dstF2 = evalBSDFCosine(rcVertexSd, rcVertexWi, allowedSampledTypes2);

    // connection point behind surface
    if (all(dstF1 == 0.f) || all(dstF2 == 0.f)) return 0.f;

    //////
    float3 dstIntegrandNoF1 = dstF2 / dstPDF2 * rcVertexIrradiance;
    float3 dstIntegrand = dstF1 / dstPDF1 * dstIntegrandNoF1; // TODO: might need to reevaluate Le for changing emissive lights

    if (isRcVertexEscapedVertex)
    {
        float misWeight = PathTracer::evalMIS(1, dstPDF1All, 1, srcReservoir.lightPdf);// dstPDF1 / (srcReservoir.lightPdf + dstPDF1);
        dstIntegrand *= misWeight;
    }

    // MIS weight
    if (isRcVertexFinal && kUseMIS)
    {
        if (srcReservoir.pathFlags.lightType() != (uint)PathTracer::LightSampleType::Analytic) // TODO: optimize way this check
        {
            float lightPdf = srcReservoir.lightPdf;
            float misWeight = PathTracer::evalMIS(1, isRcVertexNEE ? lightPdf : dstRcVertexScatterPdfAll, 1, isRcVertexNEE ? dstRcVertexScatterPdfAll : lightPdf);
            dstIntegrand = dstIntegrand * misWeight; 
            dstIntegrandNoF1 = dstIntegrandNoF1 * misWeight;
            if (!isRcVertexNEE)
                Jacobian *= dstRcVertexScatterPdf / srcRcVertexScatterPdf;
        }
    }

    // need to account for non-identity jacobian due to BSDF sampling
    if (!isRcVertexFinal && !isRcVertexEscapedVertex)
    {
        Jacobian *= dstRcVertexScatterPdf / srcRcVertexScatterPdf;
    }

    if (isJacobianInvalid(Jacobian)) return 0.f;

    // Evaluate visibility: vertex 1 <-> vertex 2 (reconnection vertex).
    if (evalVisibility)
    {
        bool isVisible = evalSegmentVisibility(dstOrigin, rcVertexSd.posW);
        if (!isVisible)
            return 0.f;
    }

    if (any(isnan(dstIntegrand) || isinf(dstIntegrand))) return 0.f;

    if (params.rejectShiftBasedOnJacobian)
    {
        if (Jacobian > 0.f && (max(Jacobian, 1 / Jacobian) > 1 + params.jacobianRejectionThreshold))
        {
            // discard based on Jacobian (unbiased)
            Jacobian = 0.f;
            dstIntegrand = 0.f;
        }
    }

    dstJacobian = Jacobian;

    return dstIntegrand;
}

float3 computeShiftedIntegrand<let IsTemporalReuse : int>(inout float4 dstMass, inout float4 srcMass,
    const RestirPathTracerParams params, out float dstJacobian, const ShadingData dstPrimarySd,
    const ShadingData srcPrimarySd, PathReservoir srcReservoir, bool evalVisibility, bool reverseEvaluation = false)
{
    dstJacobian = 0.f;
    if (srcReservoir.weight == 0.f) return 0.f;
    // A black hit or exhausted depth budget cannot win a connection from either primary.
    if (IsTemporalReuse != 0 && gEnableSharcTemporalQuery && srcReservoir.pathFlags.isSharcTerminal() &&
        all(srcReservoir.rcVertexIrradiance[0] == 0.f)) return 0.f;
#if PG_TEMPORAL_STATS || PG_SPATIAL_STATS
    InterlockedAdd(PG_SHIFT_COUNTS[reverseEvaluation ? 1u : 0u], 1u);
#endif
    float3 integrand = computeShiftedIntegrandReconnection<IsTemporalReuse>(dstMass, srcMass, params, dstJacobian, dstPrimarySd,
        srcPrimarySd, srcReservoir, evalVisibility, reverseEvaluation);
#if PG_TEMPORAL_STATS || PG_SPATIAL_STATS
    if (PathReservoir::toScalar(integrand) * dstJacobian > 0.f)
        InterlockedAdd(PG_SHIFT_COUNTS[reverseEvaluation ? 5u : 4u], 1u);
#endif
    return integrand;
}

float3 computeShiftedIntegrand<let IsTemporalReuse : int>(const RestirPathTracerParams params, out float dstJacobian, const ShadingData dstPrimarySd,
    const ShadingData srcPrimarySd, PathReservoir srcReservoir, bool evalVisibility, bool reverseEvaluation = false)
{
    float4 dstMass = -1.f, srcMass = -1.f;
    return computeShiftedIntegrand<IsTemporalReuse>(dstMass, srcMass, params, dstJacobian, dstPrimarySd, srcPrimarySd,
        srcReservoir, evalVisibility, reverseEvaluation);
}

bool shiftAndMergeReservoir<let IsTemporalReuse : int>(inout float4 dstMass, inout float4 srcMass,
    const RestirPathTracerParams params, bool temporalUpdateForDynamicScene, out float dstJacobian,
    const ShadingData dstPrimarySd, inout PathReservoir dstReservoir, const ShadingData srcPrimarySd, PathReservoir srcReservoir,
    bool evalVisibility, inout SampleGenerator sg, float misWeight = 1.f, bool forceMerge = false)
{
    const bool replayHistory = IsTemporalReuse != 0 && temporalUpdateForDynamicScene && srcReservoir.weight != 0.f;
    float survivalJacobian = 1.f;
    if (replayHistory)
    {
#if PG_TEMPORAL_STATS
        InterlockedAdd(pgTemporalCounts[6], 1u);
#endif
        survivalJacobian = gPathTracer.traceTemporalUpdate(dstPrimarySd, srcReservoir);
    }
    float3 dstIntegrand = computeShiftedIntegrand<IsTemporalReuse>(dstMass, srcMass, params, dstJacobian, dstPrimarySd, srcPrimarySd,
        srcReservoir, evalVisibility);
    dstJacobian *= survivalJacobian;
    bool selected = dstReservoir.merge(dstIntegrand, dstJacobian, srcReservoir, sg, misWeight, forceMerge);

    if (forceMerge)
    {
        if (!selected) dstReservoir.F = 0.f;
        dstReservoir.M = srcReservoir.M;
        dstReservoir.weight = srcReservoir.weight;
    }
    return selected;
}

bool shiftAndMergeReservoir<let IsTemporalReuse : int>(const RestirPathTracerParams params, bool temporalUpdateForDynamicScene, out float dstJacobian,
    const ShadingData dstPrimarySd, inout PathReservoir dstReservoir, const ShadingData srcPrimarySd, PathReservoir srcReservoir,
    bool evalVisibility, inout SampleGenerator sg, float misWeight = 1.f, bool forceMerge = false)
{
    float4 dstMass = -1.f, srcMass = -1.f;
    return shiftAndMergeReservoir<IsTemporalReuse>(dstMass, srcMass, params, temporalUpdateForDynamicScene, dstJacobian, dstPrimarySd,
        dstReservoir, srcPrimarySd, srcReservoir, evalVisibility, sg, misWeight, forceMerge);
}

bool mergeReservoirNoResampling(const RestirPathTracerParams params, inout PathReservoir dstReservoir, const PathReservoir srcReservoir, inout SampleGenerator sg)
{
    bool selected = dstReservoir.merge(srcReservoir.F, 1.f, srcReservoir, sg, 1.f);
    return selected;
}

bool mergeReservoirWithResamplingMIS(const RestirPathTracerParams params, float3 dstIntegrand, float dstJacobian, inout PathReservoir dstReservoir, const PathReservoir tempDstReservoir, const PathReservoir srcReservoir, inout SampleGenerator sg, bool isSpatialReuse, float misWeight = 1.f)
{
    bool selected = dstReservoir.mergeWithResamplingMIS(dstIntegrand, dstJacobian, tempDstReservoir, sg, misWeight);
    return selected;
}

bool mergeReservoir(const RestirPathTracerParams params, float3 dstIntegrand, float dstJacobian, inout PathReservoir dstReservoir, const PathReservoir tempDstReservoir, const PathReservoir srcReservoir, inout SampleGenerator sg, bool isSpatialReuse, float misWeight = 1.f)
{
    bool selected = dstReservoir.merge(dstIntegrand, dstJacobian, tempDstReservoir, sg, misWeight);
    return selected;
}

/// Evaluates the visibility term to the reconnection vertex from the source vertex.
bool evalSegmentVisibility(const float3 sourceVertex, const float3 targetVertex, bool isDirection = false)
{
    float3 dir;
    float tMax;
    if (isDirection)
    {
        dir = targetVertex;
        tMax = kRayTMax;
    }
    else
    {
        float3 edge = targetVertex - sourceVertex;
        tMax = length(edge);
        dir = edge / tMax;
        tMax *= 0.999f; // prevent intersection with another surface
    }

    RayDesc ray = { sourceVertex, 0.f, dir, tMax }; // TODO: use offset position instead

    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
}

bool traceVisibilityRay(const Ray ray)
{
    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    return sceneRayQuery.traceVisibilityRay(ray.toRayDesc(), RAY_FLAG_NONE, 0xff);
}

