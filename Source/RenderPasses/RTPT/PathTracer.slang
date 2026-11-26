/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#include "Scene/Material/MaterialDefines.slangh"

import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.ClothMaterial;
import Rendering.Materials.HairMaterial;
import Rendering.Materials.MaterialShading;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Volumes.PhaseFunction;
import Rendering.Utils.PixelStats;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDHelpers;
import RenderPasses.Shared.Denoising.NRDConstants;
import NRDHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import LoadShadingData;
import PathBuilder;
import PathReservoir;

#include "Utils/Math/MathConstants.slangh"

__exported import PathState;
__exported import Params;

#ifndef SHARC_TRAINING
#define SHARC_TRAINING 0
#endif
#ifndef SHARC_GENERATE_QUERY
#define SHARC_GENERATE_QUERY 0
#endif
#ifndef SHARC_GENERATE_STATS
#define SHARC_GENERATE_STATS 0
#endif
#ifndef SHARC_TEMPORAL_QUERY
#define SHARC_TEMPORAL_QUERY 0
#endif
#ifndef SHARC_TEMPORAL_STATS
#define SHARC_TEMPORAL_STATS 0
#endif
#ifndef PG_GENERATE_GUIDING
#define PG_GENERATE_GUIDING 0
#endif
#if PG_GENERATE_GUIDING
import PGDiffuse;
#endif
#if SHARC_GENERATE_QUERY || SHARC_GENERATE_STATS
import TracePassParams;
#endif
#if SHARC_TEMPORAL_QUERY
import SharcTemporalParams;
#endif

uint getAllowedBSDFFlags(bool isSpecularBounce)
{
    if (kSeparatePathBSDF) return isSpecularBounce ? 0xc : 0x3;
    else return -1;
}

struct PathTracer
{
    RestirPathTracerParams params;                  ///< Runtime parameters.
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.

    NRDBuffers outputNRD;                           ///< Output NRD data.
    RWTexture2D<float4> outputNRDDiffuseRadianceHitDist;    ///< Output resolved diffuse color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> outputNRDSpecularRadianceHitDist;   ///< Output resolved specular color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> outputNRDResidualRadianceHitDist;   ///< Output resolved residual color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.

    Texture2D<float4> directDiffuseRadianceHitDist;
    Texture2D<float4> directSpecularRadianceHitDist;
    RWTexture2D<float4> nrdDeltaDirect;        ///< Per-pixel direct delta radiance for the current ReSTIR iteration.
    RWTexture2D<float2> nrdGuideWeights;       ///< Brightest D/S guide weight across samples when spp > 1.
    bool useSplitDirectLighting;
    Texture2D<float4> directEmission;
    bool hasDirectEmission;

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.
    RWTexture2D<float4> outputDebug;                ///< Output debug data. Valid if OUTPUT_DEBUG is nonzero.
    RWTexture2D<uint> outputTime;                   ///< Output timing data. Valid if OUTPUT_TIME is nonzero.

    Texture2D<float4> directLighting;                   ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.

    RWStructuredBuffer<PathReservoir> outputReservoirs;            ///< Output paths from the path tracing pass.
    bool isLastRound;
    bool useDirectLighting;
    int  gSppId;
    int  gNumSpatialRounds;

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    //static bool kUseEnvLight = USE_ENV_LIGHT;
    //static bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    //static bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    bool kUseEnvLight;
    bool kUseEmissiveLights;
    bool kUseAnalyticLights;

    // Additional specialization.
    static const bool kOutputTime = OUTPUT_TIME;

    /** Types of samplable lights.
    */
    enum class LightSampleType
    {
        EnvMap, //0
        Emissive, //1
        Analytic //2
    };

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint    lightType;  ///< Light type this sample comes from (LightSampleType casted to uint).

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 normal;      ///< Shading normal at the vertex (zero if not on a surface).
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).

        /** Create a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] normal Shading normal.
            \param[in] faceNormal Geometry normal.
            \return Returns a path vertex.
        */
        static PathVertex create(uint index, float3 pos, float3 normal = float3(0.f), float3 faceNormal = float3(0.f))
        {
            PathVertex v = {};
            v.index = index;
            v.pos = pos;
            v.normal = normal;
            v.faceNormal = faceNormal;
            return v;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }
    };

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.thp = float3(1.f);
        path.prefixThp = float3(1.f);

        path.LDeltaDirect = float3(0.f);
        path.enableRandomReplay = false;

        // Load the primary hit info from the V-buffer.
        // If invalid, the path is still active and treated as a miss.
        uint2 pixel = path.getPixel();
        const HitInfo hit = HitInfo(vbuffer[pixel]);

        if (hit.isValid())
        {
            path.setHit(hit);
            //path.setVertexIndex(1);
        }

        // TODO: Support depth-of-field using computeRayThinlens, if aperture size > 0.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        // Create sample generator.
        static const uint itersPerShaderPass = kCandidateSamples;

        path.sg = SampleGenerator(pixel, (itersPerShaderPass + 1 + gNumSpatialRounds) * params.seed + path.getSampleIdx());

        path.pathBuilder.init(path.sg);
        path.pathReservoir.init();
    }

    // generate random replay path
    void generateRandomReplayPath(out PathState path, HitInfo hit, Ray ray, uint randomSeed, int randomReplayLength, bool isLastVertexNEE, bool isRcVertexEscapedVertex)
    {
        path = {};
        path.L = 0.f;
        path.setActive();
        path.thp = float3(1.f);
        path.prefixThp = float3(1.f);

        path.enableRandomReplay = true;
        path.setHit(hit);

        path.origin = ray.origin;
        path.dir = ray.dir;

        path.sg = SampleGenerator(randomSeed); 

        path.randomReplayIsNEE = isLastVertexNEE && !isRcVertexEscapedVertex;
        path.randomReplayIsEscaped = !isLastVertexNEE && !isRcVertexEscapedVertex;
        path.randomReplayLength = randomReplayLength;

        path.pathReservoir.init();
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Update the path throughouput
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, inout PathState path, float4 pgMass = 0.f)
    {
        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Generate a BSDF sample.
        bool valid;
        BSDFSample_ result;

        if (kUseBSDFSampling)
        {
#if PG_GENERATE_GUIDING
            valid = sampleWithDiffusePG(sd, path.sg, pgMass, !kSeparatePathBSDF, result);
#else
            valid = sampleBSDF(sd, path.sg, result, !kSeparatePathBSDF);
#endif
        }
        else valid = sampleBSDF_Reference(sd, path.sg, result);

        float pdfAll = result.pdf;
        if (kSeparatePathBSDF) result.pdf = result.pdfSingle;

        // Ignore valid on purpose for now.
        if (kOutputNRDData && path.length == 0)
        {
            path.setDiffusePrimaryHit(result.isLobe(LobeType::Diffuse));
            path.setSpecularPrimaryHit(result.isLobe(LobeType::SpecularOrDelta));
            if (!kSeparatePathBSDF)
                path.pathBuilder.pathFlags.setNRDPrimarySpecular(path.isSpecularPrimaryHit());
        }

        if (valid)
        {
            // store sampled path type (diffuse or non-diffuse)

            path.dir = result.wi;

            // add to path buffer
            // also multiply pdf

            path.thp *= result.weight;

            // this is pdfAll
            path.pdf = pdfAll;

            // daqi: save the Jacobian for rcVertex
            if (path.length == path.pathBuilder.rcVertexLength)
            {
                // TODO: use component PDF
                path.pathBuilder.cachedJacobian.y = result.pdf;
            }

            path.prevScatterPdf = result.pdf;

            if (result.pdf <= 0.f) // invalid
            {
                return false;
            }

#if _USE_LEGACY_SHADING_CODE
            // Legacy BRDF code sometimes returns infinite PDF values.
            // We clamp them to avoid getting NaNs from the evalMIS function.
            path.pdf = min(path.pdf, 1e30f);
#endif

            path.clearEventFlags();
#if SHARC_TRAINING || SHARC_GENERATE_QUERY || SHARC_TEMPORAL_QUERY
            path.previousRoughness = sd.linearRoughness;
            path.previousDiffuse = result.isLobe(LobeType::DiffuseReflection);
#endif

            // Handle reflection events.

            if (result.isLobe(LobeType::Reflection))
            {
                // We classify specular events as diffuse if the roughness is above some threshold.
                bool isDiffuse = result.isLobe(LobeType::DiffuseReflection) || sd.linearRoughness > params.specularRoughnessThreshold;
                if (isDiffuse)
                {
                    path.incrementBounces(BounceType::Diffuse);
                }
                else
                {
                    path.incrementBounces(BounceType::Specular);
                    path.setSpecular();
                }
            }

            // Handle delta events.
            if (result.isLobe(LobeType::Delta))
            {
                path.setDelta();
            }

            if (result.isLobe(LobeType::SpecularOrDelta))
            {
                path.setSpecularBounce();
            }

            // daqi: record delta and transmission event before/after rcVertex, this is important because such information will be used in computing the Jacobian in shift.slang
            if (path.length == path.pathBuilder.rcVertexLength)
            {
                path.pathBuilder.pathFlags.insertIsDeltaEvent(result.isLobe(LobeType::Delta), false);
                path.pathBuilder.pathFlags.insertIsTransmissionEvent(result.isLobe(LobeType::Transmission), false);
                if (kSeparatePathBSDF)
                    path.pathBuilder.pathFlags.insertIsSpecularBounce(path.isSpecularBounce(), false);
            }

            // Handle transmission events.
            if (result.isLobe(LobeType::Transmission))
            {
                path.incrementBounces(BounceType::Transmission);
                path.setTransmission();

                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.isThinSurface())
                {
                    uint nestedPriority = gScene.materials[sd.materialID].getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }

            // Save the shading normal. This is needed for MIS.
            path.normal = sd.N;

            // Mark the path as valid only if it has a non-zero throughput.
            valid = any(path.thp > 0.f);
            // Add to the path gaph.
            // TODO: Handle specular/delta/transmission events.
            if (valid)
            {
                if (path.length <= path.pathBuilder.rcVertexLength)
                    path.recordPrefixThp();
            }
        }
        else
        {
            path.thp = 0.f;
        }

        return valid;
    }

    // TODO: support BySampledBSDF replay
    bool generateScatterRayGivenDirection(const ShadingData sd, float3 dir, inout PathState path)
    {
        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Generate a BSDF sample.
        bool valid = true;
        BSDFSample_ result;

        bool isDeltaEvent = path.pathBuilder.pathFlags.decodeIsDeltaEvent(path.length < 1);
        bool isSpecularBounce = path.pathBuilder.pathFlags.decodeIsSpecularBounce(path.length < 1);

        if (isDeltaEvent) valid = false; // not allowing delta event to be shifted by directional reuse

        if (valid)
        {
            float pdfAll = 0.f; // pdf with all BSDF

            {
                result.wi = dir;

                uint allowedBSDFFlags = getAllowedBSDFFlags(isSpecularBounce);

                result.weight = evalBSDFCosine(sd, dir, allowedBSDFFlags);

                if (kUseBSDFSampling)
                {
                    // this is single pdf
                    result.pdf = evalPdfBSDF(sd, result.wi, pdfAll, allowedBSDFFlags, false);
                }
                else
                {
                    result.pdf = dot(sd.N, result.wi) / M_PI;
                }

                result.weight /= result.pdf;

                if (result.pdf <= 0.f)
                {
                    result.weight = 0.f;
                }
            }

            path.dir = result.wi;

            // add to path buffer
            // also multiply pdf
            path.thp *= result.weight;

            path.pdf = pdfAll;

            if (path.length == path.pathBuilder.rcVertexLength)
            {
                path.pathBuilder.cachedJacobian.y = result.pdf;
            }

            // daqi: record the scatter pdf, we will need to use that later for the cachedJacobian
            path.prevScatterPdf = result.pdf;

#if _USE_LEGACY_SHADING_CODE
            // Legacy BRDF code sometimes returns infinite PDF values.
            // We clamp them to avoid getting NaNs from the evalMIS function.
            path.pdf = min(path.pdf, 1e30f);
#endif

            path.clearEventFlags();

            // daqi: if we use direction reuse, then base and offset path reflection/transmission type is allowed to be different
            bool isTransmissionEvent = dot(sd.N, result.wi) < 0;

            // daqi: set delta/transmission path flags. we don't care about other path flags since our resampling doesn't depend on the sampled BSDF components
            // Handle transmission events.
            if (isTransmissionEvent)
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);
                path.setTransmission();

                // Update interior list and inside volume flag.
                if (!sd.isThinSurface())
                {
                    uint nestedPriority = gScene.materials[sd.materialID].getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }

            if (isSpecularBounce)
            {
                path.setSpecularBounce();
            }

            // Save the shading normal. This is needed for MIS.
            path.normal = sd.N;

            // Mark the path as valid only if it has a non-zero throughput.
            valid = any(path.thp > 0.f);

            // Add to the path gaph.
            // TODO: Handle specular/delta/transmission events.
            if (valid)
            {
                if (path.length <= path.pathBuilder.rcVertexLength)
                    path.recordPrefixThp();
            }
        }

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    static float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere (w.r.t. shading normal) should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;

        // TODO: when random number replay, this can fails with light BVH that previously returns valid samples, why?
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.normal, upperHemisphere, sg, tls)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    void analyticLightSampleSkipRandomNumber(inout SampleGenerator sg)
    {
        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return;
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        LightData light = gScene.getLight(lightIndex);

        switch (light.type)
        {
        case LightType::Rect:
        case LightType::Sphere:
        case LightType::Disc:
        case LightType::Distant:
            sampleNext2D(sg);
        default:
        }
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        The upper/lower hemisphere is defined as the union of the hemispheres w.r.t. to the shading and face normals.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightSampleType::EnvMap)
        {
            valid = generateEnvMapSample(vertex, sg, ls);
        }
        if (kUseEmissiveLights && lightType == (uint)LightSampleType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;

            {
                valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
            }
        }
        if (kUseAnalyticLights && lightType == (uint)LightSampleType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float cosTheta = dot(vertex.normal, ls.dir);
        // Flip the face normal to point in the same hemisphere as the shading normal.
        float3 faceNormal = sign(dot(vertex.normal, vertex.faceNormal)) * vertex.faceNormal;
        float cosThetaFace = dot(faceNormal, ls.dir);
        if (!sampleUpperHemisphere && (max(cosTheta, cosThetaFace) >= -kMinCosTheta)) return false;
        if (!sampleLowerHemisphere && (min(cosTheta, cosThetaFace) <= kMinCosTheta)) return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    void skipLightSampleRandomNumbers(inout SampleGenerator sg)
    {
        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightSampleType::EnvMap)
        {
            if (!kUseEnvLight) valid = false;
            else sampleNext2D(sg);
        }
        if (kUseEmissiveLights && lightType == (uint)LightSampleType::Emissive)
        {
            if (!kUseEmissiveLights) valid = false;
            else emissiveSampler.skipRandomNumber(sg);
        }
        if (kUseAnalyticLights && lightType == (uint)LightSampleType::Analytic)
        {
            analyticLightSampleSkipRandomNumber(sg);
        }
    }

    /** Trace active path to find the next path vertex.
        \param[in,out] path The path state.
    */
    void nextVertex(inout PathState path)
    {
        // Setup ray.
        RayDesc ray = path.getScatterRay();

        logTraceRay(PixelStatsRayType::ClosestHit);
        HitInfo hit;
        float hitT;
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        bool hasHit = sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff);
        if (kOutputNRDData && path.length == 0) path.hitDist = hasHit ? hitT : kNRDInvalidPathLength;
        if (hasHit)
        {
            // Store hit info.
            path.setHit(hit);
            path.length++;
            path.sceneLength += float16_t(hitT);
        }
        else
        {
            path.clearHit();
            path.sceneLength = float16_t(kNRDInvalidPathLength);
        }
    }

    /** Trace a visibility ray to evaluate light visibility.
        \param[in] ray Visibility ray.
        \return true if the light is visible, false if occluded.
    */
    bool traceVisibilityRay(const Ray ray)
    {
        // TODO rename to PixelStatsRayType::Visibility
        logTraceRay(PixelStatsRayType::Shadow);
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray.toRayDesc(), RAY_FLAG_NONE, 0xff);
    }

    /** Handle scattering in homogeneous volumes.
    \return Returns true if a scattering event took place.
*/
    bool handleHomogeneousVolumeExtinction(inout PathState path, const ShadingData sd)
    {
        if (!path.interiorList.isEmpty())
        {
            HomogeneousVolumeSampler hvs = HomogeneousVolumeSampler::create(path.interiorList.getTopMaterialID());
            const float surfaceDistance = length(path.origin - sd.posW);

            // our implementation doesn't support volume scattering
            {
                // Handle volume extinction.
                updatePathThroughput(path, exp(-hvs.sigmaT * surfaceDistance));
            }
        }

        return false;
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Compute relative index of refraction at interface.
        sd.eta = computeRelativeIoR(path.interiorList, sd.materialID, sd.frontFacing);

        // Check for false intersections.
        uint nestedPriority = gScene.materials[sd.materialID].getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeNewRayOrigin(false);
                path.length--;
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Limit specular transmission paths.
        if (kLimitTransmission && sd.specularTransmission > 0.f)
        {
            const uint transmissionBounces = path.getBounces(BounceType::Transmission);
            if (transmissionBounces >= kMaxTransmissionReflectionDepth) sd.setActiveLobes((uint)LobeType::All & ~((uint)LobeType::DeltaReflection | (uint)LobeType::SpecularReflection));
            if (transmissionBounces >= kMaxTransmissionRefractionDepth) sd.eta = 1.f;
        }

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path)
    {
        float rrVal = luminance(path.getCurrentThp());
        const float prob = max(0.f, 1.f - rrVal);
        if (sampleNext1D(path.sg) < prob)
        {
            path.terminate();
            return true;
        }

        path.russianRoulettePdf *= 1.f - prob;

        return false;
    }

    void generateReplayPrimaryScatterRay(inout PathState path, ShadingData sd, float3 dir)
    {
        // Generate the next path segment or terminate.
        bool valid = true;

        if (all(dir == 0.f))
        {
            valid = generateScatterRay(sd, path);
        }
        else // generate with given direction
        {
            valid = generateScatterRayGivenDirection(sd, dir, path);
        }

        if (!valid)
        {
            path.terminate();
        }
    }

    ShadingData loadPathSurface(PathState path)
    {
        if (kPrimaryLodMode == TexLODMode::RayDiffs && path.length == 0 && path.hit.getType() == HitType::Triangle)
        {
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return loadShadingDataFootprint(path.hit, -path.dir, true, ddx, ddy);
        }
        return loadShadingData(path.hit, -path.dir, path.length == 0, path.length > 0 ? params.lodBias : 0.f);
    }

#if SHARC_TRAINING || SHARC_GENERATE_QUERY || SHARC_TEMPORAL_QUERY
    bool allowsSharcDepth(PathState path)
    {
        return path.length >= SHARC_MIN_DEPTH && path.length < kMaximumPathLength &&
            (path.length < 4 || path.previousDiffuse) && !path.isTransmission() && !hasFinishedSurfaceBounces(path);
    }
#endif

    void handleHit(inout PathState path)
    {
        ShadingData sd = loadPathSurface(path);
        if (handleHomogeneousVolumeExtinction(path, sd)) return;
        if (!handleNestedDielectrics(sd, path)) return;

#if SHARC_GENERATE_STATS
        if (path.length >= SHARC_MIN_DEPTH) countSharc(path.length, SharcDepthStat::Arrived);
#endif
#if SHARC_GENERATE_QUERY
        if (!path.enableRandomReplay && allowsSharcDepth(path))
        {
            float3 radiance;
            bool eligible;
            bool hit = gSharcQuery.lookup(sd, path.origin, path.isDelta(), path.previousDiffuse, path.previousRoughness, radiance, eligible);
#if SHARC_GENERATE_STATS
            applySharcDebugMode(path.length, eligible, hit, radiance);
#endif
            if (hit)
            {
                float3 F = path.getCurrentThp() * radiance;
                float3 incident = path.thp * radiance;
                bool selected = path.pathBuilder.addSharcVertex(path.length, F, incident, path.russianRoulettePdf, path.pathReservoir);
                if (selected)
                {
                    path.pathReservoir.pathFlags.setSharcRoot(path.sharcRootBounces & 0xffu,
                        (path.sharcRootBounces >> 8) & 0xffu, path.sharcRootDiffuseLobe, path.sharcPrimaryCountedDiffuse);
                    path.pathReservoir.cachedJacobian.z = path.russianRoulettePdf / path.sharcRootSurvival;
                }
#if SHARC_GENERATE_STATS
                recordSharcHit(path, F, radiance, selected);
#endif
                // Cached radiance covers the remaining path even if an earlier Fresh event wins.
                path.terminate();
                return;
            }
        }
#endif
#if SHARC_TEMPORAL_QUERY
        // Replay the saved number of scattering events, then query the current surface once.
        if (path.replaySharc && path.length == path.randomReplayLength)
        {
            if (allowsSharcDepth(path))
            {
                float3 radiance;
                bool eligible;
#if SHARC_TEMPORAL_STATS
                recordSharcTemporalLookup(path.length);
#endif
                bool hit = gSharcTemporalQuery.lookup(sd, path.origin, path.isDelta(), path.previousDiffuse, path.previousRoughness, radiance, eligible);
#if SHARC_TEMPORAL_STATS
                applySharcTemporalProbe(path.length, eligible, hit, radiance);
#endif
                if (hit) path.pathReservoir.rcVertexIrradiance[0] = path.thp * radiance;
            }
            path.terminate();
            return;
        }
#endif
#if SHARC_GENERATE_STATS
        if (path.length >= SHARC_MIN_DEPTH) countSharc(path.length, SharcDepthStat::Explicit);
#endif

        logPathVertex();

        // Optionally disable emissive lights inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sd.emissive = float3(0.f);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && sd.linearRoughness <= params.specularRoughnessThreshold)
        {
            sd.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Check if this is the primary hit.
        const bool isPrimaryHit = path.length == 0;

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);
#if SHARC_TEMPORAL_QUERY
        if (path.replaySharc) computeEmissive = false;
#endif

        // Disable all constributions to direct illumination if requested. (but we need LDeltaDirect which cannot be sampled by ReSTIR DI)
        if (kDisableDirectIllumination && path.length == 1 && isLightSamplable) computeEmissive = false;

        if (path.length == 1 && !path.isTransmission() && !path.isDelta()) computeEmissive = false;

        // daqi: when doing random number replay, we will terminate the path when the length reaches the base path length and 
        // when the path types match. In this case, if the base path is a escaped path (as opposed to an NEE path), we should terminate when we 
        // find that offset path can also be an escaped path
        bool terminateRandomReplayForEscape = path.enableRandomReplay && path.length - 1 == path.randomReplayLength && path.randomReplayIsEscaped && path.randomReplayLength >= 1;

        if (computeEmissive && any(sd.emissive > 0.f))
        {
            float lightPdf = 0.f;

            float misWeight = 1.f;
            float3 Lr = 0.f;

            if (kUseEmissiveLights && kUseNEE && kUseMIS && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleHit triangleHit = path.hit.getTriangleHit();
                TriangleLightHit hit;
                hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.posW = sd.posW;
                hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                // Compute MIS weight by combining this with BRDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            if (!kDisableDirectIllumination || !isPrimaryHit || !isLightSamplable)
                Lr = path.getCurrentThp() * sd.emissive * misWeight;

            // daqi: Since ScreenSpaceReSTIR cannot handle transmission and delta reflection, we must include the contribution in the color buffer 
            if (!path.enableRandomReplay || terminateRandomReplayForEscape)
            {
                // will change this to path.IsDelta() if ScreenSpaceReSTIR can handle transmission
                if (!isLightSamplable && path.length == 1) path.LDeltaDirect += Lr;
            }

            // daqi: here we are adding the path terminated with an escaped vertex
            if (!path.enableRandomReplay || terminateRandomReplayForEscape)
            {
                path.pathBuilder.addEscapeVertex(params, max(0, path.length - 1), path.dir, Lr, path.thp * sd.emissive * misWeight, 
                    path.russianRoulettePdf, misWeight, lightPdf, (uint)LightSampleType::Emissive, path.pathReservoir, path.enableRandomReplay);
            }
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path) || terminateRandomReplayForEscape)
        {
            path.terminate();
            return;
        }

        // daqi: save as previous path origin to compute distance between vertices
        float3 prevPathOrigin = path.origin;

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Determine if BSDF supports sampling with next-event estimation.
        // The available lobes depend on the material.
        uint lobes = getBSDFLobes(sd);

#if PG_GENERATE_GUIDING
        float4 pgMass = lookupDiffusePG(sd);
#else
        const float4 pgMass = 0.f;
#endif
        bool supportsNEE = (lobes & (uint)LobeType::NonDelta) != 0;

        // Check if we should apply NEE.
        const bool applyNEE = kUseNEE && supportsNEE;

        // daqi: we don't use the integrated ScreenSpaceReSTIR as inlinepathtracer does, instead, we call ScreenSpaceReSTIRPass
        // therefore, functions with applyScreenSpaceReSTIR as condition are removed
        // Check if sample from screen space ReSTIR should be applied instead of NEE.
        //const bool applyScreenSpaceReSTIR = kUseScreenSpaceReSTIR && isPrimaryHit && !isVolumeHit && supportsNEE;

        // Check if lights should be sampled.
        bool sampleLights = applyNEE;

        // daqi: when doing random number replay, we will terminate the path when the length reaches the base path length and 
        // when the path types match. In this case, if the base path is a NEE path (as opposed to an escaped path), we should terminate when we 
        // find that offset path can also be an NEE path
        bool terminateRandomReplayForNEE = path.enableRandomReplay && path.length == path.randomReplayLength && path.randomReplayIsNEE;

        const bool is_rcVertex = path.length == 1;
        if (is_rcVertex)
        {
#if SHARC_GENERATE_QUERY
            path.sharcPrimaryCountedDiffuse = path.getBounces(BounceType::Diffuse) > 0u;
#endif
            path.pathBuilder.rcVertexHit.initFromHitInfo(path.hit);
            // daqi: save the scatter PDF as cachedJacobian
            path.pathBuilder.cachedJacobian.x = path.prevScatterPdf;
            path.pathBuilder.pathFlags.insertIsDeltaEvent(path.isDelta(), true);
            path.pathBuilder.pathFlags.insertIsTransmissionEvent(path.isTransmission(), true);
            if (kSeparatePathBSDF)
                path.pathBuilder.pathFlags.insertIsSpecularBounce(path.isSpecularBounce(), true);

            float3 disp = prevPathOrigin - sd.posW;

            // daqi: save geometry term as part of cachedJacobian
            path.pathBuilder.cachedJacobian.z = abs(dot(sd.faceN, sd.V)) / dot(disp, disp);
            path.pathBuilder.rcVertexLength = path.length;
        }

        path.setLightSampled(false, false);

        // daqi: if we are doing random number replay, we should only sample the light if we are about to terminate
        if (!(kDisableDirectIllumination && isPrimaryHit))
        {
            // generate kNEESamples light samples
            bool drawLight = !path.enableRandomReplay || terminateRandomReplayForNEE;
#if SHARC_TEMPORAL_QUERY
            // Light samplers can consume a variable number of random values.
            // Sample the light to reproduce the scatter seed, then discard it.
            drawLight = drawLight || path.replaySharc;
#endif
            if (sampleLights && drawLight)
            {
                // cache (NEE sampling) seed for temporal validation (only possible for non early direction reuse since it uses the cachedRandomSeed slot)
                if (path.length == path.pathBuilder.rcVertexLength)
                {
                    path.pathBuilder.cachedRandomSeed = path.sg.getCurrentSeed();
                }

                LightSample ls;
                bool validSample = true;
                {
                    // Setup path vertex.
                    PathVertex vertex = PathVertex.create(/*path.getVertexIndex()*/ path.length + 1, sd.posW, sd.N, sd.faceN);

                    // Determine if upper/lower hemispheres need to be sampled.
                    bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
                    if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                    bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

                    // Sample a light.

                    validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);

                    path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
                }

#if SHARC_TEMPORAL_QUERY
                if (path.replaySharc) validSample = false;
#endif
                if (validSample)
                {
                    // Apply MIS weight.
                    float misWeight = 1.f;
                    float scatterPdf = 0.f;

                    if (kUseMIS && ls.lightType != (uint)LightSampleType::Analytic)
                    {
#if PG_GENERATE_GUIDING
                        scatterPdf = evalScatterPdfWithDiffusePG(sd, ls.dir, pgMass);
#else
                        scatterPdf = evalPdfBSDF(sd, ls.dir);
#endif

                        misWeight *= evalMIS(1, ls.pdf, 1, scatterPdf);
                    }
                    ls.Li *= misWeight;

                    float3 weight = evalBSDFCosine(sd, ls.dir);

                    float3 Lr = weight * ls.Li * path.getCurrentThp();

                    if (any(Lr > 0.f))
                    {
                        const Ray ray = ls.getVisibilityRay();
                        bool visible = traceVisibilityRay(ray);
                        if (visible)
                        {

                            if (ls.lightType == (uint)LightSampleType::Analytic && ls.pdf == 0.f)
                            {
                                ls.pdf = getAnalyicSelectionProbability(); // analytic light doesn't have a solid angle PDF.
                            }

                            // The RC-side BSDF is evaluated when reconnecting the selected path.
                            if (is_rcVertex) weight = 1.f;
                            path.pathBuilder.addNeeVertex(params, path.length, ls.dir, Lr, weight * ls.Li * path.thp,
                                path.russianRoulettePdf, misWeight, ls.pdf, ls.lightType, path.pathReservoir, path.enableRandomReplay);
                        }
                    }
                }
            }
            else
                // daqi: for random number replay, we should skip the random numbers for NEE if we should not terminate on current bounce
            {
                if (path.enableRandomReplay && sampleLights)
                {
                    bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
                    if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                    bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
                    path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
                }

                skipLightSampleRandomNumbers(path.sg);
            }
        }

        // daqi: terminate the path if the bounce of random replay reaches the desired bounce and the base path is an NEE path
        if (terminateRandomReplayForNEE)
        {
            path.terminate();
            return;
        }

        // Russian roulette to terminate paths early.
        // TODO: check if this is handled correctly in ReSTIR PT
        if (kUseRussianRoulette && path.length >= 1)
        {
            bool skipRouletteTest = path.enableRandomReplay;
#if SHARC_TEMPORAL_QUERY
            if (path.replaySharc) skipRouletteTest = false;
#endif
            if (skipRouletteTest)
            {
                sampleNext1D(path.sg);
            }
            else
            {
                if (terminatePathByRussianRoulette(path)) return;
            }
        }

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, path, pgMass);

        // cache seed for temporal validation (only possible for non early direction reuse since it uses the cachedRandomSeed slot)
        if (path.length == path.pathBuilder.rcVertexLength)
        {
            path.pathBuilder.cachedRandomSeed = path.sg.getCurrentSeed();
        }

        // daqi: save the incoming direction on rcVertex
        if (is_rcVertex && valid)
        {
            path.pathBuilder.rcVertexWi[0] = path.dir;
#if SHARC_GENERATE_QUERY
            path.sharcRootBounces = path.bounceCounters;
            path.sharcRootDiffuseLobe = path.previousDiffuse;
            path.sharcRootSurvival = path.russianRoulettePdf;
#endif
        }

        if (!path.isActive()) return;

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // Disable all constributions to direct illumination if requested.
        if (kDisableDirectIllumination && path.length == 0 && isLightSamplable) computeEnv = false;

        if (computeEnv)
        {
            logPathVertex();

            float misWeight = 1.f;
            float lightPdf = 0.f;
            if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);

                // Compute MIS weight by combining this with BRDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            float3 Le = envMapSampler.eval(path.dir);
            float3 Lr = path.getCurrentThp() * Le * misWeight;

            // daqi: when doing random number replay, we will terminate the path when the length reaches the base path length and 
            // when the path types match. In this case, if the base path is a escaped path (as opposed to an NEE path), we should terminate when we 
            // find that offset path can also be an escaped path
            bool terminateRandomReplayForEscape = path.enableRandomReplay && path.length == path.randomReplayLength && path.randomReplayIsEscaped
                && path.length >= 1;

            // daqi: Since ScreenSpaceReSTIR cannot handle transmission and delta reflection, we must include the contribution in the color buffer 
            if (!path.enableRandomReplay || terminateRandomReplayForEscape)
            {

                // will change this to path.IsDelta() if ScreenSpaceReSTIR can handle transmission
                if (!isLightSamplable && path.length == 0) path.LDeltaDirect += Lr;
            }

            // daqi: here we are adding the path terminated with an escaped vertex
            if (!path.enableRandomReplay || terminateRandomReplayForEscape)
            {
                path.pathBuilder.addEscapeVertex(params, path.length, path.dir, Lr, path.thp * Le * misWeight, 
                    path.russianRoulettePdf, misWeight, lightPdf, (uint)LightSampleType::EnvMap, path.pathReservoir, path.enableRandomReplay);
            }
        }

        path.terminate();
    }

#if SHARC_TEMPORAL_QUERY
    // Reconstruct the saved scatter at x1, including the throughput used for suffix roulette.
    bool restoreSharcReplayX1(ShadingData rcPrevSd, PathReservoir history, out PathState path)
    {
        path = {};
        const ReSTIRPathFlags flags = history.pathFlags;
        HitInfo hit = history.rcVertexHit.getHitInfo();
        if (flags.rcVertexLength() != 1 || flags.pathLength() < SHARC_MIN_DEPTH || !hit.isValid() ||
            flags.decodeIsDeltaEvent(true) || flags.decodeIsDeltaEvent(false)) return false;

        float3 primaryOrigin = rcPrevSd.computeNewRayOrigin(!flags.decodeIsTransmissionEvent(true));
        ShadingData sd = loadShadingDataWithPrevVertexPosition(hit, primaryOrigin, false, params.lodBias);
        path.setActive();
        path.enableRandomReplay = true;
        path.replaySharc = true;
        path.randomReplayLength = flags.pathLength();
        path.pathBuilder.rcVertexLength = 1;
        path.pathBuilder.pathFlags = flags;
        path.thp = path.prefixThp = 1.f;
        path.russianRoulettePdf = 1.f;
        path.pathReservoir.init();
        path.origin = primaryOrigin;
        if (kUseRussianRoulette || flags.decodeIsTransmissionEvent(true))
        {
            if (!handleNestedDielectrics(rcPrevSd, path) ||
                !generateScatterRayGivenDirection(rcPrevSd, -sd.V, path)) return false;
        }

        path.length = 1;
        path.setHit(hit);
        uint diffuse = flags.sharcRootDiffuseBounces();
        uint specular = flags.sharcRootSpecularBounces();
        if (diffuse + specular > 2u) return false;
        uint priorDiffuse = uint(flags.sharcPrimaryCountedDiffuse());
        path.bounceCounters = priorDiffuse | (uint(flags.decodeIsTransmissionEvent(true)) << 16);
        handleHomogeneousVolumeExtinction(path, sd);
        if (!handleNestedDielectrics(sd, path)) return false;
        if (kDisableCaustics && priorDiffuse > 0 && sd.linearRoughness <= params.specularRoughnessThreshold)
            sd.setActiveLobes((uint)LobeType::Diffuse);

        path.bounceCounters = diffuse | (specular << 8) | ((2u - diffuse - specular) << 16);
        path.previousDiffuse = flags.sharcRootDiffuseLobe();
        path.previousRoughness = sd.linearRoughness;
        path.origin = sd.computeNewRayOrigin(!flags.decodeIsTransmissionEvent(false));
        if (!generateScatterRayGivenDirection(sd, history.rcVertexWi[0], path)) return false;
        path.sg = SampleGenerator(history.rcRandomSeed);
        return true;
    }
#endif

    // Start with the outgoing ray and RNG restored at the reconnection vertex.
    void traceReplaySuffix(inout PathState path)
    {
        while (path.isActive())
        {
#if SHARC_TEMPORAL_QUERY
            if (path.replaySharc && path.length >= path.randomReplayLength) break;
#endif
#if SHARC_TEMPORAL_STATS
            if (path.replaySharc) recordSharcTemporalRay(path.length + 1);
#endif
            nextVertex(path);
            if (!path.isHit())
            {
#if SHARC_TEMPORAL_QUERY
                if (path.replaySharc) break; // A missing cache endpoint has zero contribution.
#endif
                handleMiss(path);
                break;
            }
            handleHit(path);
        }
    }

    float traceTemporalUpdate(ShadingData rcPrevSd, inout PathReservoir tempPathReservoir)
    {
        PathState path;
#if SHARC_TEMPORAL_QUERY
        bool replaySharc = tempPathReservoir.pathFlags.isSharcTerminal();
        if (replaySharc)
        {
#if SHARC_TEMPORAL_STATS
            countSharcTemporalDepth(SharcTemporalDepthStat::Replay, tempPathReservoir.pathFlags.pathLength());
#endif
            tempPathReservoir.rcVertexIrradiance[0] = 0.f;
            if (!restoreSharcReplayX1(rcPrevSd, tempPathReservoir, path)) return 1.f;
        }
        else
#endif
        {
#if SHARC_TEMPORAL_STATS
            if (tempPathReservoir.pathFlags.lastVertexNEE()) countSharcTemporal(SharcTemporalStat::ExplicitNEE);
            else if (tempPathReservoir.pathFlags.lightType() == uint(LightSampleType::Emissive)) countSharcTemporal(SharcTemporalStat::ExplicitEmitter);
            else if (tempPathReservoir.pathFlags.lightType() == uint(LightSampleType::EnvMap)) countSharcTemporal(SharcTemporalStat::ExplicitEnvironment);
#endif

            int rcVertexLength = tempPathReservoir.pathFlags.rcVertexLength();
            ShadingData rcVertexSd = loadShadingDataWithPrevVertexPosition(tempPathReservoir.rcVertexHit.getHitInfo(), rcPrevSd.posW, false);

            //rcVertex is NEE
            if (tempPathReservoir.pathFlags.lastVertexNEE() && tempPathReservoir.pathFlags.pathLength() == rcVertexLength)
            {
                path = {};
                path.length = rcVertexLength;
                // call light sampling function
                LightSample ls;
                bool validSample = true;
                {
                    uint lobes = getBSDFLobes(rcVertexSd);
                    // Setup path vertex.
                    PathVertex vertex = PathVertex.create(/*path.getVertexIndex()*/ path.length + 1, rcVertexSd.posW, rcVertexSd.N, rcVertexSd.faceN);

                    // Determine if upper/lower hemispheres need to be sampled.
                    bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
                    bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

                    path.sg = SampleGenerator(tempPathReservoir.rcRandomSeed);
                    validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
                }

                bool visible = true;
                if (validSample)
                {
                    const Ray ray = ls.getVisibilityRay();
                    visible = traceVisibilityRay(ray);

                    if (ls.lightType == (uint)LightSampleType::Analytic && ls.pdf == 0.f)
                    {
                        ls.pdf = getAnalyicSelectionProbability(); // analytic light doesn't have a solid angle PDF.
                    }
                }

                if (validSample && visible)
                {
                    tempPathReservoir.rcVertexWi[0] = ls.dir;
                    tempPathReservoir.rcVertexIrradiance[0] = ls.Li * ls.pdf;
                    tempPathReservoir.pathFlags.insertLightType(ls.lightType);
                    tempPathReservoir.lightPdf = ls.pdf;
                }
                else tempPathReservoir.rcVertexIrradiance[0] = 0.f;
                return 1.f;
            }
            else
            {
                Ray emptyRay = Ray(float3(0.f), float3(0.f), 0.f, kRayTMax);

                // should always be false in this case
                bool isRcVertexEscapedVertex = false;
                generateRandomReplayPath(path, tempPathReservoir.rcVertexHit.getHitInfo(), emptyRay, tempPathReservoir.rcRandomSeed, tempPathReservoir.pathFlags.pathLength(), tempPathReservoir.pathFlags.lastVertexNEE(), isRcVertexEscapedVertex);
                path.length = rcVertexLength;

                uint lobes = getBSDFLobes(rcVertexSd);
                bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
                if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
                path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);

                // start from path length
                float3 scatterDir = tempPathReservoir.rcVertexWi[0];
                path.pathBuilder.rcVertexLength = rcVertexLength;
                path.origin = rcVertexSd.computeNewRayOrigin(!tempPathReservoir.pathFlags.decodeIsTransmissionEvent(false));

                path.pathBuilder.pathFlags.transferDeltaInformation(tempPathReservoir.pathFlags);
                path.pathBuilder.pathFlags.transferTransmissionInformation(tempPathReservoir.pathFlags);
                path.pathBuilder.pathFlags.transferSpecularBounceInformation(tempPathReservoir.pathFlags);
                path.pathReservoir.rcVertexIrradiance[0] = 0.f;

                generateReplayPrimaryScatterRay(path, rcVertexSd, scatterDir);
                path.sg = SampleGenerator(tempPathReservoir.rcRandomSeed);
            }
        }
        traceReplaySuffix(path);
#if SHARC_TEMPORAL_QUERY
        if (replaySharc)
        {
            float3 incident = path.pathReservoir.rcVertexIrradiance[0];
            if (!any(incident > 0.f) || !all(isfinite(incident))) return 1.f;
            float survivalRatio = tempPathReservoir.cachedJacobian.z / path.russianRoulettePdf;
            if (!(survivalRatio > 0.f) || isinf(survivalRatio)) return 1.f;
            tempPathReservoir.rcVertexIrradiance[0] = incident;
            tempPathReservoir.cachedJacobian.z = path.russianRoulettePdf;
            return survivalRatio;
        }
#endif
        tempPathReservoir.rcVertexIrradiance[0] = path.pathReservoir.rcVertexIrradiance[0];
        tempPathReservoir.lightPdf = path.pathReservoir.lightPdf;
        tempPathReservoir.pathFlags.insertLightType(path.pathReservoir.pathFlags.lightType());
        return 1.f;
    }

    /** Finalize the path state before writing to the output buffer.
    */
    void finalize(inout PathState path)
    {
        // All endpoint candidates came from one sampled path.
        path.pathReservoir.M = 1;
    }

    ShadingData loadNRDPrimarySurface(uint2 pixel)
    {
        PathState primary = {};
        primary.id = pixel.x | (pixel.y << 12);
        primary.hit = HitInfo(vbuffer[pixel]);
        primary.dir = gScene.camera.computeRayPinhole(pixel, params.frameDim).dir;
        return loadPathSurface(primary);
    }

    void writeNRDBackground(uint2 pixel)
    {
        outputNRDDiffuseRadianceHitDist[pixel] = 0.f;
        outputNRDSpecularRadianceHitDist[pixel] = 0.f;
        outputNRDResidualRadianceHitDist[pixel] = 0.f;
        outputNRD.primaryHitDiffuseReflectance[pixel] = 1.f;
        outputNRD.primaryHitSpecularReflectance[pixel] = 1.f;
        outputNRD.primaryHitEmission[pixel] = outputColor[pixel];
    }

    void writeNRDSignals(uint2 pixel, ShadingData sd, uint sampleIndex, NRDPathType type, float3 indirect,
        float hitDist, float4 deltaDirect)
    {
        NRDSignal diffuse = {};
        NRDSignal specular = {};
        float3 residual = 0.f;
        float3 emission = 0.f;
        if (useDirectLighting)
        {
            float3 total = directLighting[pixel].rgb;
            if (useSplitDirectLighting)
            {
                float4 directD = directDiffuseRadianceHitDist[pixel];
                float4 directS = directSpecularRadianceHitDist[pixel];
                diffuse.add(directD.rgb, directD.a);
                specular.add(directS.rgb, directS.a);
                // ScreenSpaceReSTIR.color contains the same D, S and primary emission.
                emission += hasDirectEmission ? directEmission[pixel].rgb :
                    min(max(0.f, total - (directD.rgb + directS.rgb)), max(0.f, sd.emissive));
            }
            else
            {
                // Legacy graphs supply a combined color. Preserve its energy and the original winner routing.
                float3 primary = hasDirectEmission ? directEmission[pixel].rgb : min(max(total, 0.f), max(sd.emissive, 0.f));
                emission += primary;
                indirect += total - primary;
            }
            specular.add(deltaDirect.rgb, deltaDirect.a);
        }

        if (type == NRDPathType::Diffuse) diffuse.add(indirect, hitDist);
        else if (type == NRDPathType::Specular) specular.add(indirect, hitDist);
        else residual = indirect;

        NRDMaterialFactors f = getNRDMaterialFactors(sd, kUseNRDDemodulation);
        if (sampleIndex == 0)
        {
            outputNRD.primaryHitEmission[pixel] = float4(emission, 1.f);
            outputNRD.primaryHitDiffuseReflectance[pixel] = float4(f.diffuse, 1.f);
            outputNRD.primaryHitSpecularReflectance[pixel] = float4(f.specular, 1.f);
        }

        float invSpp = 1.f / kSamplesPerPixel;
        float4 previousD = 0.f, previousS = 0.f;
        float3 previousResidual = 0.f;
        float2 previousWeights = 0.f;
        if (sampleIndex > 0)
        {
            previousD = outputNRDDiffuseRadianceHitDist[pixel];
            previousS = outputNRDSpecularRadianceHitDist[pixel];
            previousResidual = outputNRDResidualRadianceHitDist[pixel].rgb;
            previousWeights = nrdGuideWeights[pixel];
        }
        outputNRDDiffuseRadianceHitDist[pixel] = diffuse.accumulate(previousD, previousWeights.x, f.diffuse, invSpp);
        outputNRDSpecularRadianceHitDist[pixel] = specular.accumulate(previousS, previousWeights.y, f.specular, invSpp);
        outputNRDResidualRadianceHitDist[pixel] = float4(previousResidual + residual * invSpp, 0.f);
        if (kSamplesPerPixel > 1)
            nrdGuideWeights[pixel] = max(previousWeights, float2(diffuse.guideWeight, specular.guideWeight));
    }

    void writeNRDReservoir(uint2 pixel, ShadingData sd, PathReservoir reservoir, float4 deltaDirect)
    {
        NRDPathType type = NRDPathType::Residual;
        float hitDist = 0.f;
        float3 indirect = reservoir.F * reservoir.weight;
        if (any(isnan(indirect) || isinf(indirect) || indirect < 0.f)) indirect = 0.f;
        if (any(indirect > 0.f))
        {
            bool environment = reservoir.pathFlags.lightType() == (uint)LightSampleType::EnvMap;
            if (reservoir.rcVertexHit.instanceID != 0xffffffff || environment)
            {
                bool specular = (kSeparatePathBSDF ? reservoir.pathFlags.decodeIsSpecularBounce(true) : reservoir.pathFlags.isNRDPrimarySpecular()) ||
                    reservoir.pathFlags.decodeIsDeltaEvent(true);
                type = specular ? NRDPathType::Specular : NRDPathType::Diffuse;
                if (reservoir.rcVertexHit.instanceID == 0xffffffff) hitDist = kNRDInvalidPathLength;
                else
                {
                    ShadingData next = loadShadingDataWithPrevVertexPosition(reservoir.rcVertexHit.getHitInfo(), sd.posW, false);
                    hitDist = length(next.posW - sd.posW);
                }
            }
        }
        writeNRDSignals(pixel, sd, gSppId, type, indirect, hitDist, deltaDirect);
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(PathState path, inout PathReservoir giReservoir, int sampleId)
    {
        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();

        const uint reservoirIdx = params.getReservoirOffset(pixel);

        static const uint itersPerShaderPass = kCandidateSamples;

        if (itersPerShaderPass == 1)
        {
            float3 L = 0.f;

            if (useDirectLighting) L += path.LDeltaDirect;

            path.pathReservoir.finalizeRIS();

            if (isLastRound)
            {
                // Write color directly to frame buffer.
                float3 color = path.pathReservoir.F * path.pathReservoir.weight;

                if (any(isinf(color) || isnan(color))) color = 0.f;

                L += color;
            }

            if (useDirectLighting && isLastRound)
            {
                L += directLighting[pixel].rgb;
            }

            if (any(isinf(L) || isnan(L))) L = 0.f;

            L /= kSamplesPerPixel;

            if (kSamplesPerPixel == 1 || gSppId == 0)
                outputColor[pixel] = float4(L, 1.f);
            else
                outputColor[pixel] += float4(L, 1.f);

            outputReservoirs[reservoirIdx] = path.pathReservoir;
        }
        else
        {
            if (sampleId == 0)
            {
                giReservoir = path.pathReservoir; // we are the first one
            }
            else
            {
                if (giReservoir.mergeInSamplePixel(path.pathReservoir, path.sg))
                {
                    giReservoir.F = path.pathReservoir.F;
                }
            }

            if (sampleId == kCandidateSamples - 1) // last sample
            {
                float3 L = 0.f;
                if (useDirectLighting) L += path.LDeltaDirect;

                giReservoir.finalizeRIS();
                if (isLastRound)
                {
                    float3 color = giReservoir.F * giReservoir.weight;

                    if (any(isinf(color) || isnan(color))) color = 0.f;
                    L += color;
                }

                if (useDirectLighting && isLastRound)
                {
                    L += directLighting[pixel].rgb;
                    if (any(isnan(L) || isinf(L))) L = 0.f;
                }

                L /= kSamplesPerPixel;

                if (kSamplesPerPixel == 1 || gSppId == 0)
                    outputColor[pixel] = float4(L, 1.f);
                else
                    outputColor[pixel] += float4(L, 1.f);

                outputReservoirs[reservoirIdx] = giReservoir;
            }
        }

        if (kOutputNRDData && sampleId == itersPerShaderPass - 1)
        {
            float4 delta = float4(path.LDeltaDirect, path.hitDist);
            if (isLastRound)
            {
                PathReservoir winner = path.pathReservoir;
                if (itersPerShaderPass > 1) winner = giReservoir;
                writeNRDReservoir(pixel, loadNRDPrimarySurface(pixel), winner, delta);
            }
            else nrdDeltaDirect[pixel] = delta;
        }

    }

};
