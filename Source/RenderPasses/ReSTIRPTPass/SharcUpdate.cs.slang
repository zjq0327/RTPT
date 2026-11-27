import Scene.Scene;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Rendering.Materials.MaterialShading;
import Utils.Math.HashUtils;
#ifndef PG_TRAINING
#define PG_TRAINING 0
#endif
#ifndef SHARC_FIXED_X1_TRAINER
import LoadShadingData;
import PathTracer;
import SharcCache;
#endif

ParameterBlock<PathTracer> gPathTracer;

cbuffer CB
{
    SharcQueryData gSharcWrite;
    uint gSharcFrameIndex;
    uint gUpdateStride;
#if PG_TRAINING
    PGWrite gPGWrite;
#ifdef SHARC_FIXED_X1_TRAINER
    uint gTrainSharc;
#endif
#endif
}

float3 trainingEnvironment(PathState path, out float3 raw)
{
    raw = 0.f;
    if (!gPathTracer.kUseEnvLight) return 0.f;
    raw = gPathTracer.envMapSampler.eval(path.dir);
    float weight = 1.f;
    if (kUseNEE && path.isLightSampled() && path.isLightSamplable())
    {
        if (!kUseMIS) return 0.f;
        const float pdf = gPathTracer.getEnvMapSelectionProbability() * gPathTracer.envMapSampler.evalPdf(path.dir);
        weight = PathTracer::evalMIS(1, path.pdf, 1, pdf);
    }
    return raw * weight;
}

float3 trainingEmission(PathState path, ShadingData sd, out float3 raw)
{
    raw = 0.f;
    if (!gPathTracer.kUseEmissiveLights || !any(sd.emissive > 0.f) ||
        (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())) return 0.f;
    raw = sd.emissive;
    if (!kUseNEE || !path.isLightSampled() || !path.isLightSamplable() || path.hit.getType() != HitType::Triangle)
        return sd.emissive;
    if (!kUseMIS) return 0.f;

    const TriangleHit triangle = path.hit.getTriangleHit();
    TriangleLightHit lightHit;
    lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangle.instanceID, triangle.primitiveIndex);
    lightHit.posW = sd.posW;
    lightHit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;
    const bool upperOnly = path.isLightSampledUpper() && !path.isLightSampledLower();
    const float pdf = gPathTracer.getEmissiveSelectionProbability() *
        gPathTracer.emissiveSampler.evalPdf(path.origin, path.normal, upperOnly, lightHit);
    return sd.emissive * PathTracer::evalMIS(1, path.pdf, 1, pdf);
}

float3 trainingDirect(ShadingData sd, inout PathState path)
{
    const uint lobes = getBSDFLobes(sd);
    const bool upper = kUseNEE && (lobes & uint(LobeType::NonDeltaReflection)) != 0u &&
        (kUseLightsInDielectricVolumes || !path.isInsideDielectricVolume());
    const bool lower = kUseNEE && (lobes & uint(LobeType::NonDeltaTransmission)) != 0u;
    path.setLightSampled(upper, lower);
    if (!upper && !lower) return 0.f;
    PathTracer::LightSample light;
    const PathTracer::PathVertex vertex = PathTracer::PathVertex::create(path.length + 1, sd.posW, sd.N, sd.faceN);
    if (!gPathTracer.generateLightSample(vertex, upper, lower, path.sg, light)) return 0.f;
    float3 value = evalBSDFCosine(sd, light.dir) * light.Li;
    if (!any(value > 0.f)) return 0.f;
    if (!gPathTracer.traceVisibilityRay(light.getVisibilityRay())) return 0.f;
    if (kUseMIS && light.lightType != uint(PathTracer::LightSampleType::Analytic))
        value *= PathTracer::evalMIS(1, light.pdf, 1, evalPdfBSDF(sd, light.dir));
    return value;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if PG_TRAINING && defined(SHARC_FIXED_X1_TRAINER)
    const bool trainSharc = gTrainSharc != 0;
#else
    const bool trainSharc = true;
#endif
    const uint stride = max(1u, gUpdateStride);
    const uint2 frameDim = gPathTracer.params.frameDim;
    const uint2 tile = dispatchThreadId.xy;
    const uint2 origin = tile * stride;
    if (any(origin >= frameDim)) return;
    const uint2 extent = min(uint2(stride), frameDim - origin);
    const uint phase = jenkinsHash(tile.x ^ (tile.y * 0x9e3779b9u));
    const uint offset = (phase + gSharcFrameIndex) % (extent.x * extent.y);
    const uint2 pixel = origin + uint2(offset % extent.x, offset / extent.x);

    PathState path;
    gPathTracer.generatePath(pixel.x | (pixel.y << 12), path);
    if (!path.isHit()) return;
    path.sg = SampleGenerator(pixel, jenkinsHash(gSharcFrameIndex ^ 0xb4c193d7u));
    float3 cameraWeight = 1.f;
#if PG_TRAINING
    PGTrainingPath pg;
    pg.init();
#endif
    SharcState state;
    SharcInit(state);
    [loop] while (path.isActive())
    {
        if (!path.isHit())
        {
            float3 rawEnvironment;
            const float3 environment = trainingEnvironment(path, rawEnvironment);
#if PG_TRAINING
            pg.arrive(rawEnvironment, environment, 0.f);
#endif
            if (trainSharc && all(isfinite(environment))) SharcUpdateMiss(gSharcWrite.cache, state, environment);
            break;
        }
        ShadingData sd = gPathTracer.loadPathSurface(path);
        if (!path.interiorList.isEmpty())
        {
            gPathTracer.handleHomogeneousVolumeExtinction(path, sd);
            if (trainSharc) SharcSetThroughput(state, path.thp);
#if PG_TRAINING
            pg.attenuation *= path.thp;
#endif
            cameraWeight *= path.thp;
            path.thp = 1.f;
        }
        if (!gPathTracer.handleNestedDielectrics(sd, path))
        {
            if (path.isActive()) gPathTracer.nextVertex(path);
            continue;
        }
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && sd.linearRoughness <= gPathTracer.params.specularRoughnessThreshold)
            sd.setActiveLobes(uint(LobeType::Diffuse));
        float3 rawEmission;
        const float3 emission = trainingEmission(path, sd, rawEmission);
        if (!all(isfinite(emission)))
        {
#if PG_TRAINING
            pg.valid = false;
#endif
            break;
        }
        if (gPathTracer.hasFinishedSurfaceBounces(path))
        {
#if PG_TRAINING
            pg.arrive(rawEmission, emission, 0.f);
#endif
            if (trainSharc) SharcUpdateMiss(gSharcWrite.cache, state, emission);
            break;
        }
        const float3 direct = trainingDirect(sd, path);
#if PG_TRAINING
        pg.arrive(rawEmission, emission, direct);
#endif
        if (!all(isfinite(direct))) break;
        // Each sparse path trains its first queryable surface with the full explicit
        // remainder. Inserting later vertices into the same grid would mix shorter
        // remaining bounce budgets into the estimate at this surface.
        if (trainSharc)
        {
            const bool insert = state.pathLength == 0 && gPathTracer.allowsSharcDepth(path) &&
                gSharcWrite.allowsSurface(sd, path.origin, path.isDelta(), path.previousDiffuse, path.previousRoughness);
            if (insert)
            {
                SharcUpdateHit(gSharcWrite.cache, state, makeSharcHit(sd, emission), direct, 0.f);
            }
            else if (state.pathLength != 0)
                SharcUpdateMiss(gSharcWrite.cache, state, direct + emission);
        }

        float survival = 1.f;
        if (kUseRussianRoulette && path.length >= 1)
        {
            path.prefixThp = cameraWeight;
            path.russianRoulettePdf = 1.f;
            if (gPathTracer.terminatePathByRussianRoulette(path)) break;
            // Explicit ReSTIR candidates divide by their survival density in RIS.
            // Raw PathTracing accumulates its surviving path.L directly.
#ifndef SHARC_FIXED_X1_TRAINER
            if (PathSamplingMode(kPathSamplingMode) != PathSamplingMode::PathTracing)
#endif
                survival = path.russianRoulettePdf;
            path.prefixThp = 1.f;
        }
        path.origin = sd.computeNewRayOrigin();
        if (!gPathTracer.generateScatterRay(sd, path)) break;
        if (kUseNEE && !kUseMIS && gPathTracer.hasFinishedSurfaceBounces(path) && path.isLightSamplable()) break;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) break;
        const float3 segmentWeight = path.getCurrentThp();
        if (!all(isfinite(segmentWeight)) || !all(isfinite(path.dir)))
        {
#if PG_TRAINING
            pg.valid = false;
#endif
            break;
        }
#if PG_TRAINING
        pg.scatter(gPGWrite, sd, path.dir, path.pdf, survival, segmentWeight, path.isDelta());
#endif
        cameraWeight *= segmentWeight;
        if (trainSharc) SharcSetThroughput(state, segmentWeight / survival);
        path.thp = 1.f;
        path.prefixThp = 1.f;
        gPathTracer.nextVertex(path);
    }
#if PG_TRAINING
    pg.publish(gPGWrite);
#endif
}
