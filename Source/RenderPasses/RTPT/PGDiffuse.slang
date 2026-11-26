import PGModel;
import Rendering.Materials.MaterialShading;

cbuffer PGGenerateCB
{
    PGRead pgRead;
    float pgDiffuseFraction;
#if PG_GENERATE_STATS
    RWStructuredBuffer<uint> pgGenerateCounts;
#endif
}

float3 pgOrientedNormal(ShadingData sd)
{
    return sd.frontFacing ? sd.faceN : -sd.faceN;
}

float4 lookupDiffusePG(ShadingData sd)
{
    if ((sd.getActiveLobes() & uint(LobeType::DiffuseReflection)) == 0u ||
        !any(sd.diffuse > 0.f) || sd.metallic >= 1.f ||
        sd.specularTransmission >= 1.f || sd.diffuseTransmission >= 1.f) return 0.f;
    float4 mass;
    if (!pgLookup(pgRead, sd.posW, pgOrientedNormal(sd), mass) ||
        !all(isfinite(mass)) || any(mass < 0.f) || abs(dot(mass, float4(1.f)) - 1.f) > 1e-4f) return 0.f;
    return mass;
}

// A PG draw outside the diffuse shading hemisphere terminates with zero weight.
// Other lobes and NEE must also omit that failed proposal from their competing density.
float evalJointDiffusePGPdf(FalcorBSDF bsdf, ShadingData sd, float3 direction, float4 mass, out float nativeJoint)
{
    nativeJoint = bsdf.pDiffuseReflection * bsdf.diffuseReflection.evalPdf(sd.toLocal(sd.V), sd.toLocal(direction));
    if (!(nativeJoint > 0.f)) return 0.f;
    float guideJoint = bsdf.pDiffuseReflection * pgPdf(mass, pgOrientedNormal(sd), direction);
    return (1.f - pgDiffuseFraction) * nativeJoint + pgDiffuseFraction * guideJoint;
}

float evalOtherLobesPdf(FalcorBSDF bsdf, ShadingData sd, float3 direction)
{
    return bsdf.evalPdf(sd.toLocal(sd.V), sd.toLocal(direction), ~uint(SampledBSDFFlags::DiffuseReflection));
}

float evalScatterPdfWithDiffusePG(ShadingData sd, float3 direction, float4 mass)
{
    if (pgDiffuseFraction == 0.f || !any(mass > 0.f)) return evalPdfBSDF(sd, direction);
    FalcorBSDF bsdf;
    bsdf.setup(sd);
    float nativeJoint;
    return evalJointDiffusePGPdf(bsdf, sd, direction, mass, nativeJoint) + evalOtherLobesPdf(bsdf, sd, direction);
}

bool sampleWithDiffusePG<S : ISampleGenerator>(ShadingData sd, inout S sg, float4 mass,
    bool evaluateAllLobes, out BSDFSample_ result)
{
    FalcorBSDF bsdf;
    bsdf.setup(sd);
    bool ready = any(mass > 0.f) && bsdf.pDiffuseReflection > 0.f;
    bool usedGuide = false;
    S guideRng = sg;
    if (ready && pgDiffuseFraction > 0.f && sampleNext1D(guideRng) < bsdf.pDiffuseReflection)
    {
        // The native diffuse BxDF consumes this 1D value before sampling its 2D direction.
        usedGuide = sampleNext1D(guideRng) < pgDiffuseFraction;
    }

    bool valid;
    if (usedGuide)
    {
        result = {};
        result.lobe = uint(LobeType::DiffuseReflection);
        result.wi = pgSample(mass, pgOrientedNormal(sd), sampleNext2D(guideRng));
        sg = guideRng;
        valid = bsdf.diffuseReflection.evalPdf(sd.toLocal(sd.V), sd.toLocal(result.wi)) > 0.f;
    }
    else valid = sampleBSDF(sd, sg, result, evaluateAllLobes);

    bool diffuse = result.isLobe(LobeType::DiffuseReflection) && bsdf.pDiffuseReflection > 0.f;
#if PG_GENERATE_STATS
    if (diffuse)
    {
        InterlockedAdd(pgGenerateCounts[0], 1u);
        if (ready) InterlockedAdd(pgGenerateCounts[1], 1u);
    }
    if (usedGuide) InterlockedAdd(pgGenerateCounts[diffuse ? 2u : 3u], 1u);
    else if (result.isLobe(LobeType::SpecularOrDelta)) InterlockedAdd(pgGenerateCounts[4], 1u);
    else if (diffuse) InterlockedAdd(pgGenerateCounts[5], 1u);
#endif
    if (!valid || !ready || pgDiffuseFraction == 0.f || result.isLobe(LobeType::Delta)) return valid;

    float nativeJoint;
    float jointDiffuse = evalJointDiffusePGPdf(bsdf, sd, result.wi, mass, nativeJoint);
    // Native sampling has already evaluated all competing GGX lobes. Replace only
    // the weighted diffuse term. Sum the other lobes explicitly if subtraction
    // would lose precision, or if the guide generated the direction itself.
    float pdfAll = result.pdf + (jointDiffuse - (diffuse ? result.pdfSingle : nativeJoint));
    if (usedGuide || !isfinite(pdfAll) || pdfAll <= 0.25f * result.pdf)
        pdfAll = jointDiffuse + evalOtherLobesPdf(bsdf, sd, result.wi);
    if (!(pdfAll > 0.f) || !isfinite(pdfAll) || (diffuse && !(jointDiffuse > 0.f))) return false;
    if (usedGuide)
    {
        uint lobes = evaluateAllLobes ? ~0u : uint(SampledBSDFFlags::DiffuseReflection);
        result.weight = bsdf.eval(sd.toLocal(sd.V), sd.toLocal(result.wi), lobes) /
            (evaluateAllLobes ? pdfAll : jointDiffuse);
    }
    else if (evaluateAllLobes) result.weight *= result.pdf / pdfAll;
    else if (diffuse) result.weight *= result.pdfSingle / jointDiffuse;
    if (diffuse) result.pdfSingle = jointDiffuse;
    result.pdf = pdfAll;
    return true;
}
