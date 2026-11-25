import Rendering.Materials.MaterialShading;
import StaticParams;
import PGTraining;

// A record exists for each ray that actually left a surface. Non-trainable
// surfaces still propagate radiance to earlier vertices.
struct PGTrainingEdge
{
    float3 localRaw;
    float3 localMIS;
    float3 transport;
    uint2 key;
    uint quadrant;
    float density;
};

struct PGTrainingPath
{
    // hasFinishedSurfaceBounces() permits the final ray at the bounce limit.
    PGTrainingEdge edges[kMaxSurfaceBounces + 1];
    uint count;
    bool valid;
    float3 attenuation;
    float3 tailRaw;
    float3 tailMIS;

    [mutating] void init()
    {
        count = 0u;
        valid = true;
        attenuation = 1.f;
        tailRaw = tailMIS = 0.f;
    }

    [mutating] void arrive(float3 emission, float3 weightedEmission, float3 direct)
    {
        tailRaw = attenuation * (emission + direct);
        tailMIS = attenuation * (weightedEmission + direct);
        valid = valid && all(isfinite(tailRaw)) && all(isfinite(tailMIS));
    }

    [mutating] void scatter(PGWrite model, ShadingData sd, float3 direction, float pdf,
        float survival, float3 scatterWeight, bool delta)
    {
        if (count >= kMaxSurfaceBounces + 1u || !(survival > 0.f && survival <= 1.f))
        {
            valid = false;
            return;
        }
        PGTrainingEdge edge;
        edge.localRaw = tailRaw;
        edge.localMIS = tailMIS;
        edge.transport = attenuation * scatterWeight / survival;
        valid = valid && all(isfinite(edge.transport));
        edge.key = 0u;
        edge.quadrant = 4u;
        edge.density = 0.f;
        uint lobes = getBSDFLobes(sd) & sd.getActiveLobes();
        uint continuousReflection = uint(LobeType::NonDeltaReflection);
        if (!delta && (lobes & continuousReflection) != 0u && (lobes & ~continuousReflection) == 0u)
        {
            float3 normal = sd.frontFacing ? sd.faceN : -sd.faceN;
            uint quadrant;
            if (pgTrainingAddress(model, sd.posW, normal, direction, edge.key, quadrant))
            {
                edge.quadrant = quadrant;
                edge.density = pdf * survival;
                valid = valid && (edge.density > 0.f) && isfinite(edge.density);
            }
        }
        edges[count++] = edge;
        attenuation = 1.f;
        tailRaw = tailMIS = 0.f;
    }

    [mutating] void publish(PGWrite model)
    {
        if (!valid) return;
        // Immediate emitter/environment is a physical incident-radiance label.
        // For a more distant ancestor its light/BSDF MIS is already competing
        // with NEE at this edge's source and must be applied exactly once.
        for (uint remaining = count; remaining > 0u; --remaining)
        {
            uint i = remaining - 1u;
            PGTrainingEdge edge = edges[i];
            if (edge.quadrant < 4u)
            {
                float weight = pgTrainingWeight(tailRaw, edge.density, 1.f);
                if (!(weight >= 0.f) || !isfinite(weight)) return;
                edges[i].density = weight;
            }
            float3 reflected = edge.transport * tailMIS;
            tailRaw = edge.localRaw + reflected;
            tailMIS = edge.localMIS + reflected;
            if (!all(isfinite(tailRaw)) || !all(isfinite(tailMIS))) return;
        }
        // Publish after validating the complete returned suffix.
        for (uint i = 0u; i < count; ++i)
        {
            PGTrainingEdge edge = edges[i];
            if (edge.quadrant < 4u) pgTrainAddress(model, edge.key, edge.quadrant, edge.density);
        }
    }
};
