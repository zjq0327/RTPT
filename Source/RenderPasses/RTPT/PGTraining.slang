import PGModel;

struct PGWrite
{
    uint capacity;
    float3 cameraPosition;
    float sceneScale;
    RWStructuredBuffer<uint2> keys;
    RWStructuredBuffer<uint4> accumulation;
    RWStructuredBuffer<uint> counts;
    RWStructuredBuffer<uint> diagnostics;
};

bool pgTrainingAddress(PGWrite model, float3 position, float3 orientedNormal, float3 direction,
    out uint2 key, out uint quadrant)
{
    quadrant = 0u;
    return pgGridKey(position, orientedNormal, model.cameraPosition, model.sceneScale, key) &&
        pgDirectionBin(orientedNormal, direction, quadrant);
}

float pgTrainingWeight(float3 incident, float samplePdf, float survival)
{
    if (!all(isfinite(incident)) || !isfinite(samplePdf) || !(samplePdf > 0.f) ||
        !(survival > 0.f && survival <= 1.f)) return -1.f;
    return dot(max(incident, 0.f), float3(1.f / 3.f)) / (samplePdf * survival);
}

void pgTrainAddress(PGWrite model, uint2 key, uint quadrant, float weight)
{
    if (model.capacity == 0u) return;
#if SHARC_TRAINING_STATS
    InterlockedAdd(model.diagnostics[0], 1u);
#endif
    if (!all(key != 0u) || quadrant >= 4u || !isfinite(weight) || weight < 0.f)
    {
        InterlockedAdd(model.diagnostics[4], 1u);
        return;
    }
    uint slot = pgFirstSlot(key, model.capacity);
    for (uint probe = 0u; probe < min(model.capacity, kPGProbeCount); ++probe)
    {
        // A slot first claims its high word, then its low word. A contender with the same
        // high word can finish claiming the slot. Neither stage waits for the other lane.
        uint high;
        InterlockedCompareExchange(model.keys[slot][1], 0u, key.y, high);
        if (high == 0u || high == key.y)
        {
            uint low;
            InterlockedCompareExchange(model.keys[slot][0], 0u, key.x, low);
            if (low == 0u || low == key.x)
            {
                if (weight > 0.f)
                {
                    // Common scale leaves the normalized guide unchanged and protects the accumulator.
                    const float scaled = weight * 1e-4f;
                    uint previous = 0u;
                    for (;;)
                    {
                        const float sum = asfloat(previous) + scaled;
                        if (!isfinite(sum))
                        {
                            InterlockedAdd(model.diagnostics[4], 1u);
                            return;
                        }
                        uint observed;
                        InterlockedCompareExchange(model.accumulation[slot][quadrant], previous, asuint(sum), observed);
                        if (observed == previous) break;
                        previous = observed;
                    }
                }
                InterlockedAdd(model.counts[slot], 1u);
#if SHARC_TRAINING_STATS
                InterlockedAdd(model.diagnostics[weight > 0.f ? 1u : 2u], 1u);
#endif
                return;
            }
        }
        slot = (slot + 1u == model.capacity) ? 0u : slot + 1u;
    }
    InterlockedAdd(model.diagnostics[3], 1u);
}

