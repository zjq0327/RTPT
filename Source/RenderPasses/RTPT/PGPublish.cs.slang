cbuffer CB
{
    uint gCapacity;
    uint gCarryPrevious;
    StructuredBuffer<uint2> gKeys;
    StructuredBuffer<uint4> gAccumulation;
    StructuredBuffer<uint> gCounts;
    StructuredBuffer<uint2> gPrevKeys;
    StructuredBuffer<float4> gPrevMass;
    RWStructuredBuffer<float4> gHistoryMean;
    RWStructuredBuffer<uint> gHistoryCount;
    RWStructuredBuffer<uint2> gOutKeys;
    RWStructuredBuffer<float4> gOutMass;
};

[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint slot = threadId.x;
    if (slot >= gCapacity) return;
    const uint2 key = gKeys[slot];
    float4 mass = 0.f;
    if (all(key != 0u))
    {
        const uint observations = gCounts[slot];
        if (observations != 0u)
        {
            const float4 sums = asfloat(gAccumulation[slot]);
            // Store a mean of the four unnormalized training weights. A new window
            // can contain one sparse observation or thousands of pretraining observations.
            const uint history = gHistoryCount[slot];
            const uint priorCount = history & 0xffu;
            const float totalCount = float(priorCount) + float(observations);
            float4 mean = gHistoryMean[slot] * (float(priorCount) / totalCount) + sums / totalCount;
            // Repeated all-black windows scale every component equally. Without
            // an explicit cutoff their normalized direction PDF would survive indefinitely.
            const uint blackLimit = 32u;
            const uint blackCount = any(sums > 0.f) ? 0u :
                min(blackLimit, (history >> 8u) + min(observations, blackLimit));
            if (blackCount == blackLimit) mean = 0.f;
            const uint priorLimit = 64u;
            const uint retained = min(priorLimit, priorCount + min(observations, priorLimit));
            gHistoryMean[slot] = mean;
            gHistoryCount[slot] = retained | (blackCount << 8u);
            const float peak = max(max(mean.x, mean.y), max(mean.z, mean.w));
            if (peak > 0.f)
            {
                const float4 relative = mean / peak;
                mass = relative / dot(relative, float4(1.f));
            }
        }
        else if (gCarryPrevious != 0u && all(gPrevKeys[slot] == key))
            mass = gPrevMass[slot];
    }
    gOutKeys[slot] = key;
    gOutMass[slot] = mass;
}
