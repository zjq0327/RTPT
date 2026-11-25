/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import RenderPasses.ReSTIRPTPass.SharcCache;

cbuffer TemporalSharcCB
{
    SharcQueryData gSharcTemporalQuery;
#if SHARC_TEMPORAL_STATS
    uint gTemporalSharcDebugMode;
    uint gTemporalSharcDebugDepth;
#endif
}

#if SHARC_TEMPORAL_STATS
// Five 16-depth histograms, followed by scalar counters. Only the instrumented
// Temporal program declares this UAV.
RWStructuredBuffer<uint> gSharcTemporalStats;

enum class SharcTemporalDepthStat { Ray, Lookup, Hit, Replay, Winner };
enum class SharcTemporalStat
{
    ExplicitNEE = 80, ExplicitEmitter, ExplicitEnvironment,
    ReverseEvaluation, ReversePositive, ReverseZero, BlackHit,
    ForwardPositive, ForwardZero, ReverseVisible, ReverseOccluded
};

void countSharcTemporalDepth(SharcTemporalDepthStat stat, uint depth)
{
    InterlockedAdd(gSharcTemporalStats[16u * uint(stat) + min(depth, 15u)], 1u);
}

void countSharcTemporal(SharcTemporalStat stat)
{
    InterlockedAdd(gSharcTemporalStats[uint(stat)], 1u);
}

void recordSharcTemporalRay(uint nextDepth)
{
    countSharcTemporalDepth(SharcTemporalDepthStat::Ray, nextDepth);
}

void recordSharcTemporalLookup(uint depth)
{
    countSharcTemporalDepth(SharcTemporalDepthStat::Lookup, depth);
}

void applySharcTemporalProbe(uint depth, bool eligible, inout bool hit, inout float3 radiance)
{
    if (gTemporalSharcDebugMode != 0u)
    {
        hit = eligible && depth == gTemporalSharcDebugDepth && gTemporalSharcDebugMode >= 2u;
        radiance = gTemporalSharcDebugMode == 2u ? float3(8.f, 4.f, 2.f) : float3(0.f);
    }
    if (hit)
    {
        countSharcTemporalDepth(SharcTemporalDepthStat::Hit, depth);
        if (all(radiance == 0.f)) countSharcTemporal(SharcTemporalStat::BlackHit);
    }
}
#endif
