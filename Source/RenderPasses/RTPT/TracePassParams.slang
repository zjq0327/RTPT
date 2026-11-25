/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#ifndef SHARC_GENERATE_QUERY
#define SHARC_GENERATE_QUERY 0
#endif
#ifndef SHARC_GENERATE_STATS
#define SHARC_GENERATE_STATS 0
#endif
#if SHARC_GENERATE_QUERY
import RenderPasses.ReSTIRPTPass.SharcCache;
#endif

#if SHARC_GENERATE_STATS && SHARC_GENERATE_QUERY
import PathState;
#endif

cbuffer CB
{
    int gSampleId;
#if SHARC_GENERATE_QUERY
    SharcQueryData gSharcQuery;
#endif
#if SHARC_GENERATE_QUERY && SHARC_GENERATE_STATS
    uint gSharcDebugMode;
    uint gSharcDebugDepth;
#endif
}

#if SHARC_GENERATE_STATS
RWStructuredBuffer<uint> gSharcGenerateStats;

enum class SharcDepthStat { Arrived, Eligible, Hits, Positive, Selected, Winners, Black, Explicit };

void countSharc(uint depth, SharcDepthStat stat)
{
    InterlockedAdd(gSharcGenerateStats[min(depth, 15u) * 8u + uint(stat)], 1u);
}

#if SHARC_GENERATE_QUERY
void applySharcDebugMode(uint depth, bool eligible, inout bool hit, inout float3 radiance)
{
    if (eligible) countSharc(depth, SharcDepthStat::Eligible);
    if (gSharcDebugMode == 0u) return;
    hit = eligible && depth == gSharcDebugDepth && (gSharcDebugMode == 2u || gSharcDebugMode == 3u);
    radiance = gSharcDebugMode == 2u ? float3(8.f, 4.f, 2.f) : float3(0.f);
}

void recordSharcHit(PathState path, float3 F, float3 radiance, bool selected)
{
    countSharc(path.length, SharcDepthStat::Hits);
    if (any(F > 0.f)) countSharc(path.length, SharcDepthStat::Positive);
    if (all(radiance == 0.f)) countSharc(path.length, SharcDepthStat::Black);
    if (!selected) return;
    countSharc(path.length, SharcDepthStat::Selected);
    if (path.length != 2u) return;

    // Save one x2 sample's inputs for comparison with the stored Fresh record.
    uint claimed;
    InterlockedCompareExchange(gSharcGenerateStats[128], 0u, 1u, claimed);
    if (claimed != 0u) return;
    uint2 pixel = path.getPixel();
    gSharcGenerateStats[129] = pixel.x;
    gSharcGenerateStats[130] = pixel.y;
    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        gSharcGenerateStats[131u + i] = asuint(path.getCurrentThp()[i]);
        gSharcGenerateStats[134u + i] = asuint(path.thp[i]);
        gSharcGenerateStats[137u + i] = asuint(radiance[i]);
        gSharcGenerateStats[140u + i] = asuint(path.pathBuilder.rcVertexWi[0][i]);
    }
    gSharcGenerateStats[143] = path.pathBuilder.cachedRandomSeed;
}
#endif
#endif
