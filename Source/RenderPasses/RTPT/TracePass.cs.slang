/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Scene.Raytracing;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import PathTracer;
import PathState;
import RenderPasses.Shared.Denoising.NRDBuffers;
import Params;
import PathReservoir;

import TracePassParams;

#ifndef SHARC_GENERATE_STATS
#define SHARC_GENERATE_STATS 0
#endif

ParameterBlock<PathTracer> gPathTracer;

void tracePath(uint pathID, inout PathReservoir giReservoir, uint sampleIdx)
{
    PathState path;
    gPathTracer.generatePath(pathID, path);

    printSetPixel(path.getPixel());
    logSetPixel(path.getPixel());

    while (path.isActive())
    {
        if (path.isHit())
        {
            gPathTracer.handleHit(path);
            if (path.isTerminated()) break;
#if SHARC_GENERATE_STATS
            if (path.length >= 1u) InterlockedAdd(gSharcGenerateStats[144u + min(path.length, 15u)], 1u);
#endif
            gPathTracer.nextVertex(path);
        }
        else
        {
            gPathTracer.handleMiss(path);
        }
    }

    gPathTracer.finalize(path);
    gPathTracer.writeOutput(path, giReservoir, sampleIdx);
#if SHARC_GENERATE_STATS
    if (sampleIdx == kCandidateSamples - 1)
    {
        PathReservoir fresh = gPathTracer.outputReservoirs[gPathTracer.params.getReservoirOffset(path.getPixel())];
        if (fresh.weight > 0.f && fresh.pathFlags.isSharcTerminal()) countSharc(fresh.pathFlags.pathLength(), SharcDepthStat::Winners);
    }
#endif
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gPathTracer.params.frameDim)) return;

    // Skip pixel if there is no hit in the vbuffer.
    HitInfo hit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!hit.isValid())
    {
        if (kOutputNRDData && gPathTracer.gSppId == 0) gPathTracer.writeNRDBackground(pixel);
        return;
    }

    PathReservoir giReservoir;

    static const uint itersPerShaderPass = kCandidateSamples;

    if (itersPerShaderPass == 1)
    {
        // Handle fixed 1 spp case.
        uint pathID = pixel.x | (pixel.y << 12) | (gSampleId << 24);
        tracePath(pathID, giReservoir, 0);
    }
    else//(itersPerShaderPass > 1)
    {
        // Handle fixed multiple spp case.
        for (uint sampleIdx = 0; sampleIdx < itersPerShaderPass; ++sampleIdx)
        {
            uint pathID = pixel.x | (pixel.y << 12) | (sampleIdx << 24);
            tracePath(pathID, giReservoir, sampleIdx);
        }
    }
}
