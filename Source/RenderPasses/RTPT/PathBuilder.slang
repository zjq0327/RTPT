/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#include "Scene/Material/MaterialDefines.slangh"

import Rendering.Materials.MaterialShading;
import LoadShadingData;
import Utils.Debug.PixelDebug;
import Utils.Math.HashUtils;
import Params;
import PathReservoir;
import Rendering.Utils.PixelStats;
import Scene.RaytracingInline;
import Rendering.Materials.MaterialShading;
import Utils.Sampling.TinyUniformSampleGenerator;

static const int kRcWiCount = 1;

/// Builds a Path and a PathPrefix.
struct PathBuilder
{
    TriMeshHitInfo rcVertexHit;
    float3 rcVertexWi[kRcWiCount];
    uint cachedRandomSeed;
    TinyUniformSampleGenerator sg;
    int rcVertexLength = kMaximumPathLength;
    ReSTIRPathFlags pathFlags;
    float3 cachedJacobian = 1.f;

    [mutating]
    void init<S : ISampleGenerator>(inout S sg)
    {
        this.sg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));
        rcVertexWi[0] = float3(0.f);
        rcVertexHit.instanceID = 0xffffffff;
    }

    void storeEndpoint(uint depth, bool nee, uint lightType, float lightPdf, float3 F,
        float3 incident, float3 wi, inout PathReservoir reservoir)
    {
        reservoir.pathFlags = pathFlags;
        reservoir.pathFlags.insertPathLength(depth);
        reservoir.pathFlags.insertRcVertexLength(rcVertexLength);
        reservoir.pathFlags.insertLastVertexNEE(nee);
        reservoir.pathFlags.insertLightType(lightType);
        reservoir.F = F;
        reservoir.rcVertexIrradiance[0] = incident;
        reservoir.rcVertexWi[0] = wi;
        reservoir.rcVertexHit = rcVertexHit;
        reservoir.rcRandomSeed = cachedRandomSeed;
        reservoir.cachedJacobian = cachedJacobian;
        reservoir.lightPdf = lightPdf;
    }

    [mutating]
    bool addEscapeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)
    {
        if (pathLength < 1) return false;
        if (!forceAdd && !pathReservoir.add(pathWeight, russianroulettePDF, sg)) return false;
        const bool atRc = pathLength == rcVertexLength;
        // A shift re-evaluates the outgoing edge's MIS weight at the reconnection vertex.
        storeEndpoint(pathLength, false, lightType, lightPdf, pathWeight, atRc ? postfixWeight / misWeight : postfixWeight,
            atRc ? wi : rcVertexWi[0], pathReservoir);
        return true;
    }

    [mutating]
    bool addNeeVertex(RestirPathTracerParams params, uint pathLength, float3 wi, float3 pathWeight, float3 postfixWeight,
        float russianroulettePDF, float misWeight, float lightPdf, int lightType, inout PathReservoir pathReservoir, bool forceAdd = false)
    {
        if (pathLength < 1) return false;
        if (!forceAdd && !pathReservoir.add(pathWeight, russianroulettePDF, sg)) return false;
        const bool atRc = pathLength == rcVertexLength;
        storeEndpoint(pathLength, true, lightType, lightPdf, pathWeight, atRc ? postfixWeight * lightPdf / misWeight : postfixWeight,
            atRc ? wi : rcVertexWi[0], pathReservoir);
        return true;
    }
};

