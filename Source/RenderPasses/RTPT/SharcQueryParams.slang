/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import RenderPasses.ReSTIRPTPass.SharcCache;

cbuffer GenerateSharcCB
{
    SharcQueryData gSharcQuery;
    bool gEnableSharcGenerateQuery;
}

cbuffer TemporalSharcCB
{
    SharcQueryData gSharcTemporalQuery;
    bool gEnableSharcTemporalQuery;
}
