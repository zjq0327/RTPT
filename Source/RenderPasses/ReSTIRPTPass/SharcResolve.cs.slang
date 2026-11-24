import SharcCache;

cbuffer PerFrame
{
    SharcQueryData gSharc;
    SharcResolveParameters gResolve;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    SharcResolveEntry(dispatchThreadID.x, gSharc.cache, gResolve);
}
