import SharcCache;

StructuredBuffer<float4> gBounds;
cbuffer PerFrame
{
    SharcQueryData gSharc;
    uint gBoundsCount;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index < gSharc.cache.hashGridData.capacity)
        gSharc.invalidateAt(index, gBounds, gBoundsCount);
}
