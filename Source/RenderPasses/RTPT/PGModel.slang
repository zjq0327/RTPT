// The reader only sees complete snapshots published between rendered frames.
struct PGRead
{
    uint capacity;
    float3 cameraPosition;
    float sceneScale;
    StructuredBuffer<uint2> keys;
    StructuredBuffer<float4> mass;
};

static const uint kPGProbeCount = 16u;
static const float kPGTwoPi = 6.283185307179586f;

uint pgHashWord(uint x)
{
    x = (x + 0x7ed55d16u) + (x << 12u);
    x = (x ^ 0xc761c23cu) ^ (x >> 19u);
    x = (x + 0x165667b1u) + (x << 5u);
    x = (x + 0xd3a2646cu) ^ (x << 9u);
    x = (x + 0xfd7046c5u) + (x << 3u);
    return (x ^ 0xb55a4f09u) ^ (x >> 16u);
}

uint pgFirstSlot(uint2 key, uint capacity)
{
    return (pgHashWord(key.x) ^ pgHashWord(key.y)) % capacity;
}

bool pgGridKey(float3 position, float3 normal, float3 cameraPosition, float sceneScale, out uint2 key)
{
    key = uint2(0u);
    if (!all(isfinite(position)) || !all(isfinite(normal)) || !(dot(normal, normal) > 0.f) || !(sceneScale > 0.f)) return false;
    const float3 biasedPosition = position + float3(1e-4f);
    const float3 offset = cameraPosition - biasedPosition;
    const uint level = uint(clamp(0.5f * log2(max(dot(offset, offset), 1e-10f)), 1.f, 511.f));
    const float3 cell = floor(biasedPosition * (exp2(-float(level)) * sceneScale));
    // Signed 17-bit cells. Fall back outside the namespace instead of aliasing a distant surface.
    if (any(cell < -65536.f) || any(cell > 65535.f)) return false;
    const uint3 xyz = uint3(int3(cell)) & 0x1ffffu;
    const uint normalBits = (normal.x < -1e-3f ? 1u : 0u) |
        (normal.y < -1e-3f ? 2u : 0u) | (normal.z < -1e-3f ? 4u : 0u);
    const uint low = xyz.x | (xyz.y << 17u);
    const uint high = (xyz.y >> 15u) | (xyz.z << 2u) | (level << 19u) | (normalBits << 28u);
    // Both words must be nonzero for the two-stage atomic insertion. Bit 63 is otherwise unused.
    key = uint2(max(1u, low), high | (low == 0u ? 0x80000000u : 0u));
    return true;
}

bool pgLookup(PGRead model, float3 position, float3 orientedNormal, out float4 mass)
{
    mass = 0.f;
    if (model.capacity == 0u) return false;
    uint2 key;
    if (!pgGridKey(position, orientedNormal, model.cameraPosition, model.sceneScale, key)) return false;
    uint slot = pgFirstSlot(key, model.capacity);
    for (uint probe = 0u; probe < min(model.capacity, kPGProbeCount); ++probe)
    {
        const uint2 stored = model.keys[slot];
        if (stored.y == 0u) break;
        if (all(stored == key))
        {
            mass = model.mass[slot];
            return any(mass > 0.f);
        }
        slot = (slot + 1u == model.capacity) ? 0u : slot + 1u;
    }
    return false;
}

void pgTangentFrame(float3 n, out float3 t, out float3 b)
{
    const float3 reference = abs(n.z) < 0.999f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
    t = normalize(cross(reference, n));
    b = cross(n, t);
}

bool pgDirectionBin(float3 normal, float3 direction, out uint bin)
{
    bin = 0u;
    const float3 n = normalize(normal);
    const float z = dot(n, direction);
    if (!(z >= 0.f) || !all(isfinite(direction))) return false;
    float3 t, b;
    pgTangentFrame(n, t, b);
    // u=z, v=azimuth/(2*pi). For half circles, sign and the negative X axis classify v exactly.
    const float y = dot(b, direction);
    const bool secondHalf = y < 0.f || (y == 0.f && dot(t, direction) < 0.f);
    bin = (z >= 0.5f ? 1u : 0u) | (secondHalf ? 2u : 0u);
    return true;
}

float pgPdf(float4 mass, float3 orientedNormal, float3 direction)
{
    uint bin;
    if (!pgDirectionBin(orientedNormal, direction, bin)) return 0.f;
    // Each equal-area sector subtends pi/2 steradians. Mass is normalized at publication.
    return mass[bin] * (4.f / kPGTwoPi);
}

float3 pgSample(float4 mass, float3 orientedNormal, float2 random)
{
    const float upper = asfloat(0x3f7fffffu);
    const float2 u = clamp(random, 0.f, upper);
    float remainder = u.x;
    uint bin = 0u;
    for (; bin < 3u; ++bin)
    {
        if (remainder < mass[bin]) break;
        remainder -= mass[bin];
    }
    // The sum of four normalized floats can round slightly below one.
    while (bin > 0u && !(mass[bin] > 0.f)) --bin;
    const float z = 0.5f * (float(bin & 1u) + clamp(remainder / mass[bin], 0.f, upper));
    const float phi = (0.5f * kPGTwoPi) * (float(bin >> 1u) + u.y);
    float sine, cosine;
    sincos(phi, sine, cosine);
    const float radius = sqrt(max(0.f, 1.f - z * z));
    const float3 n = normalize(orientedNormal);
    float3 t, b;
    pgTangentFrame(n, t, b);
    return normalize(radius * (cosine * t + sine * b) + z * n);
}
