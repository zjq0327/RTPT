/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Rendering.Materials.Microfacet;
import Rendering.Materials.MaterialShading;
import RenderPasses.Shared.Denoising.NRDConstants;
import Utils.Color.ColorHelpers;
import Scene.ShadingData;

struct NRDMaterialFactors
{
    float3 diffuse;
    float3 specular;
};

NRDMaterialFactors getNRDMaterialFactors(ShadingData sd, bool demodulate)
{
    NRDMaterialFactors f;
    f.diffuse = f.specular = 1.f;
    if (demodulate)
    {
        f.diffuse = max(0.04f, (1.f - sd.diffuseTransmission) * (1.f - sd.specularTransmission) * sd.diffuse);
        if ((getBSDFLobes(sd) & (uint)LobeType::DeltaTransmission) == 0)
        {
            float3 albedo = (1.f - sd.specularTransmission) * sd.specular;
            f.specular = max(0.04f, approxSpecularIntegralGGX(albedo, sd.linearRoughness * sd.linearRoughness, saturate(dot(sd.N, sd.V))));
        }
    }
    // Reflectance outputs are RGBA16Float. Use the exact factor that final composition will read.
    f.diffuse = f16tof32(f32tof16(f.diffuse));
    f.specular = f16tof32(f32tof16(f.specular));
    return f;
}

// Choose a guide from a real path. Distances cannot be added like RGB.
struct NRDSignal
{
    float3 radiance;
    float hitDist;
    float guideWeight;

    [mutating] void add(float3 value, float distance)
    {
        radiance += value;
        float strength = luminance(value);
        if (strength > guideWeight)
        {
            guideWeight = strength;
            hitDist = clamp(distance, 0.f, kNRDInvalidPathLength);
        }
    }

    float4 accumulate(float4 previous, float previousGuideWeight, float3 factor, float invSpp)
    {
        float distance = guideWeight > previousGuideWeight ? hitDist : previous.a;
        return float4(previous.rgb + (radiance / factor) * invSpp, distance);
    }
};
