/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#include "RTPT.h"
#include "RenderGraph/RenderPassHelpers.h"
#include <cmath>
#include <fstream>

namespace
{
    const char kDesc[] = "ReSTIR path tracer with fixed first-hit reconnection";

    const std::string kGeneratePathsFilename = "RenderPasses/RTPT/GeneratePaths.cs.slang";
    const std::string kTracePassFilename = "RenderPasses/RTPT/TracePass.cs.slang";
    const std::string kReflectTypesFile = "RenderPasses/RTPT/ReflectTypes.cs.slang";
    const std::string kSpatialReusePassFile = "RenderPasses/RTPT/SpatialReuse.cs.slang";
    const std::string kTemporalReusePassFile = "RenderPasses/RTPT/TemporalReuse.cs.slang";

    // Render pass inputs and outputs.
    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "motionVectors";
    const std::string kInputDirectLighting = "directLighting";
    const std::string kInputDirectEmission = "directEmission";
    const std::string kInputDirectDiffuse = "directDiffuseRadianceHitDist";
    const std::string kInputDirectSpecular = "directSpecularRadianceHitDist";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format", false, ResourceFormat::Unknown },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */, ResourceFormat::RG32Float },
        { kInputDirectLighting,    "gDirectLighting",     "Sample count buffer (integer format)", true /* optional */, ResourceFormat::RGBA32Float },
        { kInputDirectEmission, "gDirectEmission", "Primary surface emission", true, ResourceFormat::RGBA32Float },
        { kInputDirectDiffuse, "gDirectDiffuseRadianceHitDist", "Direct diffuse radiance and hit distance", true, ResourceFormat::RGBA32Float },
        { kInputDirectSpecular, "gDirectSpecularRadianceHitDist", "Direct specular radiance and hit distance", true, ResourceFormat::RGBA32Float },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputSpecularAlbedo = "specularAlbedo";
    const std::string kOutputIndirectAlbedo = "indirectAlbedo";
    const std::string kOutputNormal = "normal";
    const std::string kOutputReflectionPosW = "reflectionPosW";
    const std::string kOutputRayCount = "rayCount";
    const std::string kOutputPathLength = "pathLength";
    const std::string kOutputDebug = "debug";
    const std::string kOutputTime = "time";
    const std::string kOutputNRDDiffuseRadianceHitDist = "nrdDiffuseRadianceHitDist";
    const std::string kOutputNRDSpecularRadianceHitDist = "nrdSpecularRadianceHitDist";
    const std::string kOutputNRDResidualRadianceHitDist = "nrdResidualRadianceHitDist";
    const std::string kOutputNRDEmission = "nrdEmission";
    const std::string kOutputNRDDiffuseReflectance = "nrdDiffuseReflectance";
    const std::string kOutputNRDSpecularReflectance = "nrdSpecularReflectance";


    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,                 "gOutputColor",                 "Output color (linear)", true /* optional */ },
        { kOutputAlbedo,                "gOutputAlbedo",                "Output albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputNormal,                "gOutputNormal",                "Output normal (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputRayCount,              "",                             "Per-pixel ray count", true /* optional */, ResourceFormat::R32Uint },
        { kOutputPathLength,            "",                             "Per-pixel path length", true /* optional */, ResourceFormat::R32Uint },
        { kOutputDebug,                 "",                             "Debug output", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputTime,                  "",                             "Per-pixel time", true /* optional */, ResourceFormat::R32Uint },
        { kOutputSpecularAlbedo,                "gOutputSpecularAlbedo",                "Output specular albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputIndirectAlbedo,                "gOutputIndirectAlbedo",                "Output indirect albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputReflectionPosW,                "gOutputReflectionPosW",                "Output reflection pos (world space)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDiffuseRadianceHitDist,     "gOutputNRDDiffuseRadianceHitDist",     "Output demodulated diffuse color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDSpecularRadianceHitDist,    "gOutputNRDSpecularRadianceHitDist",    "Output demodulated specular color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDResidualRadianceHitDist,    "gOutputNRDResidualRadianceHitDist",    "Output residual color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDEmission,                   "gOutputNRDEmission",                   "Output primary surface emission", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputNRDDiffuseReflectance,         "gOutputNRDDiffuseReflectance",         "Output primary surface diffuse reflectance", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputNRDSpecularReflectance,        "gOutputNRDSpecularReflectance",        "Output primary surface specular reflectance", true /* optional */, ResourceFormat::RGBA16Float },
    };

    // UI variables.
    const Gui::DropdownList kColorFormatList =
    {
        { (uint32_t)ColorFormat::RGBA32F, "RGBA32F (128bpp)" },
        { (uint32_t)ColorFormat::LogLuvHDR, "LogLuvHDR (32bpp)" },
    };

    const Gui::DropdownList kMISHeuristicList =
    {
        { (uint32_t)MISHeuristic::Balance, "Balance heuristic" },
        { (uint32_t)MISHeuristic::PowerTwo, "Power heuristic (exp=2)" },
        { (uint32_t)MISHeuristic::PowerExp, "Power heuristic" },
    };

    const Gui::DropdownList kSpatialReusePatternList =
    {
        { (uint32_t)SpatialReusePattern::Default, std::string("Default")},
        { (uint32_t)SpatialReusePattern::SmallWindow, std::string("Small Window")},
    };

    const Gui::DropdownList kEmissiveSamplerList =
    {
        { (uint32_t)EmissiveLightSamplerType::Uniform, "Uniform" },
        { (uint32_t)EmissiveLightSamplerType::LightBVH, "LightBVH" },
        { (uint32_t)EmissiveLightSamplerType::Power, "Power" },
    };

    const Gui::DropdownList kLODModeList =
    {
        { (uint32_t)TexLODMode::Mip0, "Mip0" },
        { (uint32_t)TexLODMode::RayDiffs, "Ray Diffs" }
    };

    // Scripting options.
    const std::string kSamplesPerPixel = "samplesPerPixel";
    const std::string kMaxSurfaceBounces = "maxSurfaceBounces";
    const std::string kMaxDiffuseBounces = "maxDiffuseBounces";
    const std::string kMaxSpecularBounces = "maxSpecularBounces";
    const std::string kMaxTransmissionBounces = "maxTransmissionBounces";
    const std::string kAdjustShadingNormals = "adjustShadingNormals";
    const std::string kLODBias = "lodBias";
    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kUseBSDFSampling = "useBSDFSampling";
    const std::string kUseNEE = "useNEE";
    const std::string kUseMIS = "useMIS";
    const std::string kUseRussianRoulette = "useRussianRoulette";
    const std::string kScreenSpaceReSTIROptions = "screenSpaceReSTIROptions";
    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kMaxNestedMaterials = "maxNestedMaterials";
    const std::string kUseLightsInDielectricVolumes = "useLightsInDielectricVolumes";
    const std::string kLimitTransmission = "limitTransmission";
    const std::string kMaxTransmissionReflectionDepth = "maxTransmissionReflectionDepth";
    const std::string kMaxTransmissionRefractionDepth = "maxTransmissionRefractionDepth";
    const std::string kDisableCaustics = "disableCaustics";
    const std::string kSpecularRoughnessThreshold = "specularRoughnessThreshold";
    const std::string kDisableDirectIllumination = "disableDirectIllumination";
    const std::string kColorFormat = "colorFormat";
    const std::string kMISHeuristic = "misHeuristic";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";
    const std::string kPrimaryLodMode = "primaryLodMode";
    const std::string kUseNRDDemodulation = "useNRDDemodulation";

    const std::string kSpatialMisKind = "spatialMisKind";
    const std::string kTemporalMisKind = "temporalMisKind";
    const std::string kShiftStrategy = "shiftStrategy";
    const std::string kRejectShiftBasedOnJacobian = "rejectShiftBasedOnJacobian";
    const std::string kJacobianRejectionThreshold = "jacobianRejectionThreshold";

    const std::string kTemporalHistoryLength = "temporalHistoryLength";
    const std::string kUseMaxHistory = "useMaxHistory";
    const std::string kSeedOffset = "seedOffset";
    const std::string kEnableTemporalReuse = "enableTemporalReuse";
    const std::string kEnableSpatialReuse = "enableSpatialReuse";
    const std::string kNumSpatialRounds = "numSpatialRounds";
    const std::string kPathSamplingMode = "pathSamplingMode";
    const std::string kEnableTemporalReprojection = "enableTemporalReprojection";
    const std::string kNoResamplingForTemporalReuse = "noResamplingForTemporalReuse";
    const std::string kSpatialNeighborCount = "spatialNeighborCount";
    const std::string kFeatureBasedRejection = "featureBasedRejection";
    const std::string kSpatialReusePattern = "spatialReusePattern";
    const std::string kSmallWindowRestirWindowRadius = "smallWindowRestirWindowRadius";
    const std::string kSpatialReuseRadius = "spatialReuseRadius";
    const std::string kUseDirectLighting = "useDirectLighting";
    const std::string kSeparatePathBSDF = "separatePathBSDF";
    const std::string kCandidateSamples = "candidateSamples";
    const std::string kTemporalUpdateForDynamicScene = "temporalUpdateForDynamicScene";
    const std::string kEnableRayStats = "enableRayStats";
    const std::string kSharcPretrainSpp = "sharcPretrainSpp";
    const std::string kAlwaysEnabledOptions[] = {
        kEnableTemporalReuse, kEnableSpatialReuse, kUseBSDFSampling, "pgSpatialCachePrimary"
    };

    const uint32_t kNeighborOffsetCount = 8192;
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("RTPT", kDesc, RTPT::create);
    ScriptBindings::registerBinding(RTPT::registerBindings);
}


void RTPT::updateDict(const Dictionary& dict)
{
    const bool pgWasEnabled = mEnablePG;
    const bool sharcWasEnabled = mEnableSharc;
    const bool collectedPGStats = mPGGenerateStats;
    const bool collectedPGTemporalStats = mPGTemporalStats;
    const bool collectedPGSpatialStats = mPGSpatialStats;
    const float previousPGFraction = mPGDiffuseFraction;
    const float previousPGSceneScale = mPGSceneScale;
    const auto trainingSettings = getSharcTrainingSettings();
    bool needToReset = parseDictionary(dict);
    updateGuidingOptions(pgWasEnabled, sharcWasEnabled);
    size_t cacheOptionCount = size_t(dict.keyExists("enablePG")) + size_t(dict.keyExists("enableSHARC")) +
        size_t(dict.keyExists("sharcUseCache")) + size_t(dict.keyExists("sharcTemporalUseCache")) +
        size_t(dict.keyExists("sharcFreeze")) +
        size_t(dict.keyExists(kSharcPretrainSpp)) + size_t(dict.keyExists("pgSceneScale")) +
        size_t(dict.keyExists("pgDiffuseFraction")) + size_t(dict.keyExists("pgGenerateStats")) +
        size_t(dict.keyExists("pgTemporalStats")) + size_t(dict.keyExists("pgSpatialStats"));
    for (const auto& key : kAlwaysEnabledOptions) cacheOptionCount += size_t(dict.keyExists(key));
    if (cacheOptionCount && dict.size() == cacheOptionCount)
    {
        mRecompile |= collectedPGStats != mPGGenerateStats ||
            collectedPGTemporalStats != mPGTemporalStats || collectedPGSpatialStats != mPGSpatialStats;
        mOptionsChanged |= previousPGFraction != mPGDiffuseFraction || previousPGSceneScale != mPGSceneScale;
        return;
    }
    if (needToReset)
    {
        validateOptions();
        mSharcNeedsClear |= trainingSettings != getSharcTrainingSettings() || dict.keyExists(kLightBVHOptions);
        mOptionsChanged = true;
        mRecompile = true;
        mParams.frameCount = 0;
    }
}

void RTPT::updateGuidingOptions(bool pgWasEnabled, bool sharcWasEnabled)
{
    if (pgWasEnabled != mEnablePG)
    {
        mRecompile = true;
        mOptionsChanged = true;
    }
    if (sharcWasEnabled != mEnableSharc)
    {
        mSharcPretrainPending = true;
        mSharcPretrainedSpp = 0;
        mRecompile = true;
        mOptionsChanged = true;
    }
}

void RTPT::initDict()
{
    const auto frameDim = mParams.frameDim;
    const auto screenTiles = mParams.screenTiles;
    Init();
    mEnablePG = true;
    mEnableSharc = true;
    mPGDiffuseFraction = kDefaultPGDiffuseFraction;
    mPGGenerateStats = false;
    mPGTemporalStats = false;
    mPGSpatialStats = false;
    mPGSceneScale = kDefaultPGSceneScale;
    releasePG();
    mSharcFreeze = false;
    mSharcNeedsClear = true;
    mSharcPretrainSpp = kDefaultSharcPretrainSpp;
    mSharcPretrainedSpp = 0;
    mSharcPretrainPending = true;
    mJacobianRejectionOverride = false;
    mTemporalUpdateOverride = false;
    mParams.frameDim = frameDim;
    mParams.screenTiles = screenTiles;
    parseDictionary({});
    validateOptions();
    if (mpScene)
    {
        const bool animated = mpScene->hasAnimation() && mpScene->isAnimated();
        mParams.rejectShiftBasedOnJacobian = animated;
        mStaticParams.temporalUpdateForDynamicScene = animated;
    }
    mOptionsChanged = true;
    mRecompile = true;
    mParams.frameCount = 0;
}

void RTPT::registerBindings(pybind11::module& m)
{
    pybind11::class_<RTPT, RenderPass, RTPT::SharedPtr> pass(m, "RTPT");
    pass.def_property_readonly("pixelStats", &RTPT::getPixelStats);
    pass.def("getSharcStats", &RTPT::getSharcStats);
    pass.def("getPGStats", &RTPT::getPGStats);
    pass.def("getSharcFreshReservoirs", &RTPT::getSharcFreshReservoirs);
    pass.def("getSharcTemporalReservoirs", &RTPT::getSharcTemporalReservoirs);
    pass.def("getSharcSpatialReservoirs", &RTPT::getSharcSpatialReservoirs);
    pass.def("clearSharc", [](RTPT& pt) { pt.mSharcNeedsClear = true; });
    pass.def_property_readonly("reservoirStride", [](const RTPT& pt)
    {
        return pt.mpOutputReservoirs ? pt.mpOutputReservoirs->getElementSize() : 0u;
    });

    pass.def_property("useFixedSeed",
        [](const RTPT* pt) { return pt->mParams.useFixedSeed ? true : false; },
        [](RTPT* pt, bool value) { pt->mParams.useFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const RTPT* pt) { return pt->mParams.fixedSeed; },
        [](RTPT* pt, uint32_t value) { pt->mParams.fixedSeed = value; }
    );
}

std::string RTPT::getDesc() { return kDesc; }

RTPT::SharedPtr RTPT::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new RTPT(dict));
}

RTPT::RTPT(const Dictionary& dict)
{
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw std::exception("Raytracing Tier 1.1 is not supported by the current device");
    }

    parseDictionary(dict);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mStaticParams.sampleGenerator);

    // Create neighbor offset texture.
    mpNeighborOffsets = createNeighborOffsetTexture(kNeighborOffsetCount);

    // Create programs.
    auto defines = mStaticParams.getDefines(*this);

    mpGeneratePaths = ComputePass::create(kGeneratePathsFilename, "main", defines, false);
    mpReflectTypes = ComputePass::create(kReflectTypesFile, "main", defines, false);

    {
        Program::Desc desc;
        desc.addShaderLibrary(kTracePassFilename).csEntry("main").setShaderModel("6_5");
        mpTracePass = ComputePass::create(desc, defines, false);
    }

    {
        Program::Desc desc;
        desc.addShaderLibrary(kSpatialReusePassFile).csEntry("main").setShaderModel("6_5");
        mpSpatialReusePass = ComputePass::create(desc, defines, false);
    }

    {
        Program::Desc desc;
        desc.addShaderLibrary(kTemporalReusePassFile).csEntry("main").setShaderModel("6_5");
        mpTemporalReusePass = ComputePass::create(desc, defines, false);
    }

    mpPixelStats = PixelStats::create();
    mpPixelDebug = PixelDebug::create(1000);
}

bool RTPT::parseDictionary(const Dictionary& dict)
{
    // Reject incompatible configurations before mutating either the cache or the reuse settings.
    for (const auto& key : kAlwaysEnabledOptions)
    {
        if (dict.keyExists(key) && !bool(dict[key]))
            throw std::invalid_argument("RTPT always enables '" + key + "'.");
    }
    if (dict.keyExists(kTemporalMisKind) && uint32_t(dict[kTemporalMisKind]) != uint32_t(StaticParams::temporalMisKind))
        throw std::invalid_argument("RTPT uses Talbot MIS for Temporal reuse.");
    if (dict.keyExists(kSpatialMisKind) && uint32_t(dict[kSpatialMisKind]) != uint32_t(StaticParams::spatialMisKind))
        throw std::invalid_argument("RTPT uses Pairwise MIS for Spatial reuse.");

    const bool enableSharc = dict.keyExists("enableSHARC") ? bool(dict["enableSHARC"]) : mEnableSharc;
    if ((dict.keyExists("sharcUseCache") && bool(dict["sharcUseCache"]) != enableSharc) ||
        (dict.keyExists("sharcTemporalUseCache") && bool(dict["sharcTemporalUseCache"]) != enableSharc))
        throw std::invalid_argument("RTPT controls both SHaRC queries with 'enableSHARC'.");
    const uint32_t pretrainSpp = dict.keyExists(kSharcPretrainSpp) ? uint32_t(dict[kSharcPretrainSpp]) : mSharcPretrainSpp;
    const float pgSceneScale = dict.keyExists("pgSceneScale") ? float(dict["pgSceneScale"]) : mPGSceneScale;
    if (!(pgSceneScale > 0.f) || !std::isfinite(pgSceneScale))
        throw std::invalid_argument("pgSceneScale must be finite and greater than zero.");
    const float pgFraction = dict.keyExists("pgDiffuseFraction") ? float(dict["pgDiffuseFraction"]) : mPGDiffuseFraction;
    if (!(pgFraction >= 0.f && pgFraction < 1.f))
        throw std::invalid_argument("pgDiffuseFraction must be in [0, 1) to preserve native diffuse sampling support.");

    bool needToReset = true;
    for (const auto& [key, value] : dict)
    {
        if (key == kSamplesPerPixel) mStaticParams.samplesPerPixel = value;
        else if (key == kMaxSurfaceBounces) mStaticParams.maxSurfaceBounces = value;
        else if (key == kMaxDiffuseBounces) mStaticParams.maxDiffuseBounces = value;
        else if (key == kMaxSpecularBounces) mStaticParams.maxSpecularBounces = value;
        else if (key == kMaxTransmissionBounces) mStaticParams.maxTransmissionBounces = value;
        else if (key == kAdjustShadingNormals) mStaticParams.adjustShadingNormals = value;
        else if (key == kLODBias) mParams.lodBias = value;
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mParams.fixedSeed = value; mParams.useFixedSeed = true; }
        else if (key == kUseNEE) mStaticParams.useNEE = value;
        else if (key == kUseMIS) mStaticParams.useMIS = value;
        else if (key == kUseRussianRoulette) mStaticParams.useRussianRoulette = value;
        else if (key == kUseAlphaTest) mStaticParams.useAlphaTest = value;
        else if (key == kMaxNestedMaterials) mStaticParams.maxNestedMaterials = value;
        else if (key == kUseLightsInDielectricVolumes) mStaticParams.useLightsInDielectricVolumes = value;
        else if (key == kLimitTransmission) mStaticParams.limitTransmission = value;
        else if (key == kMaxTransmissionReflectionDepth) mStaticParams.maxTransmissionReflectionDepth = value;
        else if (key == kMaxTransmissionRefractionDepth) mStaticParams.maxTransmissionRefractionDepth = value;
        else if (key == kDisableCaustics) mStaticParams.disableCaustics = value;
        else if (key == kSpecularRoughnessThreshold) mParams.specularRoughnessThreshold = value;
        else if (key == kDisableDirectIllumination) mStaticParams.disableDirectIllumination = value;
        else if (key == kPrimaryLodMode)  mStaticParams.primaryLodMode = static_cast<TexLODMode>(static_cast<uint32_t>(value));
        // Denoising parameters
        else if (key == kUseNRDDemodulation) mStaticParams.useNRDDemodulation = value;
        else if (key == kColorFormat) mStaticParams.colorFormat = static_cast<ColorFormat>(static_cast<uint32_t>(value));
        else if (key == kMISHeuristic) mStaticParams.misHeuristic = static_cast<MISHeuristic>(static_cast<uint32_t>(value));
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = static_cast<EmissiveLightSamplerType>(static_cast<uint32_t>(value));
        else if (key == kLightBVHOptions) mLightBVHOptions = value;
        else if (key == kShiftStrategy)
        {
            if (static_cast<uint32_t>(value) != uint32_t(ShiftMapping::Reconnection))
                throw std::invalid_argument("RTPT supports fixed reconnection only.");
        }
        else if (key == kRejectShiftBasedOnJacobian)
        {
            mParams.rejectShiftBasedOnJacobian = value;
            mJacobianRejectionOverride = true;
        }
        else if (key == kJacobianRejectionThreshold) mParams.jacobianRejectionThreshold = value;
        else if (key == kTemporalHistoryLength) mTemporalHistoryLength = value;
        else if (key == kUseMaxHistory) mUseMaxHistory = value;
        else if (key == kSeedOffset) mSeedOffset = value;
        else if (key == kNumSpatialRounds) mNumSpatialRounds = value;
        else if (key == kPathSamplingMode)
        {
            if (static_cast<uint32_t>(value) != uint32_t(PathSamplingMode::ReSTIR))
                throw std::invalid_argument("RTPT supports ReSTIR path sampling only.");
        }
        else if (key == kEnableTemporalReprojection) mEnableTemporalReprojection = value;
        else if (key == kNoResamplingForTemporalReuse) mNoResamplingForTemporalReuse = value;
        else if (key == kSpatialNeighborCount) mSpatialNeighborCount = value;
        else if (key == kFeatureBasedRejection) mFeatureBasedRejection = value;
        else if (key == kSpatialReusePattern) mSpatialReusePattern = static_cast<SpatialReusePattern>(static_cast<uint32_t>(value));
        else if (key == kSmallWindowRestirWindowRadius) mSmallWindowRestirWindowRadius = value;
        else if (key == kSpatialReuseRadius) mSpatialReuseRadius = value;
        else if (key == kUseDirectLighting) mUseDirectLighting = value;
        else if (key == kSeparatePathBSDF) mStaticParams.separatePathBSDF = value;
        else if (key == kCandidateSamples) mStaticParams.candidateSamples = value;
        else if (key == "enablePG") mEnablePG = value;
        else if (key == "enableSHARC") mEnableSharc = enableSharc;
        else if (key == "pgDiffuseFraction") mPGDiffuseFraction = pgFraction;
        else if (key == "pgGenerateStats") mPGGenerateStats = value;
        else if (key == "pgTemporalStats") mPGTemporalStats = value;
        else if (key == "pgSpatialStats") mPGSpatialStats = value;
        else if (key == "pgSceneScale")
        {
            mPGResetPending |= mPGSceneScale != pgSceneScale;
            mPGSceneScale = pgSceneScale;
        }
        else if (key == "sharcFreeze") mSharcFreeze = value;
        else if (key == kSharcPretrainSpp) mSharcPretrainSpp = pretrainSpp;
        else if (key == kTemporalUpdateForDynamicScene)
        {
            mStaticParams.temporalUpdateForDynamicScene = value;
            mTemporalUpdateOverride = true;
        }
        else if (key == kEnableRayStats) mEnableRayStats = value;
        else if (key == kTemporalMisKind || key == kSpatialMisKind || key == "sharcUseCache" || key == "sharcTemporalUseCache" ||
            std::find(std::begin(kAlwaysEnabledOptions), std::end(kAlwaysEnabledOptions), key) != std::end(kAlwaysEnabledOptions)) continue;
        else logWarning("Unknown field '" + key + "' in RTPT dictionary");
    }

    // Check for deprecated bounces configuration.
    if (dict.keyExists("maxBounces"))
    {
        logWarning("'maxBounces' is deprecated. Use 'maxSurfaceBounces' instead.");
        if (!dict.keyExists(kMaxSurfaceBounces)) mStaticParams.maxSurfaceBounces = dict["maxBounces"];
    }
    if (dict.keyExists("maxNonSpecularBounces"))
    {
        logWarning("'maxNonSpecularBounces' is deprecated. Use 'maxDiffuseBounces' instead.");
        if (!dict.keyExists(kMaxDiffuseBounces)) mStaticParams.maxDiffuseBounces = dict["maxNonSpecularBounces"];
    }

    // Initialize the other bounce counts to 'maxSurfaceBounces' if they weren't explicitly set.
    if (mStaticParams.maxDiffuseBounces == -1) mStaticParams.maxDiffuseBounces = mStaticParams.maxSurfaceBounces;
    if (mStaticParams.maxSpecularBounces == -1) mStaticParams.maxSpecularBounces = mStaticParams.maxSurfaceBounces;
    if (mStaticParams.maxTransmissionBounces == -1) mStaticParams.maxTransmissionBounces = mStaticParams.maxSurfaceBounces;

    bool maxSurfaceBouncesNeedsAdjustment =
        mStaticParams.maxSurfaceBounces < mStaticParams.maxDiffuseBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxSpecularBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxTransmissionBounces;

    // Show a warning if maxSurfaceBounces will be adjusted in validateOptions().
    if ((dict.keyExists("maxSurfaceBounces") || dict.keyExists("maxBounces")) && maxSurfaceBouncesNeedsAdjustment)
    {
        logWarning("'maxSurfaceBounces' is set lower than 'maxDiffuseBounces', 'maxSpecularBounces' or 'maxTransmissionBounces' and will be increased.");
    }

    // Show a warning for deprecated 'useNestedDielectrics'.
    if (dict.keyExists("useNestedDielectrics"))
    {
        logWarning("'useNestedDielectrics' is deprecated. Support for nested dielectrics is always enabled now.");
    }

    return needToReset;
}

void RTPT::validateOptions()
{
    mStaticParams.candidateSamples = std::clamp(mStaticParams.candidateSamples, 1u, kMaxSamplesPerPixel);
    mNumSpatialRounds = std::clamp(mNumSpatialRounds, 1, 5);
    if (mParams.specularRoughnessThreshold < 0.f || mParams.specularRoughnessThreshold > 1.f)
    {
        logError("'specularRoughnessThreshold' has invalid value. Clamping to range [0,1].");
        mParams.specularRoughnessThreshold = clamp(mParams.specularRoughnessThreshold, 0.f, 1.f);
    }

    // Static parameters.
    if (mStaticParams.samplesPerPixel < 1 || mStaticParams.samplesPerPixel > kMaxSamplesPerPixel)
    {
        logError("'samplesPerPixel' must be in the range [1, " + std::to_string(kMaxSamplesPerPixel) + "]. Clamping to this range.");
        mStaticParams.samplesPerPixel = std::clamp(mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }

    auto clampBounces = [](uint32_t& bounces, const std::string& name)
    {
        if (bounces > kMaxBounces)
        {
            logError("'" + name + "' exceeds the maximum supported bounces. Clamping to " + std::to_string(kMaxBounces));
            bounces = kMaxBounces;
        }
    };

    clampBounces(mStaticParams.maxSurfaceBounces, kMaxSurfaceBounces);
    clampBounces(mStaticParams.maxDiffuseBounces, kMaxDiffuseBounces);
    clampBounces(mStaticParams.maxSpecularBounces, kMaxSpecularBounces);
    clampBounces(mStaticParams.maxTransmissionBounces, kMaxTransmissionBounces);

    // Make sure maxSurfaceBounces is at least as many as any of diffuse, specular or transmission.
    uint32_t minSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxSurfaceBounces, minSurfaceBounces);

    if (mStaticParams.maxTransmissionReflectionDepth > mStaticParams.maxTransmissionBounces)
    {
        logWarning("'maxTransmissionReflectionDepth' exceeds `maxTransmissionBounces`. Clamping to " + std::to_string(mStaticParams.maxTransmissionBounces));
        mStaticParams.maxTransmissionReflectionDepth = mStaticParams.maxTransmissionBounces;
    }

    if (mStaticParams.maxTransmissionRefractionDepth > mStaticParams.maxTransmissionBounces)
    {
        logWarning("'maxTransmissionRefractionDepth' exceeds `maxTransmissionBounces`. Clamping to " + std::to_string(mStaticParams.maxTransmissionBounces));
        mStaticParams.maxTransmissionRefractionDepth = mStaticParams.maxTransmissionBounces;
    }

    if (mStaticParams.primaryLodMode == TexLODMode::RayCones)
    {
        logError("Unsupported tex lod mode. Defaulting to Mip0.");
        mStaticParams.primaryLodMode = TexLODMode::Mip0;
    }
}

Dictionary RTPT::getScriptingDictionary()
{
    if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Dictionary d;
    d[kSamplesPerPixel] = mStaticParams.samplesPerPixel;
    d[kMaxSurfaceBounces] = mStaticParams.maxSurfaceBounces;
    d[kMaxDiffuseBounces] = mStaticParams.maxDiffuseBounces;
    d[kMaxSpecularBounces] = mStaticParams.maxSpecularBounces;
    d[kMaxTransmissionBounces] = mStaticParams.maxTransmissionBounces;
    d[kAdjustShadingNormals] = mStaticParams.adjustShadingNormals;
    d[kLODBias] = mParams.lodBias;
    d[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed) d[kFixedSeed] = mParams.fixedSeed;
    d[kUseNEE] = mStaticParams.useNEE;
    d[kUseMIS] = mStaticParams.useMIS;
    d[kUseRussianRoulette] = mStaticParams.useRussianRoulette;
    d[kUseAlphaTest] = mStaticParams.useAlphaTest;
    d[kMaxNestedMaterials] = mStaticParams.maxNestedMaterials;
    d[kUseLightsInDielectricVolumes] = mStaticParams.useLightsInDielectricVolumes;
    d[kLimitTransmission] = mStaticParams.limitTransmission;
    d[kMaxTransmissionReflectionDepth] = mStaticParams.maxTransmissionReflectionDepth;
    d[kMaxTransmissionRefractionDepth] = mStaticParams.maxTransmissionRefractionDepth;
    d[kDisableCaustics] = mStaticParams.disableCaustics;
    d[kSpecularRoughnessThreshold] = mParams.specularRoughnessThreshold;
    d[kDisableDirectIllumination] = mStaticParams.disableDirectIllumination;
    d[kPrimaryLodMode] = uint32_t(mStaticParams.primaryLodMode);
    d[kColorFormat] = uint32_t(mStaticParams.colorFormat);
    d[kMISHeuristic] = uint32_t(mStaticParams.misHeuristic);
    d[kMISPowerExponent] = mStaticParams.misPowerExponent;
    d[kEmissiveSampler] = uint32_t(mStaticParams.emissiveSampler);
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) d[kLightBVHOptions] = mLightBVHOptions;
    d[kSpatialMisKind] = uint32_t(mStaticParams.spatialMisKind);
    d[kTemporalMisKind] = uint32_t(mStaticParams.temporalMisKind);
    d[kShiftStrategy] = uint32_t(ShiftMapping::Reconnection);
    d[kRejectShiftBasedOnJacobian] = mParams.rejectShiftBasedOnJacobian;
    d[kJacobianRejectionThreshold] = mParams.jacobianRejectionThreshold;
    d[kTemporalHistoryLength] = mTemporalHistoryLength;
    d[kUseMaxHistory] = mUseMaxHistory;
    d[kSeedOffset] = mSeedOffset;
    d[kNumSpatialRounds] = mNumSpatialRounds;
    d[kPathSamplingMode] = uint32_t(PathSamplingMode::ReSTIR);
    d[kEnableTemporalReprojection] = mEnableTemporalReprojection;
    d[kNoResamplingForTemporalReuse] = mNoResamplingForTemporalReuse;
    d[kSpatialNeighborCount] = mSpatialNeighborCount;
    d[kFeatureBasedRejection] = mFeatureBasedRejection;
    d[kSpatialReusePattern] = uint32_t(mSpatialReusePattern);
    d[kSmallWindowRestirWindowRadius] = mSmallWindowRestirWindowRadius;
    d[kSpatialReuseRadius] = mSpatialReuseRadius;
    d[kUseDirectLighting] = mUseDirectLighting;
    d[kSeparatePathBSDF] = mStaticParams.separatePathBSDF;
    d[kCandidateSamples] = mStaticParams.candidateSamples;
    d["pgDiffuseFraction"] = mPGDiffuseFraction;
    d["pgGenerateStats"] = mPGGenerateStats;
    d["pgTemporalStats"] = mPGTemporalStats;
    d["pgSpatialStats"] = mPGSpatialStats;
    d["pgSceneScale"] = mPGSceneScale;
    d["sharcFreeze"] = mSharcFreeze;
    d[kSharcPretrainSpp] = mSharcPretrainSpp;
    d[kTemporalUpdateForDynamicScene] = mStaticParams.temporalUpdateForDynamicScene;
    d[kEnableRayStats] = mEnableRayStats;
    // Denoising parameters
    d[kUseNRDDemodulation] = mStaticParams.useNRDDemodulation;
    d["enablePG"] = mEnablePG;
    d["enableSHARC"] = mEnableSharc;
    d["sharcUseCache"] = mEnableSharc;
    d["sharcTemporalUseCache"] = mEnableSharc;
    for (const auto& key : kAlwaysEnabledOptions) d[key] = true;

    return d;
}

RenderPassReflection RTPT::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void RTPT::compile(RenderContext* pContext, const CompileData& compileData)
{
    if (mParams.frameDim != compileData.defaultTexDims) mReservoirFrameCount = 0;
    mParams.frameDim = compileData.defaultTexDims;
    if (mParams.frameDim.x > kMaxFrameDimension || mParams.frameDim.y > kMaxFrameDimension)
    {
        logError("Frame dimensions up to " + std::to_string(kMaxFrameDimension) + " pixels width/height are supported.");
    }

    // Tile dimensions have to be powers-of-two.
    assert(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    assert(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    mParams.screenTiles = div_round_up(mParams.frameDim, kScreenTileDim);

    mVarsChanged = true;
}

void RTPT::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    releasePG();
    mpSharcPGUpdate.reset();
    mSharcNeedsClear = true;
    mSharcPretrainPending = true;
    mSharcPretrainedSpp = 0;
    mSharcPreviousBounds.clear();
    mSharcMotionBounds.clear();
    mpSharcUpdate.reset();
    mParams.frameCount = 0;
    mReservoirFrameCount = 0;

    resetLighting();

    if (mpScene)
    {
        if (is_set(pScene->getPrimitiveTypes(), PrimitiveTypeFlags::Custom)) logError("This render pass does not support custom primitives.");

        // check if the scene is dynamic
        const bool animated = mpScene->hasAnimation() && mpScene->isAnimated();
        if (!mJacobianRejectionOverride) mParams.rejectShiftBasedOnJacobian = animated;
        if (!mTemporalUpdateOverride) mStaticParams.temporalUpdateForDynamicScene = animated;

        // Prepare our programs for the scene.
        Shader::DefineList defines = mpScene->getSceneDefines();

        mpGeneratePaths->getProgram()->addDefines(defines);
        mpTracePass->getProgram()->addDefines(defines);
        mpReflectTypes->getProgram()->addDefines(defines);

        mpSpatialReusePass->getProgram()->addDefines(defines);
        mpTemporalReusePass->getProgram()->addDefines(defines);

        validateOptions();

        mRecompile = true;
    }

    mOptionsChanged = true;
}

void RTPT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mPGGenerateBoundVersion.reset();
    mPGTemporalBoundVersion.reset();
    mPGSpatialBoundVersions.clear();
    if (!beginFrame(pRenderContext, renderData)) return;
    renderData.getDictionary()["enableScreenSpaceReSTIR"] = mUseDirectLighting;

    const uint32_t numPasses = mStaticParams.samplesPerPixel;

    for (uint32_t restir_i = 0; restir_i < numPasses; restir_i++)
    {
        {
            // Update shader program specialization.
            updatePrograms();

            // Prepare resources.
            prepareResources(pRenderContext, renderData);

            // Prepare the path tracer parameter block.
            // This should be called after all resources have been created.
            preparePathTracer(renderData);
            if (restir_i == 0)
            {
                if (mEnablePG) preparePG(pRenderContext);
                else if (mpPGBuildKeys) releasePG();
                updateSharc(pRenderContext);
                if (mEnablePG && !mPGPublishedThisFrame)
                {
                    // Falcor collects the previous frame's timer when it encounters the same event again.
                    PROFILE("PG Publish");
                }
            }

            if (const auto& texture = renderData[kOutputTime])
            {
                pRenderContext->clearUAV(texture->getUAV().get(), uint4(0));
            }

            mpPathTracerBlock->getRootVar()["gSppId"] = restir_i;
            mpPathTracerBlock->getRootVar()["gNumSpatialRounds"] = mNumSpatialRounds;

            if (restir_i == 0)
            {
                // Generate paths at primary hits.
                generatePaths(pRenderContext, renderData, 0);
            }

            tracePass(pRenderContext, renderData, mpTracePass, "tracePass", 0);
        }

        if (mReservoirFrameCount > 0)
            PathReusePass(pRenderContext, restir_i, renderData, true, 0, false);
        for (int spatialRoundId = 0; spatialRoundId < mNumSpatialRounds; spatialRoundId++)
            PathReusePass(pRenderContext, restir_i, renderData, false, spatialRoundId, spatialRoundId == mNumSpatialRounds - 1);

        if (restir_i == numPasses - 1) mReservoirFrameCount++;
        if (mNumSpatialRounds % 2 == 0)
            pRenderContext->copyResource(mpTemporalReservoirs[restir_i].get(), mpOutputReservoirs.get());
        if (restir_i == numPasses - 1)
            pRenderContext->copyResource(mpTemporalVBuffer.get(), renderData[kInputVBuffer].get());
        mParams.seed++;
    }

    mParams.frameCount++;

    endFrame(pRenderContext, renderData);
}

void RTPT::renderUI(Gui::Widgets& widget)
{
    const auto trainingSettings = getSharcTrainingSettings();
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Stats and debug options.
    dirty |= renderStatsUI(widget);

    dirty |= renderDebugUI(widget);
    if (auto group = widget.group("PG and SHaRC"))
    {
        const bool pgWasEnabled = mEnablePG;
        const bool sharcWasEnabled = mEnableSharc;
        group.checkbox("Enable path guiding", mEnablePG);
        group.checkbox("Enable SHaRC", mEnableSharc);
        updateGuidingOptions(pgWasEnabled, sharcWasEnabled);

        if (mEnablePG)
        {
            dirty |= group.var("Diffuse guide fraction", mPGDiffuseFraction, 0.f, 0.999f);
            if (group.var("PG grid scene scale", mPGSceneScale, 0.001f, 1000000.f))
            {
                mPGResetPending = true;
                dirty = true;
            }
        }
        if (mEnablePG || mEnableSharc)
        {
            group.var("Scene-load training (spp)", mSharcPretrainSpp);
            group.tooltip("Training paths per pixel from the starting camera. Zero skips pretraining.");
        }
        if (mEnableSharc)
        {
            group.checkbox("Freeze cache updates", mSharcFreeze);
            if (group.button("Clear cache")) mSharcNeedsClear = true;
        }
    }

    if (dirty)
    {
        validateOptions();
        mSharcNeedsClear |= trainingSettings != getSharcTrainingSettings();
        mOptionsChanged = true;
    }
}

Texture::SharedPtr RTPT::createNeighborOffsetTexture(uint32_t sampleCount)
{
    std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
    const int R = 254;
    const float phi2 = 1.f / 1.3247179572447f;
    float u = 0.5f;
    float v = 0.5f;
    for (uint32_t index = 0; index < sampleCount * 2;)
    {
        u += phi2;
        v += phi2 * phi2;
        if (u >= 1.f) u -= 1.f;
        if (v >= 1.f) v -= 1.f;

        float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
        if (rSq > 0.25f) continue;

        offsets[index++] = int8_t((u - 0.5f) * R);
        offsets[index++] = int8_t((v - 0.5f) * R);
    }

    return Texture::create1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
}

bool RTPT::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (mpScene && mpScene->hasAnimation())
    {
        if (gpFramework->getGlobalClock().isPaused())
        {
            if (widget.button("Resume Animation"))
                gpFramework->getGlobalClock().play();
        }
        else
        {
            if (widget.button("Pause Animation"))
                gpFramework->getGlobalClock().pause();
        }
    }

    dirty |= widget.checkbox("Direct lighting (ReSTIR DI)", mUseDirectLighting);
    if (auto group = widget.group("Path Reuse Controls", true))
    {
        if (widget.button("Clean Reservoirs")) mReservoirFrameCount = 0;
        dirty |= widget.var("Candidate Samples", mStaticParams.candidateSamples, 1u, 64u);
        if (widget.checkbox("Reject Shift based on Jacobian", mParams.rejectShiftBasedOnJacobian))
        {
            mJacobianRejectionOverride = true;
            dirty = true;
        }
        if (mParams.rejectShiftBasedOnJacobian)
            dirty |= widget.var("Shift rejection jacobian threshold", mParams.jacobianRejectionThreshold, 0.f, 100.f);
        dirty |= widget.var("Specular roughness threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        if (auto group = widget.group("Spatial reuse controls", true))
        {
            dirty |= group.var("Num Spatial Rounds", mNumSpatialRounds, 1, 5);
            dirty |= group.dropdown("Spatial Reuse Pattern", kSpatialReusePatternList, reinterpret_cast<uint32_t&>(mSpatialReusePattern));
            dirty |= group.checkbox("Feature-based rejection", mFeatureBasedRejection);

            if (SpatialReusePattern(mSpatialReusePattern) == SpatialReusePattern::SmallWindow)
            {
                dirty |= group.var("Window radius", mSmallWindowRestirWindowRadius, 0u, 32u);
            }
            else
            {
                dirty |= group.var("Spatial Neighbor Count", mSpatialNeighborCount, 0, 6);
                dirty |= group.var("Spatial Reuse Radius", mSpatialReuseRadius, 0.f, 100.f);
            }
        }

        if (auto group = widget.group("Temporal reuse controls", true))
        {
            dirty |= group.var("Temporal History Length", mTemporalHistoryLength, 0, 100);
            dirty |= group.checkbox("Use M capping", mUseMaxHistory);
            dirty |= group.checkbox("Temporal Reprojection", mEnableTemporalReprojection);
            if (group.checkbox("Temporal Update for Dynamic Scenes", mStaticParams.temporalUpdateForDynamicScene))
            {
                mTemporalUpdateOverride = true;
                dirty = true;
            }
            group.tooltip("Resample cached radiance in reconnection vertex in temporal reservoir for dynamic scenes (eliminate lags).");
            dirty |= group.checkbox("Disable Resampling in Temporal Reuse", mNoResamplingForTemporalReuse);
        }
    }

    if (auto group = widget.group("Shared Path Sampler Options", true))
    {

        dirty |= widget.var("Samples/pixel", mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);

        widget.tooltip("Number of samples per pixel. One path is traced for each sample.\n");

        dirty |= widget.checkbox("Use Sampled BSDFs", mStaticParams.separatePathBSDF);
        widget.tooltip("Control whether to use mixture BSDF or sampled BSDF in path tracing/path reuse.\n");

        if (widget.var("Max bounces (override all)", mStaticParams.maxSurfaceBounces, 0u, kMaxBounces))
        {
            // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
            mStaticParams.maxDiffuseBounces = mStaticParams.maxSurfaceBounces;
            mStaticParams.maxSpecularBounces = mStaticParams.maxSurfaceBounces;
            mStaticParams.maxTransmissionBounces = mStaticParams.maxSurfaceBounces;
            dirty = true;
        }

        if (widget.var("Max surface bounces", mStaticParams.maxSurfaceBounces, 0u, kMaxBounces))
        {
            // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
            mStaticParams.maxDiffuseBounces = std::min(mStaticParams.maxDiffuseBounces, mStaticParams.maxSurfaceBounces);
            mStaticParams.maxSpecularBounces = std::min(mStaticParams.maxSpecularBounces, mStaticParams.maxSurfaceBounces);
            mStaticParams.maxTransmissionBounces = std::min(mStaticParams.maxTransmissionBounces, mStaticParams.maxSurfaceBounces);
            dirty = true;
        }
        widget.tooltip("Maximum number of surface bounces (diffuse + specular + transmission).\n"
            "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold are also classified as diffuse events.");

        dirty |= widget.var("Max diffuse bounces", mStaticParams.maxDiffuseBounces, 0u, kMaxBounces);
        widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

        dirty |= widget.var("Max specular bounces", mStaticParams.maxSpecularBounces, 0u, kMaxBounces);
        widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

        dirty |= widget.var("Max transmission bounces", mStaticParams.maxTransmissionBounces, 0u, kMaxBounces);
        widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

        // Sampling options.

        if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
        {
            mpSampleGenerator = SampleGenerator::create(mStaticParams.sampleGenerator);
            dirty = true;
        }

        widget.tooltip("BSDF importance sampling should normally be enabled.\n\n"
            "If disabled, cosine-weighted hemisphere sampling is used for debugging purposes");

        dirty |= widget.checkbox("Disable direct illumination", mStaticParams.disableDirectIllumination);
        widget.tooltip("If enabled, incoming radiance is collected starting from first order indirect.");

        dirty |= widget.checkbox("Russian roulette", mStaticParams.useRussianRoulette);
        widget.tooltip("Use russian roulette to terminate low throughput paths.");

        dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
        widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");

        if (mStaticParams.useNEE)
        {
            dirty |= widget.checkbox("Multiple importance sampling (MIS)", mStaticParams.useMIS);
            widget.tooltip("When enabled, BSDF sampling is combined with light sampling for the environment map and emissive lights.\n"
                "Note that MIS has currently no effect on analytic lights.");

            if (mStaticParams.useMIS)
            {
                dirty |= widget.dropdown("MIS heuristic", kMISHeuristicList, reinterpret_cast<uint32_t&>(mStaticParams.misHeuristic));

                if (mStaticParams.misHeuristic == MISHeuristic::PowerExp)
                {
                    dirty |= widget.var("MIS power exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
                }
            }

            if (mpScene && mpScene->useEmissiveLights())
            {
                if (auto group = widget.group("Emissive sampler"))
                {
                    if (widget.dropdown("Emissive sampler", kEmissiveSamplerList, (uint32_t&)mStaticParams.emissiveSampler))
                    {
                        resetLighting();
                        dirty = true;
                    }
                    widget.tooltip("Selects which light sampler to use for importance sampling of emissive geometry.", true);

                    if (mpEmissiveSampler)
                    {
                        if (mpEmissiveSampler->renderUI(group)) mOptionsChanged = true;
                    }
                }
            }
        }
    }

    if (auto group = widget.group("Material controls"))
    {
        dirty |= widget.checkbox("Alpha test", mStaticParams.useAlphaTest);
        widget.tooltip("Use alpha testing on non-opaque triangles.");

        dirty |= widget.checkbox("Adjust shading normals on secondary hits", mStaticParams.adjustShadingNormals);
        widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.\nDoes not apply to primary hits which is configured in GBuffer.", true);

        dirty |= widget.var("Max nested materials", mStaticParams.maxNestedMaterials, 2u, 4u);
        widget.tooltip("Maximum supported number of nested materials.");

        dirty |= widget.checkbox("Use deterministic BSDF evaluations", mStaticParams.useDeterministicBSDF);
        widget.tooltip("If disabled, BSDF evaluations of BSDF-sampled directions are only correct on expectation.", true);

        dirty |= widget.checkbox("Use lights in dielectric volumes", mStaticParams.useLightsInDielectricVolumes);
        widget.tooltip("Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.");

        dirty |= widget.checkbox("Limit transmission", mStaticParams.limitTransmission);
        widget.tooltip("Limit specular transmission by handling reflection/refraction events only up to a given transmission depth.");

        if (mStaticParams.limitTransmission)
        {
            dirty |= widget.var("Max transmission reflection depth", mStaticParams.maxTransmissionReflectionDepth, 0u, kMaxBounces);
            widget.tooltip("Maximum transmission depth at which to sample specular reflection.\n"
                "0: Reflection is never sampled.\n"
                "1: Reflection is only sampled on primary hits.\n"
                "N: Reflection is only sampled on the first N hits.");

            dirty |= widget.var("Max transmission refraction depth", mStaticParams.maxTransmissionRefractionDepth, 0u, kMaxBounces);
            widget.tooltip("Maximum transmission depth at which to sample specular refraction (after that, IoR is set to 1).\n"
                "0: Refraction is never sampled.\n"
                "1: Refraction is only sampled on primary hits.\n"
                "N: Refraction is only sampled on the first N hits.");
        }

        dirty |= widget.checkbox("Disable caustics", mStaticParams.disableCaustics);
        widget.tooltip("Disable sampling of caustic light paths (i.e. specular events after diffuse events).");

        dirty |= widget.checkbox("Disable direct illumination", mStaticParams.disableDirectIllumination);
        widget.tooltip("Disable computation of all direct illumination.");

        dirty |= widget.var("TexLOD bias", mParams.lodBias, -16.f, 16.f, 0.01f);
    }

    if (auto group = widget.group("Denoiser options"))
    {
        dirty |= widget.checkbox("Use NRD demodulation", mStaticParams.useNRDDemodulation);
        widget.tooltip("Global switch for NRD demodulation");
    }

    if (auto group = widget.group("Output options"))
    {
        dirty |= widget.dropdown("Color format", kColorFormatList, (uint32_t&)mStaticParams.colorFormat);
        widget.tooltip("Selects the color format used for internal per-sample color and denoiser buffers");
    }

    if (dirty) mRecompile = true;
    return dirty;
}

bool RTPT::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging", true))
    {
        dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        group.tooltip("Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging.");
        if (mParams.useFixedSeed)
        {
            dirty |= group.var("Seed", mParams.fixedSeed);
        }

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

bool RTPT::renderStatsUI(Gui::Widgets& widget)
{
    bool dirty = false;
    if (auto g = widget.group("Statistics"))
    {
        // Show ray stats
        dirty |= mpPixelStats->renderUI(g);
    }
    return dirty;
}

bool RTPT::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

void RTPT::updatePrograms()
{
    if (mRecompile == false) return;

    auto defines = mStaticParams.getDefines(*this);
    specializeSharc(defines);
    mpTracePass->addDefine("PG_GENERATE_STATS", mEnablePG && mPGGenerateStats ? "1" : "0");
    mpTemporalReusePass->addDefine("PG_TEMPORAL_STATS", mEnablePG && mPGTemporalStats ? "1" : "0");
    mpSpatialReusePass->addDefine("PG_SPATIAL_STATS", mEnablePG && mPGSpatialStats ? "1" : "0");

    // Update program specialization. This is done through defines in lieu of specialization constants.
    mpGeneratePaths->getProgram()->addDefines(defines);
    mpTracePass->getProgram()->addDefines(defines);
    mpReflectTypes->getProgram()->addDefines(defines);
    mpSpatialReusePass->getProgram()->addDefines(defines);
    mpTemporalReusePass->getProgram()->addDefines(defines);

    // Recreate program vars. This may trigger recompilation if needed.
    // Note that program versions are cached, so switching to a previously used specialization is faster.
    mpGeneratePaths->setVars(nullptr);
    mpTracePass->setVars(nullptr);
    mpReflectTypes->setVars(nullptr);
    mpSpatialReusePass->setVars(nullptr);
    mpTemporalReusePass->setVars(nullptr);

    mVarsChanged = true;
    mRecompile = false;
}

void RTPT::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Delta-direct is handed to the last reuse pass. Across multiple samples the
    // guide weights preserve the distance of the strongest D and S contributions.
    auto prepareNRDSurface = [&](Texture::SharedPtr& texture, bool needed, ResourceFormat format)
    {
        if (!needed) texture.reset();
        else if (!texture || texture->getWidth() != mParams.frameDim.x || texture->getHeight() != mParams.frameDim.y)
        {
            texture = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, format, 1, 1, nullptr,
                Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        }
    };
    prepareNRDSurface(mpNRDDeltaDirect, mOutputNRDData, ResourceFormat::RGBA32Float);
    prepareNRDSurface(mpNRDGuideWeights, mOutputNRDData && mStaticParams.samplesPerPixel > 1, ResourceFormat::RG32Float);

    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.
    uint32_t tileCount = mParams.screenTiles.x * mParams.screenTiles.y;
    const uint32_t reservoirCount = tileCount * kScreenTileDim.x * kScreenTileDim.y;
    const uint32_t screenPixelCount = mParams.frameDim.x * mParams.frameDim.y;
    const uint32_t sampleCount = reservoirCount; // we are effectively only using 1spp for ReSTIR

    auto var = mpReflectTypes->getRootVar();

    if (!mpOutputReservoirs || mpOutputReservoirs->getElementCount() != reservoirCount ||
        mpTemporalReservoirs.size() != mStaticParams.samplesPerPixel)
    {
        const auto bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess;
        auto allocateReservoirs = [&]() {
            return Buffer::createStructured(var["outputReservoirs"], reservoirCount, bindFlags, Buffer::CpuAccess::None, nullptr, false);
        };
        mpOutputReservoirs = allocateReservoirs();
        mpTemporalReservoirs.resize(mStaticParams.samplesPerPixel);
        for (auto& history : mpTemporalReservoirs) history = allocateReservoirs();
        mReservoirFrameCount = 0;
        mVarsChanged = true;
    }

    const auto hitFormat = mpScene->getHitInfo().getFormat();
    if (!mpTemporalVBuffer || mpTemporalVBuffer->getHeight() != mParams.frameDim.y ||
        mpTemporalVBuffer->getWidth() != mParams.frameDim.x || mpTemporalVBuffer->getFormat() != hitFormat)
    {
        mpTemporalVBuffer = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, hitFormat, 1, 1);
        mReservoirFrameCount = 0;
    }
}


void RTPT::setNRDData(const ShaderVar& var, const RenderData& renderData) const
{
    var["primaryHitEmission"] = renderData[kOutputNRDEmission]->asTexture();
    var["primaryHitDiffuseReflectance"] = renderData[kOutputNRDDiffuseReflectance]->asTexture();
    var["primaryHitSpecularReflectance"] = renderData[kOutputNRDSpecularReflectance]->asTexture();
}

void RTPT::preparePathTracer(const RenderData& renderData)
{
    // Create path tracer parameter block if needed.
    if (!mpPathTracerBlock || mVarsChanged)
    {
        auto reflector = mpTracePass->getProgram()->getReflector()->getParameterBlock("gPathTracer");
        mpPathTracerBlock = ParameterBlock::create(reflector);
        assert(mpPathTracerBlock);
        mVarsChanged = true;
    }

    // Bind resources.
    auto var = mpPathTracerBlock->getRootVar();
    setShaderData(var, renderData, true, false);
    var["outputReservoirs"] = mpOutputReservoirs;
    const auto& directLighting = renderData[kInputDirectLighting];
    var["directLighting"] = directLighting ? directLighting->asTexture() : nullptr;
}

void RTPT::resetLighting()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }
    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void RTPT::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for material changes and performs any necessary update.
    // For now all we need to do is to trigger a recompile so that the right defines get set.
    // In the future, we might want to do additional material-specific setup here.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::MaterialsChanged))
    {
        mRecompile = true;
    }
}
bool RTPT::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getLightCollection(pRenderContext);
            assert(pLights && pLights->getActiveLightCount() > 0);
            assert(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = EmissiveUniformSampler::create(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler = LightBVHSampler::create(pRenderContext, mpScene, mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = EmissivePowerSampler::create(pRenderContext, mpScene);
                break;
            default:
                logError("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext);
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass->getProgram()->addDefines(defines)) mRecompile = true;
        if (mpSpatialReusePass->getProgram()->addDefines(defines)) mRecompile = true;
        if (mpTemporalReusePass->getProgram()->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void RTPT::setShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (isPathTracer && mpEnvMapSampler) mpEnvMapSampler->setShaderData(var["envMapSampler"]);
    }

    // Bind runtime data.
    var["params"].setBlob(mParams);
    var["vbuffer"] = renderData[kInputVBuffer]->asTexture();
    var["outputColor"] = renderData[kOutputColor]->asTexture();


    if (mOutputNRDData && isPathTracer)
    {
        setNRDData(var["outputNRD"], renderData);
        const auto& diffuse = renderData[kInputDirectDiffuse];
        const auto& specular = renderData[kInputDirectSpecular];
        const auto& emission = renderData[kInputDirectEmission];
        var["useSplitDirectLighting"] = bool(diffuse && specular);
        var["hasDirectEmission"] = bool(emission);
        var["directDiffuseRadianceHitDist"] = diffuse ? diffuse->asTexture() : nullptr;
        var["directSpecularRadianceHitDist"] = specular ? specular->asTexture() : nullptr;
        var["directEmission"] = emission ? emission->asTexture() : nullptr;
        var["nrdDeltaDirect"] = mpNRDDeltaDirect;
        var["nrdGuideWeights"] = mpNRDGuideWeights;
        var["outputNRDDiffuseRadianceHitDist"] = renderData[kOutputNRDDiffuseRadianceHitDist]->asTexture();    ///< Output resolved diffuse color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
        var["outputNRDSpecularRadianceHitDist"] = renderData[kOutputNRDSpecularRadianceHitDist]->asTexture();  ///< Output resolved specular color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
        var["outputNRDResidualRadianceHitDist"] = renderData[kOutputNRDResidualRadianceHitDist]->asTexture();///< Output resolved residual color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    }

    if (isPathTracer)
    {
        var["isLastRound"] = false;
        var["useDirectLighting"] = mUseDirectLighting;
        var["kUseEnvLight"] = mpScene->useEnvLight();
        var["kUseEmissiveLights"] = mpScene->useEmissiveLights();
        var["kUseAnalyticLights"] = mpScene->useAnalyticLights();
    }
    else if (isPathGenerator)
    {
        var["kUseEnvBackground"] = mpScene->useEnvBackground();
    }

    if (auto outputDebug = var.findMember("outputDebug"); outputDebug.isValid())
    {
        outputDebug = renderData[kOutputDebug]->asTexture(); // Can be nullptr
    }
    if (auto outputTime = var.findMember("outputTime"); outputTime.isValid())
    {
        outputTime = renderData[kOutputTime]->asTexture(); // Can be nullptr
    }

    if (isPathTracer && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        bool success = mpEmissiveSampler->setShaderData(var["emissiveSampler"]);
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}

bool RTPT::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mOptionsChanged)
    {
        mReservoirFrameCount = 0;
    }
    // Clear outputs that need to be cleared.
    const auto& pOutputDebug = renderData[kOutputDebug]->asTexture();
    if (pOutputDebug) pRenderContext->clearUAV(pOutputDebug->getUAV().get(), float4(0.f));

    if (mpScene == nullptr)
    {
        const auto& pOutputColor = renderData[kOutputColor]->asTexture();
        assert(pOutputColor);
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        return false;
    }

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);
    if (mEnableSharc || mEnablePG) trackSharcMotion(lightingChanged);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged)
        {
            flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        }
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    bool gbufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);
    if (gbufferAdjustShadingNormals != mGBufferAdjustShadingNormals)
    {
        mGBufferAdjustShadingNormals = gbufferAdjustShadingNormals;
        mSharcNeedsClear = true;
        mRecompile = true;
    }

    // Check if NRD data should be generated.
    mOutputNRDData =
        renderData[kOutputNRDDiffuseRadianceHitDist] != nullptr
        || renderData[kOutputNRDSpecularRadianceHitDist] != nullptr
        || renderData[kOutputNRDResidualRadianceHitDist] != nullptr
        || renderData[kOutputNRDEmission] != nullptr
        || renderData[kOutputNRDDiffuseReflectance] != nullptr
        || renderData[kOutputNRDSpecularReflectance] != nullptr;

    // Check if time data should be generated.
    mOutputTime = renderData[kOutputTime] != nullptr;

    // Enable pixel stats if rayCount or pathLength outputs are connected.
    if (renderData[kOutputRayCount] != nullptr
        || renderData[kOutputPathLength] != nullptr
        || mEnableRayStats)
        mpPixelStats->setEnabled(true);

    mpPixelStats->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    mpPixelDebug->beginFrame(pRenderContext, renderData.getDefaultTextureDims());

    // Update the random seed.
    const uint32_t initialShaderPasses = mStaticParams.samplesPerPixel;
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mSeedOffset + initialShaderPasses * mParams.frameCount;

    return true;
}

void RTPT::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelStats->endFrame(pRenderContext);
    mpPixelDebug->endFrame(pRenderContext);

    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            assert(pDst && pSrc);
            assert(pDst->getFormat() == pSrc->getFormat());
            assert(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Copy pixel stats to outputs if available.
    copyTexture(renderData[kOutputRayCount]->asTexture().get(), mpPixelStats->getRayCountTexture(pRenderContext).get());
    copyTexture(renderData[kOutputPathLength]->asTexture().get(), mpPixelStats->getPathLengthTexture().get());

    mVarsChanged = false;
}

void RTPT::generatePaths(RenderContext* pRenderContext, const RenderData& renderData, int sampleId)
{
    PROFILE("generatePaths");

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    assert(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see comments in shader.
    assert(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    assert(mpGeneratePaths->getThreadGroupSize().x == tileSize);
    assert(mpGeneratePaths->getThreadGroupSize().y == 1 && mpGeneratePaths->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    mpGeneratePaths->addDefine("OUTPUT_TIME", mOutputTime ? "1" : "0");
    mpGeneratePaths->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");

    // Bind resources.
    auto var = mpGeneratePaths->getRootVar()["CB"]["gPathGenerator"];
    setShaderData(var, renderData, false, true);

    mpGeneratePaths["gScene"] = mpScene->getParameterBlock();
    var["gSampleId"] = sampleId;

    // Launch one thread per pixel.
    // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
    mpGeneratePaths->execute(pRenderContext, { mParams.screenTiles.x * tileSize, mParams.screenTiles.y, 1u });
}

void RTPT::tracePass(RenderContext* pRenderContext, const RenderData& renderData, const ComputePass::SharedPtr& pass, const std::string& passName, int sampleID)
{
    PROFILE(passName);

    // Additional specialization. This shouldn't change resource declarations.
    bool outputDebug = renderData[kOutputDebug] != nullptr;
    pass->addDefine("OUTPUT_TIME", mOutputTime ? "1" : "0");
    pass->addDefine("OUTPUT_DEBUG", outputDebug ? "1" : "0");
    pass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");

    // Bind global resources.
    auto var = pass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    if (mVarsChanged) mpSampleGenerator->setShaderData(var);

    mpPixelStats->prepareProgram(pass->getProgram(), var);
    mpPixelDebug->prepareProgram(pass->getProgram(), var);

    // Bind the path tracer.
    var["gPathTracer"] = mpPathTracerBlock;
    var["CB"]["gSampleId"] = sampleID;
    auto pg = var["PGGenerateCB"];
    bindCurrentPGSampling(pg);
    if (mEnablePG)
    {
        mPGGenerateBoundVersion = mPGSnapshots[mPGCurrent].version;
        if (mPGGenerateStats) pg["pgGenerateCounts"] = mpPGGenerateCounts;
    }

    auto sharc = var["GenerateSharcCB"];
    sharc["gEnableSharcGenerateQuery"] = mEnableSharc;
    bindSharc(sharc["gSharcQuery"]);

    // Launch the threads.
    auto frameDim = renderData.getDefaultTextureDims();
    pass->execute(pRenderContext, uint3(frameDim, 1u));
}

void RTPT::PathReusePass(RenderContext* pRenderContext, uint32_t restir_i, const RenderData& renderData, bool isTemporalReuse, int spatialRoundId, bool isLastRound)
{
    PROFILE(isTemporalReuse ? "temporalReuse" : "spatialReuse");
    const auto& pass = isTemporalReuse ? mpTemporalReusePass : mpSpatialReusePass;

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    assert(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see comments in shader.
    assert(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    assert(pass->getThreadGroupSize().x == 16);
    assert(pass->getThreadGroupSize().y == 16 && pass->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    pass->addDefine("OUTPUT_TIME", mOutputTime ? "1" : "0");
    pass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");

    // Bind resources.
    const auto root = pass->getRootVar();
    auto var = root["CB"]["gPathReusePass"];

    // TODO: refactor arguments
    setShaderData(var, renderData, false, false);

    var["outputReservoirs"] = spatialRoundId % 2 == 1 ? mpTemporalReservoirs[restir_i] : mpOutputReservoirs;
    var["temporalReservoirs"] = spatialRoundId % 2 == 0 ? mpTemporalReservoirs[restir_i] : mpOutputReservoirs;

    var["gNumSpatialRounds"] = mNumSpatialRounds;

    const auto pg = root["PGGenerateCB"];
    bindCurrentPGSampling(pg);
    if (isTemporalReuse)
    {
        if (mEnablePG)
        {
            mPGTemporalBoundVersion = mPGSnapshots[mPGCurrent].version;
            if (mPGTemporalStats)
            {
                root["pgTemporalCounts"] = mpPGTemporalCounts;
                pg["pgTemporalReplayCounts"] = mpPGTemporalReplayCounts;
            }
        }
        const auto sharc = root["TemporalSharcCB"];
        sharc["gEnableSharcTemporalQuery"] = mEnableSharc;
        bindSharc(sharc["gSharcTemporalQuery"]);
        const auto& motionVectors = renderData[kInputMotionVectors];
        var["temporalVbuffer"] = mpTemporalVBuffer;
        var["motionVectors"] = motionVectors ? motionVectors->asTexture() : nullptr;
        var["gEnableTemporalReprojection"] = mEnableTemporalReprojection && bool(motionVectors);
        var["gNoResamplingForTemporalReuse"] = mNoResamplingForTemporalReuse;
        if (!mUseMaxHistory) var["gTemporalHistoryLength"] = 1e30f;
        else var["gTemporalHistoryLength"] = (float)mTemporalHistoryLength;
    }
    else
    {
        if (mEnablePG)
        {
            mPGSpatialBoundVersions.push_back(mPGSnapshots[mPGCurrent].version);
            if (mPGSpatialStats) root["pgSpatialCounts"] = mpPGSpatialCounts;
        }
        var["gSpatialReusePattern"] = uint32_t(mSpatialReusePattern);
        var["gNeighborCount"] = mSpatialNeighborCount;
        var["gGatherRadius"] = mSpatialReuseRadius;
        var["gSpatialRoundId"] = spatialRoundId;
        var["gSmallWindowRadius"] = mSmallWindowRestirWindowRadius;
        var["gFeatureBasedRejection"] = mFeatureBasedRejection;
        var["neighborOffsets"] = mpNeighborOffsets;
    }
    const auto& directLighting = renderData[kInputDirectLighting];
    var["directLighting"] = directLighting ? directLighting->asTexture() : nullptr;
    var["useDirectLighting"] = mUseDirectLighting;
    var["gIsLastRound"] = isLastRound;

    pass["gScene"] = mpScene->getParameterBlock();
    pass["gPathTracer"] = mpPathTracerBlock;

    mpPixelStats->prepareProgram(pass->getProgram(), pass->getRootVar());
    mpPixelDebug->prepareProgram(pass->getProgram(), pass->getRootVar());

    {
        // Launch one thread per pixel.
        // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
        pass->execute(pRenderContext, { mParams.screenTiles.x * kScreenTileDim.x, mParams.screenTiles.y * kScreenTileDim.y, 1u });
    }
}

Program::DefineList RTPT::StaticParams::getDefines(const RTPT& owner) const
{
    Program::DefineList defines;

    // Path tracer configuration.
    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel)); // 0 indicates a variable sample count
    defines.add("CANDIDATE_SAMPLES", std::to_string(candidateSamples)); // 0 indicates a variable sample count    
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(maxSurfaceBounces));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(maxTransmissionBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("LIMIT_TRANSMISSION", limitTransmission ? "1" : "0");
    defines.add("MAX_TRANSMISSION_REFLECTION_DEPTH", std::to_string(maxTransmissionReflectionDepth));
    defines.add("MAX_TRANSMISSION_REFRACTION_DEPTH", std::to_string(maxTransmissionRefractionDepth));
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("DISABLE_DIRECT_ILLUMINATION", disableDirectIllumination ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("USE_NRD_DEMODULATION", useNRDDemodulation ? "1" : "0");
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("_USE_DETERMINISTIC_BSDF", useDeterministicBSDF ? "1" : "0");
    defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(kNeighborOffsetCount));
    defines.add("SHIFT_STRATEGY", std::to_string(uint32_t(ShiftMapping::Reconnection)));
    defines.add("PATH_SAMPLING_MODE", std::to_string(uint32_t(PathSamplingMode::ReSTIR)));

    // Sampling utilities configuration.
    assert(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    // We don't use the legacy shading code anymore (MaterialShading.slang).
    defines.add("_USE_LEGACY_SHADING_CODE", "0");

    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;

    // Set default (off) values for additional features.
    defines.add("OUTPUT_GUIDE_DATA", "0");
    defines.add("OUTPUT_TIME", "0");
    defines.add("OUTPUT_NRD_DATA", "0");
    defines.add("OUTPUT_NRD_ADDITIONAL_DATA", "0");

    defines.add("SPATIAL_RESTIR_MIS_KIND", std::to_string((uint32_t)spatialMisKind));
    defines.add("TEMPORAL_RESTIR_MIS_KIND", std::to_string((uint32_t)temporalMisKind));

    defines.add("TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE", temporalUpdateForDynamicScene ? "1" : "0");

    defines.add("BPR", "0");

    defines.add("SEPARATE_PATH_BSDF", separatePathBSDF ? "1" : "0");


    return defines;
}
