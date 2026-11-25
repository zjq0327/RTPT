/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/EnvMapLighting.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Volumes/GridVolumeSampler.h"
#include "Rendering/Utils/PixelStats.h"
#include "Rendering/Materials/TexLODTypes.slang"
#include "Params.slang"
#include <fstream>

using namespace Falcor;

/** Path tracer that uses TraceRayInline() in DXR 1.1.
*/
class RTPT : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<RTPT>;

    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    Dictionary getSpecializedScriptingDictionary();
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    const PixelStats::SharedPtr& getPixelStats() const { return mpPixelStats; }

    void updateDict(const Dictionary& dict) override;
    void initDict() override;

    static void registerBindings(pybind11::module& m);

private:
    RTPT(const Dictionary& dict);
    bool parseDictionary(const Dictionary& dict);
    void validateOptions();
    void updatePrograms();
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void setNRDData(const ShaderVar& var, const RenderData& renderData) const;
    void preparePathTracer(const RenderData& renderData);
    void resetLighting();
    void specializeSharc(const Program::DefineList& defines);
    void bindSharc(const ShaderVar& var) const;
    void trackSharcMotion(bool lightingChanged);
    void updateSharc(RenderContext* pRenderContext);
    void traceSharcTrainingPaths(RenderContext* pRenderContext, uint32_t stride, bool trainPG = false);
    void resolveSharcTraining(RenderContext* pRenderContext);
    void preparePG(RenderContext* pRenderContext);
    void releasePG();
    void bindPGTraining(const ShaderVar& var) const;
    void bindPGRead(const ShaderVar& var, bool previousFrame) const;
    void barrierPGTraining(RenderContext* pRenderContext) const;
    void notePGTraining(uint32_t stride, bool pretrain);
    void finishPGTraining(RenderContext* pRenderContext, bool immediate);
    void publishPG(RenderContext* pRenderContext);
    pybind11::dict getPGStats() const;
    pybind11::dict getSharcStats() const;
    pybind11::dict getSharcGenerateStats() const;
    std::vector<uint32_t> getSharcTemporalStats() const;
    pybind11::bytes getSharcFreshReservoirs() const;
    pybind11::bytes getSharcTemporalReservoirs() const;
    pybind11::bytes getSharcSpatialReservoirs() const;
    bool querySharcInGenerate() const { return mEnableSharc && mSharcUseCache; }
    bool querySharcInTemporal() const { return mEnableTemporalReuse && mSharcTemporalUseCache && querySharcInGenerate(); }
    void prepareMaterials(RenderContext* pRenderContext);
    bool prepareLighting(RenderContext* pRenderContext);
    void setShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const;
    bool renderRenderingUI(Gui::Widgets& widget);
    bool renderDebugUI(Gui::Widgets& widget);
    bool renderStatsUI(Gui::Widgets& widget);
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void generatePaths(RenderContext* pRenderContext, const RenderData& renderData, int sampleId = 0);
    void tracePass(RenderContext* pRenderContext, const RenderData& renderData, const ComputePass::SharedPtr& pass, const std::string& passName, int sampleId);
    void PathReusePass(RenderContext* pRenderContext, uint32_t restir_i, const RenderData& renderData, bool temporalReuse = false, int spatialRoundId = 0, bool isLastRound = false);
    Texture::SharedPtr createNeighborOffsetTexture(uint32_t sampleCount);

    /** Static configuration. Changing any of these options require shader recompilation.
    */
    struct StaticParams
    {
        // Rendering parameters
        uint32_t    samplesPerPixel = 1;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
        uint32_t    candidateSamples = 1;
        uint32_t    maxSurfaceBounces = 9;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth.
        uint32_t    maxDiffuseBounces = -1;                     ///< Max number of diffuse bounces (0 = direct only), up to kMaxBounces. This will be initialized at startup.
        uint32_t    maxSpecularBounces = -1;                    ///< Max number of specular bounces (0 = direct only), up to kMaxBounces. This will be initialized at startup.
        uint32_t    maxTransmissionBounces = -1;                ///< Max number of transmission bounces (0 = none), up to kMaxBounces. This will be initialized at startup.
        uint32_t    sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM; ///< Pseudorandom sample generator type.
        bool        adjustShadingNormals = false;               ///< Adjust shading normals on secondary hits.
        bool        useBSDFSampling = true;                     ///< Use BRDF importance sampling, otherwise cosine-weighted hemisphere sampling.
        bool        useNEE = true;                              ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        bool        useMIS = true;                              ///< Use multiple importance sampling (MIS) when NEE is enabled.
        bool        useRussianRoulette = false;                 ///< Use russian roulette to terminate low throughput paths.
        bool        useAlphaTest = true;                        ///< Use alpha testing on non-opaque triangles.
        uint32_t    maxNestedMaterials = 2;                     ///< Maximum supported number of nested materials.
        bool        useLightsInDielectricVolumes = false;       ///< Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.
        bool        limitTransmission = false;                  ///< Limit specular transmission by handling reflection/refraction events only up to a given transmission depth.
        uint32_t    maxTransmissionReflectionDepth = 0;         ///< Maximum transmission depth at which to sample specular reflection.
        uint32_t    maxTransmissionRefractionDepth = 0;         ///< Maximum transmission depth at which to sample specular refraction (after that, IoR is set to 1).
        bool        disableCaustics = false;                    ///< Disable sampling of caustics.
        bool        disableDirectIllumination = true;          ///< Disable all direct illumination.
        TexLODMode  primaryLodMode = TexLODMode::Mip0;          ///< Use filtered texture lookups at the primary hit.
        ColorFormat colorFormat = ColorFormat::LogLuvHDR;       ///< Color format used for internal per-sample color and denoiser buffers.
        MISHeuristic misHeuristic = MISHeuristic::Balance;      ///< MIS heuristic.
        float       misPowerExponent = 2.f;                     ///< MIS exponent for the power heuristic. This is only used when 'PowerExp' is chosen.
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Power;  ///< Emissive light sampler to use for NEE.

        bool        useDeterministicBSDF = true;                    ///< Evaluate all compatible lobes at BSDF sampling time.

        ReSTIRMISKind    spatialMisKind = ReSTIRMISKind::Pairwise;
        ReSTIRMISKind    temporalMisKind = ReSTIRMISKind::Talbot;

        bool            temporalUpdateForDynamicScene = false;


        bool            separatePathBSDF = true;


		// Denoising parameters
		bool        useNRDDemodulation = true;                  ///< Global switch for NRD demodulation.

        Program::DefineList getDefines(const RTPT& owner) const;
    };


    void Init()
    {
        mStaticParams = StaticParams();
        mParams = RestirPathTracerParams();
        mEnableTemporalReuse = true;
        mEnableSpatialReuse = true;
        mSpatialReusePattern = SpatialReusePattern::Default;
        mSmallWindowRestirWindowRadius = 2;
        mSpatialNeighborCount = 3;
        mSpatialReuseRadius = 20.f;
        mNumSpatialRounds = 1;
        mEnableTemporalReprojection = false;
        mUseMaxHistory = true;
        mUseDirectLighting = true;
        mTemporalHistoryLength = 20;
        mNoResamplingForTemporalReuse = false;
    }

    // Cache training depends on transport settings even when the visible path is unchanged.
    auto getSharcTrainingSettings() const
    {
        const auto& p = mStaticParams;
        return std::make_tuple(p.maxSurfaceBounces, p.maxDiffuseBounces, p.maxSpecularBounces, p.maxTransmissionBounces,
            p.useNEE, p.useMIS, p.misHeuristic, p.misPowerExponent, p.disableCaustics, p.useBSDFSampling,
            p.useDeterministicBSDF, p.separatePathBSDF, p.useRussianRoulette, p.adjustShadingNormals, p.useAlphaTest,
            p.maxNestedMaterials, p.useLightsInDielectricVolumes, p.limitTransmission, p.maxTransmissionReflectionDepth,
            p.maxTransmissionRefractionDepth, p.disableDirectIllumination, p.primaryLodMode, p.emissiveSampler,
            mParams.lodBias, mParams.specularRoughnessThreshold);
    }

    // Configuration
    RestirPathTracerParams          mParams;                    ///< Runtime path tracer parameters.
    StaticParams                    mStaticParams;              ///< Static parameters. These are set as compile-time constants in the shaders.
    bool                            mJacobianRejectionOverride = false;
    bool                            mTemporalUpdateOverride = false;
    LightBVHSampler::Options        mLightBVHOptions;           ///< Current options for the light BVH sampler.

    // Internal state
    Scene::SharedPtr                mpScene;                    ///< The current scene, or nullptr if no scene loaded.
    SampleGenerator::SharedPtr      mpSampleGenerator;          ///< GPU pseudo-random sample generator.
    EnvMapSampler::SharedPtr        mpEnvMapSampler;            ///< Environment map sampler or nullptr if not used.
    EmissiveLightSampler::SharedPtr mpEmissiveSampler;          ///< Emissive light sampler or nullptr if not used.
    PixelStats::SharedPtr           mpPixelStats;               ///< Utility class for collecting pixel stats.
    PixelDebug::SharedPtr           mpPixelDebug;               ///< Utility class for pixel debugging (print in shaders).
    ParameterBlock::SharedPtr       mpPathTracerBlock;          ///< Parameter block for the path tracer.

    bool                            mEnableSharc = false;
    bool                            mSharcFreeze = false;
    bool                            mSharcUseCache = false;
    bool                            mSharcTemporalUseCache = true;
    bool                            mSharcGenerateStats = false;
    uint32_t                        mSharcDebugMode = 0;
    uint32_t                        mSharcDebugDepth = 2;
    uint32_t                        mSharcTemporalDebugMode = 0;
    uint32_t                        mSharcTemporalDebugDepth = 2;
    Buffer::SharedPtr               mpSharcGenerateStats;
    Buffer::SharedPtr               mpSharcTrainingStats;
    Buffer::SharedPtr               mpSharcTemporalStats;
    bool                            mSharcNeedsClear = true;
    bool                            mSharcPretrainPending = true;
    static constexpr uint32_t       kDefaultSharcPretrainSpp = 8;
    uint32_t                        mSharcPretrainSpp = kDefaultSharcPretrainSpp;
    uint32_t                        mSharcPretrainedSpp = 0;
    uint32_t                        mSharcFrameIndex = 0;
    uint64_t                        mSharcUpdateCount = 0, mSharcResolveCount = 0;
    uint64_t                        mSharcClearCount = 0, mSharcInvalidateCount = 0;
    struct SharcGrid
    {
        float3 cameraPosition = float3(0.f);
        float sceneScale = 16.f;
        float logarithmBase = 2.f;
        float levelBias = 0.f;
    };
    SharcGrid                       mSharcPublishedGrid;
    uint64_t                        mSharcPublishedVersion = 0;
    float3                          mSharcPreviousCamera = float3(0.f);
    std::vector<AABB>               mSharcPreviousBounds;
    std::vector<float4>             mSharcMotionBounds;
    ComputePass::SharedPtr          mpSharcUpdate, mpSharcResolve, mpSharcInvalidate;
    Buffer::SharedPtr               mpSharcKeys, mpSharcLocks, mpSharcAccumulation, mpSharcResolved, mpSharcMotionBounds;

    bool                            mEnablePG = false;
    ComputePass::SharedPtr          mpSharcPGUpdate, mpPGPublish;
    Buffer::SharedPtr               mpPGBuildKeys, mpPGAccumulation, mpPGCounts, mpPGDiagnostics;
    Buffer::SharedPtr               mpPGHistoryMean, mpPGHistoryCount;
    static constexpr uint32_t       kPGCapacity = 65536;
    static constexpr float          kDefaultPGSceneScale = 8.f;
    float                           mPGSceneScale = kDefaultPGSceneScale;
    bool                            mPGResetPending = false, mPGCarryPrevious = true;
    struct PGSnapshot
    {
        Buffer::SharedPtr keys, mass;
        float3 cameraPosition = float3(0.f);
        float sceneScale = kDefaultPGSceneScale;
        uint64_t version = 0;
    };
    PGSnapshot                      mPGSnapshots[2];
    uint32_t                        mPGCurrent = 0, mPGPrevious = 0;
    bool                            mPGPublishPending = false, mPGPublishedThisFrame = false;
    float3                          mPGTrainingCamera = float3(0.f);
    uint64_t                        mPGRuntimeFrameCount = 0, mPGPublications = 0;
    uint64_t                        mPGPretrainDispatches = 0, mPGRuntimeDispatches = 0;
    uint64_t                        mPGPretrainPaths = 0, mPGRuntimePaths = 0;
    uint64_t                        mPGLastTrainingFrame = 0;
    uint32_t                        mPGLastTrainingSample = 0;

    // internal below
    bool                            mRecompile = false;         ///< Set to true when program specialization has changed.
    bool                            mVarsChanged = true;        ///< This is set to true whenever the program vars have changed and resources need to be rebound.
    bool                            mOptionsChanged = false;    ///< True if the config has changed since last frame.
    bool                            mGBufferAdjustShadingNormals = false; ///< True if GBuffer/VBuffer has adjusted shading normals enabled.
    bool                            mOutputTime = false;        ///< True if time data should be generated as output.
    bool                            mOutputNRDData = false;     ///< True if NRD diffuse/specular data should be generated as outputs.
    bool                            mEnableRayStats = false;      ///< Set to true when the stats tab in the UI is open. This may negatively affect performance.

    uint64_t                        mAccumulatedRayCount = 0;
    uint64_t                        mAccumulatedClosestHitRayCount = 0;
    uint64_t                        mAccumulatedShadowRayCount = 0;

    // params below
    bool                            mEnableTemporalReuse = true;
    bool                            mEnableSpatialReuse = true;
    SpatialReusePattern             mSpatialReusePattern = SpatialReusePattern::Default;
    uint32_t                        mSmallWindowRestirWindowRadius = 2;
    int                             mSpatialNeighborCount = 3;
    float                           mSpatialReuseRadius = 20.f;
    int                             mNumSpatialRounds = 1;

    bool                            mEnableTemporalReprojection = true;
    bool                            mFeatureBasedRejection = true;

    bool                            mUseMaxHistory = true;

    int                             mReservoirFrameCount = 0; // internal

    bool                            mUseDirectLighting = true;

    int                             mTemporalHistoryLength = 20;
    bool                            mNoResamplingForTemporalReuse = false;
    int                             mSeedOffset = 0;


    bool mResetRenderPassFlags = false;

    ComputePass::SharedPtr          mpSpatialReusePass;      ///< Merges reservoirs.
    ComputePass::SharedPtr          mpTemporalReusePass;      ///< Merges reservoirs.


    ComputePass::SharedPtr          mpGeneratePaths;                ///< Fullscreen compute pass generating paths starting at primary hits.
    ComputePass::SharedPtr          mpTracePass;                    ///< Main tracing pass.

    ComputePass::SharedPtr          mpReflectTypes;             ///< Helper for reflecting structured buffer types.

    GpuFence::SharedPtr             mpReadbackFence;            ///< GPU fence for synchronizing stats readback.

    // Data
    Buffer::SharedPtr               mpCounters;                 ///< Atomic counters (32-bit).
    Buffer::SharedPtr               mpCountersReadback;         ///< Readback buffer for the counters for stats gathering.

    Buffer::SharedPtr               mpOutputReservoirs;               ///< Output paths from the path sampling stage.
    // enable multiple temporal reservoirs for spp > 1 (multiple ReSTIR chains)
    std::vector<Buffer::SharedPtr>               mpTemporalReservoirs;               ///< Output paths from the path sampling stage.

    Texture::SharedPtr              mpTemporalVBuffer;
    Texture::SharedPtr mpNRDDeltaDirect;
    Texture::SharedPtr mpNRDGuideWeights;

    Texture::SharedPtr              mpNeighborOffsets;

};
