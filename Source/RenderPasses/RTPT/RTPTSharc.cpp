/***************************************************************************
 # Copyright (c) 2022, Daqi Lin. All rights reserved.
 **************************************************************************/
#include "RTPT.h"
#include <array>
#include <cstring>

namespace
{
    constexpr uint32_t kSharcCapacity = 1u << 20;
    constexpr uint32_t kSharcUpdateStride = 5;
    constexpr uint32_t kSharcPretrainBatchSpp = 8;
    constexpr uint32_t kGenerateDepthCount = 16;
    constexpr uint32_t kGenerateMetricCount = 8;
    constexpr uint32_t kGenerateProbeOffset = kGenerateDepthCount * kGenerateMetricCount;
    constexpr uint32_t kGenerateRayCountOffset = kGenerateProbeOffset + 16;
    constexpr uint32_t kGenerateCounterCount = kGenerateRayCountOffset + kGenerateDepthCount;
    // Five depth histograms followed by sixteen scalar slots in the Temporal shader.
    constexpr uint32_t kTemporalCounterCount = 6u * kGenerateDepthCount;
    const char kSharcUpdateFile[] = "RenderPasses/RTPT/SharcUpdate.cs.slang";
    const char kSharcResolveFile[] = "RenderPasses/ReSTIRPTPass/SharcResolve.cs.slang";
    const char kSharcInvalidateFile[] = "RenderPasses/ReSTIRPTPass/SharcInvalidate.cs.slang";

    pybind11::bytes readReservoirs(const Buffer::SharedPtr& reservoirs)
    {
        // One tile-padded image of the last sample-per-pixel chain.
        const auto data = static_cast<const char*>(reservoirs->map(Buffer::MapType::Read));
        pybind11::bytes result(data, reservoirs->getSize());
        reservoirs->unmap();
        return result;
    }
}

void RTPT::specializeSharc(const Program::DefineList& defines)
{
    if (mSharcGenerateStats && !mpSharcGenerateStats)
        mpSharcGenerateStats = Buffer::createStructured(sizeof(uint32_t), kGenerateCounterCount,
            Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    else if (!mSharcGenerateStats) mpSharcGenerateStats.reset();

    const bool collectTemporal = mSharcGenerateStats && querySharcInTemporal();
    if (collectTemporal && !mpSharcTemporalStats)
        mpSharcTemporalStats = Buffer::createStructured(sizeof(uint32_t), kTemporalCounterCount,
            Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    else if (!collectTemporal) mpSharcTemporalStats.reset();

    if (!mEnableSharc)
    {
        mpSharcUpdate.reset();
        mpSharcResolve.reset();
        mpSharcInvalidate.reset();
        mpSharcMotionBounds.reset();
        mpSharcKeys = mpSharcLocks = mpSharcAccumulation = mpSharcResolved = nullptr;
        mSharcPreviousBounds.clear();
        mSharcMotionBounds.clear();
        mSharcNeedsClear = true;
        mSharcPretrainPending = true;
        mSharcPretrainedSpp = 0;
        return;
    }
    auto updateDefines = defines;
    updateDefines.add("SHARC_TRAINING", "1");
    updateDefines.add(mpScene->getSceneDefines());
    if (mpEmissiveSampler) updateDefines.add(mpEmissiveSampler->getDefines());
    if (!mpSharcUpdate)
    {
        Program::Desc desc;
        desc.addShaderLibrary(kSharcUpdateFile).csEntry("main").setShaderModel("6_5");
        mpSharcUpdate = ComputePass::create(desc, updateDefines, true);
    }
    else
    {
        mpSharcUpdate->getProgram()->setDefines(updateDefines);
        mpSharcUpdate->setVars(nullptr);
    }
    if (!mpSharcResolve)
    {
        Program::Desc desc;
        desc.addShaderLibrary(kSharcResolveFile).csEntry("main").setShaderModel("6_5");
        mpSharcResolve = ComputePass::create(desc, {}, true);
    }
}

void RTPT::bindSharc(const ShaderVar& var) const
{
    auto cache = var["cache"];
    auto grid = cache["hashGridParameters"];
    grid["cameraPosition"] = mSharcPublishedGrid.cameraPosition;
    grid["logarithmBase"] = mSharcPublishedGrid.logarithmBase;
    grid["sceneScale"] = mSharcPublishedGrid.sceneScale;
    grid["levelBias"] = mSharcPublishedGrid.levelBias;
    auto table = cache["hashGridData"];
    table["capacity"] = kSharcCapacity;
    table["hashEntriesBuffer"] = mpSharcKeys;
    table["lockBuffer"] = mpSharcLocks;
    cache["accumulationBuffer"] = mpSharcAccumulation;
    cache["resolvedBuffer"] = mpSharcResolved;
    cache["radianceScale"] = 1000.f;
    var["minRoughness"] = 0.4f;
}

void RTPT::updateSharc(RenderContext* pRenderContext)
{
    if (!mpSharcKeys)
    {
        const auto flags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess;
        mpSharcKeys = Buffer::createStructured(8u, kSharcCapacity, flags, Buffer::CpuAccess::None, nullptr, false);
        mpSharcLocks = Buffer::createStructured(4u, kSharcCapacity, flags, Buffer::CpuAccess::None, nullptr, false);
        mpSharcAccumulation = Buffer::createStructured(16u, kSharcCapacity, flags, Buffer::CpuAccess::None, nullptr, false);
        mpSharcResolved = Buffer::createStructured(16u, kSharcCapacity, flags, Buffer::CpuAccess::None, nullptr, false);
        mSharcNeedsClear = true;
    }
    if (mSharcFreeze && !mSharcPretrainPending && !mSharcNeedsClear && mSharcMotionBounds.empty()) return;
    // Invalidation uses world-space keys. A frozen grid keeps its last resolved camera/LOD.
    if (mSharcNeedsClear || !mSharcFreeze) mSharcPublishedGrid.cameraPosition = mpScene->getCamera()->getPosition();
    const auto buffers = {mpSharcKeys, mpSharcLocks, mpSharcAccumulation, mpSharcResolved};
    if (mSharcNeedsClear)
    {
        PROFILE("SHaRC Clear");
        ++mSharcClearCount;
        mSharcFrameIndex = 0;
        mSharcMotionBounds.clear();
        for (const auto& buffer : buffers)
        {
            pRenderContext->clearUAV(buffer->getUAV().get(), uint4(0));
            pRenderContext->uavBarrier(buffer.get());
        }
        mSharcPreviousCamera = mpScene->getCamera()->getPosition();
        mSharcNeedsClear = false;
    }
    else if (!mSharcMotionBounds.empty())
    {
        PROFILE("SHaRC Invalidate");
        ++mSharcInvalidateCount;
        if (!mpSharcInvalidate)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kSharcInvalidateFile).csEntry("main").setShaderModel("6_5");
            mpSharcInvalidate = ComputePass::create(desc, {}, true);
        }
        const auto count = uint32_t(mSharcMotionBounds.size());
        if (!mpSharcMotionBounds || mpSharcMotionBounds->getElementCount() < count)
            mpSharcMotionBounds = Buffer::createStructured(sizeof(float4), count, Resource::BindFlags::ShaderResource,
                Buffer::CpuAccess::None, nullptr, false);
        mpSharcMotionBounds->setBlob(mSharcMotionBounds.data(), 0, count * sizeof(float4));
        auto var = mpSharcInvalidate->getRootVar();
        var["gBounds"] = mpSharcMotionBounds;
        var["PerFrame"]["gBoundsCount"] = count / 2u;
        bindSharc(var["PerFrame"]["gSharc"]);
        mpSharcInvalidate->execute(pRenderContext, uint3(kSharcCapacity, 1u, 1u));
        pRenderContext->uavBarrier(mpSharcAccumulation.get());
        pRenderContext->uavBarrier(mpSharcResolved.get());
    }
    if (mSharcPretrainPending)
    {
        // The upstream V-buffer has the scene's initial pose. Only the training RNG advances.
        while (mSharcPretrainedSpp < mSharcPretrainSpp)
        {
            {
                PROFILE("SHaRC Pretrain");
                const uint32_t batch = std::min(mSharcPretrainSpp - mSharcPretrainedSpp, kSharcPretrainBatchSpp);
                for (uint32_t i = 0; i < batch; ++i)
                {
                    traceSharcTrainingPaths(pRenderContext, 1u);
                    resolveSharcTraining(pRenderContext);
                    ++mSharcPretrainedSpp;
                }
            }
            // Bound outstanding GPU work and transient allocations for long pretrains.
            if (mSharcPretrainedSpp < mSharcPretrainSpp) pRenderContext->flush(true);
        }
        mSharcPretrainPending = false;
        if (mSharcPretrainedSpp > 0) return;
    }
    if (mSharcFreeze)
    {
        ++mSharcPublishedVersion;
        return;
    }
    {
        PROFILE("SHaRC SparseUpdate");
        traceSharcTrainingPaths(pRenderContext, kSharcUpdateStride);
    }
    {
        PROFILE("SHaRC Resolve");
        resolveSharcTraining(pRenderContext);
    }
}

void RTPT::traceSharcTrainingPaths(RenderContext* pRenderContext, uint32_t stride)
{
    ++mSharcUpdateCount;
    auto var = mpSharcUpdate->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    mpSampleGenerator->setShaderData(var);
    var["gPathTracer"] = mpPathTracerBlock;
    auto cb = var["CB"];
    bindSharc(cb["gSharcWrite"]);
    cb["gSharcFrameIndex"] = mSharcFrameIndex;
    cb["gUpdateStride"] = stride;
    mpSharcUpdate->execute(pRenderContext, uint3(div_round_up(mParams.frameDim, uint2(stride)), 1u));
    for (const auto& buffer : {mpSharcKeys, mpSharcLocks, mpSharcAccumulation, mpSharcResolved})
        pRenderContext->uavBarrier(buffer.get());
}

void RTPT::resolveSharcTraining(RenderContext* pRenderContext)
{
    ++mSharcResolveCount;
    auto cb = mpSharcResolve->getRootVar()["PerFrame"];
    bindSharc(cb["gSharc"]);
    auto resolve = cb["gResolve"];
    resolve["cameraPositionPrev"] = mSharcPreviousCamera;
    resolve["accumulationFrameNum"] = 32u;
    resolve["responsiveFrameNum"] = 1u;
    resolve["staleFrameNumMax"] = 128u;
    resolve["frameIndex"] = mSharcFrameIndex;
    mpSharcResolve->execute(pRenderContext, uint3(kSharcCapacity, 1u, 1u));
    for (const auto& buffer : {mpSharcKeys, mpSharcLocks, mpSharcAccumulation, mpSharcResolved})
        pRenderContext->uavBarrier(buffer.get());
    mSharcPreviousCamera = mpScene->getCamera()->getPosition();
    ++mSharcFrameIndex;
    ++mSharcPublishedVersion;
}

void RTPT::trackSharcMotion(bool lightingChanged)
{
    using Updates = Scene::UpdateFlags;
    const auto updates = mpScene->getUpdates();
    const auto globalChanges = Updates::GeometryChanged | Updates::DisplacementChanged | Updates::MaterialsChanged |
        Updates::LightsMoved | Updates::LightIntensityChanged | Updates::LightPropertiesChanged | Updates::LightCountChanged |
        Updates::LightCollectionChanged | Updates::EnvMapChanged | Updates::EnvMapPropertiesChanged | Updates::RenderSettingsChanged |
        Updates::CustomPrimitivesMoved;
    const bool graphChanged = is_set(updates, Updates::SceneGraphChanged);
    const bool geometryMoved = graphChanged || is_set(updates, Updates::MeshesMoved) || is_set(updates, Updates::CurvesMoved);
    const auto* animation = mpScene->getAnimationController();
    // Vertex-deformed bounds live on the GPU in this Falcor version.
    const bool deformed = (graphChanged && animation->getPrevVertexData()) ||
        (is_set(updates, Updates::CurvesMoved) && animation->hasAnimatedCurveCaches());
    mSharcNeedsClear |= lightingChanged || (updates & globalChanges) != Updates::None || deformed;
    mSharcMotionBounds.clear();

    const uint32_t meshes = mpScene->getMeshInstanceCount();
    const uint32_t curves = mpScene->getCurveInstanceCount();
    const bool first = mSharcPreviousBounds.size() != meshes + curves;
    if (!first && !geometryMoved) return;
    mSharcPreviousBounds.resize(meshes + curves);
    mSharcNeedsClear |= first;
    const auto& transforms = animation->getGlobalMatrices();
    auto track = [&](uint32_t index, uint32_t node, const AABB& local)
    {
        if (!first && !animation->isMatrixChanged(node)) return;
        const AABB world = local.transform(transforms[node]);
        if (!mSharcNeedsClear)
        {
            AABB swept = world;
            swept.include(mSharcPreviousBounds[index]);
            if (swept.valid())
            {
                mSharcMotionBounds.emplace_back(swept.minPoint, 0.f);
                mSharcMotionBounds.emplace_back(swept.maxPoint, 0.f);
            }
            else mSharcNeedsClear = true;
        }
        mSharcPreviousBounds[index] = world;
    };
    for (uint32_t i = 0; i < meshes; ++i)
    {
        const auto& instance = mpScene->getMeshInstance(i);
        track(i, instance.globalMatrixID, mpScene->getMeshBounds(instance.meshID));
    }
    for (uint32_t i = 0; i < curves; ++i)
    {
        const auto& instance = mpScene->getCurveInstance(i);
        track(meshes + i, instance.globalMatrixID, mpScene->getCurveBounds(instance.curveID));
    }
}

// The large readback is opt-in. It never runs as part of a rendered frame.
pybind11::dict RTPT::getSharcStats() const
{
    uint32_t occupied = 0, valid = 0, lit = 0, invalidValues = 0;
    uint64_t digest = 0;
    if (mpSharcKeys && mEnableSharc)
    {
        // SHARC_SH_ENCODING is disabled. Resolved layout is fp16 RGB + sample count, then two packed uints.
        const auto* keys = static_cast<const uint64_t*>(mpSharcKeys->map(Buffer::MapType::Read));
        const auto* resolved = static_cast<const uint32_t*>(mpSharcResolved->map(Buffer::MapType::Read));
        digest = 14695981039346656037ull;
        auto mix = [&](uint64_t value) { digest = (digest ^ value) * 1099511628211ull; };
        for (uint32_t i = 0; i < kSharcCapacity; ++i)
        {
            if (!keys[i]) continue;
            ++occupied;
            const uint32_t* entry = resolved + 4 * i;
            mix(i);
            mix(keys[i]);
            for (uint32_t j = 0; j < 4; ++j) mix(entry[j]);
            const uint16_t halves[] = { uint16_t(entry[0]), uint16_t(entry[0] >> 16),
                uint16_t(entry[1]), uint16_t(entry[1] >> 16) };
            const bool invalid = std::any_of(std::begin(halves), std::end(halves),
                [](uint16_t value) { return (value & 0x7c00u) == 0x7c00u || value > 0x8000u; });
            if (invalid) ++invalidValues;
            // Positive half-floats are ordered by their unsigned encoding. SDK's threshold is strictly > 4.
            if (invalid || halves[3] <= 0x4400u) continue;
            ++valid;
            if ((halves[0] | halves[1] | halves[2]) & 0x7fffu) ++lit;
        }
        mpSharcResolved->unmap();
        mpSharcKeys->unmap();
    }
    pybind11::dict stats;
    stats["capacity"] = mpSharcKeys && mEnableSharc ? kSharcCapacity : 0u;
    stats["occupied"] = occupied;
    stats["valid"] = valid;
    stats["lit"] = lit;
    stats["invalidValues"] = invalidValues;
    stats["digest"] = digest;
    stats["publishedVersion"] = mSharcPublishedVersion;
    const auto camera = mSharcPublishedGrid.cameraPosition;
    stats["cameraPosition"] = pybind11::make_tuple(camera.x, camera.y, camera.z);
    stats["sceneScale"] = mSharcPublishedGrid.sceneScale;
    stats["logarithmBase"] = mSharcPublishedGrid.logarithmBase;
    stats["levelBias"] = mSharcPublishedGrid.levelBias;
    stats["updates"] = mSharcUpdateCount;
    stats["pretrainedSpp"] = mSharcPretrainedSpp;
    stats["resolves"] = mSharcResolveCount;
    stats["clears"] = mSharcClearCount;
    stats["invalidations"] = mSharcInvalidateCount;
    stats["lastMotionBounds"] = mSharcMotionBounds.size() / 2;
    return stats;
}

pybind11::dict RTPT::getSharcGenerateStats() const
{
    // Read only when requested by a diagnostic client. Counter storage is cleared once each rendered frame.
    std::array<uint32_t, kGenerateCounterCount> values{};
    if (mSharcGenerateStats && mpSharcGenerateStats)
    {
        auto data = static_cast<const uint32_t*>(mpSharcGenerateStats->map(Buffer::MapType::Read));
        std::copy_n(data, values.size(), values.begin());
        mpSharcGenerateStats->unmap();
    }
    pybind11::dict result;
    const char* names[kGenerateMetricCount] = {"arrived", "eligible", "hits", "positive", "selected", "winners", "black", "explicit"};
    for (uint32_t metric = 0; metric < kGenerateMetricCount; ++metric)
    {
        std::vector<uint32_t> depths(kGenerateDepthCount);
        for (uint32_t depth = 0; depth < kGenerateDepthCount; ++depth)
            depths[depth] = values[depth * kGenerateMetricCount + metric];
        result[names[metric]] = depths;
    }
    result["suffixRays"] = std::vector<uint32_t>(values.begin() + kGenerateRayCountOffset, values.end());
    result["probe"] = pybind11::none();
    if (values[kGenerateProbeOffset])
    {
        const auto* probe = values.data() + kGenerateProbeOffset;
        auto vectorAt = [&](uint32_t index)
        {
            float rgb[3];
            std::memcpy(rgb, probe + index, sizeof(rgb));
            return pybind11::make_tuple(rgb[0], rgb[1], rgb[2]);
        };
        pybind11::dict sample;
        sample["pixel"] = pybind11::make_tuple(probe[1], probe[2]);
        sample["fullThroughput"] = vectorAt(3);
        sample["suffixThroughput"] = vectorAt(6);
        sample["radiance"] = vectorAt(9);
        sample["wi"] = vectorAt(12);
        sample["seed"] = probe[15];
        result["probe"] = sample;
    }
    return result;
}

pybind11::bytes RTPT::getSharcFreshReservoirs() const
{
    if (!mSharcGenerateStats || !mpOutputReservoirs) return pybind11::bytes("", 0);
    if (mEnableTemporalReuse || mEnableSpatialReuse)
        throw std::logic_error("Fresh reservoir readback requires Temporal and Spatial reuse disabled.");
    return readReservoirs(mpOutputReservoirs);
}

std::vector<uint32_t> RTPT::getSharcTemporalStats() const
{
    std::vector<uint32_t> values(kTemporalCounterCount);
    if (mSharcGenerateStats && mpSharcTemporalStats)
    {
        const auto* data = static_cast<const uint32_t*>(mpSharcTemporalStats->map(Buffer::MapType::Read));
        std::copy_n(data, values.size(), values.begin());
        mpSharcTemporalStats->unmap();
    }
    return values;
}

pybind11::bytes RTPT::getSharcTemporalReservoirs() const
{
    if (!mSharcGenerateStats || !mpOutputReservoirs) return pybind11::bytes("", 0);
    if (!mEnableTemporalReuse || mEnableSpatialReuse)
        throw std::logic_error("Temporal reservoir readback requires Temporal reuse enabled and Spatial reuse disabled.");
    return readReservoirs(mpOutputReservoirs);
}

pybind11::bytes RTPT::getSharcSpatialReservoirs() const
{
    if (!mSharcGenerateStats || !mpOutputReservoirs) return pybind11::bytes("", 0);
    if (!mEnableSpatialReuse) throw std::logic_error("Spatial reservoir readback requires Spatial reuse enabled.");
    // Odd round counts end in the per-sample buffer, even counts in the shared output.
    return readReservoirs(mNumSpatialRounds % 2 ? mpTemporalReservoirs.back() : mpOutputReservoirs);
}
