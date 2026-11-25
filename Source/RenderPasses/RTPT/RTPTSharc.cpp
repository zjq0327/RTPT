/***************************************************************************
 # Copyright (c) 2022, Daqi Lin. All rights reserved.
 **************************************************************************/
#include "RTPT.h"

namespace
{
    constexpr uint32_t kSharcCapacity = 1u << 20;
    constexpr uint32_t kSharcUpdateStride = 5;
    const char kSharcUpdateFile[] = "RenderPasses/RTPT/SharcUpdate.cs.slang";
    const char kSharcResolveFile[] = "RenderPasses/ReSTIRPTPass/SharcResolve.cs.slang";
    const char kSharcInvalidateFile[] = "RenderPasses/ReSTIRPTPass/SharcInvalidate.cs.slang";
}

void RTPT::specializeSharc(const Program::DefineList& defines)
{
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
        return;
    }
    auto updateDefines = defines;
    updateDefines.add("SHARC_TRAINING", "1");
    updateDefines.add("SHARC_MIN_DEPTH", "3");
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
    if (mSharcFreeze && !mSharcNeedsClear && mSharcMotionBounds.empty()) return;
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
    if (mSharcFreeze)
    {
        ++mSharcPublishedVersion;
        return;
    }
    {
        PROFILE("SHaRC SparseUpdate");
        ++mSharcUpdateCount;
        auto var = mpSharcUpdate->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
        mpSampleGenerator->setShaderData(var);
        var["gPathTracer"] = mpPathTracerBlock;
        auto cb = var["CB"];
        bindSharc(cb["gSharcWrite"]);
        cb["gSharcFrameIndex"] = mSharcFrameIndex;
        cb["gUpdateStride"] = kSharcUpdateStride;
        mpSharcUpdate->execute(pRenderContext, uint3(div_round_up(mParams.frameDim, uint2(kSharcUpdateStride)), 1u));
        for (const auto& buffer : buffers) pRenderContext->uavBarrier(buffer.get());
    }
    {
        PROFILE("SHaRC Resolve");
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
        for (const auto& buffer : buffers) pRenderContext->uavBarrier(buffer.get());
    }
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
    stats["resolves"] = mSharcResolveCount;
    stats["clears"] = mSharcClearCount;
    stats["invalidations"] = mSharcInvalidateCount;
    stats["lastMotionBounds"] = mSharcMotionBounds.size() / 2;
    return stats;
}
