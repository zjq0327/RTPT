/***************************************************************************
 # Copyright (c) 2022, Daqi Lin. All rights reserved.
 **************************************************************************/
#include "RTPT.h"
#include <cmath>
#include <cstring>

namespace
{
    void clearPGBuffer(RenderContext* context, const Buffer::SharedPtr& buffer)
    {
        context->clearUAV(buffer->getUAV().get(), uint4(0));
        context->uavBarrier(buffer.get());
    }
}

void RTPT::releasePG()
{
    mpPGPublish.reset();
    if (mpSharcPGUpdate) mpSharcPGUpdate->setVars(nullptr);
    mpPGBuildKeys = mpPGAccumulation = mpPGCounts = nullptr;
    mpPGHistoryMean = mpPGHistoryCount = nullptr;
    mpPGGenerateCounts = mpPGTemporalCounts = mpPGTemporalReplayCounts = nullptr;
    mpPGSpatialCounts.reset();
    mPGGenerateBoundVersion.reset();
    mPGTemporalBoundVersion.reset();
    mPGSpatialBoundVersions.clear();
    for (auto& snapshot : mPGSnapshots) snapshot = {};
    mPGCurrent = mPGPrevious = 0;
    mPGPublishPending = false;
    mPGResetPending = false;
    mPGCarryPrevious = true;
    mPGRuntimeFrameCount = mPGPublications = 0;
    mPGPretrainDispatches = mPGRuntimeDispatches = 0;
    mPGPretrainPaths = mPGRuntimePaths = 0;
    mPGLastTrainingFrame = mPGLastTrainingSample = 0;
    mPGNextTrainingSample = 0;
}

void RTPT::preparePG(RenderContext* context)
{
    mPGPublishedThisFrame = false;
    auto prepareCounters = [&](Buffer::SharedPtr& buffer, uint32_t count, bool enabled)
    {
        if (!enabled) { buffer.reset(); return; }
        if (!buffer)
            buffer = Buffer::createStructured(sizeof(uint32_t), count,
                Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        clearPGBuffer(context, buffer);
    };
    prepareCounters(mpPGGenerateCounts, kPGGenerateCounterCount, mPGGenerateStats);
    prepareCounters(mpPGTemporalCounts, kPGTemporalCounterCount, mPGTemporalStats);
    prepareCounters(mpPGTemporalReplayCounts, kPGGenerateCounterCount, mPGTemporalStats);
    prepareCounters(mpPGSpatialCounts, kPGSpatialCounterCount, mPGSpatialStats);
    if (!mpPGBuildKeys)
    {
        const auto flags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess;
        auto allocate = [&](uint32_t stride, uint32_t count)
        {
            auto buffer = Buffer::createStructured(stride, count, flags, Buffer::CpuAccess::None, nullptr, false);
            clearPGBuffer(context, buffer);
            return buffer;
        };
        mpPGBuildKeys = allocate(8u, kPGCapacity);
        mpPGAccumulation = allocate(16u, kPGCapacity);
        mpPGCounts = allocate(4u, kPGCapacity);
        mpPGHistoryMean = allocate(16u, kPGCapacity);
        mpPGHistoryCount = allocate(4u, kPGCapacity);
        for (auto& snapshot : mPGSnapshots)
        {
            snapshot.keys = allocate(8u, kPGCapacity);
            snapshot.mass = allocate(16u, kPGCapacity);
            snapshot.cameraPosition = mpScene->getCamera()->getPosition();
            snapshot.sceneScale = mPGSceneScale;
        }
        Program::Desc desc;
        desc.addShaderLibrary("RenderPasses/RTPT/PGPublish.cs.slang").csEntry("main").setShaderModel("6_5");
        mpPGPublish = ComputePass::create(desc, {}, true);
        mPGResetPending = false;
    }
    else if (mPGResetPending || mSharcNeedsClear)
    {
        // A scene-wide transport change invalidates the previous observations.
        // Keep the previous publication available for diagnostics.
        clearPGBuffer(context, mpPGBuildKeys);
        clearPGBuffer(context, mpPGHistoryMean);
        clearPGBuffer(context, mpPGHistoryCount);
        clearPGBuffer(context, mpPGAccumulation);
        clearPGBuffer(context, mpPGCounts);
        mPGResetPending = false;
        mPGPublishPending = false;
        mPGPrevious = mPGCurrent;
        mPGCarryPrevious = false;
        mPGCurrent ^= 1u;
        auto& current = mPGSnapshots[mPGCurrent];
        clearPGBuffer(context, current.keys);
        clearPGBuffer(context, current.mass);
        current.version = ++mPGPublications;
        current.cameraPosition = mpScene->getCamera()->getPosition();
        current.sceneScale = mPGSceneScale;
        return;
    }
    mPGPrevious = mPGCurrent;
    if (mPGPublishPending) publishPG(context);
}

void RTPT::bindPGTraining(const ShaderVar& var) const
{
    var["capacity"] = kPGCapacity;
    var["cameraPosition"] = mpScene->getCamera()->getPosition();
    var["sceneScale"] = mPGSceneScale;
    var["keys"] = mpPGBuildKeys;
    var["accumulation"] = mpPGAccumulation;
    var["counts"] = mpPGCounts;
}

void RTPT::bindCurrentPGSampling(const ShaderVar& cb) const
{
    // Generate, temporal shifts and replay, and every spatial round use this frame's immutable publication.
    const auto& snapshot = mPGSnapshots[mPGCurrent];
    cb["pgEnabled"] = mEnablePG && (mPGDiffuseFraction > 0.f || mPGGenerateStats || mPGTemporalStats || mPGSpatialStats);
    cb["pgDiffuseFraction"] = mPGDiffuseFraction;
    const auto var = cb["pgRead"];
    var["capacity"] = mEnablePG && snapshot.keys ? kPGCapacity : 0u;
    var["cameraPosition"] = snapshot.cameraPosition;
    var["sceneScale"] = snapshot.sceneScale;
    var["keys"] = mEnablePG ? snapshot.keys : nullptr;
    var["mass"] = mEnablePG ? snapshot.mass : nullptr;
}

void RTPT::barrierPGTraining(RenderContext* context) const
{
    for (const auto& buffer : {mpPGBuildKeys, mpPGAccumulation, mpPGCounts})
        context->uavBarrier(buffer.get());
}

void RTPT::notePGTraining(uint32_t stride, bool pretrain)
{
    const uint2 tiles = div_round_up(mParams.frameDim, uint2(stride));
    const uint64_t paths = uint64_t(tiles.x) * tiles.y;
    if (pretrain)
    {
        ++mPGPretrainDispatches;
        mPGPretrainPaths += paths;
    }
    else
    {
        ++mPGRuntimeDispatches;
        mPGRuntimePaths += paths;
    }
    mPGLastTrainingFrame = mPGRuntimeFrameCount;
    mPGTrainingCamera = mpScene->getCamera()->getPosition();
}

void RTPT::finishPGTraining(RenderContext* context, bool immediate)
{
    if (immediate) publishPG(context);
    else mPGPublishPending = true;
}

void RTPT::publishPG(RenderContext* context)
{
    PROFILE("PG Publish");
    mPGPublishedThisFrame = true;
    const auto& source = mPGSnapshots[mPGCurrent];
    const uint32_t outputIndex = mPGCurrent ^ 1u;
    auto& output = mPGSnapshots[outputIndex];
    auto cb = mpPGPublish->getRootVar()["CB"];
    cb["gCapacity"] = kPGCapacity;
    cb["gCarryPrevious"] = uint32_t(mPGCarryPrevious);
    cb["gKeys"] = mpPGBuildKeys;
    cb["gAccumulation"] = mpPGAccumulation;
    cb["gCounts"] = mpPGCounts;
    cb["gHistoryMean"] = mpPGHistoryMean;
    cb["gHistoryCount"] = mpPGHistoryCount;
    cb["gPrevKeys"] = source.keys;
    cb["gPrevMass"] = source.mass;
    cb["gOutKeys"] = output.keys;
    cb["gOutMass"] = output.mass;
    mpPGPublish->execute(context, uint3(kPGCapacity, 1u, 1u));
    context->uavBarrier(output.keys.get());
    context->uavBarrier(output.mass.get());
    context->uavBarrier(mpPGHistoryMean.get());
    context->uavBarrier(mpPGHistoryCount.get());
    clearPGBuffer(context, mpPGAccumulation);
    clearPGBuffer(context, mpPGCounts);
    output.version = ++mPGPublications;
    output.cameraPosition = mPGTrainingCamera;
    output.sceneScale = mPGSceneScale;
    mPGCurrent = outputIndex;
    mPGPublishPending = false;
    mPGCarryPrevious = true;
}

pybind11::dict RTPT::getPGStats() const
{
    pybind11::dict stats;
    stats["enabled"] = mEnablePG;
    const bool guidingEnabled = mEnablePG && mPGDiffuseFraction > 0.f;
    stats["generateGuidingEnabled"] = guidingEnabled;
    stats["temporalGuidingEnabled"] = guidingEnabled;
    stats["spatialGuidingEnabled"] = guidingEnabled;
    stats["generateBoundVersion"] = mPGGenerateBoundVersion ? pybind11::cast(*mPGGenerateBoundVersion) : pybind11::none();
    stats["temporalBoundVersion"] = mPGTemporalBoundVersion ? pybind11::cast(*mPGTemporalBoundVersion) : pybind11::none();
    stats["spatialBoundVersions"] = mPGSpatialBoundVersions;
    stats["spatialBoundVersion"] = mPGSpatialBoundVersions.empty() ? pybind11::none() : pybind11::cast(mPGSpatialBoundVersions.back());
    stats["diffuseFraction"] = mPGDiffuseFraction;
    stats["spatialCachePrimary"] = true;
    const char* generateCounters[kPGGenerateCounterCount] = {
        "diffuseEligible", "diffuseTableHits", "pgDiffuseDraws", "pgSpecularDraws", "nativeSpecularDraws", "nativeDiffuseDraws"
    };
    const char* temporalCounters[kPGTemporalCounterCount] = {
        "temporalForwardShiftEvals", "temporalReverseShiftEvals", "temporalForwardReadyLookups", "temporalReverseReadyLookups",
        "temporalForwardPositiveTargets", "temporalReversePositiveTargets", "temporalForwardReplayCalls", "temporalHistoryWins"
    };
    const char* replayCounters[kPGGenerateCounterCount] = {
        "temporalReplayDiffuseEligible", "temporalReplayDiffuseTableHits", "temporalReplayPGDiffuseDraws",
        "temporalReplayPGSpecularDraws", "temporalReplayNativeSpecularDraws", "temporalReplayNativeDiffuseDraws"
    };
    const char* spatialCounters[kPGSpatialCounterCount] = {
        "spatialNeighborToCenterEvals", "spatialCenterToNeighborEvals", "spatialNeighborToCenterReadyLookups",
        "spatialCenterToNeighborReadyLookups", "spatialNeighborToCenterPositiveTargets",
        "spatialCenterToNeighborPositiveTargets", "spatialPartnerVisits", "spatialNeighborWins",
        "spatialCenterPrimaryLookupRequests", "spatialNeighborPrimaryLookupRequests"
    };
    auto readCounters = [&](const Buffer::SharedPtr& buffer, const auto& names)
    {
        const auto* counts = buffer ? static_cast<const uint32_t*>(buffer->map(Buffer::MapType::Read)) : nullptr;
        uint32_t i = 0;
        for (const char* name : names) { stats[name] = counts ? counts[i] : 0u; ++i; }
        if (counts) buffer->unmap();
    };
    readCounters(mpPGGenerateCounts, generateCounters);
    readCounters(mpPGTemporalCounts, temporalCounters);
    readCounters(mpPGTemporalReplayCounts, replayCounters);
    readCounters(mpPGSpatialCounts, spatialCounters);
    stats["spatialPrimaryLookupRequests"] = stats["spatialCenterPrimaryLookupRequests"].cast<uint64_t>() +
        stats["spatialNeighborPrimaryLookupRequests"].cast<uint64_t>();
    stats["capacity"] = mpPGBuildKeys ? kPGCapacity : 0u;
    stats["sceneScale"] = mPGSceneScale;
    stats["publications"] = mPGPublications;
    stats["pending"] = mPGPublishPending;
    stats["displayedFrames"] = mPGRuntimeFrameCount;
    stats["pretrainDispatches"] = mPGPretrainDispatches;
    stats["pretrainPaths"] = mPGPretrainPaths;
    stats["runtimeDispatches"] = mPGRuntimeDispatches;
    stats["runtimePaths"] = mPGRuntimePaths;
    stats["trainingDispatches"] = mPGPretrainDispatches + mPGRuntimeDispatches;
    stats["trainingPaths"] = mPGPretrainPaths + mPGRuntimePaths;
    stats["lastTrainingFrame"] = mPGLastTrainingFrame;
    stats["lastTrainingSample"] = mPGLastTrainingSample;

    auto readSnapshot = [](const PGSnapshot& snapshot)
    {
        pybind11::dict info;
        uint32_t ready = 0, occupied = 0, invalid = 0;
        uint64_t digest = 14695981039346656037ull;
        info["example"] = pybind11::none();
        if (snapshot.keys)
        {
            const auto* keys = static_cast<const uint32_t*>(snapshot.keys->map(Buffer::MapType::Read));
            const auto* masses = static_cast<const float*>(snapshot.mass->map(Buffer::MapType::Read));
            for (uint32_t i = 0; i < kPGCapacity; ++i)
            {
                const bool keyed = keys[2 * i] && keys[2 * i + 1];
                const float* mass = masses + 4 * i;
                float sum = 0.f;
                bool valid = true;
                for (uint32_t j = 0; j < 4; ++j)
                {
                    valid &= std::isfinite(mass[j]) && mass[j] >= 0.f;
                    sum += mass[j];
                }
                if (!valid || (sum > 0.f && (!keyed || std::abs(sum - 1.f) > 1e-4f))) ++invalid;
                if (!keyed) continue;
                ++occupied;
                auto mix = [&](uint32_t word) { digest = (digest ^ word) * 1099511628211ull; };
                mix(i);
                mix(keys[2 * i]);
                mix(keys[2 * i + 1]);
                for (uint32_t j = 0; j < 4; ++j)
                {
                    uint32_t bits;
                    std::memcpy(&bits, mass + j, sizeof(bits));
                    mix(bits);
                }
                if (!(sum > 0.f) || !valid) continue;
                if (!ready)
                {
                    pybind11::dict sample;
                    sample["slot"] = i;
                    sample["key"] = pybind11::make_tuple(keys[2 * i], keys[2 * i + 1]);
                    sample["mass"] = pybind11::make_tuple(mass[0], mass[1], mass[2], mass[3]);
                    info["example"] = sample;
                }
                ++ready;
            }
            snapshot.mass->unmap();
            snapshot.keys->unmap();
        }
        info["readyRegions"] = ready;
        info["occupied"] = occupied;
        info["invalidValues"] = invalid;
        info["digest"] = digest;
        return info;
    };
    const auto& current = mPGSnapshots[mPGCurrent];
    const auto& previous = mPGSnapshots[mPGPrevious];
    const auto curr = readSnapshot(current);
    const auto prev = mPGCurrent == mPGPrevious ? curr : readSnapshot(previous);
    stats["version"] = current.version;
    stats["previousVersion"] = previous.version;
    stats["previousSceneScale"] = previous.sceneScale;
    stats["cameraPosition"] = pybind11::make_tuple(current.cameraPosition.x, current.cameraPosition.y, current.cameraPosition.z);
    stats["previousCameraPosition"] = pybind11::make_tuple(previous.cameraPosition.x, previous.cameraPosition.y, previous.cameraPosition.z);
    stats["occupied"] = curr["occupied"];
    stats["readyRegions"] = curr["readyRegions"];
    stats["previousReadyRegions"] = prev["readyRegions"];
    stats["invalidValues"] = curr["invalidValues"];
    stats["previousInvalidValues"] = prev["invalidValues"];
    stats["digest"] = curr["digest"];
    stats["previousDigest"] = prev["digest"];
    stats["example"] = curr["example"];
    stats["previousExample"] = prev["example"];
    return stats;
}
