/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "../../src/utils/aclsolver_handle_internal.h"
#include "cann_ops_solver.h"

namespace
{
constexpr size_t TEST_BLOCK_BYTES = sizeof(aclsolverFloatComplex);
constexpr int MOCK_RUNTIME_ERROR = 1;
constexpr int PERSISTENT_FAILURE = -1;
enum class SyncFailure
{
    None,
    BeforeFree,
    AfterFree
};
struct Allocation
{
    aclrtStream stream;
    int attempts = 0;
    bool released = false;
    aclrtContext context = nullptr;
};
std::map<void*, Allocation> allocations;
std::map<aclrtStream, bool> completed;
aclrtStream failingStream = nullptr;
bool streamFailure = false;
bool deviceFailure = false;
void* failingFree = nullptr;
int freeFailures = 0;
int deviceSyncCalls = 0;
int realReleaseCalls = 0;
void* realScratch = nullptr;
aclrtContext currentContext = nullptr;
bool restoreFailure = false;
bool eventFailure = false;
bool eventDestroyFailure = false;
int eventSyncCalls = 0;
int eventDestroyCalls = 0;

aclrtContext OwningContext() { return reinterpret_cast<aclrtContext>(&deviceSyncCalls); }
aclrtContext OtherContext() { return reinterpret_cast<aclrtContext>(&freeFailures); }

void Require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

aclrtStream OtherStream() { return reinterpret_cast<aclrtStream>(&deviceSyncCalls); }

_aclsolver_handle* NewHandle()
{
    for (const auto& entry : allocations)
    {
        Require(entry.second.released, "previous case retained an allocation");
    }
    Require(realScratch == nullptr, "previous case retained real scratch");
    realReleaseCalls = 0;
    allocations.clear();
    completed.clear();
    streamFailure = false;
    deviceFailure = false;
    failingFree = nullptr;
    freeFailures = 0;
    deviceSyncCalls = 0;
    currentContext = OwningContext();
    restoreFailure = eventFailure = eventDestroyFailure = false;
    eventSyncCalls = eventDestroyCalls = 0;
    aclsolverHandle_t handle = nullptr;
    Require(aclsolverCreate(&handle) == ACLSOLVER_STATUS_SUCCESS, "create failed");
    return static_cast<_aclsolver_handle*>(handle);
}

void* AddBlock(_aclsolver_handle* handle, aclrtStream stream)
{
    void* ptr = std::malloc(TEST_BLOCK_BYTES);
    Require(ptr != nullptr, "mock allocation failed");
    allocations.emplace(ptr, Allocation{stream});
    allocations.at(ptr).context = currentContext;
    completed[stream] = false;
    handle->choleskyScratch.push_back({ptr, TEST_BLOCK_BYTES, stream});
    return ptr;
}

void* AddRealBlock(_aclsolver_handle* handle, bool recorded)
{
    void* ptr = std::malloc(TEST_BLOCK_BYTES);
    Require(ptr != nullptr, "mock LU allocation failed");
    allocations.emplace(ptr, Allocation{OtherStream(), 0, false, OtherContext()});
    completed[OtherStream()] = false;
    _real_lu_workspace cache;
    cache.context = OtherContext();
    cache.data = ptr;
    cache.capacity = TEST_BLOCK_BYTES;
    cache.recorded = recorded;
    cache.completion = recorded ? reinterpret_cast<aclrtEvent>(ptr) : nullptr;
    cache.untracked = !recorded;
    cache.pendingStream = recorded ? nullptr : OtherStream();
    handle->realLuWorkspaces.push_back(cache);
    return ptr;
}

void ReleaseRealScratch(void* ptr)
{
    Require(ptr == realScratch && realReleaseCalls == 0, "real scratch released incorrectly");
    delete static_cast<int*>(ptr);
    realScratch = nullptr;
    ++realReleaseCalls;
}

void AddRealScratch(_aclsolver_handle* handle)
{
    realScratch = new int(7);
    handle->solverScratch = realScratch;
    handle->releaseScratch = ReleaseRealScratch;
}

void RequireReleased()
{
    for (const auto& entry : allocations)
    {
        Require(entry.second.released, "allocation was not released");
    }
}

void StreamFallback(bool useDefault)
{
    auto* handle = NewHandle();
    failingStream = useDefault ? nullptr : OtherStream();
    AddBlock(handle, failingStream);
    AddBlock(handle, failingStream);
    streamFailure = true;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_EXECUTION_FAILED, "lost synchronization error");
    Require(deviceSyncCalls == 1 && handle->choleskyScratch.empty(), "device fallback did not reclaim all blocks");
    RequireReleased();
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "empty handle retry failed");
}

void FreeFailure(bool persistent, SyncFailure syncFailure)
{
    auto* handle = NewHandle();
    failingFree = AddBlock(handle, nullptr);
    void* other = AddBlock(handle, OtherStream());
    freeFailures = persistent ? PERSISTENT_FAILURE : 1;
    failingStream = syncFailure == SyncFailure::BeforeFree ? nullptr : OtherStream();
    streamFailure = syncFailure != SyncFailure::None;
    auto expected =
        syncFailure == SyncFailure::BeforeFree ? ACLSOLVER_STATUS_EXECUTION_FAILED : ACLSOLVER_STATUS_INTERNAL_ERROR;
    Require(aclsolverDestroy(handle) == expected, "first error was not preserved");
    Require(handle->choleskyScratch.size() == 1 && handle->choleskyScratch[0].ptr == failingFree,
            "failed release lost ownership or blocked another release");
    Require(allocations[other].released && allocations[other].attempts == 1, "other block was not reclaimed");
    if (persistent)
    {
        Require(aclsolverDestroy(handle) == expected, "persistent release error was hidden");
        Require(handle->choleskyScratch.size() == 1, "persistent failure lost ownership");
    }
    streamFailure = false;
    freeFailures = 0;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "release retry failed");
    RequireReleased();
    Require(allocations[other].attempts == 1, "retry released a completed block twice");
}

void IncompleteStream()
{
    auto* handle = NewHandle();
    failingStream = nullptr;
    void* pending = AddBlock(handle, failingStream);
    void* ready = AddBlock(handle, OtherStream());
    streamFailure = deviceFailure = true;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_EXECUTION_FAILED, "synchronization failure was hidden");
    Require(allocations[pending].attempts == 0, "freed a block without a completed barrier");
    Require(allocations[ready].released && handle->choleskyScratch.size() == 1, "other stream cleanup was blocked");
    streamFailure = deviceFailure = false;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "pending stream retry failed");
    RequireReleased();
}

void MixedSuccess()
{
    auto* handle = NewHandle();
    AddBlock(handle, nullptr);
    AddRealBlock(handle, true);
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "mixed cleanup failed");
    Require(currentContext == OwningContext(), "LU cleanup did not restore the owning context");
    Require(eventSyncCalls == 1 && eventDestroyCalls == 1, "LU event lifecycle changed");
    RequireReleased();
}

void MixedLuFailure(bool event, bool destroyEvent)
{
    auto* handle = NewHandle();
    void* chol = AddBlock(handle, nullptr);
    void* real = AddRealBlock(handle, event);
    failingStream = OtherStream();
    streamFailure = !event;
    eventFailure = event && !destroyEvent;
    eventDestroyFailure = destroyEvent;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_EXECUTION_FAILED, "LU cleanup error was hidden");
    Require(allocations[chol].attempts == 0 && !completed[nullptr], "Cholesky ran after failed LU cleanup");
    streamFailure = eventFailure = eventDestroyFailure = false;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "mixed LU retry failed");
    Require(allocations[real].attempts == 1, "LU retry released a completed block twice");
    RequireReleased();
}

void MixedRestoreFailure()
{
    auto* handle = NewHandle();
    void* chol = AddBlock(handle, nullptr);
    AddRealBlock(handle, false);
    restoreFailure = true;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_EXECUTION_FAILED, "context restore error was hidden");
    Require(currentContext == OtherContext() && allocations[chol].attempts == 0,
            "Cholesky cleanup used the wrong context");
    restoreFailure = false;
    Require(aclrtSetCurrentContext(OwningContext()) == ACL_SUCCESS, "caller context recovery failed");
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "context recovery retry failed");
    RequireReleased();
}

void MixedCholeskyFailure()
{
    auto* handle = NewHandle();
    failingFree = AddBlock(handle, nullptr);
    void* real = AddRealBlock(handle, false);
    freeFailures = 1;
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_INTERNAL_ERROR, "Cholesky release error was hidden");
    Require(allocations[real].released, "LU cleanup was lost");
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "mixed Cholesky retry failed");
    Require(allocations[real].attempts == 1, "Cholesky retry released the LU block twice");
    RequireReleased();
}

void RealScratchOnly()
{
    auto* handle = NewHandle();
    AddRealScratch(handle);
    Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "real-only cleanup failed");
    Require(realReleaseCalls == 1 && realScratch == nullptr, "real-only callback was lost");
}

void MixedScratch(bool failFree, bool failSync)
{
    auto* handle = NewHandle();
    failingFree = AddBlock(handle, nullptr);
    AddBlock(handle, OtherStream());
    AddRealScratch(handle);
    freeFailures = failFree ? 1 : 0;
    streamFailure = deviceFailure = failSync;
    auto status = aclsolverDestroy(handle);
    if (failFree || failSync)
    {
        Require(status == (failSync ? ACLSOLVER_STATUS_EXECUTION_FAILED : ACLSOLVER_STATUS_INTERNAL_ERROR),
                "mixed cleanup lost the first error");
        Require(realReleaseCalls == 0 && handle->solverScratch == realScratch,
                "failed complex cleanup discarded real ownership");
        streamFailure = deviceFailure = false;
        freeFailures = 0;
        status = aclsolverDestroy(handle);
    }
    Require(status == ACLSOLVER_STATUS_SUCCESS, "mixed cleanup retry failed");
    Require(realReleaseCalls == 1 && realScratch == nullptr, "mixed cleanup lost or repeated real release");
    RequireReleased();
}
}  // namespace

namespace Cholesky
{
aclsolverStatus_t ReportError(aclsolverStatus_t status, const char*) { return status; }
}  // namespace Cholesky

extern "C" aclError aclrtGetCurrentContext(aclrtContext* context)
{
    *context = currentContext;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSetCurrentContext(aclrtContext context)
{
    if (restoreFailure && context == OwningContext() && currentContext != context)
    {
        return MOCK_RUNTIME_ERROR;
    }
    currentContext = context;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSynchronizeEvent(aclrtEvent event)
{
    ++eventSyncCalls;
    if (eventFailure)
    {
        return MOCK_RUNTIME_ERROR;
    }
    completed[allocations.at(event).stream] = true;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyEvent(aclrtEvent)
{
    ++eventDestroyCalls;
    return eventDestroyFailure ? MOCK_RUNTIME_ERROR : ACL_SUCCESS;
}
extern "C" aclError aclrtFreeHost(void* ptr)
{
    std::free(ptr);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSynchronizeStream(aclrtStream stream)
{
    if (streamFailure && stream == failingStream)
    {
        return MOCK_RUNTIME_ERROR;
    }
    completed[stream] = true;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSynchronizeDevice()
{
    ++deviceSyncCalls;
    if (deviceFailure)
    {
        return MOCK_RUNTIME_ERROR;
    }
    for (auto& entry : completed)
    {
        entry.second = true;
    }
    return ACL_SUCCESS;
}
extern "C" aclError aclrtFree(void* ptr)
{
    auto& allocation = allocations.at(ptr);
    Require(!allocation.released && completed[allocation.stream], "unsafe or duplicate release");
    Require(currentContext == allocation.context, "release used the wrong context");
    ++allocation.attempts;
    if (ptr == failingFree && freeFailures != 0)
    {
        if (freeFailures > 0)
        {
            --freeFailures;
        }
        return MOCK_RUNTIME_ERROR;
    }
    allocation.released = true;
    std::free(ptr);
    return ACL_SUCCESS;
}

int main()
{
    try
    {
        auto* handle = NewHandle();
        AddBlock(handle, nullptr);
        AddBlock(handle, OtherStream());
        Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "ordinary cleanup failed");
        RequireReleased();
        StreamFallback(false);
        StreamFallback(true);
        FreeFailure(false, SyncFailure::None);
        FreeFailure(true, SyncFailure::None);
        FreeFailure(false, SyncFailure::BeforeFree);
        FreeFailure(false, SyncFailure::AfterFree);
        IncompleteStream();
        handle = NewHandle();
        streamFailure = deviceFailure = true;
        Require(aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS, "empty handle cleanup touched Runtime");
        Require(deviceSyncCalls == 0, "empty handle performed a device barrier");
        Require(aclsolverDestroy(nullptr) == ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, "null handle status changed");
        MixedSuccess();
        MixedLuFailure(false, false);
        MixedLuFailure(true, false);
        MixedLuFailure(true, true);
        MixedRestoreFailure();
        MixedCholeskyFailure();
        RealScratchOnly();
        MixedScratch(false, false);
        MixedScratch(true, false);
        MixedScratch(false, true);
        std::cout << "Destroy ownership, fallback, retry and mixed scratch: 20 cases PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
