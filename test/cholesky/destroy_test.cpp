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
};
std::map<void*, Allocation> allocations;
std::map<aclrtStream, bool> completed;
aclrtStream failingStream = nullptr;
bool streamFailure = false;
bool deviceFailure = false;
void* failingFree = nullptr;
int freeFailures = 0;
int deviceSyncCalls = 0;

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
    allocations.clear();
    completed.clear();
    streamFailure = false;
    deviceFailure = false;
    failingFree = nullptr;
    freeFailures = 0;
    deviceSyncCalls = 0;
    aclsolverHandle_t handle = nullptr;
    Require(aclsolverCreate(&handle) == ACLSOLVER_STATUS_SUCCESS, "create failed");
    return static_cast<_aclsolver_handle*>(handle);
}

void* AddBlock(_aclsolver_handle* handle, aclrtStream stream)
{
    void* ptr = std::malloc(TEST_BLOCK_BYTES);
    Require(ptr != nullptr, "mock allocation failed");
    allocations.emplace(ptr, Allocation{stream});
    completed[stream] = false;
    handle->choleskyScratch.push_back({ptr, TEST_BLOCK_BYTES, stream});
    return ptr;
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
}  // namespace

namespace Cholesky
{
aclsolverStatus_t ReportError(aclsolverStatus_t status, const char*) { return status; }
}  // namespace Cholesky

extern "C" aclError aclrtGetCurrentContext(aclrtContext* context)
{
    *context = reinterpret_cast<aclrtContext>(&deviceSyncCalls);
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
        std::cout << "Destroy ownership, fallback, retry and safe release: 10 cases PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
