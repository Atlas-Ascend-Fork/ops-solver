/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <map>
#include <new>
#include <vector>

#include "aclsolver_handle_internal.h"
#include "cann_ops_solver.h"

extern "C" void real_lu_fill_info_do(void *info, int count, int value, aclrtStream stream);
extern "C" void real_lu_factor_do(void *a, int m, int n, int lda, void *piv, void *info, aclrtStream stream);
extern "C" void real_lu_thin_factor_do(void *a, void *workspace, int m, int n, int lda, void *piv, void *info,
                                       aclrtStream stream);
extern "C" void real_lu_inverse_do(const void *a, const void *c, const void *piv, void *info, int n, int lda, int ldc,
                                   int batch, bool arrays, bool factor, aclrtStream stream);
extern "C" void real_lu_copy_do(const void *src, void *dst, const void *info, int n, int lda, aclrtStream stream);
extern "C" void real_lu_small_do(const void *a, void *c, const void *piv, void *info, int n, int batch, bool factor,
                                 bool identityPivots, aclrtStream stream);
extern "C" void real_lu_finite_do(const void *a, void *flags, int m, int n, int lda, int batch, bool arrays, int blocks,
                                  aclrtStream stream);
extern "C" int real_lu_vector_cores();
extern "C" int real_lu_cube_cores();
extern "C" int real_lu_blocked_factor_do(void *a, void *workspace, void *piv, void *info, int m, int n, int lda,
                                         aclrtStream stream);
extern "C" int real_lu_gemm_do(const void *a, const void *b, void *c, const void *info, int pitch, int rows,
                               int columns, int depth, int64_t matrixStride, int batch, int64_t offsetA,
                               int64_t offsetB, int64_t offsetC, bool transposeB, aclrtStream stream);
extern "C" int real_lu_blocked_inverse_do(const void *a, const void *c, const void *piv, void *info, void *workspace,
                                          int n, int lda, int ldc, int batch, bool arrays, bool identityPivots,
                                          aclrtStream stream);

namespace real_lu
{
inline aclrtStream Stream(aclsolverHandle_t handle) { return static_cast<_aclsolver_handle *>(handle)->stream; }
inline bool Device(const void *pointer)
{
    if (pointer == nullptr)
    {
        return false;
    }
    aclrtPtrAttributes attr{};
    return aclrtPointerGetAttributes(pointer, &attr) == ACL_SUCCESS &&
           attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
}
inline aclsolverStatus_t Launched()
{
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS ? ACLSOLVER_STATUS_SUCCESS
                                                                    : ACLSOLVER_STATUS_EXECUTION_FAILED;
}
inline aclsolverStatus_t Invalid(aclsolverHandle_t handle, int *info, int count, int argument)
{
    if (count > 0 && reinterpret_cast<uintptr_t>(info) % alignof(int) == 0 && Device(info))
    {
        void *base = nullptr;
        size_t length = 0;
        auto offset = reinterpret_cast<uintptr_t>(info);
        if (aclrtMemGetAddressRange(info, &base, &length) == ACL_SUCCESS && offset >= reinterpret_cast<uintptr_t>(base))
        {
            offset -= reinterpret_cast<uintptr_t>(base);
            if (offset <= length && static_cast<size_t>(count) * sizeof(int) <= length - offset)
            {
                real_lu_fill_info_do(info, count, -argument, Stream(handle));
            }
        }
    }
    return ACLSOLVER_STATUS_INVALID_VALUE;
}
inline bool Overlap(const void *a, size_t aBytes, const void *b, size_t bBytes)
{
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < aBytes : x - y < bBytes;
}
class Ranges
{
   public:
    bool Fits(const void *pointer, size_t bytes)
    {
        uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
        if (pointer == nullptr || bytes > UINTPTR_MAX - address)
        {
            return false;
        }
        auto upper = ranges.upper_bound(address);
        if (upper != ranges.begin())
        {
            auto previous = std::prev(upper);
            if (address < previous->second && bytes <= previous->second - address)
            {
                return true;
            }
        }
        void *base = nullptr;
        size_t length = 0;
        if (!Device(pointer) || aclrtMemGetAddressRange(const_cast<void *>(pointer), &base, &length) != ACL_SUCCESS)
        {
            return false;
        }
        uintptr_t first = reinterpret_cast<uintptr_t>(base);
        if (length > UINTPTR_MAX - first || address < first || bytes > first + length - address)
        {
            return false;
        }
        ranges[first] = first + length;
        return true;
    }

   private:
    std::map<uintptr_t, uintptr_t> ranges;
};
inline bool InfoFits(int *info, int count)
{
    Ranges ranges;
    return reinterpret_cast<uintptr_t>(info) % alignof(int) == 0 &&
           ranges.Fits(info, static_cast<size_t>(count) * sizeof(int));
}
inline aclsolverStatus_t EmptyInfo(aclsolverHandle_t handle, int *info, int count)
{
    if (!InfoFits(info, count))
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    real_lu_fill_info_do(info, count, 0, Stream(handle));
    return Launched();
}
struct ProtectedView
{
    const void *data;
    size_t bytes;
    int argument;
};
inline bool AlignedFits(Ranges &ranges, const void *pointer, size_t bytes, size_t alignment)
{
    return alignment != 0 && reinterpret_cast<uintptr_t>(pointer) % alignment == 0 && ranges.Fits(pointer, bytes);
}
inline aclsolverStatus_t CheckViewOverlaps(aclsolverHandle_t handle, int *info, int count,
                                           const ProtectedView (&views)[4])
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            if (views[i].bytes && views[j].bytes &&
                Overlap(views[i].data, views[i].bytes, views[j].data, views[j].bytes))
            {
                if (j == 3)
                {
                    return ACLSOLVER_STATUS_INVALID_VALUE;
                }
                return Invalid(handle, info, count, views[j].argument);
            }
        }
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

#if defined(REAL_LU_REGISTER_LAYOUT_950)
inline bool FullLocalFits950(int m, int n)
{
    // Match real_lu_factor_kernel's matrix, scratch, pivot and two stubs.
    int64_t stride = (static_cast<int64_t>(m) + 63) / 64 * 64;
    int64_t steps = (static_cast<int64_t>(std::min(m, n)) + 7) / 8 * 8;
    int64_t bytes = (stride * (static_cast<int64_t>(n) + 1) + steps + 16) * sizeof(float);
    return bytes <= 237568;
}
#endif
inline bool BlockedFactor(int m, int n)
{
    bool legacy = m >= 64 && n >= 64 && static_cast<int64_t>((m + 7) / 8 * 8) * n > 32768;
#if defined(REAL_LU_REGISTER_LAYOUT_950)
    bool tall =
        m >= 64 && n >= 8 && n < 64 && static_cast<int64_t>((m + 7) / 8 * 8) * n > 32768 && !FullLocalFits950(m, n);
    return legacy || tall;
#else
    return legacy;
#endif
}
inline bool ThinFactor(int m, int n) { return m <= 32 && n > 128 && n > m; }
inline int FactorWorkspace(int m, int n)
{
    int64_t size = ((static_cast<int64_t>(m) + 7) / 8 * 8) * n;
    if (BlockedFactor(m, n))
    {
        int64_t pitch = (static_cast<int64_t>(m) + 127) / 128 * 128 + 128;
        int64_t columns = (static_cast<int64_t>(n) + 127) / 128 * 128 + 128;
        size = 2 * pitch * columns + (static_cast<int64_t>(std::min(m, n)) + 7) / 8 * 8;
    }
    return size > INT_MAX ? -1 : static_cast<int>(size);
}
inline bool BlockedInverse(int n) { return n > 32; }
inline int InverseWorkspace(int n)
{
    int64_t size = static_cast<int64_t>(n) * n;
    if (BlockedInverse(n))
    {
        int64_t pitch = (static_cast<int64_t>(n) + 63) / 64 * 64;
        size = 5 * pitch * pitch + (pitch / 64 + 7) / 8 * 8;
    }
    return size > INT_MAX ? -1 : static_cast<int>(size);
}
inline int InverseBatchChunk(int n, int batch)
{
    if (n <= 0 || batch <= 0)
    {
        return 0;
    }
    constexpr size_t limit = 2UL * 1024 * 1024 * 1024;
    size_t perMatrix = static_cast<size_t>(InverseWorkspace(n)) * sizeof(float);
    if (perMatrix == 0)
    {
        return 0;
    }
    return std::min<size_t>(batch, std::max<size_t>(1, limit / perMatrix));
}
inline aclsolverStatus_t ContextWorkspace(aclsolverHandle_t handle, _real_lu_workspace *&cache)
{
    aclrtContext context = nullptr;
    if (aclrtGetCurrentContext(&context) != ACL_SUCCESS || context == nullptr)
    {
        return ACLSOLVER_STATUS_NOT_INITIALIZED;
    }
    auto *internal = static_cast<_aclsolver_handle *>(handle);
    cache = nullptr;
    for (auto &slot : internal->realLuWorkspaces)
    {
        if (slot.context == context)
        {
            cache = &slot;
        }
    }
    if (cache == nullptr)
    {
        try
        {
            internal->realLuWorkspaces.push_back({context});
        }
        catch (const std::bad_alloc &)
        {
            return ACLSOLVER_STATUS_ALLOC_FAILED;
        }
        cache = &internal->realLuWorkspaces.back();
    }
    return RealLuDrainValidation(*cache);
}
class Metadata
{
   public:
    Metadata(aclsolverHandle_t handle, size_t bytes) : stream(Stream(handle))
    {
        ready = ContextWorkspace(handle, cache);
        if (ready == ACLSOLVER_STATUS_SUCCESS)
        {
            status = aclrtMallocHost(&data, bytes);
        }
    }
    ~Metadata()
    {
        if (data != nullptr && aclrtFreeHost(data) != ACL_SUCCESS)
        {
            cache->validationHost = data;
        }
    }
    aclsolverStatus_t Sync(aclError copy, void *flags = nullptr)
    {
        // A failed synchronization does not prove that an asynchronous copy
        // or the finite-check kernel has stopped accessing these buffers.
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        {
            cache->validationStream = stream;
            cache->validationPending = true;
            cache->validationHost = data;
            cache->validationFlags = flags;
            data = nullptr;
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        if (flags != nullptr && aclrtFree(flags) != ACL_SUCCESS)
        {
            cache->validationFlags = flags;
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        return copy == ACL_SUCCESS ? ACLSOLVER_STATUS_SUCCESS : ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    Metadata(const Metadata &) = delete;
    Metadata &operator=(const Metadata &) = delete;
    void *data = nullptr;
    aclError status = ACL_SUCCESS;
    aclsolverStatus_t ready = ACLSOLVER_STATUS_SUCCESS;

   private:
    _real_lu_workspace *cache = nullptr;
    aclrtStream stream = nullptr;
};
inline aclsolverStatus_t ScratchBeforeValidation(aclsolverHandle_t handle, _real_lu_workspace *&cache)
{
    auto ready = ContextWorkspace(handle, cache);
    if (ready != ACLSOLVER_STATUS_SUCCESS)
    {
        return ready;
    }
    if (cache->untracked)
    {
        if (aclrtSynchronizeStream(cache->pendingStream) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache->untracked = false;
        cache->recorded = false;
        cache->pendingStream = nullptr;
    }
    if (cache->recorded && aclrtStreamWaitEvent(Stream(handle), cache->completion) != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}
// Call only after the validation prefix synchronized the caller stream. Its
// preceding event wait also completed any earlier use in a different stream.
inline aclsolverStatus_t ScratchAfterValidation(_real_lu_workspace &cache, size_t bytes)
{
    if (cache.completion == nullptr && aclrtCreateEvent(&cache.completion) != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    if (cache.data != nullptr && cache.capacity < bytes)
    {
        if (aclrtFree(cache.data) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.data = nullptr;
        cache.capacity = 0;
    }
    if (cache.data == nullptr)
    {
        if (aclrtMalloc(&cache.data, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_ALLOC_FAILED;
        }
        cache.capacity = bytes;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}
inline aclsolverStatus_t ScratchFinished(aclsolverHandle_t handle, _real_lu_workspace &cache)
{
    if (aclrtRecordEvent(cache.completion, Stream(handle)) != ACL_SUCCESS)
    {
        cache.pendingStream = Stream(handle);
        cache.untracked = aclrtSynchronizeStream(cache.pendingStream) != ACL_SUCCESS;
        if (!cache.untracked)
        {
            cache.recorded = false;
            cache.pendingStream = nullptr;
        }
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    cache.recorded = true;
    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace real_lu
