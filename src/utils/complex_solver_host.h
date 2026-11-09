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
#include <cstring>
#include <map>
#include <vector>

#include "cann_ops_solver.h"
#include "complex_solver_launch.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace ComplexSolver
{
constexpr int MAX_ORDER = 4096;
constexpr int MAX_BATCH = 1000000;
// cmatinv_batched 专用上限：n<=32 的小矩阵直接求逆，单批内存占用小，上限放宽到 10000000（与
// cmatinv_batched.md 文档口径一致）；cgetri_batched 维持 MAX_BATCH（文档口径 1000000）。issue #329。
constexpr int MAX_BATCH_MATINV = 10000000;

inline aclsolverStatus_t Context(aclsolverHandle_t handle, aclrtStream& stream, uint32_t& blocks, const char* op)
{
    if (handle == nullptr)
    {
        OP_LOGE(op, "handle is null");
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        OP_LOGE(op, "GetStream failed: %d", status);
        return status;
    }
    const char* soc = aclrtGetSocName();
    if (soc == nullptr || std::strncmp(soc, "Ascend950", 9) != 0)
    {
        OP_LOGE(op, "requires Ascend950, actual SoC=%s", soc == nullptr ? "unknown" : soc);
        return ACLSOLVER_STATUS_ARCH_MISMATCH;
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr || platform->GetCoreNumAiv() == 0)
    {
        OP_LOGE(op, "AIV core count query failed");
        return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    blocks = platform->GetCoreNumAiv();
    return ACLSOLVER_STATUS_SUCCESS;
}

inline aclsolverStatus_t DevicePointer(const void* pointer, const char* op, int argument)
{
    if (pointer == nullptr)
    {
        OP_LOGE(op, "argument %d is null", argument);
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    aclrtPtrAttributes attributes{};
    aclError error = aclrtPointerGetAttributes(pointer, &attributes);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "argument %d pointer query failed: %d", argument, error);
        return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    int device = -1;
    error = aclrtGetDevice(&device);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "GetDevice failed: %d", error);
        return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE || attributes.location.id != device)
    {
        OP_LOGE(op, "argument %d must belong to current Device %d", argument, device);
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline aclsolverStatus_t DeviceSpan(const void* pointer, uint64_t bytes, size_t alignment, const char* op, int argument)
{
    auto status = DevicePointer(pointer, op, argument);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    void* base = nullptr;
    size_t size = 0;
    aclError error = aclrtMemGetAddressRange(const_cast<void*>(pointer), &base, &size);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "argument %d allocation range query failed: %d", argument, error);
        return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    if (alignment == 0 || address % alignment != 0 || address < begin || address - begin > size ||
        bytes > size - (address - begin) || bytes > std::numeric_limits<uintptr_t>::max() - address)
    {
        OP_LOGE(op, "argument %d has invalid alignment or insufficient Device capacity", argument);
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline int ErrorInfoCount(int batch, int limit) { return batch < 0 || batch > limit ? 1 : batch; }

inline aclsolverStatus_t LaunchStatus(const char* op)
{
    aclError error = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "kernel launch failed: %d", error);
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline aclsolverStatus_t CompleteGetri(const int* pivots, const int* info, int count, aclrtStream stream,
                                       const char* op)
{
    auto status = LaunchStatus(op);
    if (status != ACLSOLVER_STATUS_SUCCESS || pivots == nullptr) return status;
    // Device pivot errors must also be reported by the synchronous API status.
    // Copy only per-matrix info and preserve independent batch results.
    int* host = nullptr;
    size_t bytes = static_cast<size_t>(count) * sizeof(int);
    aclError error = aclrtMallocHost(reinterpret_cast<void**>(&host), bytes);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "info metadata allocation failed: %d", error);
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    error = aclrtMemcpyAsync(host, bytes, info, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclError syncError = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS || syncError != ACL_SUCCESS)
    {
        OP_LOGE(op, "info metadata copy/sync failed: %d/%d", error, syncError);
        status = ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    else if (std::find(host, host + count, -4) != host + count)
    {
        OP_LOGE(op, "invalid argument 4: Device pivot value outside its row range");
        status = ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (aclrtFreeHost(host) != ACL_SUCCESS)
    {
        // 释放失败只记日志、不覆盖已确定的业务错误码（issue #310）：
        // 调用方应拿到 pivot/拷贝阶段的真实失败原因，而非笼统的 INTERNAL_ERROR。
        OP_LOGE(op, "info metadata release failed");
    }
    return status;
}

inline aclsolverStatus_t Reject(const char* op, int argument, int* info, int count, aclrtStream stream, uint32_t blocks)
{
    OP_LOGE(op, "invalid argument %d", argument);
    if (info != nullptr && count > 0)
    {
        auto status = DeviceSpan(info, static_cast<uint64_t>(count) * sizeof(int), alignof(int), op, argument);
        if (status == ACLSOLVER_STATUS_INTERNAL_ERROR) return status;
        if (status != ACLSOLVER_STATUS_SUCCESS) return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (info != nullptr && count > 0)
    {
        ComplexSolverFillInfo(info, count, -argument, std::min<uint32_t>(blocks, count), stream);
        auto status = LaunchStatus(op);
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    }
    return ACLSOLVER_STATUS_INVALID_VALUE;
}

inline aclsolverStatus_t PointerArgument(const void* pointer, bool optional, const char* op, int argument, int* info,
                                         int count, aclrtStream stream, uint32_t blocks, uint64_t bytes = 1,
                                         size_t alignment = 1)
{
    if (pointer == nullptr && optional) return ACLSOLVER_STATUS_SUCCESS;
    auto status = DeviceSpan(pointer, bytes, alignment, op, argument);
    if (status == ACLSOLVER_STATUS_INVALID_VALUE) return Reject(op, argument, info, count, stream, blocks);
    return status;
}

inline bool Overlap(const void* a, uint64_t aBytes, const void* b, uint64_t bBytes)
{
    auto left = reinterpret_cast<uintptr_t>(a);
    auto right = reinterpret_cast<uintptr_t>(b);
    return left <= right ? right - left < aBytes : left - right < bBytes;
}

// 附加校验视图：指针/字节足迹/归参序号（issue #371 跨缓冲重叠校验用，footprint 为 0
// 表示该视图缺省，跳过比对）。
struct ExtraView
{
    const void* data;
    uint64_t bytes;
    int argument;
};

// 跨缓冲重叠校验（issue #371，对齐实数 batched 家族 CheckIntervals 的全区间口径）：
// 把 pivots/info 等附加视图与 A/C 指针数组指向的全部矩阵区间（按 lda*n / ldc*n 逻辑
// 足迹，与 CheckMatrixMetadata 一致）两两比对，附加视图之间亦两两比对；重叠时 bad
// 记为对应视图的归参序号。调用前附加视图必须已通过 DeviceSpan/PointerArgument 校验。
inline aclsolverStatus_t CheckExtraViewOverlaps(const uint64_t* host, ComplexSolverShape shape, const ExtraView* extra,
                                                size_t count, int& bad)
{
    const uint64_t matrixBytes[] = {static_cast<uint64_t>(shape.lda) * shape.n * sizeof(aclFloatComplex),
                                    static_cast<uint64_t>(shape.ldc) * shape.n * sizeof(aclFloatComplex)};
    for (size_t e = 0; e < count; ++e)
    {
        if (extra[e].bytes == 0) continue;
        for (int output = 0; output < 2; ++output)
        {
            const uint64_t* array = host + static_cast<size_t>(output) * shape.batch;
            for (int batch = 0; batch < shape.batch; ++batch)
            {
                const void* matrix = reinterpret_cast<const void*>(array[batch]);
                if (Overlap(matrix, matrixBytes[output], extra[e].data, extra[e].bytes))
                {
                    bad = extra[e].argument;
                    return ACLSOLVER_STATUS_INVALID_VALUE;
                }
            }
        }
        for (size_t f = e + 1; f < count; ++f)
        {
            if (extra[f].bytes != 0 && Overlap(extra[e].data, extra[e].bytes, extra[f].data, extra[f].bytes))
            {
                bad = extra[f].argument;
                return ACLSOLVER_STATUS_INVALID_VALUE;
            }
        }
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

struct MatrixRange
{
    uintptr_t begin;
    uintptr_t end;
    bool output;
};

inline int CheckRanges(std::vector<MatrixRange>& ranges, int inputArgument, int outputArgument)
{
    std::sort(ranges.begin(), ranges.end(),
              [](const MatrixRange& a, const MatrixRange& b) { return a.begin < b.begin; });
    uintptr_t inputEnd = 0;
    uintptr_t outputEnd = 0;
    for (const auto& range : ranges)
    {
        if (range.end < range.begin) return range.output ? outputArgument : inputArgument;
        if (range.begin < outputEnd || (range.output && range.begin < inputEnd)) return outputArgument;
        if (range.output)
            outputEnd = std::max(outputEnd, range.end);
        else
            inputEnd = std::max(inputEnd, range.end);
    }
    return 0;
}

inline aclsolverStatus_t ReadPointerArrays(const void* a, const void* c, int count, uint64_t** host, aclrtStream stream,
                                           const char* op)
{
    const size_t bytes = static_cast<size_t>(count) * sizeof(uint64_t);
    aclError error = aclrtMallocHost(reinterpret_cast<void**>(host), 2 * bytes);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "pointer metadata allocation failed: %d", error);
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    error = aclrtMemcpyAsync(*host, bytes, a, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (error == ACL_SUCCESS)
        error = aclrtMemcpyAsync(*host + count, bytes, c, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    // Required to return INVALID_VALUE for inner null pointers and cross-batch overlap.
    // Matrix data stays on Device. Computing kernels remain asynchronous on the caller's stream.
    aclError syncError = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS || syncError != ACL_SUCCESS)
    {
        OP_LOGE(op, "pointer metadata copy/sync failed: %d/%d", error, syncError);
        return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline aclsolverStatus_t MatrixAllocation(uintptr_t address, uint64_t bytes, std::map<uintptr_t, size_t>& cache,
                                          const char* op, int argument)
{
    if (address % alignof(aclFloatComplex) != 0 || bytes > std::numeric_limits<uintptr_t>::max() - address)
    {
        OP_LOGE(op, "argument %d matrix address is misaligned or overflows", argument);
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    auto range = cache.upper_bound(address);
    if (range != cache.begin())
    {
        --range;
        uint64_t offset = address - range->first;
        if (offset < range->second && bytes <= range->second - offset) return ACLSOLVER_STATUS_SUCCESS;
    }
    auto status = DevicePointer(reinterpret_cast<void*>(address), op, argument);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    void* base = nullptr;
    size_t size = 0;
    aclError error = aclrtMemGetAddressRange(reinterpret_cast<void*>(address), &base, &size);
    if (error != ACL_SUCCESS)
    {
        OP_LOGE(op, "argument %d allocation range query failed: %d", argument, error);
        return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    if (address < begin || address - begin > size || bytes > size - (address - begin))
    {
        OP_LOGE(op, "argument %d matrix exceeds its Device allocation", argument);
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    cache.emplace(begin, size);
    return ACLSOLVER_STATUS_SUCCESS;
}

inline int CheckSortedMatrices(const uint64_t* host, ComplexSolverShape shape, uint64_t inputBytes,
                               uint64_t outputBytes, int outputArgument)
{
    const uint64_t* output = host + shape.batch;
    for (int i = 1; i < shape.batch; ++i)
        if (output[i] - output[i - 1] < outputBytes) return outputArgument;
    int i = 0;
    int j = 0;
    while (i < shape.batch && j < shape.batch)
    {
        if (host[i] <= output[j])
        {
            if (output[j] - host[i] < inputBytes) return outputArgument;
            ++i;
        }
        else
        {
            if (host[i] - output[j] < outputBytes) return outputArgument;
            ++j;
        }
    }
    return 0;
}

inline aclsolverStatus_t CheckMatrixMetadata(const uint64_t* host, ComplexSolverShape shape, int inputArgument,
                                             int outputArgument, int& bad, const char* op)
{
    std::map<uintptr_t, size_t> allocations;
    uint64_t bytes[] = {static_cast<uint64_t>(shape.lda) * shape.n * sizeof(aclFloatComplex),
                        static_cast<uint64_t>(shape.ldc) * shape.n * sizeof(aclFloatComplex)};
    bool sorted = true;
    for (int output = 0; output < 2; ++output)
    {
        const uint64_t* array = host + static_cast<size_t>(output) * shape.batch;
        for (int batch = 0; batch < shape.batch; ++batch)
        {
            bad = output ? outputArgument : inputArgument;
            auto status = MatrixAllocation(array[batch], bytes[output], allocations, op, bad);
            if (status != ACLSOLVER_STATUS_SUCCESS) return status;
            if (batch > 0 && array[batch] < array[batch - 1]) sorted = false;
        }
    }
    if (sorted)
        bad = CheckSortedMatrices(host, shape, bytes[0], bytes[1], outputArgument);
    else
    {
        std::vector<MatrixRange> ranges;
        ranges.reserve(static_cast<size_t>(shape.batch) * 2);
        for (int output = 0; output < 2; ++output)
            for (int batch = 0; batch < shape.batch; ++batch)
            {
                uintptr_t address = host[batch + static_cast<size_t>(output) * shape.batch];
                ranges.push_back({address, address + bytes[output], output != 0});
            }
        bad = CheckRanges(ranges, inputArgument, outputArgument);
    }
    return bad == 0 ? ACLSOLVER_STATUS_SUCCESS : ACLSOLVER_STATUS_INVALID_VALUE;
}

inline aclsolverStatus_t ValidateMatrices(const void* a, const void* c, ComplexSolverShape shape, int inputArgument,
                                          int outputArgument, int& bad, aclrtStream stream, const char* op,
                                          const ExtraView* extraViews = nullptr, size_t extraViewCount = 0)
{
    uint64_t* host = nullptr;
    auto status = ReadPointerArrays(a, c, shape.batch, &host, stream, op);
    try
    {
        if (status == ACLSOLVER_STATUS_SUCCESS)
            status = CheckMatrixMetadata(host, shape, inputArgument, outputArgument, bad, op);
        if (status == ACLSOLVER_STATUS_SUCCESS && extraViews != nullptr)
            status = CheckExtraViewOverlaps(host, shape, extraViews, extraViewCount, bad);
    }
    catch (const std::bad_alloc&)
    {
        OP_LOGE(op, "matrix metadata allocation failed");
        status = ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    if (host != nullptr && aclrtFreeHost(host) != ACL_SUCCESS)
    {
        // 同 CompleteGetri：释放失败只记日志，不覆盖已确定的校验错误码（issue #310）。
        OP_LOGE(op, "pointer metadata release failed");
    }
    return status;
}
}  // namespace ComplexSolver
