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

#include "real_lu_runtime.h"

namespace real_lu
{
inline int BatchedOutputArgument(int n, const void *c, int ldc, int *info, int batch, bool factor)
{
    if (n > 0 && batch > 0 && !Device(c))
    {
        return factor ? 4 : 5;
    }
    if (ldc < std::max(1, n))
    {
        return factor ? 5 : 6;
    }
    if (batch > 0 && !Device(info))
    {
        return factor ? 6 : 7;
    }
    if (batch < 0)
    {
        return factor ? 7 : 8;
    }
    return 0;
}
inline int BatchedArgument(int n, const void *a, int lda, const void *piv, const void *c, int ldc, int *info, int batch,
                           bool factor)
{
    if (n < 0 || n > (factor ? 32 : 4096))
    {
        return 1;
    }
    if (n > 0 && batch > 0 && !Device(a))
    {
        return 2;
    }
    if (lda < std::max(1, n))
    {
        return 3;
    }
    if (!factor && piv != nullptr && !Device(piv))
    {
        return 4;
    }
    return BatchedOutputArgument(n, c, ldc, info, batch, factor);
}
inline aclsolverStatus_t CheckSingle(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *piv, float *work,
                                     int workElements, int *info, bool inverse)
{
    Ranges ranges;
    if (!AlignedFits(ranges, info, sizeof(int), alignof(int)))
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    size_t aBytes = static_cast<size_t>(lda) * n * sizeof(float);
    size_t workBytes = static_cast<size_t>(workElements) * sizeof(float);
    size_t pivotBytes = static_cast<size_t>(std::min(m, n)) * sizeof(int);
    if (!AlignedFits(ranges, a, aBytes, alignof(float)))
    {
        return Invalid(handle, info, 1, inverse ? 2 : 3);
    }
    if (!AlignedFits(ranges, work, workBytes, alignof(float)))
    {
        return Invalid(handle, info, 1, 5);
    }
    if (piv != nullptr && !AlignedFits(ranges, piv, pivotBytes, alignof(int)))
    {
        return Invalid(handle, info, 1, inverse ? 4 : 6);
    }
    ProtectedView views[] = {{a, aBytes, inverse ? 2 : 3},
                             {work, workBytes, 5},
                             {piv, piv == nullptr ? 0 : pivotBytes, inverse ? 4 : 6},
                             {info, sizeof(int), 7}};
    return CheckViewOverlaps(handle, info, 1, views);
}
inline aclsolverStatus_t CheckFinite(aclsolverHandle_t handle, const void *a, int m, int n, int lda, int batch,
                                     bool arrays, int *info, int argument)
{
    if (m <= 0 || n <= 0 || lda <= 0 || batch <= 0)
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    int64_t jobs = arrays || lda > 8192
                       ? static_cast<int64_t>(batch) * n
                       : (static_cast<int64_t>(batch) * lda * n + (8192 / lda * lda) - 1) / (8192 / lda * lda);
    int blocks = std::min<int64_t>(real_lu_vector_cores(), jobs);
    Metadata staging(handle, blocks * sizeof(int));
    if (staging.ready != ACLSOLVER_STATUS_SUCCESS)
    {
        return staging.ready;
    }
    if (staging.status != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    void *flags = nullptr;
    if (aclrtMalloc(&flags, blocks * sizeof(int), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    real_lu_finite_do(a, flags, m, n, lda, batch, arrays, blocks, Stream(handle));
    aclError copy = aclrtMemcpyAsync(staging.data, blocks * sizeof(int), flags, blocks * sizeof(int),
                                     ACL_MEMCPY_DEVICE_TO_HOST, Stream(handle));
    auto sync = staging.Sync(copy, flags);
    if (sync != ACLSOLVER_STATUS_SUCCESS)
    {
        return sync;
    }
    const int *values = static_cast<const int *>(staging.data);
    for (int i = 0; i < blocks; ++i)
    {
        if (values[i] != 0)
        {
            return Invalid(handle, info, batch, argument);
        }
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline aclsolverStatus_t CheckPivots(aclsolverHandle_t handle, const int *piv, int n, int *info, int batch = 1,
                                     bool *identity = nullptr)
{
    if (identity != nullptr)
    {
        *identity = true;
    }
    if (piv == nullptr)
    {
        return ACLSOLVER_STATUS_SUCCESS;
    }
    size_t bytes = static_cast<size_t>(n) * batch * sizeof(int);
    Ranges ranges;
    if (!ranges.Fits(piv, bytes))
    {
        return Invalid(handle, info, batch, 4);
    }
    Metadata staging(handle, bytes);
    if (staging.ready != ACLSOLVER_STATUS_SUCCESS)
    {
        return staging.ready;
    }
    if (staging.status != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    aclError copy = aclrtMemcpyAsync(staging.data, bytes, piv, bytes, ACL_MEMCPY_DEVICE_TO_HOST, Stream(handle));
    auto sync = staging.Sync(copy);
    if (sync != ACLSOLVER_STATUS_SUCCESS)
    {
        return sync;
    }
    const int *values = static_cast<const int *>(staging.data);
    for (int b = 0; b < batch; ++b)
    {
        for (int k = 0; k < n; ++k)
        {
            if (values[static_cast<size_t>(b) * n + k] < k + 1 || values[static_cast<size_t>(b) * n + k] > n)
            {
                return Invalid(handle, info, batch, 4);
            }
            if (identity != nullptr && values[static_cast<size_t>(b) * n + k] != k + 1)
            {
                *identity = false;
            }
        }
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

struct Interval
{
    uintptr_t begin, end;
    int role;  // 0: read-only input, 1: output, other: argument number of metadata.
};
struct BatchedLayout
{
    const void *input = nullptr;
    void *output = nullptr;
    bool contiguous = true;
    bool identityPivots = true;
};
class ArrayValidation
{
   public:
    ArrayValidation(aclsolverHandle_t handle, int n, const void *a, int lda, const int *piv, const void *c, int ldc,
                    int *info, int batch, bool factor)
        : handle(handle),
          n(n),
          a(a),
          piv(piv),
          c(c),
          info(info),
          batch(batch),
          outArg(factor ? 4 : 5),
          pointerBytes(static_cast<size_t>(batch) * sizeof(uintptr_t)),
          pivotBytes(piv == nullptr ? 0 : static_cast<size_t>(batch) * n * sizeof(int)),
          aBytes(static_cast<size_t>(lda) * n * sizeof(float)),
          cBytes(static_cast<size_t>(ldc) * n * sizeof(float)),
          metadata{{a, pointerBytes, 2},
                   {c, pointerBytes, outArg},
                   {piv, pivotBytes, 4},
                   {info, static_cast<size_t>(batch) * sizeof(int), factor ? 6 : 7}}
    {
    }
    aclsolverStatus_t Run(BatchedLayout &layout)
    {
        auto status = CheckMetadata();
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
        Metadata staging(handle, 2 * pointerBytes + pivotBytes);
        if (staging.ready != ACLSOLVER_STATUS_SUCCESS)
        {
            return staging.ready;
        }
        if (staging.status != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_ALLOC_FAILED;
        }
        auto *bytes = static_cast<uint8_t *>(staging.data);
        status = CopyMetadata(staging, bytes);
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
        const auto *inputs = reinterpret_cast<const uintptr_t *>(bytes);
        const auto *outputs = reinterpret_cast<const uintptr_t *>(bytes + pointerBytes);
        const auto *pivots = reinterpret_cast<const int *>(bytes + 2 * pointerBytes);
        layout.input = reinterpret_cast<const void *>(inputs[0]);
        layout.output = reinterpret_cast<void *>(outputs[0]);
        try
        {
            std::vector<Interval> intervals;
            intervals.reserve(static_cast<size_t>(batch) * 2 + 4);
            status = CollectIntervals(inputs, outputs, pivots, layout, intervals);
            return status == ACLSOLVER_STATUS_SUCCESS ? CheckIntervals(intervals) : status;
        }
        catch (const std::bad_alloc &)
        {
            return ACLSOLVER_STATUS_ALLOC_FAILED;
        }
    }

   private:
    aclsolverStatus_t CheckMetadata()
    {
        if (!AlignedFits(ranges, a, pointerBytes, alignof(uintptr_t)))
        {
            return Invalid(handle, info, batch, 2);
        }
        if (!AlignedFits(ranges, c, pointerBytes, alignof(uintptr_t)))
        {
            return Invalid(handle, info, batch, outArg);
        }
        if (!AlignedFits(ranges, info, metadata[3].bytes, alignof(int)))
        {
            return ACLSOLVER_STATUS_INVALID_VALUE;  // Only verified info storage may be written.
        }
        if (piv != nullptr && !AlignedFits(ranges, piv, pivotBytes, alignof(int)))
        {
            return Invalid(handle, info, batch, 4);
        }
        return CheckViewOverlaps(handle, info, batch, metadata);
    }
    aclsolverStatus_t CopyMetadata(Metadata &staging, uint8_t *bytes)
    {
        aclError copy =
            aclrtMemcpyAsync(bytes, pointerBytes, a, pointerBytes, ACL_MEMCPY_DEVICE_TO_HOST, Stream(handle));
        if (copy == ACL_SUCCESS)
        {
            copy = aclrtMemcpyAsync(bytes + pointerBytes, pointerBytes, c, pointerBytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                    Stream(handle));
        }
        if (copy == ACL_SUCCESS && piv != nullptr)
        {
            copy = aclrtMemcpyAsync(bytes + 2 * pointerBytes, pivotBytes, piv, pivotBytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                    Stream(handle));
        }
        // Complete an already enqueued prefix even if a later copy could not start.
        return staging.Sync(copy);
    }
    aclsolverStatus_t CheckMatrixMetadata(const void *input, const void *output)
    {
        if (Overlap(input, aBytes, info, metadata[3].bytes) || Overlap(output, cBytes, info, metadata[3].bytes))
        {
            return ACLSOLVER_STATUS_INVALID_VALUE;
        }
        for (int i = 0; i < 3; ++i)
        {
            if (metadata[i].bytes && (Overlap(input, aBytes, metadata[i].data, metadata[i].bytes) ||
                                      Overlap(output, cBytes, metadata[i].data, metadata[i].bytes)))
            {
                return Invalid(handle, info, batch, metadata[i].argument);
            }
        }
        return ACLSOLVER_STATUS_SUCCESS;
    }
    aclsolverStatus_t CheckPivotRow(const int *pivots, int b, BatchedLayout &layout)
    {
        if (piv != nullptr)
        {
            for (int k = 0; k < n; ++k)
            {
                int value = pivots[static_cast<size_t>(b) * n + k];
                if (value < k + 1 || value > n)
                {
                    return Invalid(handle, info, batch, 4);
                }
                if (value != k + 1)
                {
                    layout.identityPivots = false;
                }
            }
        }
        return ACLSOLVER_STATUS_SUCCESS;
    }
    aclsolverStatus_t CheckMatrix(const uintptr_t *inputs, const uintptr_t *outputs, int b, BatchedLayout &layout)
    {
        if (inputs[b] != inputs[0] + static_cast<size_t>(b) * aBytes ||
            outputs[b] != outputs[0] + static_cast<size_t>(b) * cBytes)
        {
            layout.contiguous = false;
        }
        auto *input = reinterpret_cast<const void *>(inputs[b]);
        auto *output = reinterpret_cast<const void *>(outputs[b]);
        if (!AlignedFits(ranges, input, aBytes, alignof(float)))
        {
            return Invalid(handle, info, batch, 2);
        }
        if (!AlignedFits(ranges, output, cBytes, alignof(float)))
        {
            return Invalid(handle, info, batch, outArg);
        }
        return CheckMatrixMetadata(input, output);
    }
    aclsolverStatus_t CollectIntervals(const uintptr_t *inputs, const uintptr_t *outputs, const int *pivots,
                                       BatchedLayout &layout, std::vector<Interval> &intervals)
    {
        for (int b = 0; b < batch; ++b)
        {
            auto status = CheckMatrix(inputs, outputs, b, layout);
            if (status != ACLSOLVER_STATUS_SUCCESS)
            {
                return status;
            }
            intervals.push_back({inputs[b], inputs[b] + aBytes, 0});
            intervals.push_back({outputs[b], outputs[b] + cBytes, 1});
            status = CheckPivotRow(pivots, b, layout);
            if (status != ACLSOLVER_STATUS_SUCCESS)
            {
                return status;
            }
        }
        AddInterval(intervals, metadata[0]);
        AddInterval(intervals, metadata[1]);
        AddInterval(intervals, metadata[3]);
        if (piv != nullptr)
        {
            AddInterval(intervals, metadata[2]);
        }
        return ACLSOLVER_STATUS_SUCCESS;
    }
    static void AddInterval(std::vector<Interval> &intervals, const ProtectedView &view)
    {
        uintptr_t first = reinterpret_cast<uintptr_t>(view.data);
        intervals.push_back({first, first + view.bytes, view.argument});
    }
    aclsolverStatus_t CheckIntervals(std::vector<Interval> &intervals)
    {
        std::sort(intervals.begin(), intervals.end(), [](const Interval &x, const Interval &y)
                  { return x.begin < y.begin || (x.begin == y.begin && x.role < y.role); });
        uintptr_t readEnd = 0, writeEnd = 0;
        int previousRole = outArg;
        for (const auto &span : intervals)
        {
            if (span.role == 0)
            {
                if (span.begin < writeEnd)
                {
                    return Invalid(handle, info, batch, previousRole);
                }
                readEnd = std::max(readEnd, span.end);
            }
            else
            {
                if (span.begin < readEnd || span.begin < writeEnd)
                {
                    return Invalid(handle, info, batch, span.role == 1 ? outArg : span.role);
                }
                writeEnd = span.end;
                previousRole = span.role == 1 ? outArg : span.role;
            }
        }
        return ACLSOLVER_STATUS_SUCCESS;
    }
    aclsolverHandle_t handle;
    int n;
    const void *a;
    const int *piv;
    const void *c;
    int *info;
    int batch;
    int outArg;
    size_t pointerBytes, pivotBytes, aBytes, cBytes;
    ProtectedView metadata[4];
    Ranges ranges;
};
inline aclsolverStatus_t CheckArrays(aclsolverHandle_t handle, int n, const void *a, int lda, const int *piv,
                                     const void *c, int ldc, int *info, int batch, bool factor, BatchedLayout &layout)
{
    ArrayValidation validation(handle, n, a, lda, piv, c, ldc, info, batch, factor);
    return validation.Run(layout);
}
}  // namespace real_lu
