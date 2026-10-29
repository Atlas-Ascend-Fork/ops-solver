/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <algorithm>

#include "../utils/kernel/real_lu.hpp"
#include "../utils/kernel/real_lu_small.hpp"
using namespace RealLuKernel;
extern "C" int real_lu_vector_cores();

__global__ __aicore__ void real_lu_small_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR pivot, GM_ADDR info, int n,
                                                int batch, bool factor, bool identityPivots, int groupLanes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SmallInverse op(pipe, n, groupLanes, factor);
    int lanes = op.Lanes();
    for (int64_t first = static_cast<int64_t>(GetBlockIdx()) * lanes; first < batch;
         first += static_cast<int64_t>(GetBlockNum()) * lanes)
    {
        int count = batch - first < lanes ? batch - first : lanes;
        op.Run(input, output, pivot, info, first, count, factor, identityPivots);
    }
}
namespace
{
struct SmallGroupCapacity
{
    int lanes = 8;
    int maximum = 8;
    int aligned = 8;
};
#ifdef REAL_LU_REGISTER_LAYOUT_950
bool SmallLaneFits(int n, int candidate, bool factor)
{
    int packedStride = n == 2 ? n * n : (((n * n + 7) / 8) | 1) * 8;
    // Match the contiguous inverse-from-LU n=32 storage in SmallInverse.
    bool compact = (factor && n >= 17 && n <= 32) || (!factor && n == 32);
    int rowStride = n * candidate;
    size_t masks = 2 * ((rowStride + 255) / 256) * 32;
    size_t floats = compact ? static_cast<size_t>(packedStride) * 8 + static_cast<size_t>(n * n + 4 * n + 4) * candidate
                            : static_cast<size_t>(packedStride + 2 * n * n + 4 * n + 4) * candidate;
    size_t bytes = floats * sizeof(float);
    // Conservatively reserve placeholders plus the compiled VF stack.
    return bytes + masks + 224 <= 253952 - REAL_LU_VF_STACK_BYTES;
}
int ShortFactorLanes(int n, int batch, int cores, int lanes, int maximum)
{
    if (cores == 0 || cores < 0)
    {
        return lanes;
    }
    int selectedLanes = lanes;
    int64_t bestScore = -1;
    for (int candidate = 8; candidate <= maximum; candidate *= 2)
    {
        if (!SmallLaneFits(n, candidate, true))
        {
            continue;
        }
        const int64_t groups = (static_cast<int64_t>(batch) + candidate - 1) / candidate;
        const int64_t worstGroups = (groups + static_cast<int64_t>(cores) - 1) / cores;
        const int64_t wavesPerGroup = (static_cast<int64_t>(candidate) + 63) / 64;
        const int64_t score = worstGroups * (4 * wavesPerGroup + 1);
        if (bestScore < 0 || score < bestScore || (score == bestScore && candidate < selectedLanes))
        {
            bestScore = score;
            selectedLanes = candidate;
        }
    }
    return selectedLanes;
}
int CompactFactorLanes(int n, int batch, int cores, int lanes, int aligned)
{
    if (cores == 0 || cores < 0)
    {
        return lanes;
    }
    int selectedLanes = lanes;
    int64_t bestCriticalWaves = -1, bestWorstGroups = -1;
    // Compact storage permits aligned groups beyond the power-of-two bound.
    for (int64_t candidate = 8; candidate <= static_cast<int64_t>(aligned); candidate += 8)
    {
        if (!SmallLaneFits(n, static_cast<int>(candidate), true))
        {
            continue;
        }
        const int64_t groups = (static_cast<int64_t>(batch) + candidate - 1) / candidate;
        const int64_t worstGroups = (groups + static_cast<int64_t>(cores) - 1) / cores;
        const int64_t wavesPerGroup = (candidate + 63) / 64;
        const int64_t criticalWaves = worstGroups * wavesPerGroup;
        if (bestCriticalWaves < 0 || criticalWaves < bestCriticalWaves ||
            (criticalWaves == bestCriticalWaves && worstGroups < bestWorstGroups) ||
            (criticalWaves == bestCriticalWaves && worstGroups == bestWorstGroups && candidate < selectedLanes))
        {
            bestCriticalWaves = criticalWaves;
            bestWorstGroups = worstGroups;
            selectedLanes = static_cast<int>(candidate);
        }
    }
    return selectedLanes;
}
#endif
SmallGroupCapacity SmallCapacity(int n, int batch, bool factor)
{
    SmallGroupCapacity capacity;
    if (n == 0)
    {
        capacity.lanes = 0;
        return capacity;
    }
    if (n < 0 || n > 32)
    {
        capacity.lanes = 0;
        return capacity;
    }
#ifdef REAL_LU_REGISTER_LAYOUT_950
    capacity.aligned = std::min<int64_t>(2048, (static_cast<int64_t>(batch) + 7) / 8 * 8);
    while (capacity.lanes * 2 <= capacity.aligned && SmallLaneFits(n, capacity.lanes * 2, factor))
    {
        capacity.lanes *= 2;
    }
    capacity.maximum = capacity.lanes;
#else
    (void)factor;
    int lanes = std::min(1024, 8192 / (n * n) / 8 * 8);
    lanes = std::min<int64_t>(lanes, (static_cast<int64_t>(batch) + 7) / 8 * 8);
    while (capacity.lanes * 2 <= lanes)
    {
        capacity.lanes *= 2;
    }
#endif
    return capacity;
}
int SpreadSmallLanes(int batch, int cores, int lanes)
{
    if (cores == 0)
    {
        return lanes;
    }
    if (cores < 0)
    {
        return lanes;
    }
    // Spread small batches over AIVs; retain full groups for large batches.
    if (static_cast<int64_t>(batch) <= static_cast<int64_t>(cores) * lanes)
    {
        int target = (static_cast<int64_t>(batch) + cores - 1) / cores;
        int parallelLanes = 8;
        while (parallelLanes < target && parallelLanes * 2 <= lanes)
        {
            parallelLanes *= 2;
        }
        return parallelLanes;
    }
    return lanes;
}
}  // namespace

extern "C" void real_lu_small_do(GM_ADDR a, GM_ADDR c, GM_ADDR piv, GM_ADDR info, int n, int batch, bool factor,
                                 bool identityPivots, aclrtStream stream)
{
    int cores = real_lu_vector_cores();
    if (n <= 0 || n > 32 || batch <= 0 || cores <= 0)
    {
        return;
    }
    auto capacity = SmallCapacity(n, batch, factor);
    int lanes = SpreadSmallLanes(batch, cores, capacity.lanes);
#ifdef REAL_LU_REGISTER_LAYOUT_950
    if (factor && n >= 3 && n <= 8)
    {
        lanes = ShortFactorLanes(n, batch, cores, lanes, capacity.maximum);
    }
    if (factor && n >= 17 && n <= 32)
    {
        lanes = CompactFactorLanes(n, batch, cores, lanes, capacity.aligned);
    }
#endif
    if (lanes == 0)
    {
        return;
    }
    if (lanes < 0)
    {
        return;
    }
    int blocks = std::min<int64_t>(cores, (static_cast<int64_t>(batch) + lanes - 1) / lanes);
    real_lu_small_kernel<<<blocks, nullptr, stream>>>(a, c, piv, info, n, batch, factor, identityPivots, lanes);
}

__global__ __aicore__ void real_lu_generic_small_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR pivot, GM_ADDR info,
                                                        int n, int lda, int ldc, int batch, bool arrays, bool factor)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    TPipe pipe;
    // Generic pointer arrays and leading dimensions use aligned UB columns;
    // the same LAPACK-order computation serves every independent matrix.
    SmallInverse op(pipe, n, 8, factor, true);
    for (int64_t first = static_cast<int64_t>(GetBlockIdx()) * 8; first < batch;
         first += static_cast<int64_t>(GetBlockNum()) * 8)
    {
        int count = batch - first < 8 ? batch - first : 8;
        op.Run(input, output, pivot, info, first, count, factor, !factor && pivot == nullptr, arrays, lda, ldc);
    }
#endif
}

namespace
{
__aicore__ inline int InversePointerArgument(uint64_t input, uint64_t output, uint64_t inputBytes, uint64_t outputBytes,
                                             bool factor)
{
    if (input == 0)
    {
        return -2;
    }
    if (output == 0)
    {
        return factor ? -4 : -5;
    }
    if (input <= output ? output - input < inputBytes : input - output < outputBytes)
    {
        return factor ? -4 : -5;
    }
    return 0;
}
}  // namespace

__aicore__ inline int InverseTileWidth(int n)
{
    if (n == 0)
    {
        return 0;
    }
    if (n < 0)
    {
        return 0;
    }
    int width = (32768 / n / 8) * 8;
    if (width > 128)
    {
        width = 128;
    }
    return width > Align(n) ? Align(n) : width;
}

__global__ __aicore__ void real_lu_inverse_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR pivot, GM_ADDR info, int n,
                                                  int lda, int ldc, int batch, bool arrays, bool factor)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (n <= 0 || batch <= 0)
    {
        return;
    }
    TPipe pipe;
    Inverse op(pipe, n, factor);
    GlobalTensor<uint64_t> inputs, outputs;
    inputs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input));
    outputs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output));
    GlobalTensor<int> infos;
    infos.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 32);
    auto scalar = buf.Get<int>();
    int tiles = op.Tiles();
    if (tiles == 0)
    {
        return;
    }
    if (tiles < 0)
    {
        return;
    }
    int tileWidth = InverseTileWidth(n);
    for (int64_t job = GetBlockIdx(); job < static_cast<int64_t>(batch) * tiles; job += GetBlockNum())
    {
        int index = job / tiles, tile = job % tiles;
        uint64_t in = arrays ? inputs.GetValue(index) : reinterpret_cast<uint64_t>(input);
        uint64_t out = arrays ? outputs.GetValue(index) : reinterpret_cast<uint64_t>(output);
        uint64_t inBytes = static_cast<uint64_t>(lda) * n * sizeof(float);
        uint64_t outBytes = static_cast<uint64_t>(ldc) * n * sizeof(float);
        int invalid = InversePointerArgument(in, out, inBytes, outBytes, factor);
        if (invalid)
        {
            if (tile == 0)
            {
                StoreInt(infos[index], scalar, invalid);
            }
            continue;
        }
        GM_ADDR p = pivot == nullptr ? nullptr : pivot + static_cast<int64_t>(index) * n * sizeof(int);
        op.Run(reinterpret_cast<GM_ADDR>(in), reinterpret_cast<GM_ADDR>(out), p,
               info + static_cast<int64_t>(index) * sizeof(int), lda, ldc, tile * tileWidth, factor,
               !arrays && !factor && n <= 32);
    }
}
__global__ __aicore__ void real_lu_copy_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR info, int n, int lda)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int> status;
    status.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    if (status.GetValue(0) != 0)
    {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, Align(n) * sizeof(float));
    auto local = buf.Get<float>();
    GlobalTensor<float> a, b;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
    for (int j = GetBlockIdx(); j < n; j += GetBlockNum())
    {
        Load(local, a[static_cast<int64_t>(j) * n], n);
        Store(b[static_cast<int64_t>(j) * lda], local, n);
    }
}
extern "C" void real_lu_inverse_do(GM_ADDR a, GM_ADDR c, GM_ADDR piv, GM_ADDR info, int n, int lda, int ldc, int batch,
                                   bool arrays, bool factor, aclrtStream stream)
{
    if (n == 0)
    {
        return;
    }
    if (n < 0 || batch <= 0)
    {
        return;
    }
#ifdef REAL_LU_REGISTER_LAYOUT_950
    if (n == 2 || n == 4 || n == 8 || (factor && n >= 3 && n <= 8))
    {
        int blocks = std::min<int64_t>(real_lu_vector_cores(), (static_cast<int64_t>(batch) + 7) / 8);
        real_lu_generic_small_kernel<<<blocks, nullptr, stream>>>(a, c, piv, info, n, lda, ldc, batch, arrays, factor);
        return;
    }
#endif
    int width = std::min(128, (32768 / n / 8) * 8);
    width = std::min(width, (n + 7) / 8 * 8);
    if (width == 0)
    {
        return;
    }
    if (width < 0)
    {
        return;
    }
    int64_t jobs = static_cast<int64_t>(batch) * ((n + width - 1) / width);
    int blocks = std::min<int64_t>(real_lu_vector_cores(), jobs);
    real_lu_inverse_kernel<<<blocks, nullptr, stream>>>(a, c, piv, info, n, lda, ldc, batch, arrays, factor);
}
extern "C" void real_lu_copy_do(GM_ADDR src, GM_ADDR dst, GM_ADDR info, int n, int lda, aclrtStream stream)
{
    real_lu_copy_kernel<<<std::min(real_lu_vector_cores(), n), nullptr, stream>>>(src, dst, info, n, lda);
}
