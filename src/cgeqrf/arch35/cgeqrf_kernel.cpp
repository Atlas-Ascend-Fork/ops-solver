/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../cunmqr/arch35/cunmqr_apply_helpers.h"
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "cgeqrf_device_helpers.h"
#include "cgeqrf_internal.h"

namespace
{
__simt_vf__ __aicore__ LAUNCH_BOUND(32) void ReflectorVf(__gm__ float* a, __gm__ float* tau, uint32_t m, uint64_t lda,
                                                         uint32_t k)
{
    ReflectorBody(a, tau, m, lda, k);
}

__global__ __aicore__ void ReflectorKernel(__gm__ float* a, __gm__ float* tau, uint32_t m, uint64_t lda, uint32_t k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<ReflectorVf>(dim3{32, 1, 1}, a, tau, m, lda, k);
}
}  // namespace

void LaunchReflectorKernel(uint32_t blocks, void* stream, float* a, float* tau, uint32_t m, uint64_t lda, uint32_t k)
{
    ReflectorKernel<<<blocks, nullptr, stream>>>(a, tau, m, lda, k);
}

namespace
{
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void CachedApplyVf(__ubuf__ float* vectors, __ubuf__ float* target,
                                                            __ubuf__ float* coefficients, uint32_t rows,
                                                            uint32_t stride, uint32_t width, uint32_t panel,
                                                            bool adjoint = true)
{
    for (uint32_t j = 0; j < panel; ++j)
    {
        uint32_t k = adjoint ? j : panel - 1 - j;
        if (rows - k >= (width == 1 ? 512U : 1536U))
        {
            for (uint32_t column = 0; column < width; ++column)
            {
                CooperativeApply(vectors, coefficients, target, coefficients + 32, rows, stride, k, column, adjoint);
            }
        }
        else
        {
            if (threadIdx.x / 32 < width)
            {
                ApplyBody(vectors, coefficients, target, rows, stride, stride, k, threadIdx.x / 32, false, adjoint);
            }
            asc_syncthreads();
        }
    }
}

__global__ __aicore__ void CachedApplyKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ float* c,
                                             uint64_t lda, uint64_t ldc, uint32_t rows, uint32_t cols, uint32_t first,
                                             uint32_t panel, uint32_t columnStart)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    uint32_t stride = (rows + 3) / 4 * 4, tileColumns = 1;
    if (GetBlockIdx() * tileColumns >= cols)
    {
        return;
    }
    TPipe pipe;
    __ubuf__ float cache[32768 + 16384 + 128];
    LocalTensor<float> storage(TPosition::VECCALC, static_cast<uint32_t>(reinterpret_cast<uint64_t>(cache)),
                               32768 + 16384 + 128);
    auto vectors = storage, target = storage[32768], localTau = storage[32768 + 16384];
    GlobalTensor<float> aGm, tauGm, cGm;
    aGm.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    tauGm.SetGlobalBuffer(const_cast<__gm__ float*>(tau));
    cGm.SetGlobalBuffer(c);
    ComplexQrTransfer::Read(vectors, aGm, 2 * (first + uint64_t(first) * lda), panel, 2 * rows, 2 * lda, 2 * stride);
    ComplexQrTransfer::Read(localTau, tauGm, 2 * first, 1, 2 * panel, 2 * panel, 32);
    for (uint32_t start = GetBlockIdx() * tileColumns; start < cols; start += GetBlockNum() * tileColumns)
    {
        uint32_t width = cols - start < tileColumns ? cols - start : tileColumns;
        uint64_t offset = 2 * (first + uint64_t(columnStart + start) * ldc);
        ComplexQrTransfer::Sync<HardEvent::V_MTE2>();
        ComplexQrTransfer::Read(target, cGm, offset, width, 2 * rows, 2 * ldc, 2 * stride);
        ComplexQrTransfer::Sync<HardEvent::MTE2_V>();
        asc_vf_call<CachedApplyVf>(dim3{512, 1, 1}, cache, cache + 32768, cache + 32768 + 16384, rows, stride, width,
                                   panel, true);
        ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
        ComplexQrTransfer::Write(cGm, offset, target, width, 2 * rows, 2 * ldc, 2 * stride);
        ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace

void LaunchCachedApplyKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c, uint64_t lda,
                             uint64_t ldc, uint32_t rows, uint32_t cols, uint32_t first, uint32_t panel,
                             uint32_t columnStart)
{
    const uint64_t stride = (uint64_t(rows) + 3) / 4 * 4;
    const uint64_t bufferBytes = (2 * stride * (uint64_t(panel) + 1) + 2 * panel + 64) * sizeof(float);
    if (bufferBytes <= 196864)
    {
        LaunchCachedApplySimdKernel(blocks, stream, a, tau, c, lda, ldc, rows, cols, first, panel, columnStart);
        return;
    }
    CachedApplyKernel<<<blocks, nullptr, stream>>>(a, tau, c, lda, ldc, rows, cols, first, panel, columnStart);
}
