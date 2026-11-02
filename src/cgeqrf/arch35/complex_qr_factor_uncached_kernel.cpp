/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define COMPLEX_QR_COOPERATIVE_THREADS 128
#include "../../cunmqr/arch35/cunmqr_apply_helpers.h"
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "cgeqrf_device_helpers.h"
#include "cgeqrf_internal.h"

namespace
{
__simt_vf__ __aicore__ LAUNCH_BOUND(kCooperativeThreads) void FactorPanelVf(__gm__ float* a, __gm__ float* tau,
                                                                            uint32_t m, uint32_t n, uint64_t lda,
                                                                            uint32_t first, uint32_t last)
{
    for (uint32_t k = first; k < last; ++k)
    {
        if (threadIdx.x < 32)
        {
            ReflectorBody(a, tau, m, lda, k);
        }
        asc_syncthreads();
        for (uint32_t col = k + 1 + threadIdx.x / 32; col < n; col += kCooperativeWarps)
        {
            ApplyBody(a, tau, a, m, lda, lda, k, col, false, true);
        }
        asc_syncthreads();
    }
}

__global__ __aicore__ void FactorPanelUncachedKernel(__gm__ float* a, __gm__ float* tau, uint32_t m, uint32_t n,
                                                     uint64_t lda, uint32_t first, uint32_t last)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<FactorPanelVf>(dim3{kCooperativeThreads, 1, 1}, a, tau, m, n, lda, first, last);
}
}  // namespace

void LaunchFactorPanelUncachedKernel(uint32_t blocks, void* stream, float* a, float* tau, uint32_t m, uint32_t n,
                                     uint64_t lda, uint32_t first, uint32_t last)
{
    FactorPanelUncachedKernel<<<blocks, nullptr, stream>>>(a, tau, m, n, lda, first, last);
}
