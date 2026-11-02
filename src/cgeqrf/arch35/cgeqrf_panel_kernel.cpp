/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../cunmqr/arch35/complex_qr_panel_simd.h"
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "../../utils/complex_qr/arch35/complex_qr_finite.h"
#include "../../utils/complex_qr/arch35/complex_qr_launch.h"
#include "../../utils/complex_qr/arch35/complex_qr_transfer.h"
#include "cgeqrf_internal.h"
#include "cgeqrf_reflector_simd.h"
#include "kernel_operator.h"

namespace
{
constexpr uint32_t kPanelCacheFloats = 32768;
constexpr uint32_t kPanelTauFloats = 2 * COMPLEX_QR_PANEL;
template <uint32_t Chunks>
__aicore__ inline void FactorCachedStep(__ubuf__ float* cache, __ubuf__ float* coefficients, uint32_t rows,
                                        uint32_t stride, uint32_t columns, uint32_t k)
{
    asc_vf_call<ComplexQrSimd::Reflector<Chunks>>(cache, coefficients, rows, stride, k);
    for (uint32_t column = k + 1; column < columns; ++column)
    {
        asc_vf_call<ComplexQrSimd::Apply<Chunks>>(cache, coefficients, rows, stride, k, cache + 2 * column * stride,
                                                  true);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
template <uint32_t Chunks>
__aicore__ inline void FactorCachedPanel(__ubuf__ float* cache, __ubuf__ float* coefficients, uint32_t rows,
                                         uint32_t stride, uint32_t columns, uint32_t reflectors)
{
    // Retain separate VF calls and their original synchronization. As the
    // panel shrinks, avoid masked chunks without changing reduction order.
    for (uint32_t k = 0; k < reflectors; ++k)
    {
        if constexpr (Chunks == 8)
        {
            if (rows - k <= 256)
            {
                FactorCachedStep<4>(cache, coefficients, rows, stride, columns, k);
            }
            else if (rows - k <= 320)
            {
                FactorCachedStep<5>(cache, coefficients, rows, stride, columns, k);
            }
            else if (rows - k <= 384)
            {
                FactorCachedStep<6>(cache, coefficients, rows, stride, columns, k);
            }
            else
            {
                FactorCachedStep<8>(cache, coefficients, rows, stride, columns, k);
            }
        }
        else
        {
            FactorCachedStep<Chunks>(cache, coefficients, rows, stride, columns, k);
        }
    }
}
template <uint32_t Chunks>
__global__ __aicore__ void FactorPanelKernel(__gm__ float* a, __gm__ float* tau, uint32_t m, uint32_t n, uint64_t lda,
                                             uint32_t first, uint32_t last)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    uint32_t stride = (m - first + 3) / 4 * 4;
    {
        TPipe pipe;
        TBuf<QuePosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, (kPanelCacheFloats + kPanelTauFloats + 64) * sizeof(float));
        auto local = buffer.Get<float>();
        auto cache = reinterpret_cast<__ubuf__ float*>(local.GetPhyAddr());
        GlobalTensor<float> matrix, coefficients;
        matrix.SetGlobalBuffer(a);
        coefficients.SetGlobalBuffer(tau);
        uint64_t offset = 2 * (first + uint64_t(first) * lda);
        ComplexQrTransfer::Read(local, matrix, offset, n - first, 2 * (m - first), 2 * lda, 2 * stride);
        ComplexQrTransfer::Sync<HardEvent::MTE2_V>();
        FactorCachedPanel<Chunks>(cache, cache + kPanelCacheFloats, m - first, stride, n - first, last - first);
        ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
        ComplexQrTransfer::Write(matrix, offset, local, n - first, 2 * (m - first), 2 * lda, 2 * stride);
        ComplexQrTransfer::Write(coefficients, 2 * first, local[kPanelCacheFloats], 1, 2 * (last - first),
                                 2 * (last - first), (2 * (last - first) + 7) / 8 * 8);
        ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace

void LaunchFactorPanelKernel(uint32_t blocks, void* stream, float* a, float* tau, uint32_t m, uint32_t n, uint64_t lda,
                             uint32_t first, uint32_t last)
{
    const uint32_t stride = (m - first + 3) / 4 * 4;
    const uint64_t bytes = uint64_t(stride) * (n - first) * 2 * sizeof(float);
    if (bytes > kPanelCacheFloats * sizeof(float) || last - first > COMPLEX_QR_PANEL)
    {
        LaunchFactorPanelUncachedKernel(blocks, stream, a, tau, m, n, lda, first, last);
        return;
    }
#define QR_LAUNCH_PANEL(chunks) FactorPanelKernel<chunks><<<blocks, nullptr, stream>>>(a, tau, m, n, lda, first, last)
    if (stride <= 64)
    {
        QR_LAUNCH_PANEL(1);
    }
    else if (stride <= 128)
    {
        QR_LAUNCH_PANEL(2);
    }
    else if (stride <= 256)
    {
        QR_LAUNCH_PANEL(4);
    }
    else if (stride <= 512)
    {
        QR_LAUNCH_PANEL(8);
    }
    else
    {
        QR_LAUNCH_PANEL(0);
    }
#undef QR_LAUNCH_PANEL
}

namespace
{
template <uint32_t Chunks>
__global__ __aicore__ void CachedApplySimdKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ float* c,
                                                 uint64_t lda, uint64_t ldc, uint32_t rows, uint32_t cols,
                                                 uint32_t first, uint32_t panel, uint32_t columnStart)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    if (GetBlockIdx() >= cols)
    {
        return;
    }
    const uint32_t stride = (rows + 3) / 4 * 4;
    const uint32_t targetOffset = 2 * stride * panel;
    const uint32_t tauOffset = targetOffset + 2 * stride;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (tauOffset + 2 * panel + 64) * sizeof(float));
    auto storage = buffer.Get<float>();
    auto cache = reinterpret_cast<__ubuf__ float*>(storage.GetPhyAddr());
    GlobalTensor<float> aGm, tauGm, cGm;
    aGm.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    tauGm.SetGlobalBuffer(const_cast<__gm__ float*>(tau));
    cGm.SetGlobalBuffer(c);
    ComplexQrTransfer::Read(storage, aGm, 2 * (first + uint64_t(first) * lda), panel, 2 * rows, 2 * lda, 2 * stride);
    ComplexQrTransfer::Read(storage[tauOffset], tauGm, 2 * first, 1, 2 * panel, 2 * panel, (2 * panel + 7) / 8 * 8);
    for (uint32_t column = GetBlockIdx(); column < cols; column += GetBlockNum())
    {
        const uint64_t offset = 2 * (first + uint64_t(columnStart + column) * ldc);
        ComplexQrTransfer::Sync<HardEvent::V_MTE2>();
        ComplexQrTransfer::Read(storage[targetOffset], cGm, offset, 1, 2 * rows, 2 * ldc, 2 * stride);
        ComplexQrTransfer::Sync<HardEvent::MTE2_V>();
        for (uint32_t k = 0; k < panel; ++k)
        {
            asc_vf_call<ComplexQrSimd::Apply<Chunks>>(cache, cache + tauOffset, rows, stride, k, cache + targetOffset,
                                                      true);
            PipeBarrier<PIPE_V>();
        }
        ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
        ComplexQrTransfer::Write(cGm, offset, storage[targetOffset], 1, 2 * rows, 2 * ldc, 2 * stride);
        ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace

void LaunchCachedApplySimdKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c,
                                 uint64_t lda, uint64_t ldc, uint32_t rows, uint32_t cols, uint32_t first,
                                 uint32_t panel, uint32_t columnStart)
{
#define QR_LAUNCH_APPLY(chunks)   \
    CachedApplySimdKernel<chunks> \
        <<<blocks, nullptr, stream>>>(a, tau, c, lda, ldc, rows, cols, first, panel, columnStart)
    if (rows <= 64)
    {
        QR_LAUNCH_APPLY(1);
    }
    else if (rows <= 128)
    {
        QR_LAUNCH_APPLY(2);
    }
    else if (rows <= 256)
    {
        QR_LAUNCH_APPLY(4);
    }
    else if (rows <= 512)
    {
        QR_LAUNCH_APPLY(8);
    }
    else
    {
        QR_LAUNCH_APPLY(0);
    }
#undef QR_LAUNCH_APPLY
}

namespace
{
using namespace AscendC;
__global__ __aicore__ void ValidateKernel(__gm__ const float* a, __gm__ const float*, __gm__ const float*,
                                          __gm__ int* info, ComplexQrShape s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FiniteCheckStorage check;
    const uint32_t rows = s.m;
    bool bad = check.Nonfinite(a, rows, s.n, s.lda, false);
    int error = bad ? 3 : 0;
    check.Report(info, error);
}
}  // namespace
void CgeqrfValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrInfo(info, 0x7fffffff, stream);
    ValidateKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(a, tau, b, info, s);
    LaunchFinalizeInfoKernel(1, stream, info);
}
