/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../utils/complex_qr/arch35/complex_qr_finite.h"
#include "../../utils/complex_qr/arch35/complex_qr_launch.h"
#include "complex_qr_panel_simd.h"
#include "cunmqr_apply_helpers.h"
#include "cunmqr_internal.h"
#include "cunmqr_launch.h"
#include "cunmqr_transfer_helpers.h"
#include "kernel_operator.h"

namespace
{
__simt_vf__ __aicore__ LAUNCH_BOUND(32) void ApplyVf(__gm__ const float* a, __gm__ const float* tau, __gm__ float* c,
                                                     uint32_t order, uint64_t lda, uint64_t ldc, uint32_t k,
                                                     uint32_t vector, bool right, bool adjoint)
{
    ApplyBody(a, tau, c, order, lda, ldc, k, vector, right, adjoint);
}

__global__ __aicore__ void ApplyKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ float* c, uint32_t order,
                                       uint64_t lda, uint64_t ldc, uint32_t k, uint32_t start, uint32_t count,
                                       bool right, bool adjoint)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    for (uint32_t v = AscendC::GetBlockIdx(); v < count; v += AscendC::GetBlockNum())
    {
        asc_vf_call<ApplyVf>(dim3{32, 1, 1}, a, tau, c, order, lda, ldc, k, start + v, right, adjoint);
    }
}
}  // namespace

void LaunchApplyKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c, uint32_t order,
                       uint64_t lda, uint64_t ldc, uint32_t k, uint32_t start, uint32_t count, bool right, bool adjoint)
{
    ApplyKernel<<<blocks, nullptr, stream>>>(a, tau, c, order, lda, ldc, k, start, count, right, adjoint);
}

namespace
{
__simt_vf__ __aicore__ LAUNCH_BOUND(32) void ApplySequenceVf(__gm__ const float* a, __gm__ const float* tau,
                                                             __gm__ float* c, uint32_t order, uint64_t lda,
                                                             uint64_t ldc, uint32_t kCount, uint32_t vector, bool right,
                                                             bool adjoint)
{
    const bool forward = right != adjoint;
    for (uint32_t j = 0; j < kCount; ++j)
    {
        uint32_t k = forward ? j : kCount - 1 - j;
        ApplyBody(a, tau, c, order, lda, ldc, k, vector, right, adjoint);
        asc_syncthreads();
    }
}

__global__ __aicore__ void ApplySequenceKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ float* c,
                                               uint32_t order, uint64_t lda, uint64_t ldc, uint32_t kCount,
                                               uint32_t count, bool right, bool adjoint)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    for (uint32_t v = AscendC::GetBlockIdx(); v < count; v += AscendC::GetBlockNum())
    {
        asc_vf_call<ApplySequenceVf>(dim3{32, 1, 1}, a, tau, c, order, lda, ldc, kCount, v, right, adjoint);
    }
}
}  // namespace

void LaunchApplySequenceKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c,
                               uint32_t order, uint64_t lda, uint64_t ldc, uint32_t kCount, uint32_t count, bool right,
                               bool adjoint)
{
    ApplySequenceKernel<<<blocks, nullptr, stream>>>(a, tau, c, order, lda, ldc, kCount, count, right, adjoint);
}

namespace
{
using namespace AscendC;
// RIGHT is evaluated as a left operation on C^H, without changing A or tau.
template <bool Pack>
__simd_vf__ inline void ConjugateRow(__ubuf__ float* strided, __ubuf__ float* contiguous, uint32_t length)
{
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, rowOffsets, contiguousOffsets;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    auto full = Reg::CreateMask<float>();
    Reg::Muls(rowOffsets, lane, uint32_t(8), full);
    Reg::Muls(contiguousOffsets, lane, uint32_t(2), full);
    Reg::RegTensor<float> real, imag;
    for (uint32_t offset = 0; offset < length; offset += 64)
    {
        uint32_t remaining = length - offset;
        auto mask = Reg::UpdateMask<float>(remaining);
        if constexpr (Pack)
        {
            Reg::Gather(real, strided + 8 * offset, rowOffsets, mask);
            Reg::Gather(imag, strided + 8 * offset + 1, rowOffsets, mask);
            Reg::Muls(imag, imag, -1.0f, mask);
            Reg::Scatter(contiguous + 2 * offset, real, contiguousOffsets, mask);
            Reg::Scatter(contiguous + 2 * offset + 1, imag, contiguousOffsets, mask);
        }
        else
        {
            Reg::Gather(real, contiguous + 2 * offset, contiguousOffsets, mask);
            Reg::Gather(imag, contiguous + 2 * offset + 1, contiguousOffsets, mask);
            Reg::Muls(imag, imag, -1.0f, mask);
            Reg::Scatter(strided + 8 * offset, real, rowOffsets, mask);
            Reg::Scatter(strided + 8 * offset + 1, imag, rowOffsets, mask);
        }
    }
}

__aicore__ inline void ReadTarget(LocalTensor<float> rowTile, LocalTensor<float> target, GlobalTensor<float> cGm,
                                  __ubuf__ float* rowCache, __ubuf__ float* cache, uint32_t order, uint64_t ldc,
                                  uint32_t col, bool right)
{
    if (right)
    {
        for (uint32_t offset = 0; offset < order; offset += 256)
        {
            const uint32_t length = order - offset < 256 ? order - offset : 256;
            ComplexQrTransfer::Sync<HardEvent::V_MTE2>();
            ComplexQrTransfer::Read(rowTile, cGm, 2 * (col + uint64_t(offset) * ldc), length, 2, 2 * ldc, 8);
            ComplexQrTransfer::Sync<HardEvent::MTE2_V>();
            asc_vf_call<ConjugateRow<true>>(rowCache, cache + 32768 + 2 * offset, length);
        }
    }
    else
    {
        ComplexQrTransfer::Sync<HardEvent::V_MTE2>();
        ComplexQrTransfer::Read(target, cGm, 2 * uint64_t(col) * ldc, 1, 2 * order, 2 * ldc, (2 * order + 7) / 8 * 8);
    }
}

__aicore__ inline void WriteTarget(LocalTensor<float> rowTile, LocalTensor<float> target, GlobalTensor<float> cGm,
                                   __ubuf__ float* rowCache, __ubuf__ float* cache, uint32_t order, uint64_t ldc,
                                   uint32_t col, bool right)
{
    if (right)
    {
        for (uint32_t offset = 0; offset < order; offset += 256)
        {
            const uint32_t length = order - offset < 256 ? order - offset : 256;
            asc_vf_call<ConjugateRow<false>>(rowCache, cache + 32768 + 2 * offset, length);
            ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
            ComplexQrTransfer::Write(cGm, 2 * (col + uint64_t(offset) * ldc), rowTile, length, 2, 2 * ldc, 8);
            ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
        }
    }
    else
    {
        ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
        ComplexQrTransfer::Write(cGm, 2 * uint64_t(col) * ldc, target, 1, 2 * order, 2 * ldc, (2 * order + 7) / 8 * 8);
        ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
    }
}

__global__ __aicore__ void CachedSequenceKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ float* c,
                                                uint32_t order, uint64_t lda, uint64_t ldc, uint32_t kCount,
                                                uint32_t count, bool adjoint, bool identity, bool right)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (32768 + 16384 + 128 + 2048) * sizeof(float));
    auto storage = buffer.Get<float>();
    auto cache = reinterpret_cast<__ubuf__ float*>(storage.GetPhyAddr());
    auto vectors = storage, target = storage[32768], localTau = storage[32768 + 16384];
    auto rowTile = storage[32768 + 16384 + 128];
    auto rowCache = reinterpret_cast<__ubuf__ float*>(rowTile.GetPhyAddr());
    const bool leftAdjoint = right ? !adjoint : adjoint;
    GlobalTensor<float> aGm, tauGm, cGm;
    aGm.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    tauGm.SetGlobalBuffer(const_cast<__gm__ float*>(tau));
    cGm.SetGlobalBuffer(c);
    uint32_t width = 16;
    while (width * ((order + 3) / 4 * 4) > 16384)
    {
        width /= 2;
    }
    for (uint32_t col = GetBlockIdx(); col < count; col += GetBlockNum())
    {
        uint32_t columnK = identity && col < kCount ? col + 1 : kCount;
        uint32_t panels = (columnK + width - 1) / width;
        ReadTarget(rowTile, target, cGm, rowCache, cache, order, ldc, col, right);
        for (uint32_t p = 0; p < panels; ++p)
        {
            uint32_t first = (leftAdjoint ? p : panels - 1 - p) * width;
            uint32_t panel = columnK - first < width ? columnK - first : width;
            uint32_t rows = order - first, stride = (rows + 3) / 4 * 4;
            ComplexQrTransfer::Sync<HardEvent::V_MTE2>();
            ComplexQrTransfer::Read(vectors, aGm, 2 * (first + uint64_t(first) * lda), panel, 2 * rows, 2 * lda,
                                    2 * stride);
            ComplexQrTransfer::Read(localTau, tauGm, 2 * first, 1, 2 * panel, 2 * panel, 32);
            ComplexQrTransfer::Sync<HardEvent::MTE2_V>();
            for (uint32_t j = 0; j < panel; ++j)
            {
                const uint32_t k = leftAdjoint ? j : panel - 1 - j;
                asc_vf_call<ComplexQrSimd::Apply<0>>(cache, cache + 32768 + 16384, rows, stride, k,
                                                     cache + 32768 + 2 * first, leftAdjoint);
                PipeBarrier<PIPE_V>();
            }
        }
        WriteTarget(rowTile, target, cGm, rowCache, cache, order, ldc, col, right);
    }
}
}  // namespace

void LaunchCachedSequenceKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c,
                                uint32_t order, uint64_t lda, uint64_t ldc, uint32_t kCount, uint32_t count,
                                bool adjoint, bool identity, bool right)
{
    CachedSequenceKernel<<<blocks, nullptr, stream>>>(a, tau, c, order, lda, ldc, kCount, count, adjoint, identity,
                                                      right);
}

namespace
{
using namespace AscendC;
__global__ __aicore__ void ValidateKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ const float* b,
                                          __gm__ int* info, ComplexQrShape s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FiniteCheckStorage check;
    const uint32_t rows = s.side ? s.n : s.m;
    bool bad = check.Nonfinite(a, rows, s.k, s.lda, true);
    int error = bad ? 6 : 0;
    if (check.Nonfinite(tau, s.k, 1, s.k, false) && !error)
    {
        error = 8;
    }
    if (check.Nonfinite(b, s.m, s.n, s.ldb, false) && !error)
    {
        error = 9;
    }
    check.Report(info, error);
}
}  // namespace
void CunmqrValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrInfo(info, 0x7fffffff, stream);
    ValidateKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(a, tau, b, info, s);
    LaunchFinalizeInfoKernel(1, stream, info);
}
