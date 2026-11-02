/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "../../utils/complex_qr/arch35/complex_qr_finite.h"
#include "../../utils/complex_qr/arch35/complex_qr_launch.h"
#include "../../utils/complex_qr/arch35/complex_qr_transfer.h"
#include "ccgels_device_helpers.h"
#include "ccgels_internal.h"
#include "kernel_operator.h"

namespace
{
using namespace AscendC;
__global__ __aicore__ void ValidateKernel(__gm__ const float* a, __gm__ const float*, __gm__ const float* b,
                                          __gm__ int* info, ComplexQrShape s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FiniteCheckStorage check;
    const uint32_t rows = s.m;
    bool bad = check.Nonfinite(a, rows, s.n, s.lda, false);
    int error = bad ? 4 : 0;
    if (check.Nonfinite(b, s.m, s.rhs, s.ldb, false) && !error)
    {
        error = 6;
    }
    check.Report(info, error);
}
}  // namespace
void CcgelsValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrInfo(info, 0x7fffffff, stream);
    ValidateKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(a, tau, b, info, s);
    LaunchFinalizeInfoKernel(1, stream, info);
}

namespace
{
__simt_vf__ __aicore__ LAUNCH_BOUND(32) void SolveVf(__gm__ const float* a, __gm__ const float* y, __gm__ float* x,
                                                     uint32_t n, uint64_t lda, uint64_t ldy, uint64_t ldx, uint32_t col,
                                                     __gm__ int* info)
{
    uint32_t zero = ZeroDiagonal(a, n, lda);
    if (col == 0 && threadIdx.x == 0)
    {
        *info = static_cast<int>(zero);
    }
    if (zero)
    {
        return;
    }
    for (int64_t r = int64_t(n) - 1; r >= 0; --r)
    {
        Complex sum = {0, 0};
        for (uint32_t c = r + 1 + threadIdx.x; c < n; c += 32)
        {
            sum = Add(sum, Mul(Load(a, r + uint64_t(c) * lda), Load(x, c + uint64_t(col) * ldx)));
        }
        sum = {Sum(sum.r), Sum(sum.i)};
        if (threadIdx.x == 0)
        {
            Store(x, r + uint64_t(col) * ldx,
                  Divide(Sub(Load(y, r + uint64_t(col) * ldy), sum), Load(a, r + uint64_t(r) * lda)));
        }
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) void CachedSolveVf(__gm__ const float* a, __gm__ const float* y,
                                                            __gm__ float* x, __ubuf__ float* cache, uint32_t n,
                                                            uint64_t lda, uint64_t ldy, uint64_t ldx, uint32_t col,
                                                            __gm__ int* info)
{
    if (threadIdx.x < 32)
    {
        uint32_t zero = ZeroDiagonal(a, n, lda);
        if (threadIdx.x == 0)
        {
            cache[0] = static_cast<float>(zero);
            if (col == 0)
            {
                *info = static_cast<int>(zero);
            }
        }
    }
    asc_syncthreads();
    const bool singular = cache[0] != 0;
    // Every thread must finish reading the flag before cache[0] becomes RHS data.
    asc_syncthreads();
    if (singular)
    {
        return;
    }
    for (uint32_t r = threadIdx.x; r < n; r += 512)
    {
        Store(cache, r, Load(y, r + uint64_t(col) * ldy));
    }
    asc_syncthreads();
    for (uint32_t step = 0; step < n; ++step)
    {
        uint32_t column = n - 1 - step;
        if (threadIdx.x == 0)
        {
            Store(cache, column, Divide(Load(cache, column), Load(a, column + uint64_t(column) * lda)));
        }
        asc_syncthreads();
        Complex value = Load(cache, column);
        for (uint32_t r = threadIdx.x; r < column; r += 512)
        {
            Store(cache, r, Sub(Load(cache, r), Mul(Load(a, r + uint64_t(column) * lda), value)));
        }
        asc_syncthreads();
    }
    for (uint32_t r = threadIdx.x; r < n; r += 512)
    {
        Store(x, r + uint64_t(col) * ldx, Load(cache, r));
    }
}

__global__ __aicore__ void SolveKernel(__gm__ const float* a, __gm__ const float* y, __gm__ float* x, __gm__ int* info,
                                       ComplexQrShape s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float cache[2 * 8192];
    for (uint32_t c = AscendC::GetBlockIdx(); c < s.rhs; c += AscendC::GetBlockNum())
    {
        if (s.n <= 8192)
        {
            asc_vf_call<CachedSolveVf>(dim3{512, 1, 1}, a, y, x, cache, s.n, s.lda, s.m, s.ldx, c, info);
        }
        else
        {
            asc_vf_call<SolveVf>(dim3{32, 1, 1}, a, y, x, s.n, s.lda, s.m, s.ldx, c, info);
        }
    }
}
}  // namespace

void LaunchSolveKernel(uint32_t blocks, void* stream, const float* a, const float* y, float* x, int* info,
                       ComplexQrShape s)
{
    if (s.n <= 8192)
    {
        LaunchSolveSimdKernel(blocks, stream, a, y, x, info, s);
        return;
    }
    SolveKernel<<<blocks, nullptr, stream>>>(a, y, x, info, s);
}

using namespace AscendC;

namespace
{
__simd_vf__ inline void FindZero(__ubuf__ float* diagonal, __ubuf__ uint32_t* result, uint32_t count, uint32_t start,
                                 uint32_t order)
{
    Reg::RegTensor<uint32_t> lane, offsets, indices, minimum, sentinel;
    Reg::RegTensor<float> real, imag;
    auto full = Reg::CreateMask<uint32_t>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, uint32_t(8), full);
    Reg::Duplicate(minimum, order);
    Reg::Duplicate(sentinel, order);
    for (uint32_t offset = 0; offset < count; offset += 64)
    {
        uint32_t remaining = count - offset;
        auto valid = Reg::UpdateMask<float>(remaining);
        Reg::Gather(real, diagonal + 8 * offset, offsets, valid);
        Reg::Gather(imag, diagonal + 8 * offset + 1, offsets, valid);
        Reg::MaskReg realZero, imagZero, bothZero;
        Reg::Compares<float, CMPMODE::EQ>(realZero, real, 0.0f, valid);
        Reg::Compares<float, CMPMODE::EQ>(imagZero, imag, 0.0f, valid);
        Reg::And(bothZero, realZero, imagZero, valid);
        Reg::Adds(indices, lane, start + offset, full);
        Reg::Select(indices, indices, sentinel, bothZero);
        Reg::Min(minimum, minimum, indices, full);
    }
    Reg::Reduce<Reg::ReduceType::MIN>(minimum, minimum, full);
    uint32_t one = 1;
    auto first = Reg::UpdateMask<uint32_t>(one);
    Reg::StoreAlign(result, minimum, first);
}

template <typename Float, typename UInt>
__simd_callee__ inline void UpdateSolvedColumn(__ubuf__ float* values, __ubuf__ float* rhs, uint32_t column,
                                               UInt& offsets, Float& xr, Float& xi)
{
    Reg::RegTensor<float> ar, ai, br, bi, temp, other;
    for (uint32_t row = 0; row < column; row += 64)
    {
        uint32_t remaining = column - row;
        auto valid = Reg::UpdateMask<float>(remaining);
        Reg::Gather(ar, values + 2 * row, offsets, valid);
        Reg::Gather(ai, values + 2 * row + 1, offsets, valid);
        Reg::Gather(br, rhs + 2 * row, offsets, valid);
        Reg::Gather(bi, rhs + 2 * row + 1, offsets, valid);
        Reg::Mul(temp, ar, xr, valid);
        Reg::Mul(other, ai, xi, valid);
        Reg::Sub(temp, temp, other, valid);
        Reg::Sub(br, br, temp, valid);
        Reg::Mul(temp, ar, xi, valid);
        Reg::Mul(other, ai, xr, valid);
        Reg::Add(temp, temp, other, valid);
        Reg::Sub(bi, bi, temp, valid);
        Reg::Scatter(rhs + 2 * row, br, offsets, valid);
        Reg::Scatter(rhs + 2 * row + 1, bi, offsets, valid);
    }
}

__simd_vf__ inline void SolvePanel(__ubuf__ float* matrix, __ubuf__ float* rhs, uint32_t stride, uint32_t firstColumn,
                                   uint32_t columns)
{
    Reg::RegTensor<uint32_t> lane, offsets, zeros;
    auto full = Reg::CreateMask<float>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, uint32_t(2), full);
    Reg::Duplicate(zeros, uint32_t(0));
    uint32_t one = 1;
    auto firstLane = Reg::UpdateMask<float>(one);
    for (uint32_t step = 0; step < columns; ++step)
    {
        const uint32_t localColumn = columns - 1 - step;
        const uint32_t column = firstColumn + localColumn;
        auto values = matrix + 2 * localColumn * stride;
        Reg::RegTensor<float> br, bi, ar, ai, scale, temp, other, denominator, xr, xi;
        Reg::Gather(br, values + 2 * column, zeros, full);
        Reg::Gather(bi, values + 2 * column + 1, zeros, full);
        Reg::Gather(ar, rhs + 2 * column, zeros, full);
        Reg::Gather(ai, rhs + 2 * column + 1, zeros, full);
        // Match the scaled complex division used by the original solver.
        Reg::Abs(scale, br, full);
        Reg::Abs(temp, bi, full);
        Reg::Max(scale, scale, temp, full);
        Reg::Div(br, br, scale, full);
        Reg::Div(bi, bi, scale, full);
        Reg::Div(ar, ar, scale, full);
        Reg::Div(ai, ai, scale, full);
        Reg::Mul(denominator, br, br, full);
        Reg::Mul(temp, bi, bi, full);
        Reg::Add(denominator, denominator, temp, full);
        Reg::Mul(xr, ar, br, full);
        Reg::Mul(temp, ai, bi, full);
        Reg::Add(xr, xr, temp, full);
        Reg::Div(xr, xr, denominator, full);
        Reg::Mul(xi, ai, br, full);
        Reg::Mul(temp, ar, bi, full);
        Reg::Sub(xi, xi, temp, full);
        Reg::Div(xi, xi, denominator, full);
        Reg::Scatter(rhs + 2 * column, xr, zeros, firstLane);
        Reg::Scatter(rhs + 2 * column + 1, xi, zeros, firstLane);
        UpdateSolvedColumn(values, rhs, column, offsets, xr, xi);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

__aicore__ inline uint32_t FindZeroDiagonal(LocalTensor<float> diagonal, LocalTensor<uint32_t> result,
                                            GlobalTensor<float> aGm, ComplexQrShape s)
{
    using namespace ComplexQrTransfer;
    uint32_t zero = 0;
    for (uint32_t start = 0; start < s.n; start += 128)
    {
        const uint32_t count = s.n - start < 128 ? s.n - start : 128;
        Sync<HardEvent::V_MTE2>();
        Read(diagonal, aGm, 2 * uint64_t(start) * (s.lda + 1), count, 2, 2 * (s.lda + 1), 8);
        Sync<HardEvent::MTE2_V>();
        asc_vf_call<FindZero>(reinterpret_cast<__ubuf__ float*>(diagonal.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()), count, start, s.n);
        Sync<HardEvent::V_S>();
        const uint32_t index = result.GetValue(0);
        Sync<HardEvent::S_V>();
        if (index < s.n)
        {
            zero = index + 1;
            break;
        }
    }
    return zero;
}

__aicore__ inline void StoreSolveInfo(__gm__ int* info, uint32_t zero)
{
    if (GetBlockIdx() == 0)
    {
        GlobalTensor<int> status;
        status.SetGlobalBuffer(info);
        status.SetValue(0, static_cast<int>(zero));
        DataCacheCleanAndInvalid<int, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(status);
    }
}

__aicore__ inline uint32_t SolvePanelWidth(uint32_t stride, uint32_t columns)
{
    if (stride == 0)
    {
        return 1;
    }
    uint32_t width = 16384 / stride;
    if (width > 128)
    {
        width = 128;
    }
    return width > columns ? columns : width;
}

__global__ __aicore__ void SolveSimdKernel(__gm__ const float* a, __gm__ const float* y, __gm__ float* x,
                                           __gm__ int* info, ComplexQrShape s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!s.n || GetBlockIdx() >= s.rhs)
    {
        return;
    }
    using namespace ComplexQrTransfer;
    const uint32_t stride = (s.n + 3) / 4 * 4;
    const uint32_t width = SolvePanelWidth(stride, s.n);
    const uint32_t matrixFloats = 2 * stride * width;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (matrixFloats + 2 * stride + 1024 + 64) * sizeof(float));
    auto matrix = buffer.Get<float>();
    auto rhs = matrix[matrixFloats];
    auto diagonal = rhs[2 * stride];
    auto result = buffer.Get<uint32_t>()[matrixFloats + 2 * stride + 1024];
    auto matrixAddress = reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr());
    auto rhsAddress = reinterpret_cast<__ubuf__ float*>(rhs.GetPhyAddr());
    GlobalTensor<float> aGm, yGm, xGm;
    aGm.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    yGm.SetGlobalBuffer(const_cast<__gm__ float*>(y));
    xGm.SetGlobalBuffer(x);
    const uint32_t zero = FindZeroDiagonal(diagonal, result, aGm, s);
    StoreSolveInfo(info, zero);
    if (zero)
    {
        return;
    }
    const uint32_t panels = (s.n + width - 1) / width;
    for (uint32_t column = GetBlockIdx(); column < s.rhs; column += GetBlockNum())
    {
        Sync<HardEvent::V_MTE2>();
        Read(rhs, yGm, 2 * uint64_t(column) * s.m, 1, 2 * s.n, 2 * s.n, 2 * stride);
        for (uint32_t step = 0; step < panels; ++step)
        {
            const uint32_t first = (panels - 1 - step) * width;
            const uint32_t count = s.n - first < width ? s.n - first : width;
            Sync<HardEvent::V_MTE2>();
            Read(matrix, aGm, 2 * uint64_t(first) * s.lda, count, 2 * s.n, 2 * s.lda, 2 * stride);
            Sync<HardEvent::MTE2_V>();
            asc_vf_call<SolvePanel>(matrixAddress, rhsAddress, stride, first, count);
        }
        Sync<HardEvent::V_MTE3>();
        Write(xGm, 2 * uint64_t(column) * s.ldx, rhs, 1, 2 * s.n, 2 * s.n, 2 * stride);
        Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace

void LaunchSolveSimdKernel(uint32_t blocks, void* stream, const float* a, const float* y, float* x, int* info,
                           ComplexQrShape s)
{
    SolveSimdKernel<<<blocks, nullptr, stream>>>(a, y, x, info, s);
}
