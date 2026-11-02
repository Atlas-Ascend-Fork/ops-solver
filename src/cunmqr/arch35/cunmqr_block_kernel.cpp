/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "complex_qr_algorithms.h"
#include "complex_qr_matmul.h"
#include "complex_qr_transform_simd.h"
#include "cunmqr_launch.h"
#include "cunmqr_transfer_helpers.h"
#include "kernel_operator.h"

using namespace AscendC;
namespace
{
__global__ __aicore__ void FormTransformKernel(__gm__ const float* tau, __gm__ const float* gram, __gm__ float* high,
                                               __gm__ float* low, uint32_t first, uint32_t panel, uint32_t actualPanel,
                                               bool adjoint)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace ComplexQrTransfer;
    using namespace ComplexQrTransform;
    constexpr uint32_t gramFloats = 2 * COMPLEX_QR_PANEL * Pitch;
    constexpr uint32_t outputFloats = 4 * Plane;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (gramFloats + 4 * COMPLEX_QR_PANEL + 2 * outputFloats) * sizeof(float));
    auto cache = storage.Get<float>();
    auto coefficients = cache[gramFloats];
    auto weights = coefficients[2 * COMPLEX_QR_PANEL];
    auto highLocal = weights[2 * COMPLEX_QR_PANEL];
    auto lowLocal = highLocal[outputFloats];
    GlobalTensor<float> source, tauGlobal, highGlobal, lowGlobal;
    source.SetGlobalBuffer(const_cast<__gm__ float*>(gram));
    tauGlobal.SetGlobalBuffer(const_cast<__gm__ float*>(tau));
    highGlobal.SetGlobalBuffer(high);
    lowGlobal.SetGlobalBuffer(low);
    Read(cache, source, 0, panel, 2 * panel, 2 * panel, 2 * Pitch);
    if (actualPanel)
    {
        Read(coefficients, tauGlobal, 2 * first, 1, 2 * actualPanel, 2 * actualPanel, 2 * COMPLEX_QR_PANEL);
    }
    Sync<HardEvent::MTE2_V>();
    for (uint32_t column = GetBlockIdx(); column < panel; column += GetBlockNum())
    {
        asc_vf_call<FormColumn>(reinterpret_cast<__ubuf__ float*>(cache.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(weights.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(highLocal.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(lowLocal.GetPhyAddr()), panel, actualPanel, column,
                                adjoint);
        Sync<HardEvent::V_MTE3>();
        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
        {
            uint64_t offset = uint64_t(quadrant / 2) * 2 * panel * panel + (quadrant % 2) * panel + column;
            Write(highGlobal, offset, highLocal[quadrant * Plane], panel, 1, 2 * panel, 8);
            Write(lowGlobal, offset, lowLocal[quadrant * Plane], panel, 1, 2 * panel, 8);
        }
        Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace

void LaunchFormTransformKernel(uint32_t blocks, void* stream, const float* tau, const float* gram, float* high,
                               float* low, uint32_t first, uint32_t panel, uint32_t actualPanel, bool adjoint)
{
    FormTransformKernel<<<blocks, nullptr, stream>>>(tau, gram, high, low, first, panel, actualPanel, adjoint);
}

using namespace AscendC;

namespace
{
__simd_vf__ inline void SplitWeightsVf(__ubuf__ float* weights, __ubuf__ float* high, __ubuf__ float* low,
                                       uint32_t count)
{
    Reg::RegTensor<float> value, leading, residual;
    Reg::RegTensor<uint32_t> bits, maskBits;
    Reg::Duplicate(maskBits, 0xffffe000U);
    for (uint32_t offset = 0; offset < count; offset += 64)
    {
        uint32_t remaining = count - offset;
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(value, weights + offset);
        Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(value), maskBits, mask);
        leading = reinterpret_cast<Reg::RegTensor<float>&>(bits);
        Reg::Sub(residual, value, leading, mask);
        Reg::StoreAlign(high + offset, leading, mask);
        Reg::StoreAlign(low + offset, residual, mask);
    }
}

__global__ __aicore__ void SplitWeightsKernel(__gm__ const float* weights, __gm__ float* high, __gm__ float* low,
                                              uint64_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 3 * 256 * sizeof(float));
    auto input = storage.Get<float>();
    auto leading = input[256];
    auto residual = input[512];
    GlobalTensor<float> source, highResult, lowResult;
    source.SetGlobalBuffer(const_cast<__gm__ float*>(weights));
    highResult.SetGlobalBuffer(high);
    lowResult.SetGlobalBuffer(low);
    using namespace ComplexQrTransfer;
    for (uint64_t offset = uint64_t(GetBlockIdx()) * 256; offset < count; offset += uint64_t(GetBlockNum()) * 256)
    {
        uint32_t length = count - offset < 256 ? static_cast<uint32_t>(count - offset) : 256;
        Read(input, source, offset, 1, length, length, 256);
        Sync<HardEvent::MTE2_V>();
        asc_vf_call<SplitWeightsVf>(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ float*>(leading.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ float*>(residual.GetPhyAddr()), length);
        Sync<HardEvent::V_MTE3>();
        Write(highResult, offset, leading, 1, length, length, 256);
        Write(lowResult, offset, residual, 1, length, length, 256);
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
}
}  // namespace

void LaunchSplitWeightsKernel(uint32_t blocks, void* stream, const float* weights, float* high, float* low,
                              uint64_t count)
{
    SplitWeightsKernel<<<blocks, nullptr, stream>>>(weights, high, low, count);
}

using namespace AscendC;

namespace
{
__simd_vf__ inline void UnpackGramVf(__ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* output, uint32_t count)
{
    Reg::RegTensor<uint32_t> lane, source, destination;
    Reg::RegTensor<float> value;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    auto mask = Reg::UpdateMask<float>(count);
    auto fullMask = Reg::CreateMask<uint32_t>();
    Reg::Muls(source, lane, uint32_t(8), fullMask);
    Reg::Muls(destination, lane, uint32_t(2), fullMask);
    Reg::Gather(value, real, source, mask);
    Reg::Scatter(output, value, destination, mask);
    Reg::Gather(value, imag, source, mask);
    Reg::Scatter(output + 1, value, destination, mask);
}

__global__ __aicore__ void UnpackGramKernel(__gm__ const float* hh, __gm__ const float* hl, __gm__ const float* lh,
                                            __gm__ float* gram, uint32_t panel)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (2 * 64 * 8 + 128) * sizeof(float));
    auto real = storage.Get<float>();
    auto imag = real[64 * 8];
    auto output = real[2 * 64 * 8];
    GlobalTensor<float> input, result;
    input.SetGlobalBuffer(const_cast<__gm__ float*>(hh));
    result.SetGlobalBuffer(gram);
    using namespace ComplexQrTransfer;
    for (uint32_t column = GetBlockIdx(); column < panel; column += GetBlockNum())
    {
        for (uint32_t first = 0; first < panel; first += 64)
        {
            uint32_t count = panel - first < 64 ? panel - first : 64;
            Read(real, input, uint64_t(first) * 2 * panel + column, count, 1, 2 * panel, 8);
            Read(imag, input, uint64_t(first + panel) * 2 * panel + column, count, 1, 2 * panel, 8);
            Sync<HardEvent::MTE2_V>();
            asc_vf_call<UnpackGramVf>(reinterpret_cast<__ubuf__ float*>(real.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float*>(imag.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), count);
            Sync<HardEvent::V_MTE3>();
            Write(result, 2 * (uint64_t(column) * panel + first), output, 1, 2 * count, 2 * count, 128);
            Sync<HardEvent::MTE3_V>();
            Sync<HardEvent::V_MTE2>();
        }
    }
}
}  // namespace

void LaunchUnpackGramKernel(uint32_t blocks, void* stream, const float* hh, const float* hl, const float* lh,
                            float* gram, uint32_t panel)
{
    UnpackGramKernel<<<blocks, nullptr, stream>>>(hh, hl, lh, gram, panel);
}

namespace
{
__simd_vf__ inline void AccumulateSplit(__ubuf__ float* input, __ubuf__ float* sum, uint32_t length, bool first)
{
    AscendC::MicroAPI::RegTensor<float> value, previous;
    for (uint16_t start = 0; start < length; start += 64)
    {
        uint32_t remaining = length - start;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
        AscendC::MicroAPI::LoadAlign(value, input + start);
        if (!first)
        {
            AscendC::MicroAPI::LoadAlign(previous, sum + start);
            AscendC::MicroAPI::Add(value, value, previous, mask);
        }
        AscendC::MicroAPI::StoreAlign(sum + start, value, mask);
    }
}
__global__ __aicore__ void ReduceSplitKernel(__gm__ const float* partial, __gm__ float* output, uint32_t elements,
                                             uint32_t parts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    constexpr uint32_t tile = 4096;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> storage;
    pipe.InitBuffer(storage, 2 * tile * sizeof(float));
    auto input = storage.Get<float>(), sum = input[tile];
    GlobalTensor<float> source, target;
    source.SetGlobalBuffer(const_cast<__gm__ float*>(partial));
    target.SetGlobalBuffer(output);
    for (uint32_t offset = GetBlockIdx() * tile; offset < elements; offset += GetBlockNum() * tile)
    {
        uint32_t count = elements - offset < tile ? elements - offset : tile, aligned = (count + 7) / 8 * 8;
        for (uint32_t part = 0; part < parts; ++part)
        {
            ComplexQrTransfer::Sync<HardEvent::V_MTE2>();
            ComplexQrTransfer::Read(input, source, uint64_t(part) * elements + offset, 1, count, count, aligned);
            ComplexQrTransfer::Sync<HardEvent::MTE2_V>();
            asc_vf_call<AccumulateSplit>(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()), count, part == 0);
        }
        ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
        ComplexQrTransfer::Write(target, offset, sum, 1, count, count, aligned);
        ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
    }
}
__global__ __aicore__ void PackTransposedKernel(__gm__ const float* a, __gm__ float* high, __gm__ float* low,
                                                uint64_t lda, uint32_t first, uint32_t columnStart, uint32_t rows,
                                                uint32_t columns, uint32_t packedRows, uint32_t packedColumns)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ComplexQrTransfer::PackTransposedDma(a, high, low, lda, first, columnStart, rows, columns, packedRows,
                                         packedColumns);
}
template <uint32_t RowTile>
__global__ __aicore__ void PackTrailingKernel(__gm__ const float* a, __gm__ float* high, __gm__ float* low,
                                              uint64_t lda, uint32_t first, uint32_t end, uint32_t rows, uint32_t cols,
                                              uint32_t packedRows, uint32_t packedCols, bool rightLayout)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ComplexQrTransfer::PackDma<RowTile>(a, high, low, lda, first, end, rows, cols, packedRows, packedCols, rightLayout);
}
template <uint32_t RowTile>
__global__ __aicore__ void UpdateTrailingKernel(__gm__ float* a, __gm__ const float* product, uint64_t lda,
                                                uint32_t first, uint32_t end, uint32_t rows, uint32_t cols,
                                                uint32_t packedRows, uint32_t packedCols, bool rightLayout)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ComplexQrTransfer::UpdateDma<RowTile>(a, product, lda, first, end, rows, cols, packedRows, packedCols, rightLayout);
}
}  // namespace

void ComplexQrReduceSplit(const float* partial, float* output, uint32_t elements, uint32_t parts, void* stream)
{
    ReduceSplitKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(partial, output, elements, parts);
}

void ComplexQrPackTrailing(const float* a, float* high, float* low, uint64_t lda, uint32_t first, uint32_t end,
                           uint32_t rows, uint32_t cols, uint32_t packedRows, uint32_t packedCols, void* stream,
                           bool rightLayout)
{
    if (!rightLayout && rows >= 512 && cols >= 512)
    {
        PackTrailingKernel<128><<<ComplexQrVectorCores(), nullptr, stream>>>(a, high, low, lda, first, end, rows, cols,
                                                                             packedRows, packedCols, false);
    }
    else
    {
        PackTrailingKernel<64><<<ComplexQrVectorCores(), nullptr, stream>>>(a, high, low, lda, first, end, rows, cols,
                                                                            packedRows, packedCols, rightLayout);
    }
}
void ComplexQrPackTransposed(const float* a, float* high, float* low, uint64_t lda, uint32_t first,
                             uint32_t columnStart, uint32_t rows, uint32_t columns, uint32_t packedRows,
                             uint32_t packedColumns, void* stream)
{
    PackTransposedKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(a, high, low, lda, first, columnStart, rows,
                                                                      columns, packedRows, packedColumns);
}
void ComplexQrUpdateTrailing(float* a, const float* product, uint64_t lda, uint32_t first, uint32_t end, uint32_t rows,
                             uint32_t cols, uint32_t packedRows, uint32_t packedCols, void* stream, bool rightLayout)
{
    if (!rightLayout && rows >= 512 && cols >= 512)
    {
        UpdateTrailingKernel<128><<<ComplexQrVectorCores(), nullptr, stream>>>(a, product, lda, first, end, rows, cols,
                                                                               packedRows, packedCols, false);
    }
    else
    {
        UpdateTrailingKernel<64><<<ComplexQrVectorCores(), nullptr, stream>>>(a, product, lda, first, end, rows, cols,
                                                                              packedRows, packedCols, rightLayout);
    }
}
