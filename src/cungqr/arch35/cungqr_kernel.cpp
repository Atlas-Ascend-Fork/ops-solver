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
#include "cungqr_internal.h"
#include "kernel_operator.h"

namespace
{
__simd_vf__ inline void ClearWeights(__ubuf__ float* high, __ubuf__ float* low)
{
    using namespace AscendC;
    Reg::RegTensor<float> zero;
    Reg::Duplicate(zero, 0.0f);
    auto mask = Reg::CreateMask<float>();
    for (uint32_t offset = 0; offset < 256; offset += 64)
    {
        Reg::StoreAlign(high + offset, zero, mask);
        Reg::StoreAlign(low + offset, zero, mask);
    }
}

__global__ __aicore__ void IdentityWeightsKernel(__gm__ const float* ah, __gm__ const float* al, __gm__ float* bh,
                                                 __gm__ float* bl, uint32_t rows, uint32_t panel, uint32_t columns,
                                                 uint32_t packedColumns)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    using namespace ComplexQrTransfer;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 512 * sizeof(float));
    auto high = buffer.Get<float>();
    auto low = high[256];
    GlobalTensor<float> sourceHigh, sourceLow, outputHigh, outputLow;
    sourceHigh.SetGlobalBuffer(const_cast<__gm__ float*>(ah));
    sourceLow.SetGlobalBuffer(const_cast<__gm__ float*>(al));
    outputHigh.SetGlobalBuffer(bh);
    outputLow.SetGlobalBuffer(bl);
    const uint32_t tiles = (packedColumns + 255) / 256;
    for (uint64_t tile = GetBlockIdx(); tile < 2 * uint64_t(panel) * tiles; tile += GetBlockNum())
    {
        const uint32_t row = tile / tiles;
        const uint32_t column = (tile % tiles) * 256;
        const uint32_t width = packedColumns - column < 256 ? packedColumns - column : 256;
        const uint32_t valid = column < columns ? (columns - column < width ? columns - column : width) : 0;
        asc_vf_call<ClearWeights>(reinterpret_cast<__ubuf__ float*>(high.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(low.GetPhyAddr()));
        Sync<HardEvent::V_MTE2>();
        if (valid)
        {
            const uint64_t source = uint64_t(row) * 2 * rows + column;
            Read(high, sourceHigh, source, 1, valid, valid, 256);
            Read(low, sourceLow, source, 1, valid, valid, 256);
        }
        Sync<HardEvent::MTE2_V>();
        Sync<HardEvent::V_MTE3>();
        const uint64_t destination = uint64_t(row) * packedColumns + column;
        Write(outputHigh, destination, high, 1, width, width, 256);
        Write(outputLow, destination, low, 1, width, width, 256);
        Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace

void LaunchIdentityWeightsKernel(uint32_t blocks, void* stream, const float* ah, const float* al, float* bh, float* bl,
                                 uint32_t rows, uint32_t panel, uint32_t columns, uint32_t packedColumns)
{
    IdentityWeightsKernel<<<blocks, nullptr, stream>>>(ah, al, bh, bl, rows, panel, columns, packedColumns);
}

namespace
{
__simd_vf__ inline void FillIdentityColumn(__ubuf__ float* output, uint32_t count, uint32_t diagonal)
{
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<float> zero, one, value;
    Reg::MaskReg selected;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Duplicate(zero, 0.0f);
    Reg::Duplicate(one, 1.0f);
    for (uint16_t start = 0; start < count; start += 64)
    {
        uint32_t remaining = count - start;
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::Compares<uint32_t, CMPMODE::EQ>(selected, lane, diagonal - start, mask);
        Reg::Select(value, one, zero, selected);
        Reg::StoreAlign(output + start, value, mask);
    }
}
__global__ __aicore__ void CopyMatrixKernel(__gm__ const float* source, __gm__ float* target, uint32_t rows,
                                            uint32_t columns, uint64_t sourceStride, uint64_t targetStride,
                                            bool identity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    constexpr uint32_t tileRows = 4096;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> storage;
    pipe.InitBuffer(storage, 2 * tileRows * sizeof(float));
    auto local = storage.Get<float>();
    GlobalTensor<float> input, output;
    input.SetGlobalBuffer(const_cast<__gm__ float*>(source));
    output.SetGlobalBuffer(target);
    uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
    for (uint64_t tile = GetBlockIdx(); tile < uint64_t(rowTiles) * columns; tile += GetBlockNum())
    {
        uint32_t column = tile / rowTiles, row = tile % rowTiles * tileRows;
        uint32_t count = rows - row < tileRows ? rows - row : tileRows, aligned = (2 * count + 7) / 8 * 8;
        if (identity)
        {
            uint32_t diagonal = column >= row && column - row < count ? 2 * (column - row) : UINT32_MAX;
            asc_vf_call<FillIdentityColumn>(reinterpret_cast<__ubuf__ float*>(local.GetPhyAddr()), 2 * count, diagonal);
            ComplexQrTransfer::Sync<HardEvent::V_MTE3>();
        }
        else
        {
            ComplexQrTransfer::Sync<HardEvent::MTE3_MTE2>();
            ComplexQrTransfer::Read(local, input, 2 * (row + uint64_t(column) * sourceStride), 1, 2 * count, 2 * count,
                                    aligned);
            ComplexQrTransfer::Sync<HardEvent::MTE2_MTE3>();
        }
        ComplexQrTransfer::Write(output, 2 * (row + uint64_t(column) * targetStride), local, 1, 2 * count, 2 * count,
                                 aligned);
        ComplexQrTransfer::Sync<HardEvent::MTE3_V>();
    }
}
}  // namespace
void ComplexQrCopyMatrix(const float* source, float* target, uint32_t rows, uint32_t columns, uint64_t sourceStride,
                         uint64_t targetStride, bool identity, void* stream)
{
    CopyMatrixKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(source, target, rows, columns, sourceStride,
                                                                  targetStride, identity);
}

namespace
{
using namespace AscendC;
__global__ __aicore__ void ValidateKernel(__gm__ const float* a, __gm__ const float* tau, __gm__ const float*,
                                          __gm__ int* info, ComplexQrShape s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FiniteCheckStorage check;
    const uint32_t rows = s.m;
    bool bad = check.Nonfinite(a, rows, s.k, s.lda, true);
    int error = bad ? 4 : 0;
    if (check.Nonfinite(tau, s.k, 1, s.k, false) && !error)
    {
        error = 6;
    }
    check.Report(info, error);
}
}  // namespace
void CungqrValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrInfo(info, 0x7fffffff, stream);
    ValidateKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(a, tau, b, info, s);
    LaunchFinalizeInfoKernel(1, stream, info);
}
