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
#include "complex_qr_transfer.h"
namespace
{
using namespace AscendC;
constexpr uint32_t VALIDATE_ROWS = 512;
constexpr uint32_t VALIDATE_FLOATS = 2 * VALIDATE_ROWS;

__simd_vf__ inline void ClearMaximum(__ubuf__ uint32_t* maximum)
{
    Reg::RegTensor<uint32_t> zero;
    Reg::Duplicate(zero, uint32_t(0));
    auto mask = Reg::CreateMask<uint32_t>();
    Reg::StoreAlign(maximum, zero, mask);
}

__simd_vf__ inline void ScanFinite(__ubuf__ float* input, __ubuf__ uint32_t* maximum, uint32_t count)
{
    Reg::RegTensor<uint32_t> value, magnitude, signMask, peak, zero;
    Reg::Duplicate(signMask, 0x7fffffffU);
    Reg::Duplicate(zero, uint32_t(0));
    auto fullMask = Reg::CreateMask<uint32_t>();
    Reg::LoadAlign(peak, maximum);
    for (uint32_t offset = 0; offset < count; offset += 64)
    {
        uint32_t remaining = count - offset;
        auto mask = Reg::UpdateMask<uint32_t>(remaining);
        Reg::LoadAlign(value, reinterpret_cast<__ubuf__ uint32_t*>(input + offset));
        Reg::And(magnitude, value, signMask, fullMask);
        Reg::Select(magnitude, magnitude, zero, mask);
        Reg::Max(peak, peak, magnitude, fullMask);
    }
    Reg::StoreAlign(maximum, peak, fullMask);
}

__simd_vf__ inline void ReduceMaximum(__ubuf__ uint32_t* maximum)
{
    Reg::RegTensor<uint32_t> peak, reduced;
    Reg::LoadAlign(peak, maximum);
    auto mask = Reg::CreateMask<uint32_t>();
    Reg::Reduce<Reg::ReduceType::MAX>(reduced, peak, mask);
    uint32_t one = 1;
    auto first = Reg::UpdateMask<uint32_t>(one);
    Reg::StoreAlign(maximum, reduced, first);
}

__aicore__ inline bool HasNonfiniteMatrix(__gm__ const float* source, uint32_t rows, uint32_t columns, uint64_t ld,
                                          bool strictLower, LocalTensor<float> input, LocalTensor<uint32_t> maximum)
{
    if (!rows || !columns)
    {
        return false;
    }
    using namespace ComplexQrTransfer;
    auto inputAddress = reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr());
    auto maximumAddress = reinterpret_cast<__ubuf__ uint32_t*>(maximum.GetPhyAddr());
    asc_vf_call<ClearMaximum>(maximumAddress);
    GlobalTensor<float> global;
    global.SetGlobalBuffer(const_cast<__gm__ float*>(source));
    const uint64_t tilesPerColumn = (uint64_t(rows) + VALIDATE_ROWS - 1) / VALIDATE_ROWS;
    for (uint64_t tile = GetBlockIdx(); tile < tilesPerColumn * columns; tile += GetBlockNum())
    {
        uint32_t column = tile / tilesPerColumn;
        uint32_t first = (tile % tilesPerColumn) * VALIDATE_ROWS;
        uint32_t length = rows - first < VALIDATE_ROWS ? rows - first : VALIDATE_ROWS;
        uint32_t end = first + length;
        // The diagonal and upper triangle of reflectors are implicit, not inputs.
        if (strictLower && first <= column)
        {
            first = column + 1;
        }
        if (first >= end)
        {
            continue;
        }
        uint32_t count = 2 * (end - first);
        Read(input, global, 2 * (uint64_t(column) * ld + first), 1, count, count, VALIDATE_FLOATS);
        Sync<HardEvent::MTE2_V>();
        asc_vf_call<ScanFinite>(inputAddress, maximumAddress, count);
        Sync<HardEvent::V_MTE2>();
    }
    asc_vf_call<ReduceMaximum>(maximumAddress);
    Sync<HardEvent::V_S>();
    const bool bad = maximum.GetValue(0) >= 0x7f800000U;
    Sync<HardEvent::S_V>();
    return bad;
}

// Shared scratch storage and status reduction; input contracts remain in each operator.
class FiniteCheckStorage
{
   public:
    __aicore__ inline FiniteCheckStorage() { pipe.InitBuffer(storage, (VALIDATE_FLOATS + 64 + 8) * sizeof(float)); }
    __aicore__ inline bool Nonfinite(__gm__ const float* source, uint32_t rows, uint32_t columns, uint64_t stride,
                                     bool strictLower)
    {
        return HasNonfiniteMatrix(source, rows, columns, stride, strictLower, storage.Get<float>(),
                                  storage.Get<uint32_t>()[VALIDATE_FLOATS]);
    }
    __aicore__ inline void Report(__gm__ int* info, int error)
    {
        if (!error)
        {
            return;
        }
        auto errorValue = storage.Get<int>()[VALIDATE_FLOATS + 64];
        errorValue.SetValue(0, error);
        ComplexQrTransfer::Sync<HardEvent::S_MTE3>();
        GlobalTensor<int> result;
        result.SetGlobalBuffer(info, 1);
        SetAtomicMin<int>();
        DataCopyPad(result, errorValue, DataCopyExtParams(1, sizeof(int), 0, 0, 0));
        SetAtomicNone();
        ComplexQrTransfer::Sync<HardEvent::MTE3_S>();
    }

   private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
};

}  // namespace
