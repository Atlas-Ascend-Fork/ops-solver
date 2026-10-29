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

namespace RealLuKernel
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::InplaceLowerInverse()
{
    auto matrix = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto work = reinterpret_cast<__ubuf__ float *>(tmp.GetPhyAddr());
    for (int column = n - 2; column >= 0; --column)
    {
        for (int wave = 0; wave < lanes; wave += 64)
        {
            uint32_t count = lanes - wave < 64 ? lanes - wave : 64;
            SmallInplaceLowerColumnVF(matrix + wave, work + wave, n, lanes, column, count);
        }
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline bool SmallInverse::InplaceUpperLow()
{
    if (n == 17)
    {
        SmallLapack17InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 18)
    {
        SmallLapack18InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 19)
    {
        SmallLapack19InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 20)
    {
        SmallLapack20InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 21)
    {
        SmallLapack21InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 22)
    {
        SmallLapack22InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 23)
    {
        SmallLapack23InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 24)
    {
        SmallLapack24InplaceUpper(lu, tmp, lanes);
        return true;
    }
    return false;
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline bool SmallInverse::InplaceUpperHigh()
{
    if (n == 25)
    {
        SmallLapack25InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 26)
    {
        SmallLapack26InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 27)
    {
        SmallLapack27InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 28)
    {
        SmallLapack28InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 29)
    {
        SmallLapack29InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 30)
    {
        SmallLapack30InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 31)
    {
        SmallLapack31InplaceUpper(lu, tmp, lanes);
        return true;
    }
    if (n == 32)
    {
        SmallLapack32InplaceUpper(lu, tmp, lanes);
        return true;
    }
    return false;
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline bool SmallInverse::SeparateInverseLow()
{
    if (n == 4)
    {
        SmallLapack4VF(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 8)
    {
        SmallLapack8VF(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 16)
    {
        SmallLapack16Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 17)
    {
        SmallLapack17Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 18)
    {
        SmallLapack18Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 19)
    {
        SmallLapack19Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 20)
    {
        SmallLapack20Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 21)
    {
        SmallLapack21Chunked(lu, inv, tmp, lanes);
        return true;
    }
    return false;
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline bool SmallInverse::SeparateInverseMiddle()
{
    if (n == 22)
    {
        SmallLapack22Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 23)
    {
        SmallLapack23Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 24)
    {
        SmallLapack24Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 25)
    {
        SmallLapack25Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 26)
    {
        SmallLapack26Chunked(lu, inv, tmp, lanes);
        return true;
    }
    return false;
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline bool SmallInverse::SeparateInverseHigh()
{
    if (n == 27)
    {
        SmallLapack27Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 28)
    {
        SmallLapack28Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 29)
    {
        SmallLapack29Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 30)
    {
        SmallLapack30Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 31)
    {
        SmallLapack31Chunked(lu, inv, tmp, lanes);
        return true;
    }
    if (n == 32)
    {
        SmallLapack32Chunked(lu, inv, tmp, lanes);
        return true;
    }
    return false;
}

#endif

__aicore__ inline void SmallInverse::UpdateRow(LocalTensor<float> target, LocalTensor<float> source,
                                               LocalTensor<float> coefficient, int count)
{
    Gather(tmp, coefficient, broadcastIndex, 0, count);
    PipeBarrier<PIPE_V>();
    Mul(tmp[rowStride], tmp, source, count);
    PipeBarrier<PIPE_V>();
    Sub(target, target, tmp[rowStride], count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SmallInverse::SwapRows(LocalTensor<float> matrix, int k)
{
    Gather(tmp, piv[k * lanes], broadcastIndex, 0, rowStride);
    PipeBarrier<PIPE_V>();
    for (int row = k + 1; row < n; ++row)
    {
        CompareScalar(mask, tmp, static_cast<float>(row + 1), CMPMODE::EQ, rowStride);
        PipeBarrier<PIPE_V>();
        auto first = matrix[k * rowStride], second = matrix[row * rowStride];
        Adds(tmp[rowStride], first, 0.0f, rowStride);
        PipeBarrier<PIPE_V>();
        Select(first, mask, second, first, SELMODE::VSEL_TENSOR_TENSOR_MODE, rowStride);
        Select(second, mask, tmp[rowStride], second, SELMODE::VSEL_TENSOR_TENSOR_MODE, rowStride);
        PipeBarrier<PIPE_V>();
    }
}

#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
__aicore__ inline void SmallInverse::ChoosePivotGeneric(int k)
{
    auto best = tmp[2 * rowStride], current = tmp[2 * rowStride + lanes], constant = tmp[2 * rowStride + 2 * lanes];
    Abs(best, L(k, k), lanes);
    Duplicate(piv[k * lanes], static_cast<float>(k + 1), lanes);
    PipeBarrier<PIPE_V>();
    for (int row = k + 1; row < n; ++row)
    {
        Abs(current, L(row, k), lanes);
        Duplicate(constant, static_cast<float>(row + 1), lanes);
        PipeBarrier<PIPE_V>();
        Compare(mask, current, best, CMPMODE::GT, lanes);
        PipeBarrier<PIPE_V>();
        Select(best, mask, current, best, SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        Select(piv[k * lanes], mask, constant, piv[k * lanes], SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        PipeBarrier<PIPE_V>();
    }
}

#endif
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
__aicore__ inline void SmallInverse::FactorGeneric()
{
    for (int k = 0; k < n; ++k)
    {
        ChoosePivotGeneric(k);
        // Every selected pivot is at least k + 1. The group needs no
        // swaps precisely when its largest pivot equals that value.
        if (k + 1 < n)
        {
            ReduceMax(tmp, piv[k * lanes], tmp[rowStride], lanes);
            Fence<HardEvent::V_S>();
            float largestPivot = tmp.GetValue(0);
            Fence<HardEvent::S_V>();
            if (largestPivot != static_cast<float>(k + 1))
            {
                SwapRows(lu, k);
            }
        }
        for (int row = k + 1; row < n; ++row)
        {
            Divide(L(row, k), L(row, k), L(k, k), lanes);
        }
        PipeBarrier<PIPE_V>();
        for (int row = k + 1; row < n; ++row)
        {
            UpdateRow(L(row, k + 1), L(k, k + 1), L(row, k), (n - k - 1) * lanes);
        }
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::ApplyLeftColumn(uint16_t k)
{
    auto factors = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto pivotValues = reinterpret_cast<__ubuf__ float *>(piv.GetPhyAddr());
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t count = lanes - wave < 64 ? lanes - wave : 64;
        SmallLeftColumnVF(factors + wave, pivotValues + wave, n, lanes, k, count);
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::ChoosePivotGroup(uint16_t k, bool leftLooking, bool rangeDivision,
                                                      __ubuf__ float *groupFlags)
{
    auto factors = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto pivotValues = reinterpret_cast<__ubuf__ float *>(piv.GetPhyAddr());
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t count = lanes - wave < 64 ? lanes - wave : 64;
        if (rangeDivision)
        {
            SmallChoosePivotRangeVF(factors + wave, pivotValues + wave, groupFlags + wave, n, lanes, k, count);
        }
        else if (leftLooking)
        {
            SmallChoosePivotNormalVF(factors + wave, pivotValues + wave, groupFlags + wave, n, lanes, k, count);
        }
        else
        {
            SmallChoosePivotVF(factors + wave, pivotValues + wave, n, lanes, k, count);
        }
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::SwapPivotGroup(uint16_t k, bool leftLooking)
{
    auto factors = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto pivotValues = reinterpret_cast<__ubuf__ float *>(piv.GetPhyAddr());
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t count = lanes - wave < 64 ? lanes - wave : 64;
        SmallSwapPivotVF(factors + wave, pivotValues + wave, n, lanes, k, count, leftLooking ? k + 1 : n);
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::EliminateColumnGroup(uint16_t k, bool leftLooking, bool rangeDivision,
                                                          bool normalColumn, bool boundedColumn)
{
    auto factors = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t count = lanes - wave < 64 ? lanes - wave : 64;
        if (leftLooking && normalColumn)
        {
            SmallScaleNormalGroupVF(factors + wave, n, lanes, k, count);
        }
        else if (leftLooking)
        {
            SmallScaleColumnVF(factors + wave, n, lanes, k, count);
        }
        else if (rangeDivision && boundedColumn)
        {
            SmallEliminatePivotRangeVF<true>(factors + wave, n, lanes, k, count);
        }
        else if (rangeDivision)
        {
            SmallEliminatePivotRangeVF<false>(factors + wave, n, lanes, k, count);
        }
        else
        {
            SmallEliminatePivotVF(factors + wave, n, lanes, k, count);
        }
    }
}

#endif
__aicore__ inline void SmallInverse::Factor()
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    bool leftLooking = n >= 3 && n <= 8;
    bool rangeDivision = n >= 9 && n <= 32;
    // Left-looking refinement has backed up original A's entire first
    // row in pivIndex. Its raw first row will be overwritten by diagonal
    // reciprocals anyway, then restored before the refinement reads A.
    // n>=9 has no original-A refinement and retains its existing flags.
    auto groupFlagTensor = leftLooking ? raw : pivIndex.ReinterpretCast<float>();
    auto groupFlags = reinterpret_cast<__ubuf__ float *>(groupFlagTensor.GetPhyAddr());
    for (uint16_t k = 0; k < n; ++k)
    {
        if (leftLooking && k != 0)
        {
            ApplyLeftColumn(k);
        }
        ChoosePivotGroup(k, leftLooking, rangeDivision, groupFlags);
        if (k + 1 == n)
        {
            break;
        }
        PipeBarrier<PIPE_V>();
        ReduceMax(tmp, piv[k * lanes], tmp[rowStride], lanes);
        if (rangeDivision || leftLooking)
        {
            // Both reductions share scratch, so serialize them. Keep the
            // two results in separate 32-byte UB blocks before V->S.
            PipeBarrier<PIPE_V>();
            ReduceMin(tmp[8], groupFlagTensor, tmp[rowStride], lanes);
        }
        Fence<HardEvent::V_S>();
        float largestPivot = tmp.GetValue(0);
        bool boundedColumn = rangeDivision && tmp.GetValue(8) == 1.0f;
        bool normalColumn = leftLooking && tmp.GetValue(8) == 1.0f;
        Fence<HardEvent::S_V>();
        if (largestPivot != static_cast<float>(k + 1))
        {
            SwapPivotGroup(k, leftLooking);
        }
        EliminateColumnGroup(k, leftLooking, rangeDivision, normalColumn, boundedColumn);
    }
#else
    FactorGeneric();
#endif
}
}  // namespace RealLuKernel
