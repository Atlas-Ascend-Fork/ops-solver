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
#include "kernel_operator.h"

namespace RealLuKernel
{
using namespace AscendC;
__aicore__ inline int Align(int n) { return (n + 7) / 8 * 8; }
#if (__NPU_ARCH__ == 3510)
constexpr DivConfig IEEE_DIVIDE = {DivAlgo::PRECISION_0ULP_FTZ_FALSE};
constexpr DivConfig NORMAL_DIVIDE = {DivAlgo::PRECISION_0ULP_FTZ_TRUE};
#endif
__aicore__ inline void Divide(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, int count)
{
#if (__NPU_ARCH__ == 3510)
    Div<float, IEEE_DIVIDE>(dst, a, b, count);
#else
    Div(dst, a, b, count);
#endif
}
template <HardEvent Event>
__aicore__ inline void Fence()
{
    // These kernels exclusively own their pipe and do not use TQue events.
    SetFlag<Event>(0);
    WaitFlag<Event>(0);
}
__aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<float> src, int count)
{
    Fence<HardEvent::V_MTE2>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(Align(count) - count), 0};
    DataCopyPad(dst, src, copy, pad);
    Fence<HardEvent::MTE2_S>();
}
__aicore__ inline void Store(GlobalTensor<float> dst, LocalTensor<float> src, int count)
{
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(dst, src, copy);
    Fence<HardEvent::MTE3_S>();
}
__aicore__ inline void StoreInt(GlobalTensor<int> dst, LocalTensor<int> tmp, int value)
{
    tmp.SetValue(0, value);
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{1, sizeof(int), 0, 0, 0};
    DataCopyPad(dst, tmp, copy);
    Fence<HardEvent::MTE3_S>();
}
__aicore__ inline int Pivot(LocalTensor<float> col, int first, int m, bool pivoting)
{
    int row = first;
    if (pivoting)
    {
        float best = col.GetValue(first);
        best = best < 0 ? -best : best;
        for (int i = first + 1; i < m; ++i)
        {
            float value = col.GetValue(i);
            value = value < 0 ? -value : value;
            if (value > best)
            {
                best = value;
                row = i;
            }
        }
    }
    return row;
}
__aicore__ inline int ReducePivotMagnitude(LocalTensor<float> column, LocalTensor<float> scratch,
                                           LocalTensor<float> reduction, int first, int rows)
{
    int start = first / 8 * 8, count = rows - start;
    Fence<HardEvent::S_V>();
    Abs(scratch, column[start], count);
    Fence<HardEvent::V_S>();
    for (int i = start; i < first; ++i)
    {
        scratch.SetValue(i - start, -1.0f);
    }
    Fence<HardEvent::S_V>();
    ReduceMax(reduction, scratch, scratch, count, true);
    Fence<HardEvent::V_S>();
    return start;
}

// RF owns a separate column scratch and a 32-byte reduction buffer. Preserve
// the scalar pivot's first-row tie rule and its NaN behavior explicitly.
__aicore__ inline int PivotRF(LocalTensor<float> column, LocalTensor<float> scratch, LocalTensor<float> reduction,
                              int first, int rows, bool pivoting)
{
    if (!pivoting)
    {
        return first;
    }
    float initial = column.GetValue(first);
    if (initial != initial)
    {
        return first;
    }
    int start = ReducePivotMagnitude(column, scratch, reduction, first, rows);
    int count = rows - start;
    float best = reduction.GetValue(0);
    int index = reduction.ReinterpretCast<int>().GetValue(1);
    if (best != best || index < 0 || index >= count)
    {
        return Pivot(column, first, rows, true);
    }
    int candidate = start + index;
    if (candidate < first || candidate >= rows)
    {
        return Pivot(column, first, rows, true);
    }
    float value = column.GetValue(candidate);
    float magnitude = value < 0 ? -value : value;
    if (magnitude != best)
    {
        return Pivot(column, first, rows, true);
    }
    // ReduceMax tie order is unspecified. Return the earliest equal magnitude.
    for (int r = first; r < candidate; ++r)
    {
        value = column.GetValue(r);
        if ((value < 0 ? -value : value) == best)
        {
            return r;
        }
    }
    return candidate;
}

#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
// Full-local RF columns reserve a 64-float tail. Reuse the blocked panel's
// first-maximum reduction without changing the local Div or FMA sequence.
__aicore__ inline void FactorLocalArgMaximum(LocalTensor<float> reduction, LocalTensor<float> column, int first,
                                             int rows)
{
    int start = first / 64 * 64;
    auto input = reinterpret_cast<__ubuf__ float *>(column.GetPhyAddr()) + start;
    auto indexOutput = reinterpret_cast<__ubuf__ float *>(reduction.GetPhyAddr());
    uint16_t limit = rows - start;
    float startIndex = static_cast<float>(start), firstIndex = static_cast<float>(first);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> value, magnitude, maximum, selected, index;
        MicroAPI::RegTensor<float> missing, one, nanSeen, global, broadcast, result;
        MicroAPI::MaskReg active, greater, replace, unordered, equal;
        auto full = MicroAPI::CreateMask<float>();
        auto scalar = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate<float>(maximum, -1.0f, full);
        MicroAPI::Duplicate<float>(missing, 8192.0f, full);
        MicroAPI::Duplicate<float>(selected, 8192.0f, full);
        MicroAPI::Duplicate<float>(nanSeen, 0.0f, full);
        MicroAPI::Duplicate<float>(one, 1.0f, full);
        MicroAPI::Arange<float>(index, startIndex);
        uint32_t remaining = limit;
        for (uint16_t offset = 0; offset < limit; offset += 64)
        {
            auto valid = MicroAPI::UpdateMask<float>(remaining);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, input + offset, 1u, valid);
            MicroAPI::Abs<float>(magnitude, value, valid);
            MicroAPI::CompareScalar<float, CMPMODE::GE>(active, index, firstIndex, valid);
            MicroAPI::Compare<float, CMPMODE::GT>(greater, magnitude, maximum, valid);
            MicroAPI::MaskAnd(replace, greater, active, valid);
            MicroAPI::Select<float>(maximum, magnitude, maximum, replace);
            MicroAPI::Select<float>(selected, index, selected, replace);
            MicroAPI::Compare<float, CMPMODE::NE>(unordered, value, value, valid);
            MicroAPI::MaskAnd(unordered, unordered, active, valid);
            MicroAPI::Select<float>(nanSeen, one, nanSeen, unordered);
            MicroAPI::Adds<float>(index, index, 64.0f, full);
        }
        // Each lane retained its first maximum; the global minimum index
        // among equal lane maxima therefore gives the first row exactly.
        MicroAPI::Reduce<MicroAPI::ReduceType::MAX>(global, maximum, full);
        MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST>(broadcast, global, full);
        MicroAPI::Compare<float, CMPMODE::EQ>(equal, maximum, broadcast, full);
        MicroAPI::Select<float>(selected, selected, missing, equal);
        MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(result, selected, full);
        MicroAPI::Reduce<MicroAPI::ReduceType::MAX>(global, nanSeen, full);
        MicroAPI::CompareScalar<float, CMPMODE::GT>(unordered, global, 0.0f, scalar);
        MicroAPI::Select<float>(result, missing, result, unordered);
        MicroAPI::StoreAlign<float>(indexOutput, result, scalar);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline int PivotLocalRF(LocalTensor<float> column, LocalTensor<float> scratch, LocalTensor<float> reduction,
                                   int first, int rows, bool pivoting)
{
    if (!pivoting)
    {
        return first;
    }
    Fence<HardEvent::S_V>();
    FactorLocalArgMaximum(reduction, column, first, rows);
    Fence<HardEvent::V_S>();
    int selected = static_cast<int>(reduction.GetValue(0));
    if (selected >= first && selected < rows)
    {
        return selected;
    }
    // An active NaN keeps the complete RF pivot validation and fallback.
    return PivotRF(column, scratch, reduction, first, rows, pivoting);
}
#endif

__aicore__ inline void Swap(LocalTensor<float> col, int k, int p)
{
    if (k == p)
    {
        return;
    }
    float value = col.GetValue(k);
    col.SetValue(k, col.GetValue(p));
    col.SetValue(p, value);
}
__aicore__ inline void ScaleColumn(LocalTensor<float> col, LocalTensor<float> scratch, int k, int m, float diagonal)
{
    int start = (k + 1) / 8 * 8;
    float saved[8];
    for (int i = start; i <= k; ++i)
    {
        saved[i - start] = col.GetValue(i);
    }
    Fence<HardEvent::S_V>();
    Duplicate(scratch, diagonal, Align(m - start));
    PipeBarrier<PIPE_V>();
    Divide(col[start], col[start], scratch, m - start);
    Fence<HardEvent::V_S>();
    for (int i = start; i <= k; ++i)
    {
        col.SetValue(i, saved[i - start]);
    }
}
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
// Write an explicit 0/1 for every actual Div input lane. Ordered bounds
// reject zero, NaN and infinity; masked loads also cover Align8 GM tails.
__aicore__ inline void NormalDivisionFlagsRF(LocalTensor<float> flags, LocalTensor<float> column, int first, int count)
{
    auto output = reinterpret_cast<__ubuf__ float *>(flags.GetPhyAddr());
    auto input = reinterpret_cast<__ubuf__ float *>(column.GetPhyAddr()) + first;
    uint16_t limit = count;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> value, magnitude, one, zero, flag;
        MicroAPI::MaskReg above, below, safe;
        uint32_t remaining = count;
        for (uint16_t offset = 0; offset < limit; offset += 64)
        {
            auto valid = MicroAPI::UpdateMask<float>(remaining);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, input + offset, 1u, valid);
            MicroAPI::Abs<float>(magnitude, value, valid);
            MicroAPI::CompareScalar<float, CMPMODE::GE>(above, magnitude, 2.3283064365386963e-10f, valid);
            MicroAPI::CompareScalar<float, CMPMODE::LE>(below, magnitude, 4294967296.0f, valid);
            MicroAPI::MaskAnd(safe, above, below, valid);
            MicroAPI::Duplicate<float>(one, 1.0f, valid);
            MicroAPI::Duplicate<float>(zero, 0.0f, valid);
            MicroAPI::Select<float>(flag, one, zero, safe);
            MicroAPI::StoreAlign<float>(output + offset, flag, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}
#endif

__aicore__ inline void ScaleColumnRF(LocalTensor<float> column, LocalTensor<float> scratch,
                                     LocalTensor<float> reduction, int k, int rows, float diagonal)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    float magnitude = diagonal < 0 ? -diagonal : diagonal;
    if (magnitude >= 2.3283064365386963e-10f && magnitude <= 4294967296.0f)
    {
        int start = (k + 1) / 8 * 8, count = rows - start;
        Fence<HardEvent::S_V>();
        NormalDivisionFlagsRF(scratch, column, start, count);
        PipeBarrier<PIPE_V>();
        ReduceMin(reduction, scratch, scratch, count);
        Fence<HardEvent::V_S>();
        if (reduction.GetValue(0) == 1.0f)
        {
            // Both magnitudes lie in [2^-32,2^32], so every nonzero result
            // lies in [2^-64,2^64] and cannot be subnormal or overflow.
            float saved[8];
            for (int i = start; i <= k; ++i)
            {
                saved[i - start] = column.GetValue(i);
            }
            Fence<HardEvent::S_V>();
            Duplicate(scratch, diagonal, Align(count));
            PipeBarrier<PIPE_V>();
            Div<float, NORMAL_DIVIDE>(column[start], column[start], scratch, count);
            Fence<HardEvent::V_S>();
            for (int i = start; i <= k; ++i)
            {
                column.SetValue(i, saved[i - start]);
            }
            return;
        }
    }
#endif
    ScaleColumn(column, scratch, k, rows, diagonal);
}
__aicore__ inline void UpdateColumn(LocalTensor<float> dst, LocalTensor<float> col, int k, int m, float coefficient)
{
    int start = (k + 1) / 8 * 8;
    float saved[8];
    for (int i = start; i <= k; ++i)
    {
        saved[i - start] = dst.GetValue(i);
    }
    Fence<HardEvent::S_V>();
    Axpy(dst[start], col[start], -coefficient, m - start);
    Fence<HardEvent::V_S>();
    for (int i = start; i <= k; ++i)
    {
        dst.SetValue(i, saved[i - start]);
    }
}
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
// Every column reserves a full 64-float register at its last row chunk.
// Masked stores preserve the pivot row and all rows above it.
__simd_callee__ inline void UpdateFactorColumnsStep(__ubuf__ float *matrix, int stride, uint16_t row, int k,
                                                    uint16_t columnBegin, uint16_t columnEnd,
                                                    MicroAPI::RegTensor<float> &negative, MicroAPI::MaskReg valid)
{
    MicroAPI::RegTensor<float> target, coefficient;
    for (uint16_t j = columnBegin; j < columnEnd; ++j)
    {
        MicroAPI::LoadAlign<float>(target, matrix + j * stride + row);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(coefficient, matrix + j * stride + k);
        MicroAPI::MulAddDst<float>(target, negative, coefficient, valid);
        MicroAPI::StoreAlign<float>(matrix + j * stride + row, target, valid);
    }
}

__aicore__ inline void UpdateFactorLocalVF(LocalTensor<float> panel, int stride, int rows, int width, int k)
{
    auto matrix = reinterpret_cast<__ubuf__ float *>(panel.GetPhyAddr());
    int begin = (k + 1) / 64 * 64;
    uint16_t rowBegin = begin, rowEnd = rows, columnBegin = k + 1, columnEnd = width;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> column, negative;
        MicroAPI::RegTensor<int32_t> index;
        MicroAPI::MaskReg valid, below;
        uint32_t remaining = rows - begin;
        auto full = MicroAPI::CreateMask<float>();
        for (uint16_t r = rowBegin; r < rowEnd; r += 64)
        {
            auto tail = MicroAPI::UpdateMask<float>(remaining);
            MicroAPI::Arange<int32_t, MicroAPI::IndexOrder::INCREASE_ORDER>(index, r);
            MicroAPI::Compares<int32_t, CMPMODE::GT>(below, index, k, full);
            MicroAPI::MaskAnd(valid, below, tail, full);
            MicroAPI::LoadAlign<float>(column, matrix + k * stride + r);
            MicroAPI::Neg<float>(negative, column, valid);
            UpdateFactorColumnsStep(matrix, stride, r, k, columnBegin, columnEnd, negative, valid);
        }
    }
}
#endif

template <bool UseVectorPivot>
__aicore__ inline int SelectFactorPivot(LocalTensor<float> column, LocalTensor<float> scratch,
                                        LocalTensor<float> reduction, int k, int rows, int stride, bool pivoting)
{
    if constexpr (UseVectorPivot)
    {
#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        if (rows >= 64 && rows <= 8192 && stride % 64 == 0)
        {
            return PivotLocalRF(column, scratch, reduction, k, rows, pivoting);
        }
#endif
        return PivotRF(column, scratch, reduction, k, rows, pivoting);
    }
    else
    {
        return Pivot(column, k, rows, pivoting);
    }
}

__aicore__ inline void UpdateFactorColumns(LocalTensor<float> matrix, LocalTensor<float> column, int stride, int rows,
                                           int columns, int k)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (stride % 64 == 0 && k + 1 < columns)
    {
        Fence<HardEvent::S_V>();
        UpdateFactorLocalVF(matrix, stride, rows, columns, k);
        Fence<HardEvent::V_S>();
        return;
    }
#endif
    for (int j = k + 1; j < columns; ++j)
    {
        UpdateColumn(matrix[j * stride], column, k, rows, matrix[j * stride].GetValue(k));
    }
}

template <bool UseVectorPivot>
__aicore__ inline int FactorLocalCore(LocalTensor<float> matrix, LocalTensor<float> scratch, int m, int n, int stride,
                                      LocalTensor<int> piv, bool pivoting, LocalTensor<float> reduction)
{
    int info = 0;
    int steps = m < n ? m : n;
    for (int k = 0; k < steps; ++k)
    {
        auto col = matrix[k * stride];
        int p = SelectFactorPivot<UseVectorPivot>(col, scratch, reduction, k, m, stride, pivoting);
        piv.SetValue(k, p + 1);
        for (int j = 0; j < n; ++j)
        {
            Swap(matrix[j * stride], k, p);
        }
        float diagonal = col.GetValue(k);
        if (diagonal == 0)
        {
            if (info == 0)
            {
                info = k + 1;
            }
            continue;
        }
        if (k + 1 == m)
        {
            continue;
        }
        if constexpr (UseVectorPivot)
        {
            ScaleColumnRF(col, scratch, reduction, k, m, diagonal);
        }
        else
        {
            ScaleColumn(col, scratch, k, m, diagonal);
        }
        UpdateFactorColumns(matrix, col, stride, m, n, k);
    }
    return info;
}

__aicore__ inline int FactorLocal(LocalTensor<float> matrix, LocalTensor<float> scratch, int m, int n, int stride,
                                  LocalTensor<int> piv, bool pivoting)
{
    return FactorLocalCore<false>(matrix, scratch, m, n, stride, piv, pivoting, scratch);
}

__aicore__ inline int FactorLocalRF(LocalTensor<float> matrix, LocalTensor<float> scratch, int m, int n, int stride,
                                    LocalTensor<int> piv, bool pivoting, LocalTensor<float> reduction)
{
    return FactorLocalCore<true>(matrix, scratch, m, n, stride, piv, pivoting, reduction);
}
}  // namespace RealLuKernel
#include "real_lu_inverse.hpp"
