/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <lib/matmul/matmul.h>
#include <tiling/tiling_api.h>

#include <algorithm>

#include "../utils/kernel/real_lu.hpp"
using namespace RealLuKernel;
extern "C" int real_lu_vector_cores();
extern "C" int real_lu_cube_cores();

// Include all five panel buffers and leave 16 KiB of the 950 UB unused.
static constexpr int PANEL_950_LOCAL_BYTES = 237568;

__global__ __aicore__ void real_lu_pack_factor_kernel(GM_ADDR input, GM_ADDR work, GM_ADDR info, int m, int n, int lda,
                                                      int pitch, int columns)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, Align(pitch) * sizeof(float));
    auto local = buffer.Get<float>();
    GlobalTensor<float> a, w;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work));
    for (int j = GetBlockIdx(); j < columns; j += GetBlockNum())
    {
        Duplicate(local, 0.0f, pitch);
        Fence<HardEvent::V_S>();
        if (j < n)
        {
            Load(local, a[static_cast<int64_t>(j) * lda], m);
        }
        Store(w[static_cast<int64_t>(j) * pitch], local, pitch);
    }
    if (GetBlockIdx() == 0)
    {
        GlobalTensor<int> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
        StoreInt(state, buffer.Get<int>(), 0);
    }
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void LoadPanel(LocalTensor<float> panel, GlobalTensor<float> work, int64_t offset, int pitch,
                                 int rows, int stride, int width)
{
    int alignedRows = Align(rows);
    DataCopyExtParams copy{static_cast<uint16_t>(width), static_cast<uint32_t>(rows * sizeof(float)),
                           static_cast<uint32_t>((pitch - rows) * sizeof(float)),
                           static_cast<uint32_t>((stride - alignedRows) / 8), 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(alignedRows - rows), 0.0f};
    Fence<HardEvent::V_MTE2>();
    DataCopyPad(panel, work[offset], copy, pad);
    Fence<HardEvent::MTE2_S>();
}

__aicore__ inline void StorePanel(GlobalTensor<float> work, LocalTensor<float> panel, int64_t offset, int pitch,
                                  int rows, int stride, int width)
{
    // GM strides count bytes; UB strides count 32-byte blocks after padding.
    DataCopyExtParams copy{static_cast<uint16_t>(width), static_cast<uint32_t>(rows * sizeof(float)),
                           static_cast<uint32_t>((stride - Align(rows)) / 8),
                           static_cast<uint32_t>((pitch - rows) * sizeof(float)), 0};
    Fence<HardEvent::S_MTE3>();
    DataCopyPad(work[offset], panel, copy);
    Fence<HardEvent::MTE3_S>();
}

__aicore__ inline void NegatePanelLocal(LocalTensor<float> panel, int stride, int rows, int width)
{
    auto matrix = reinterpret_cast<__ubuf__ float *>(panel.GetPhyAddr());
    uint16_t columns = width, limit = rows;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> value;
        for (uint16_t j = 0; j < columns; ++j)
        {
            uint32_t remaining = limit;
            for (uint16_t r = 0; r < limit; r += 64)
            {
                auto valid = MicroAPI::UpdateMask<float>(remaining);
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, matrix + j * stride + r, 1u,
                                                                                    valid);
                MicroAPI::Muls<float>(value, value, -1.0f, valid);
                MicroAPI::StoreAlign<float>(matrix + j * stride + r, value, valid);
            }
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void PanelArgMaximum(LocalTensor<float> reduction, LocalTensor<float> column, int first, int rows)
{
    FactorLocalArgMaximum(reduction, column, first, rows);
}

__aicore__ inline void PanelMaximumIndices(LocalTensor<float> scratch, LocalTensor<float> column, int start, int first,
                                           int count, float best)
{
    auto input = reinterpret_cast<__ubuf__ float *>(column.GetPhyAddr()) + start;
    auto output = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());
    uint16_t limit = count;
    float startIndex = static_cast<float>(start), firstIndex = static_cast<float>(first);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> value, magnitude, index, missing, selected;
        MicroAPI::MaskReg equal, active, match;
        uint32_t remaining = count;
        auto full = MicroAPI::CreateMask<float>();
        MicroAPI::Arange<float>(index, startIndex);
        for (uint16_t offset = 0; offset < limit; offset += 64)
        {
            auto valid = MicroAPI::UpdateMask<float>(remaining);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, input + offset, 1u, valid);
            MicroAPI::Abs<float>(magnitude, value, valid);
            MicroAPI::CompareScalar<float, CMPMODE::EQ>(equal, magnitude, best, valid);
            MicroAPI::CompareScalar<float, CMPMODE::GE>(active, index, firstIndex, valid);
            MicroAPI::MaskAnd(match, equal, active, valid);
            MicroAPI::Duplicate<float>(missing, 8192.0f, valid);
            MicroAPI::Select<float>(selected, index, missing, match);
            MicroAPI::StoreAlign<float>(output + offset, selected, valid);
            MicroAPI::Adds<float>(index, index, 64.0f, full);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void SwapPanelRows(LocalTensor<float> panel, int stride, int columns, int first, int selected)
{
    if (first == selected)
    {
        return;
    }
    auto matrix = reinterpret_cast<__ubuf__ float *>(panel.GetPhyAddr());
    uint16_t waves = (columns + 63) / 64;
    Fence<HardEvent::S_V>();
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> lane, base, firstIndex, selectedIndex;
        MicroAPI::RegTensor<float> firstValue, selectedValue;
        auto full = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
        MicroAPI::Muls(base, lane, static_cast<uint32_t>(stride), full);
        uint32_t remaining = columns;
        for (uint16_t wave = 0; wave < waves; ++wave)
        {
            auto valid = MicroAPI::UpdateMask<float>(remaining);
            auto current = matrix + static_cast<uint32_t>(wave) * 64 * stride;
            MicroAPI::Adds(firstIndex, base, static_cast<uint32_t>(first), valid);
            MicroAPI::Adds(selectedIndex, base, static_cast<uint32_t>(selected), valid);
            MicroAPI::Gather(firstValue, current, firstIndex, valid);
            MicroAPI::Gather(selectedValue, current, selectedIndex, valid);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::Scatter(current, selectedValue, firstIndex, valid);
            MicroAPI::Scatter(current, firstValue, selectedIndex, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
    Fence<HardEvent::V_S>();
}

__simd_callee__ inline void ReverseStagedSwapPlan(MicroAPI::RegTensor<uint32_t> &source,
                                                  MicroAPI::RegTensor<uint32_t> &keep,
                                                  MicroAPI::RegTensor<uint32_t> &target,
                                                  MicroAPI::RegTensor<uint32_t> &index,
                                                  MicroAPI::RegTensor<uint32_t> &zero, __ubuf__ int32_t *pivot,
                                                  uint16_t steps, int pivotOffset, MicroAPI::MaskReg valid)
{
    MicroAPI::RegTensor<uint32_t> current;
    MicroAPI::RegTensor<int32_t> selected;
    MicroAPI::MaskReg firstMatch, selectedMatch, later, duplicate;
    for (uint16_t reverse = steps; reverse > 0; --reverse)
    {
        uint16_t k = reverse - 1;
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(selected, pivot + k);
        MicroAPI::Adds<int32_t>(selected, selected, pivotOffset, valid);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(firstMatch, source, static_cast<uint32_t>(k), valid);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(selectedMatch, source,
                                                 reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(selected), valid);
        MicroAPI::Duplicate<uint32_t>(current, static_cast<uint32_t>(k), valid);
        MicroAPI::Select<uint32_t>(source, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(selected), source,
                                   firstMatch);
        MicroAPI::Select<uint32_t>(source, current, source, selectedMatch);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(duplicate, target,
                                                 reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(selected), valid);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(later, index, static_cast<uint32_t>(steps + k), valid);
        MicroAPI::MaskAnd(duplicate, duplicate, later, valid);
        MicroAPI::Select<uint32_t>(keep, zero, keep, duplicate);
    }
}
__simd_callee__ inline void InitializeStagedSwapPlan(__ubuf__ uint32_t *targetPlan, __ubuf__ uint32_t *sourcePlan,
                                                     uint16_t padded, MicroAPI::RegTensor<uint32_t> &missing,
                                                     MicroAPI::MaskReg full)
{
    for (uint16_t r = 0; r < padded; r += 64)
    {
        MicroAPI::StoreAlign<uint32_t>(targetPlan + r, missing, full);
        MicroAPI::StoreAlign<uint32_t>(sourcePlan + r, missing, full);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}
__aicore__ inline void BuildStagedSwapPlan(LocalTensor<uint32_t> plan, LocalTensor<int> pivots, int stride, int first,
                                           int width, int columns, int physical)
{
    auto targetPlan = reinterpret_cast<__ubuf__ uint32_t *>(plan.GetPhyAddr());
    auto sourcePlan = targetPlan + physical;
    auto pivot = reinterpret_cast<__ubuf__ int32_t *>(pivots.GetPhyAddr());
    int pivotOffset = -first - 1;
    uint16_t steps = width, count = columns, span = 2 * Align(width), padded = physical;
    uint32_t logical = 2 * width;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> index, target, source, keep, zero, one, missing;
        MicroAPI::RegTensor<uint32_t> targetOffset, sourceOffset;
        MicroAPI::RegTensor<int32_t> pivotIndex, selected;
        MicroAPI::MaskReg external, inside, invalid, retained;
        auto full = MicroAPI::CreateMask<uint32_t>();
        auto valid = MicroAPI::UpdateMask<uint32_t>(logical);
        MicroAPI::Duplicate<uint32_t>(missing, 0xffffffffu, full);
        InitializeStagedSwapPlan(targetPlan, sourcePlan, padded, missing, full);
        MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(index), 0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GE>(external, index, static_cast<uint32_t>(steps), valid);
        MicroAPI::MaskAnd(external, external, valid, full);
        MicroAPI::Adds<int32_t>(pivotIndex, reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(index), -steps, valid);
        MicroAPI::Gather<int32_t>(selected, pivot, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(pivotIndex),
                                  external);
        MicroAPI::Adds<int32_t>(selected, selected, pivotOffset, external);
        MicroAPI::Select<uint32_t>(target, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(selected), index,
                                   external);
        MicroAPI::Adds<uint32_t>(source, target, 0u, valid);
        MicroAPI::Duplicate<uint32_t>(zero, 0u, full);
        MicroAPI::Duplicate<uint32_t>(one, 1u, full);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(inside, target, static_cast<uint32_t>(steps), valid);
        MicroAPI::MaskAnd(invalid, external, inside, valid);
        MicroAPI::Select<uint32_t>(keep, zero, one, invalid);
        // Reverse tracking retains each final target's original source.
        ReverseStagedSwapPlan(source, keep, target, index, zero, pivot, steps, pivotOffset, valid);

        MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(retained, keep, 0u, valid);
        MicroAPI::MaskAnd(retained, retained, valid, full);
        for (uint16_t c = 0; c < count; ++c)
        {
            uint32_t base = static_cast<uint32_t>(c) * stride;
            MicroAPI::Adds<uint32_t>(targetOffset, target, base, valid);
            MicroAPI::Adds<uint32_t>(sourceOffset, source, base, valid);
            MicroAPI::Select<uint32_t>(targetOffset, targetOffset, missing, retained);
            MicroAPI::Select<uint32_t>(sourceOffset, sourceOffset, missing, retained);
            MicroAPI::StoreAlign<uint32_t>(targetPlan + c * span, targetOffset, valid);
            MicroAPI::StoreAlign<uint32_t>(sourcePlan + c * span, sourceOffset, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void ApplyStagedSwapPlan(LocalTensor<float> stage, LocalTensor<float> cache,
                                           LocalTensor<uint32_t> plan, int physical, int width, int columns)
{
    auto matrix = reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr());
    auto saved = reinterpret_cast<__ubuf__ float *>(cache.GetPhyAddr());
    auto targetPlan = reinterpret_cast<__ubuf__ uint32_t *>(plan.GetPhyAddr());
    auto sourcePlan = targetPlan + physical;
    uint16_t limit = columns * 2 * Align(width);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> target, source;
        MicroAPI::RegTensor<float> value;
        MicroAPI::MaskReg present, active;
        auto full = MicroAPI::CreateMask<uint32_t>();
        uint32_t remaining = limit;
        for (uint16_t r = 0; r < limit; r += 64)
        {
            auto valid = MicroAPI::UpdateMask<uint32_t>(remaining);
            MicroAPI::LoadAlign<uint32_t>(target, targetPlan + r);
            MicroAPI::LoadAlign<uint32_t>(source, sourcePlan + r);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(present, target, 0xffffffffu, full);
            MicroAPI::MaskAnd(active, present, valid, full);
            MicroAPI::Gather<float>(value, matrix, source, active);
            MicroAPI::StoreAlign<float>(saved + r, value, active);
        }
        // All original stage values must be read before the first overwrite;
        // the complete cached values must also precede their subsequent loads.
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        remaining = limit;
        for (uint16_t r = 0; r < limit; r += 64)
        {
            auto valid = MicroAPI::UpdateMask<uint32_t>(remaining);
            MicroAPI::LoadAlign<uint32_t>(target, targetPlan + r);
            MicroAPI::LoadAlign<float>(value, saved + r);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(present, target, 0xffffffffu, full);
            MicroAPI::MaskAnd(active, present, valid, full);
            MicroAPI::Scatter(matrix, value, target, active);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void SwapStagedColumns(LocalTensor<float> stage, LocalTensor<int> pivots, int stride, int first,
                                         int width, int columns)
{
    auto matrix = reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr());
    auto pivot = reinterpret_cast<__ubuf__ int32_t *>(pivots.GetPhyAddr());
    uint16_t steps = width;
    uint32_t count = columns;
    int pivotOffset = -first - 1;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> lane, base, firstIndex, selectedIndex;
        MicroAPI::RegTensor<int32_t> selected;
        MicroAPI::RegTensor<float> firstValue, selectedValue;
        auto full = MicroAPI::CreateMask<uint32_t>();
        auto valid = MicroAPI::UpdateMask<float>(count);
        MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
        MicroAPI::Muls(base, lane, static_cast<uint32_t>(stride), full);
        for (uint16_t k = 0; k < steps; ++k)
        {
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(selected, pivot + k);
            MicroAPI::Adds(selected, selected, pivotOffset, valid);
            MicroAPI::Adds(firstIndex, base, static_cast<uint32_t>(k), valid);
            MicroAPI::Add(selectedIndex, base, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(selected), valid);
            MicroAPI::Gather(firstValue, matrix, firstIndex, valid);
            MicroAPI::Gather(selectedValue, matrix, selectedIndex, valid);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::Scatter(matrix, selectedValue, firstIndex, valid);
            MicroAPI::Scatter(matrix, firstValue, selectedIndex, valid);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
    }
}
#endif

__aicore__ inline int PanelPivot(LocalTensor<float> column, LocalTensor<float> scratch, LocalTensor<float> reduction,
                                 int first, int rows, bool pivoting)
{
    if (!pivoting)
    {
        return first;
    }
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    Fence<HardEvent::S_V>();
    PanelArgMaximum(reduction, column, first, rows);
    Fence<HardEvent::V_S>();
    int selected = static_cast<int>(reduction.GetValue(0));
    if (selected >= first && selected < rows)
    {
        return selected;
    }
    // Preserve the previous reduction path if an intermediate column has NaN.
#endif
    int start = ReducePivotMagnitude(column, scratch, reduction, first, rows);
    int count = rows - start;
    float best = reduction.GetValue(0);
    int candidate = start + reduction.ReinterpretCast<int>().GetValue(1);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (candidate > first && candidate < rows && best == best)
    {
        Fence<HardEvent::S_V>();
        PanelMaximumIndices(scratch, column, start, first, count, best);
        PipeBarrier<PIPE_V>();
        ReduceMin(reduction, scratch, scratch, count);
        Fence<HardEvent::V_S>();
        int selected = static_cast<int>(reduction.GetValue(0));
        if (selected >= first && selected < rows)
        {
            return selected;
        }
    }
#endif
    // Reduction tie order is unspecified; enforce the first row explicitly.
    for (int r = first; r < candidate; ++r)
    {
        float value = column.GetValue(r);
        if ((value < 0 ? -value : value) == best)
        {
            return r;
        }
    }
    return candidate;
}

__aicore__ inline void PreparePanelReciprocal(LocalTensor<float> scratch, float diagonal)
{
    float magnitude = diagonal < 0 ? -diagonal : diagonal;
    Fence<HardEvent::S_V>();
    Duplicate(scratch, 1.0f, 8);
    Duplicate(scratch[8], diagonal, 8);
    PipeBarrier<PIPE_V>();
#if (__NPU_ARCH__ == 3510)
    if (magnitude <= 1.0e37f)
    {
        Div<float, NORMAL_DIVIDE>(scratch, scratch, scratch[8], 1);
    }
    else
    {
        Divide(scratch, scratch, scratch[8], 1);
    }
#else
    Divide(scratch, scratch, scratch[8], 1);
#endif
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void ScalePanelColumnVector(LocalTensor<float> column, LocalTensor<float> scratch, int first,
                                              int rows)
{
    // Reuse the exact reciprocal just computed, and mask out the LU prefix.
    // This avoids round-tripping the reciprocal and prefix through scalar UB.
    PipeBarrier<PIPE_V>();
    int begin = (first + 1) / 64 * 64;
    auto input = reinterpret_cast<__ubuf__ float *>(column.GetPhyAddr()) + begin;
    auto inverse = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());
    uint16_t limit = rows - begin;
    float beginIndex = static_cast<float>(begin), firstIndex = static_cast<float>(first);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> value, scaled, reciprocal, index;
        MicroAPI::MaskReg below, active;
        auto full = MicroAPI::CreateMask<float>();
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(reciprocal, inverse);
        MicroAPI::Arange<float>(index, beginIndex);
        uint32_t remaining = limit;
        for (uint16_t offset = 0; offset < limit; offset += 64)
        {
            auto valid = MicroAPI::UpdateMask<float>(remaining);
            MicroAPI::CompareScalar<float, CMPMODE::GT>(below, index, firstIndex, valid);
            MicroAPI::MaskAnd(active, below, valid, full);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, input + offset, 1u, valid);
            MicroAPI::Mul<float>(scaled, value, reciprocal, active);
            MicroAPI::StoreAlign<float>(input + offset, scaled, active);
            MicroAPI::Adds<float>(index, index, 64.0f, full);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
    Fence<HardEvent::V_S>();
}
#else
__aicore__ inline void ScalePanelColumnGeneric(LocalTensor<float> column, LocalTensor<float> scratch, int first,
                                               int rows)
{
    Fence<HardEvent::V_S>();
    float reciprocal = scratch.GetValue(0);
    int begin = (first + 1) / 8 * 8;
    float saved[8];
    for (int i = begin; i <= first; ++i)
    {
        saved[i - begin] = column.GetValue(i);
    }
    Fence<HardEvent::S_V>();
    Muls(column[begin], column[begin], reciprocal, rows - begin);
    Fence<HardEvent::V_S>();
    for (int i = begin; i <= first; ++i)
    {
        column.SetValue(i, saved[i - begin]);
    }
}
#endif
__aicore__ inline void ScalePanelColumn(LocalTensor<float> column, LocalTensor<float> scratch, int first, int rows,
                                        float diagonal)
{
    float magnitude = diagonal < 0 ? -diagonal : diagonal;
    if (magnitude < 1.1754943508222875e-38f)
    {
        ScaleColumn(column, scratch, first, rows, diagonal);
        return;
    }
    PreparePanelReciprocal(scratch, diagonal);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    ScalePanelColumnVector(column, scratch, first, rows);
#else
    ScalePanelColumnGeneric(column, scratch, first, rows);
#endif
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
template <bool PackedToRows>
__no_simd_vf_fusion__ __simd_vf__ inline void TransposePanelRight(__ubuf__ float *rows, __ubuf__ float *packed,
                                                                  uint16_t width, uint32_t packedStride,
                                                                  uint32_t columnOffset, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> lane, packedIndex, rowIndex;
    MicroAPI::RegTensor<float> value;
    auto full = MicroAPI::CreateMask<uint32_t>();
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    MicroAPI::Muls(packedIndex, lane, packedStride, full);
    MicroAPI::Adds(rowIndex, lane, columnOffset, full);
    auto valid = MicroAPI::UpdateMask<float>(count);
    for (uint16_t row = 0; row < width; ++row)
    {
        if constexpr (PackedToRows)
        {
            MicroAPI::Gather(value, packed + row, packedIndex, valid);
            MicroAPI::Scatter(rows + row * 32, value, rowIndex, valid);
        }
        else
        {
            MicroAPI::Gather(value, rows + row * 32, rowIndex, valid);
            MicroAPI::Scatter(packed + row, value, packedIndex, valid);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__simd_callee__ inline void SolvePanelStep(__ubuf__ float *rhs, __ubuf__ float *diagonal, uint16_t k, uint16_t first,
                                           uint16_t steps, int diagonalStride, MicroAPI::MaskReg valid)
{
    MicroAPI::RegTensor<float> source, negative, coefficient, target;
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(source, rhs + k * 32, 1u, valid);
    MicroAPI::Neg<float>(negative, source, valid);
    for (uint16_t r = first; r < steps; ++r)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(target, rhs + r * 32, 1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(coefficient, diagonal + k * diagonalStride + r);
        MicroAPI::MulAddDst<float>(target, negative, coefficient, valid);
        MicroAPI::StoreAlign<float>(rhs + r * 32, target, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}
__aicore__ inline void SolvePanelLocal(LocalTensor<float> right, LocalTensor<float> lower, int width,
                                       int diagonalStride, int columns)
{
    auto rhs = reinterpret_cast<__ubuf__ float *>(right.GetPhyAddr());
    auto diagonal = reinterpret_cast<__ubuf__ float *>(lower.GetPhyAddr());

#if defined(REAL_LU_REGISTER_LAYOUT_950)
    uint16_t steps = width;
    uint32_t count = columns;
    __VEC_SCOPE__
    {
        auto valid = MicroAPI::UpdateMask<float>(count);
        for (uint16_t k = 0; k < steps; ++k)
        {
            SolvePanelStep(rhs, diagonal, k, k + 1, steps, diagonalStride, valid);
        }
    }
#else
    for (int k = 0; k < width; ++k)
    {
        uint16_t first = k + 1, steps = width;
        uint32_t count = columns;
        __VEC_SCOPE__
        {
            auto valid = MicroAPI::UpdateMask<float>(count);
            SolvePanelStep(rhs, diagonal, static_cast<uint16_t>(k), first, steps, diagonalStride, valid);
        }
    }
#endif
}
#endif

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
template <bool Scale>
__aicore__ inline void UpdatePanelLocal(LocalTensor<float> panel, LocalTensor<float> reciprocal, int stride, int rows,
                                        int width, int k)
{
    auto matrix = reinterpret_cast<__ubuf__ float *>(panel.GetPhyAddr());
    auto inverse = reinterpret_cast<__ubuf__ float *>(reciprocal.GetPhyAddr());
    int begin = (k + 1) / 64 * 64;
    uint16_t rowBegin = begin, rowEnd = rows, columnBegin = k + 1, columnEnd = width;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> column, negative, scale;
        MicroAPI::RegTensor<int32_t> index;
        MicroAPI::MaskReg valid, below;
        uint32_t remaining = rows - begin;
        auto full = MicroAPI::CreateMask<float>();
        if constexpr (Scale)
        {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale, inverse);
        }
        for (uint16_t r = rowBegin; r < rowEnd; r += 64)
        {
            auto tail = MicroAPI::UpdateMask<float>(remaining);
            MicroAPI::Arange<int32_t, MicroAPI::IndexOrder::INCREASE_ORDER>(index, r);
            MicroAPI::Compares<int32_t, CMPMODE::GT>(below, index, k, full);
            MicroAPI::MaskAnd(valid, below, tail, full);
            MicroAPI::LoadAlign<float>(column, matrix + k * stride + r);
            if constexpr (Scale)
            {
                // Keep the separately rounded FP32 product used by the update.
                MicroAPI::Mul<float>(column, column, scale, valid);
                MicroAPI::StoreAlign<float>(matrix + k * stride + r, column, valid);
            }
            MicroAPI::Neg<float>(negative, column, valid);
            UpdateFactorColumnsStep(matrix, stride, r, k, columnBegin, columnEnd, negative, valid);
        }
    }
}
#endif

namespace
{
class FactorPanel
{
   public:
    __aicore__ inline FactorPanel(int m, int pitch, int first, int width, bool pivoting)
        : m(m), pitch(pitch), first(first), width(width), pivoting(pivoting)
    {
    }
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR work, GM_ADDR pivots, GM_ADDR info)
    {
        // A VF may load a full 64-float register at the last chunk. Reserve its
        // readable padding in every column; masked stores retain the LU prefix.
        rows = m - first;
        stride = (rows + 63) / 64 * 64;
#if defined(REAL_LU_REGISTER_LAYOUT_950)
        int panelBytes = (stride * (width + 2) + Align(width) + 8) * sizeof(float);
        inLocal = panelBytes <= PANEL_950_LOCAL_BYTES;
#else
        inLocal = stride * width <= 32768;
#endif
        pipe.InitBuffer(panelBuf, inLocal ? stride * width * sizeof(float) : 32);
        pipe.InitBuffer(columnBuf, stride * sizeof(float));
        pipe.InitBuffer(scratchBuf, (stride > 16 ? stride : 16) * sizeof(float));
        pipe.InitBuffer(pivotBuf, Align(width) * sizeof(int));
        pipe.InitBuffer(resultBuf, 32);
        panel = panelBuf.Get<float>();
        column = columnBuf.Get<float>();
        scratch = scratchBuf.Get<float>();
        piv = pivotBuf.Get<int>();
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work));
        p.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
        state.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
        bad = state.GetValue(0);
    }
    __aicore__ inline void Run(GM_ADDR negative)
    {
        LoadData();
        for (int k = 0; k < width; ++k)
        {
            Step(k);
        }
        StoreData();
        StoreNegative(negative);
        Finish();
    }

   private:
    __aicore__ inline void LoadData()
    {
        if (inLocal)
        {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
            LoadPanel(panel, w, static_cast<int64_t>(first) * pitch + first, pitch, rows, stride, width);
#else
            for (int j = 0; j < width; ++j)
            {
                Load(panel[j * stride], w[static_cast<int64_t>(first + j) * pitch + first], rows);
            }
#endif
        }
    }
    __aicore__ inline void SwapRows(LocalTensor<float> col, int k, int selected)
    {
        if (inLocal)
        {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
            SwapPanelRows(panel, stride, width, k, selected);
#else
            for (int j = 0; j < width; ++j)
            {
                Swap(panel[j * stride], k, selected);
            }
#endif
        }
        else
        {
            Swap(col, k, selected);
        }
    }
    __aicore__ inline bool FusedUpdate(int k, float diagonal)
    {
        bool fused = false;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        float magnitude = diagonal < 0 ? -diagonal : diagonal;
        if (inLocal && diagonal != 0 && k + 1 < rows && magnitude >= 1.1754943508222875e-38f)
        {
            PreparePanelReciprocal(scratch, diagonal);
            PipeBarrier<PIPE_V>();
            UpdatePanelLocal<true>(panel, scratch, stride, rows, width, k);
            Fence<HardEvent::V_S>();
            fused = true;
        }
#else
        (void)k;
        (void)diagonal;
#endif
        return fused;
    }
    __aicore__ inline void Update(LocalTensor<float> col, int k, int selected, float diagonal, bool fused)
    {
        if (inLocal && !fused && diagonal != 0 && k + 1 < rows)
        {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
            Fence<HardEvent::S_V>();
            UpdatePanelLocal<false>(panel, scratch, stride, rows, width, k);
            Fence<HardEvent::V_S>();
#else
            for (int j = k + 1; j < width; ++j)
            {
                UpdateColumn(panel[j * stride], col, k, rows, panel[j * stride].GetValue(k));
            }
#endif
        }
        else if (!inLocal)
        {
            for (int j = 0; j < width; ++j)
            {
                if (j != k)
                {
                    Load(scratch, w[static_cast<int64_t>(first + j) * pitch + first], rows);
                    Swap(scratch, k, selected);
                    if (j > k && diagonal != 0 && k + 1 < rows)
                    {
                        UpdateColumn(scratch, col, k, rows, scratch.GetValue(k));
                    }
                    Store(w[static_cast<int64_t>(first + j) * pitch + first], scratch, rows);
                }
            }
        }
    }
    __aicore__ inline void Step(int k)
    {
        if (!inLocal)
        {
            Load(column, w[static_cast<int64_t>(first + k) * pitch + first], rows);
        }
        auto col = inLocal ? panel[k * stride] : column;
        int selected = PanelPivot(col, scratch, resultBuf.Get<float>(), k, rows, pivoting);
        piv.SetValue(k, first + selected + 1);
        SwapRows(col, k, selected);
        float diagonal = col.GetValue(k);
        if (diagonal == 0 && bad == 0)
        {
            bad = first + k + 1;
        }
        bool fused = FusedUpdate(k, diagonal);
        if (!fused && diagonal != 0 && k + 1 < rows)
        {
            ScalePanelColumn(col, scratch, k, rows, diagonal);
        }
        if (!inLocal)
        {
            Store(w[static_cast<int64_t>(first + k) * pitch + first], col, rows);
        }
        Update(col, k, selected, diagonal, fused);
    }
    __aicore__ inline void StoreData()
    {
        if (inLocal)
        {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
            StorePanel(w, panel, static_cast<int64_t>(first) * pitch + first, pitch, rows, stride, width);
#else
            for (int j = 0; j < width; ++j)
            {
                Store(w[static_cast<int64_t>(first + j) * pitch + first], panel[j * stride], rows);
            }
#endif
        }
    }
    __aicore__ inline void StoreNegative(GM_ADDR negative)
    {
        if (negative != nullptr)
        {
            GlobalTensor<float> minus;
            minus.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(negative));
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
            if (inLocal)
            {
                // The completed work copy must precede modifying its UB source.
                Fence<HardEvent::S_V>();
                NegatePanelLocal(panel, stride, rows, width);
                Fence<HardEvent::V_S>();
                StorePanel(minus, panel, static_cast<int64_t>(first) * pitch + first, pitch, rows, stride, width);
            }
            else
#endif
            {
                for (int j = 0; j < width; ++j)
                {
                    auto source = inLocal ? panel[j * stride] : column;
                    int64_t offset = static_cast<int64_t>(first + j) * pitch + first;
                    if (!inLocal)
                    {
                        Load(source, w[offset], rows);
                    }
                    Fence<HardEvent::S_V>();
                    Muls(source, source, -1.0f, rows);
                    Fence<HardEvent::V_S>();
                    Store(minus[offset], source, rows);
                }
            }
        }
    }
    __aicore__ inline void Finish()
    {
        Fence<HardEvent::S_MTE3>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(width * sizeof(int)), 0, 0, 0};
        DataCopyPad(p[first], piv, copy);
        Fence<HardEvent::MTE3_S>();
        StoreInt(state, resultBuf.Get<int>(), bad);
    }
    int m, pitch, first, width, rows = 0, stride = 0, bad = 0;
    bool pivoting, inLocal = false;
    TBuf<TPosition::VECCALC> panelBuf, columnBuf, scratchBuf, pivotBuf, resultBuf;
    LocalTensor<float> panel, column, scratch;
    LocalTensor<int> piv;
    GlobalTensor<float> w;
    GlobalTensor<int> p, state;
};
}  // namespace

__global__ __aicore__ void real_lu_panel_kernel(GM_ADDR work, GM_ADDR negative, GM_ADDR pivots, GM_ADDR info, int m,
                                                int pitch, int first, int width, bool pivoting)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    FactorPanel op(m, pitch, first, width, pivoting);
    op.Init(pipe, work, pivots, info);
    op.Run(negative);
}

namespace
{
class FactorSwapTrsm
{
   public:
    __aicore__ inline FactorSwapTrsm(int m, int n, int pitch, int first, int width)
        : m(m), n(n), pitch(pitch), first(first), width(width)
    {
    }
    __aicore__ inline bool Init(TPipe &pipe, GM_ADDR work, GM_ADDR pivots)
    {
        if (!InitShape(pipe))
        {
            return false;
        }
        InitBuffers(pipe);
        LoadPivots(work, pivots);
        Prepare();
        return true;
    }
    __aicore__ inline void Run()
    {
        int tiles = (n + RHS_WIDTH - 1) / RHS_WIDTH;
        for (int tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum())
        {
            int begin = tile * RHS_WIDTH, count = n - begin < RHS_WIDTH ? n - begin : RHS_WIDTH;
            if (needsSwap)
            {
                SwapColumns(begin, count);
            }
            int activeBegin = begin > first + width ? begin : first + width;
            int activeCount = begin + count - activeBegin;
            if (activeCount <= 0)
            {
                continue;
            }
            SolveTile(begin, count, activeBegin, activeCount);
        }
    }

   private:
    __aicore__ inline bool InitShape(TPipe &pipe)
    {
        diagonalStride = Align(width);
        columnRows = m;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        // Every pivot is at least first+k: earlier rows remain in place.
        columnRows -= first;
#endif
        columnStride = Align(columnRows);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        int fixedBytes = (width * diagonalStride + (width + diagonalStride) * RHS_WIDTH + Align(width)) * sizeof(float);
        plannedSwap = width <= 32;
        int bytesPerColumn = columnStride * sizeof(float), reserve = 4096;
        if (plannedSwap)
        {
            bytesPerColumn += 4 * Align(width) * sizeof(uint32_t);
            reserve += 512;
        }
        if (bytesPerColumn <= 0)
        {
            return false;
        }
        stagedColumns = (PANEL_950_LOCAL_BYTES - reserve - fixedBytes) / bytesPerColumn;
        stagedColumns = stagedColumns < 1 ? 1 : (stagedColumns > 8 ? 8 : stagedColumns);
        planStride = (stagedColumns * 2 * Align(width) + 63) / 64 * 64;
        if (plannedSwap)
        {
            pipe.InitBuffer(planBuf, 2 * planStride * sizeof(uint32_t));
        }
#endif
        return true;
    }
    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        pipe.InitBuffer(columnBuf, stagedColumns * columnStride * sizeof(float));
        pipe.InitBuffer(diagonalBuf, width * diagonalStride * sizeof(float));
        pipe.InitBuffer(rhsBuf, width * RHS_WIDTH * sizeof(float));
        pipe.InitBuffer(packedBuf, diagonalStride * RHS_WIDTH * sizeof(float));
        pipe.InitBuffer(pivotBuf, Align(width) * sizeof(int));
        column = columnBuf.Get<float>();
        diagonal = diagonalBuf.Get<float>();
        rhs = rhsBuf.Get<float>();
        packed = packedBuf.Get<float>();
        piv = pivotBuf.Get<int>();
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        plan = plannedSwap ? planBuf.Get<uint32_t>() : pivotBuf.Get<uint32_t>();
#endif
    }
    __aicore__ inline void LoadPivots(GM_ADDR work, GM_ADDR pivots)
    {
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work));
        p.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
        DataCopyExtParams pc{1, static_cast<uint32_t>(width * sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int> pp{true, 0, static_cast<uint8_t>(Align(width) - width), 0};
        DataCopyPad(piv, p[first], pc, pp);
        Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Prepare()
    {
        for (int k = 0; k < width; ++k)
        {
            if (piv.GetValue(k) != first + k + 1)
            {
                needsSwap = true;
                break;
            }
        }
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        if (needsSwap && plannedSwap)
        {
            Fence<HardEvent::S_V>();
            BuildStagedSwapPlan(plan, piv, columnStride, first, width, stagedColumns, planStride);
            Fence<HardEvent::V_S>();
        }
#endif
#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        LoadPanel(diagonal, w, static_cast<int64_t>(first) * pitch + first, pitch, width, diagonalStride, width);
#else
        for (int k = 0; k < width; ++k)
        {
            Load(diagonal[k * diagonalStride], w[static_cast<int64_t>(first + k) * pitch + first], width);
        }
#endif
    }
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void SwapGroup(int j, int group)
    {
        DataCopyExtParams loadColumns{static_cast<uint16_t>(group), static_cast<uint32_t>(columnRows * sizeof(float)),
                                      static_cast<uint32_t>((pitch - columnRows) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padColumns{true, 0, static_cast<uint8_t>(columnStride - columnRows), 0.0f};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(column, w[static_cast<int64_t>(j) * pitch + first], loadColumns, padColumns);
        Fence<HardEvent::MTE2_V>();
        if (plannedSwap)
        {
            ApplyStagedSwapPlan(column, packed, plan, planStride, width, group);
        }
        else
        {
            SwapStagedColumns(column, piv, columnStride, first, width, group);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams storeColumns{static_cast<uint16_t>(group), static_cast<uint32_t>(columnRows * sizeof(float)),
                                       0, static_cast<uint32_t>((pitch - columnRows) * sizeof(float)), 0};
        DataCopyPad(w[static_cast<int64_t>(j) * pitch + first], column, storeColumns);
        Fence<HardEvent::MTE3_S>();
    }
#endif
    __aicore__ inline void SwapColumns(int begin, int count)
    {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
#if defined(REAL_LU_REGISTER_LAYOUT_950)
        // Completed left columns are restored once by the final unpack.
        int j = begin > first + width ? begin : first + width;
#else
        int j = begin;
#endif
        while (j < begin + count)
        {
            if (j >= first && j < first + width)
            {
                j = first + width;
                continue;
            }
            int end = begin + count;
            if (j < first && first < end)
            {
                end = first;
            }
            int group = end - j < stagedColumns ? end - j : stagedColumns;
            SwapGroup(j, group);
            j += group;
        }
#else
        for (int c = 0; c < count; ++c)
        {
            int j = begin + c;
            if (j >= first && j < first + width)
            {
                continue;
            }
            Load(column, w[static_cast<int64_t>(j) * pitch], m);
            for (int k = 0; k < width; ++k)
            {
                Swap(column, first + k, piv.GetValue(k) - 1);
            }
            Store(w[static_cast<int64_t>(j) * pitch], column, m);
        }
#endif
    }
    __aicore__ inline void SolveRight(int count, int activeCount, int columnOffset)
    {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        Fence<HardEvent::MTE2_V>();
        TransposePanelRight<true>(reinterpret_cast<__ubuf__ float *>(rhs.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float *>(packed.GetPhyAddr()), width, diagonalStride,
                                  columnOffset, activeCount);
        SolvePanelLocal(rhs, diagonal, width, diagonalStride, count);
#else
        Fence<HardEvent::MTE2_S>();
        for (int c = 0; c < activeCount; ++c)
        {
            for (int r = 0; r < width; ++r)
            {
                rhs.SetValue(r * RHS_WIDTH + c + columnOffset, packed.GetValue(c * diagonalStride + r));
            }
        }
        Fence<HardEvent::S_V>();
        for (int k = 0; k < width; ++k)
        {
            for (int r = k + 1; r < width; ++r)
            {
                Axpy(rhs[r * RHS_WIDTH], rhs[k * RHS_WIDTH], -diagonal.GetValue(k * diagonalStride + r), count);
            }
            PipeBarrier<PIPE_V>();
        }
#endif
    }
    __aicore__ inline void PackRight(int activeCount, int columnOffset)
    {
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        TransposePanelRight<false>(reinterpret_cast<__ubuf__ float *>(rhs.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ float *>(packed.GetPhyAddr()), width, diagonalStride,
                                   columnOffset, activeCount);
        Fence<HardEvent::V_MTE3>();
#else
        Fence<HardEvent::V_S>();
        for (int c = 0; c < activeCount; ++c)
        {
            for (int r = 0; r < width; ++r)
            {
                packed.SetValue(c * diagonalStride + r, rhs.GetValue(r * RHS_WIDTH + c + columnOffset));
            }
        }
        Fence<HardEvent::S_MTE3>();
#endif
    }
    __aicore__ inline void SolveTile(int begin, int count, int activeBegin, int activeCount)
    {
        int columnOffset = activeBegin - begin;
        Duplicate(rhs, 0.0f, width * RHS_WIDTH);
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams load{static_cast<uint16_t>(activeCount), static_cast<uint32_t>(width * sizeof(float)),
                               static_cast<uint32_t>((pitch - width) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(diagonalStride - width), 0.0f};
        int64_t offset = static_cast<int64_t>(activeBegin) * pitch + first;
        DataCopyPad(packed, w[offset], load, pad);
        SolveRight(count, activeCount, columnOffset);
        PackRight(activeCount, columnOffset);
        DataCopyExtParams store{static_cast<uint16_t>(activeCount), static_cast<uint32_t>(width * sizeof(float)), 0,
                                static_cast<uint32_t>((pitch - width) * sizeof(float)), 0};
        DataCopyPad(w[offset], packed, store);
        Fence<HardEvent::MTE3_S>();
    }
    static constexpr int RHS_WIDTH = 32;
    int m, n, pitch, first, width;
    int diagonalStride = 0, columnRows = 0, columnStride = 0, stagedColumns = 1, planStride = 0;
    bool needsSwap = false, plannedSwap = false;
    TBuf<TPosition::VECCALC> planBuf, columnBuf, diagonalBuf, rhsBuf, packedBuf, pivotBuf;
    LocalTensor<float> column, diagonal, rhs, packed;
    LocalTensor<int> piv;
    LocalTensor<uint32_t> plan;
    GlobalTensor<float> w;
    GlobalTensor<int> p;
};
}  // namespace

__global__ __aicore__ void real_lu_swap_trsm_kernel(GM_ADDR work, GM_ADDR pivots, int m, int n, int pitch, int first,
                                                    int width)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (m <= first || n <= 0 || pitch <= 0 || width <= 0)
    {
        return;
    }
    TPipe pipe;
    FactorSwapTrsm op(m, n, pitch, first, width);
    if (op.Init(pipe, work, pivots))
    {
        op.Run();
    }
}

// Column-major C=A*B is row-major C^T=B^T*A^T. The M and N tails therefore
// correspond to original columns and rows, respectively.
__global__ __aicore__ void real_lu_factor_product_kernel(GM_ADDR work, GM_ADDR product, int m, int n, int pitch,
                                                         int first, int width, bool accumulate,
                                                         AscendC::tiling::TCubeTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if ASCEND_IS_AIC
    {
        TPipe pipe;
        SetHF32Mode(HF32Mode::DISABLE);
        using Matrix = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        MatmulImpl<Matrix, Matrix, Matrix> multiply;
        multiply.Init(&tiling, &pipe);
        multiply.SetHF32(false, 0);
        multiply.SetOrgShape(pitch, pitch, pitch);
        multiply.SetSingleShape(128, 128, width);
        GlobalTensor<float> w, c;
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(product));
        int begin = first + width, rowTiles = (m - begin + 127) / 128;
        if (rowTiles <= 0)
        {
            return;
        }
        int columnTiles = (n - begin + 127) / 128;
        for (int tile = GetBlockIdx(); tile < rowTiles * columnTiles; tile += GetBlockNum())
        {
            int row = begin + (tile % rowTiles) * 128;
            int column = begin + (tile / rowTiles) * 128;
            multiply.SetTensorA(w[static_cast<int64_t>(column) * pitch + first]);
            multiply.SetTensorB((accumulate ? c : w)[static_cast<int64_t>(first) * pitch + row]);
            int tailM = n - column < 128 ? n - column : 128;
            int tailN = m - row < 128 ? m - row : 128;
            multiply.SetTail(tailM, tailN, width);
            // The full dot product is rounded once before adding it to the
            // trailing matrix. Distinct tiles never share output elements.
            multiply.IterateAll((accumulate ? w : c)[static_cast<int64_t>(column) * pitch + row], accumulate ? 1 : 0);
            multiply.End();
        }
    }
}

__global__ __aicore__ void real_lu_factor_subtract_kernel(GM_ADDR work, GM_ADDR product, int m, int n, int pitch,
                                                          int begin, int columnCapacity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (columnCapacity == 0)
    {
        return;
    }
    if (columnCapacity < 0)
    {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> aBuf, bBuf;
    int count = m - begin, stride = Align(count);
    pipe.InitBuffer(aBuf, stride * columnCapacity * sizeof(float));
    pipe.InitBuffer(bBuf, stride * columnCapacity * sizeof(float));
    auto a = aBuf.Get<float>(), b = bBuf.Get<float>();
    GlobalTensor<float> w, p;
    w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work));
    p.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(product));
    int tiles = (n - begin + columnCapacity - 1) / columnCapacity;
    for (int tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum())
    {
        int j = begin + tile * columnCapacity;
        int columns = n - j < columnCapacity ? n - j : columnCapacity;
        int64_t offset = static_cast<int64_t>(j) * pitch + begin;
        DataCopyExtParams load{static_cast<uint16_t>(columns), static_cast<uint32_t>(count * sizeof(float)),
                               static_cast<uint32_t>((pitch - count) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(stride - count), 0.0f};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(a, w[offset], load, pad);
        DataCopyPad(b, p[offset], load, pad);
        Fence<HardEvent::MTE2_V>();
        Sub(a, a, b, columns * stride);
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams store{static_cast<uint16_t>(columns), static_cast<uint32_t>(count * sizeof(float)), 0,
                                static_cast<uint32_t>((pitch - count) * sizeof(float)), 0};
        DataCopyPad(w[offset], a, store);
        Fence<HardEvent::MTE3_S>();
    }
}

#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)

__aicore__ inline int DeferredPanelWidth950(int m, int steps, int first)
{
    int stride = (m - first + 63) / 64 * 64;
    int width = 8;
    while (width < 128)
    {
        int candidate = width * 2;
        int bytes = (stride * (candidate + 2) + Align(candidate) + 8) * sizeof(float);
        if (bytes > PANEL_950_LOCAL_BYTES)
        {
            break;
        }
        width = candidate;
    }
    int remaining = steps - first;
    return width < remaining ? width : remaining;
}

__aicore__ inline void InitializeDeferredMap950(LocalTensor<uint32_t> rowMap, int rows)
{
    auto target = reinterpret_cast<__ubuf__ uint32_t *>(rowMap.GetPhyAddr());
    uint16_t padded = (rows + 63) / 64 * 64;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> index;
        auto full = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(index), 0);
        for (uint16_t r = 0; r < padded; r += 64)
        {
            MicroAPI::StoreAlign<uint32_t>(target + r, index, full);
            MicroAPI::Adds<uint32_t>(index, index, 64u, full);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void ScatterDeferredColumn950(LocalTensor<float> cache, LocalTensor<float> data,
                                                LocalTensor<uint32_t> rowMap, int rows)
{
    auto input = reinterpret_cast<__ubuf__ float *>(data.GetPhyAddr());
    auto output = reinterpret_cast<__ubuf__ float *>(cache.GetPhyAddr());
    auto mapping = reinterpret_cast<__ubuf__ uint32_t *>(rowMap.GetPhyAddr());
    uint16_t limit = rows;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> value;
        MicroAPI::RegTensor<uint32_t> index;
        uint32_t remaining = rows;
        for (uint16_t r = 0; r < limit; r += 64)
        {
            auto valid = MicroAPI::UpdateMask<float>(remaining);
            // All three buffers reserve a complete last 64-lane register.
            MicroAPI::LoadAlign<float>(value, input + r);
            MicroAPI::LoadAlign<uint32_t>(index, mapping + r);
            MicroAPI::Scatter(output, value, index, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}
#endif

namespace
{
class FactorUnpack
{
   public:
    __aicore__ inline FactorUnpack(int m, int n, int lda, int pitch) : m(m), n(n), lda(lda), pitch(pitch) {}
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR work, GM_ADDR output, GM_ADDR outPivots)
    {
#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        steps = m < n ? m : n;
        deferred = outPivots != nullptr && steps > 0 && steps > DeferredPanelWidth950(m, steps, 0);
        localCount = deferred ? (m + 63) / 64 * 64 : Align(m);
#else
        localCount = Align(m);
#endif
        pipe.InitBuffer(buffer, localCount * sizeof(float));
        local = buffer.Get<float>();
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work));
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
    }
    __aicore__ inline void Run(TPipe &pipe, GM_ADDR pivots, GM_ADDR outPivots)
    {
#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        if (deferred)
        {
            InitDeferred(pipe, pivots);
            RunDeferred();
        }
        else
#else
        (void)pipe;
#endif
        {
            RunPlain();
        }
        CopyPivots(pivots, outPivots);
    }

   private:
#if defined(REAL_LU_REGISTER_LAYOUT_950) && (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void InitDeferred(TPipe &pipe, GM_ADDR pivots)
    {
        pipe.InitBuffer(cacheBuf, localCount * sizeof(float));
        pipe.InitBuffer(mapBuf, localCount * sizeof(uint32_t));
        pipe.InitBuffer(pivotBuf, Align(steps) * sizeof(int));
        pipe.InitBuffer(endBuf, Align((steps + 7) / 8) * sizeof(int));
        cache = cacheBuf.Get<float>();
        rowMap = mapBuf.Get<uint32_t>();
        localPivots = pivotBuf.Get<int>();
        panelEnds = endBuf.Get<int>();
        GlobalTensor<int> pivotSource;
        pivotSource.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
        DataCopyExtParams copy{1, static_cast<uint32_t>(steps * sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int> pad{true, 0, static_cast<uint8_t>(Align(steps) - steps), 0};
        DataCopyPad(localPivots, pivotSource, copy, pad);
        Fence<HardEvent::MTE2_S>();
        panelCount = 0;
        for (int first = 0; first < steps;)
        {
            first += DeferredPanelWidth950(m, steps, first);
            panelEnds.SetValue(panelCount++, first);
        }
        InitializeDeferredMap950(rowMap, m);
        Fence<HardEvent::V_S>();
    }
    __aicore__ inline int PanelStop(int j, int &panelIndex)
    {
        int stop = steps;
        if (j < steps)
        {
            while (panelIndex > 0 && j < panelEnds.GetValue(panelIndex - 1))
            {
                --panelIndex;
            }
            stop = panelEnds.GetValue(panelIndex);
        }
        return stop;
    }
    __aicore__ inline void ApplyMap(int &next, int stop)
    {
        while (next > stop)
        {
            int k = --next;
            int selected = localPivots.GetValue(k) - 1;
            if (selected != k)
            {
                uint32_t a = rowMap.GetValue(k), b = rowMap.GetValue(selected);
                rowMap.SetValue(k, b);
                rowMap.SetValue(selected, a);
            }
        }
    }
    __aicore__ inline void RunDeferred()
    {
        int next = steps, panelIndex = panelCount - 1;
        int block = static_cast<int>(GetBlockIdx()), blocks = static_cast<int>(GetBlockNum());
        for (int j = n - 1 - block; j >= 0; j -= blocks)
        {
            int stop = PanelStop(j, panelIndex);
            ApplyMap(next, stop);
            Load(local, w[static_cast<int64_t>(j) * pitch], m);
            if (stop == steps)
            {
                Store(a[static_cast<int64_t>(j) * lda], local, m);
            }
            else
            {
                // Load's MTE2->S and this S->V publish data and the scalar map.
                Fence<HardEvent::S_V>();
                ScatterDeferredColumn950(cache, local, rowMap, m);
                // Complete every map read before the next scalar map update.
                Fence<HardEvent::V_S>();
                Store(a[static_cast<int64_t>(j) * lda], cache, m);
            }
        }
    }
#endif
    __aicore__ inline void RunPlain()
    {
        for (int j = GetBlockIdx(); j < n; j += GetBlockNum())
        {
            Load(local, w[static_cast<int64_t>(j) * pitch], m);
            Store(a[static_cast<int64_t>(j) * lda], local, m);
        }
    }
    __aicore__ inline void CopyPivots(GM_ADDR pivots, GM_ADDR outPivots)
    {
        if (GetBlockIdx() == 0 && outPivots != nullptr)
        {
            GlobalTensor<float> p, out;
            p.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pivots));
            out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outPivots));
            int count = m < n ? m : n;
            Load(local, p, count);
            Store(out, local, count);
        }
    }
    int m, n, lda, pitch, localCount = 0, steps = 0, panelCount = 0;
    bool deferred = false;
    TBuf<TPosition::VECCALC> buffer, cacheBuf, mapBuf, pivotBuf, endBuf;
    LocalTensor<float> local, cache;
    LocalTensor<uint32_t> rowMap;
    LocalTensor<int> localPivots, panelEnds;
    GlobalTensor<float> w, a;
};
}  // namespace

__global__ __aicore__ void real_lu_unpack_factor_kernel(GM_ADDR work, GM_ADDR output, GM_ADDR pivots, GM_ADDR outPivots,
                                                        int m, int n, int lda, int pitch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    FactorUnpack op(m, n, lda, pitch);
    op.Init(pipe, work, output, outPivots);
    op.Run(pipe, pivots, outPivots);
}

namespace
{
class BlockedFactorLaunch
{
   public:
    BlockedFactorLaunch(GM_ADDR a, GM_ADDR workspace, GM_ADDR piv, GM_ADDR info, int m, int n, int lda,
                        aclrtStream stream)
        : a(a), workspace(workspace), piv(piv), info(info), m(m), n(n), lda(lda), stream(stream)
    {
        pitch = (m + 127) / 128 * 128 + 128;
        columns = (n + 127) / 128 * 128 + 128;
        int64_t size = static_cast<int64_t>(pitch) * columns;
        product = workspace + size * sizeof(float);
        internalPiv = product + size * sizeof(float);
        vectorCores = real_lu_vector_cores();
        cubeCores = real_lu_cube_cores();
    }
    int InitTilings()
    {
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform == nullptr)
        {
            return -1;
        }
        for (int index = 0; index < 5; ++index)
        {
            int width = 8 << index;
            matmul_tiling::MatmulApiTiling tiler(*platform);
            tiler.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
            tiler.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
            tiler.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
            tiler.SetShape(128, 128, width);
            tiler.SetOrgShape(pitch, pitch, pitch);
            tiler.SetBias(false);
            tiler.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
            tiler.SetFixSplit(128, 128, std::min(64, width));
            if (tiler.GetTiling(cubeTilings[index]) != 0)
            {
                return -1;
            }
        }
        return 0;
    }
    int Run()
    {
        real_lu_pack_factor_kernel<<<std::min(vectorCores, columns), nullptr, stream>>>(a, workspace, info, m, n, lda,
                                                                                        pitch, columns);
        int steps = std::min(m, n);
        for (int first = 0; first < steps;)
        {
            int width = PanelWidth(first, steps);
            if (width <= 0)
            {
                return -1;
            }
            GM_ADDR negative = nullptr;
#if defined(REAL_LU_REGISTER_LAYOUT_950)
            negative = product;
#endif
            real_lu_panel_kernel<<<1, nullptr, stream>>>(workspace, negative, internalPiv, info, m, pitch, first, width,
                                                         piv != nullptr);
            real_lu_swap_trsm_kernel<<<std::min(vectorCores, (n + 31) / 32), nullptr, stream>>>(
                workspace, internalPiv, m, n, pitch, first, width);
            if (first + width < m && first + width < n && Update(first, width, negative) != 0)
            {
                return -1;
            }
            first += width;
        }

        real_lu_unpack_factor_kernel<<<std::min(vectorCores, n), nullptr, stream>>>(workspace, a, internalPiv, piv, m,
                                                                                    n, lda, pitch);
        return 0;
    }

   private:
    int PanelWidth(int first, int steps)
    {
        int width = 8;
#if defined(REAL_LU_REGISTER_LAYOUT_950)
        int panelStride = (m - first + 63) / 64 * 64;
        auto fitsPanel = [panelStride](int candidate)
        {
            // Match the device's panel, column, scratch, pivot and info TBufs.
            int floats = panelStride * (candidate + 2) + (candidate + 7) / 8 * 8 + 8;
            return floats * sizeof(float) <= PANEL_950_LOCAL_BYTES;
        };
        while (width < 128 && fitsPanel(width * 2))
        {
            width *= 2;
        }
#else
        int alignedRows = (m - first + 7) / 8 * 8;
        if (alignedRows <= 0)
        {
            return 0;
        }
        int capacity = 32768 / alignedRows;
        while (width * 2 <= capacity && width < 128)
        {
            width *= 2;
        }
#endif
        return std::min(width, steps - first);
    }
    int Subtract(int first, int width)
    {
#if defined(REAL_LU_REGISTER_LAYOUT_950)
        constexpr int floatsPerBuffer = 28672;
#else
        constexpr int floatsPerBuffer = 16384;
#endif

        int remainingRows = m - first - width;
        int stride = (remainingRows + 7) / 8 * 8;
        if (stride <= 0)
        {
            return -1;
        }
        int columns = std::min(n - first - width, std::min(32, floatsPerBuffer / stride));
        if (columns == 0)
        {
            return -1;
        }
        if (columns < 0)
        {
            return -1;
        }
        int updateTiles = (n - first - width + columns - 1) / columns;
        real_lu_factor_subtract_kernel<<<std::min(vectorCores, updateTiles), nullptr, stream>>>(
            workspace, product, m, n, pitch, first + width, columns);
        return 0;
    }
    int Update(int first, int width, GM_ADDR negative)
    {
        int jobs = ((m - first - width + 127) / 128) * ((n - first - width + 127) / 128);
        int index = 0;
        for (int value = width; value > 8; value /= 2)
        {
            ++index;
        }
        real_lu_factor_product_kernel<<<std::min(cubeCores, jobs), nullptr, stream>>>(
            workspace, product, m, n, pitch, first, width, negative != nullptr, cubeTilings[index]);
        // Two column batches fit in UB while leaving room for compiler
        // temporaries. Each valid element retains the same single Sub.
        if (negative == nullptr)
        {
            return Subtract(first, width);
        }
        return 0;
    }
    GM_ADDR a;
    GM_ADDR workspace;
    GM_ADDR piv;
    GM_ADDR info;
    GM_ADDR product = nullptr;
    GM_ADDR internalPiv = nullptr;
    int m, n, lda, pitch = 0, columns = 0, vectorCores = 0, cubeCores = 0;
    aclrtStream stream;
    AscendC::tiling::TCubeTiling cubeTilings[5];
};
}  // namespace

extern "C" int real_lu_blocked_factor_do(GM_ADDR a, GM_ADDR workspace, GM_ADDR piv, GM_ADDR info, int m, int n, int lda,
                                         aclrtStream stream)
{
    if (m <= 0 || n <= 0 || lda <= 0)
    {
        return -1;
    }
    BlockedFactorLaunch op(a, workspace, piv, info, m, n, lda, stream);
    return op.InitTilings() == 0 ? op.Run() : -1;
}

// The inverse uses compact 64x64 tiles. Diagonal inverses are stored row-major
// and read transposed; all other work matrices use column-major storage.
template <bool TransposeB>
__global__ __aicore__ void real_lu_gemm_kernel(GM_ADDR inputA, GM_ADDR inputB, GM_ADDR output, GM_ADDR info, int pitch,
                                               int rows, int columns, int depth, int64_t matrixStride, int batch,
                                               int64_t offsetA, int64_t offsetB, int64_t offsetC,
                                               AscendC::tiling::TCubeTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if ASCEND_IS_AIC
    {
        TPipe pipe;
        using AType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, TransposeB>;
        MatmulImpl<AType, BType, AType> multiply;
        multiply.Init(&tiling, &pipe);
        multiply.SetHF32(false, 0);
        multiply.SetOrgShape(pitch, pitch, pitch, pitch, pitch);
        multiply.SetSingleShape(64, 64, depth);
        GlobalTensor<int> states;
        states.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
        int rowTiles = (rows + 63) / 64, columnTiles = (columns + 63) / 64;
        int tiles = rowTiles * columnTiles;
        if (rowTiles <= 0 || tiles <= 0)
        {
            return;
        }
        int64_t jobs = static_cast<int64_t>(tiles) * batch;
        for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum())
        {
            int matrix = job / tiles, tile = job % tiles;
            if (states.GetValue(matrix) != 0)
            {
                continue;
            }
            int row = tile % rowTiles * 64, column = tile / rowTiles * 64;
            GlobalTensor<float> a, b, c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputA) + matrixStride * matrix);
            b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputB) + matrixStride * matrix);
            c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + matrixStride * matrix);
            multiply.SetTensorA(a[offsetA + static_cast<int64_t>(row) * pitch]);
            multiply.SetTensorB(b[offsetB + (TransposeB ? static_cast<int64_t>(column) * pitch : column)], TransposeB);
            int tailM = rows - row < 64 ? rows - row : 64;
            int tailN = columns - column < 64 ? columns - column : 64;
            multiply.SetTail(tailM, tailN, depth);
            multiply.IterateAll(c[offsetC + static_cast<int64_t>(row) * pitch + column]);
            multiply.End();
        }
    }
}

extern "C" int real_lu_gemm_do(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR info, int pitch, int rows, int columns,
                               int depth, int64_t matrixStride, int batch, int64_t offsetA, int64_t offsetB,
                               int64_t offsetC, bool transposeB, aclrtStream stream)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr)
    {
        return -1;
    }
    matmul_tiling::MatmulApiTiling tiler(*platform);
    tiler.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiler.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                   transposeB);
    tiler.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiler.SetShape(64, 64, depth);
    tiler.SetOrgShape(pitch, pitch, pitch, pitch);
    tiler.SetBias(false);
    tiler.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    tiler.SetFixSplit(64, 64, std::min(64, depth));
    AscendC::tiling::TCubeTiling tiling;
    if (tiler.GetTiling(tiling) != 0)
    {
        return -1;
    }
    int64_t jobs = static_cast<int64_t>((rows + 63) / 64) * ((columns + 63) / 64) * batch;
    int blocks = std::min<int64_t>(real_lu_cube_cores(), jobs);
    if (transposeB)
    {
        real_lu_gemm_kernel<true><<<blocks, nullptr, stream>>>(a, b, c, info, pitch, rows, columns, depth, matrixStride,
                                                               batch, offsetA, offsetB, offsetC, tiling);
    }
    else
    {
        real_lu_gemm_kernel<false><<<blocks, nullptr, stream>>>(a, b, c, info, pitch, rows, columns, depth,
                                                                matrixStride, batch, offsetA, offsetB, offsetC, tiling);
    }
    return 0;
}
