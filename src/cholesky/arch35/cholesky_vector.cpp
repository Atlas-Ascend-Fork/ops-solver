/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cholesky_internal.h"
#include "kernel_operator.h"
#include "transpose_transdata_5hd_021axis.h"

namespace Cholesky
{
constexpr int DENSE_PITCH = 128 + 8;
constexpr int VECTOR_PITCH = PANEL + 8;  // A 32-byte pad distributes strided gathers across UB banks.

using LocalFloat = AscendC::LocalTensor<float>;
using LocalInt = AscendC::LocalTensor<int32_t>;
using GlobalFloat = AscendC::GlobalTensor<float>;
using GlobalInt = AscendC::GlobalTensor<int>;

__aicore__ inline __gm__ float *MatrixPointer(GM_ADDR single, GM_ADDR arrayAddr, int64_t index)
{
    if (!arrayAddr) return (__gm__ float *)single;
    AscendC::GlobalTensor<uint64_t> array;
    array.SetGlobalBuffer((__gm__ uint64_t *)arrayAddr);
    return (__gm__ float *)array.GetValue(index);
}

template <bool Reverse, bool TransposedTri, bool AccumulateDot, typename RegF, typename RegI>
__simd_callee__ void UpdateTriSegment(__ubuf__ float *tri, RegF &values, RegF &solved, RegF &dot, RegI &lanes,
                                      uint32_t width, uint32_t pitch, uint32_t k, uint32_t r)
{
    using namespace AscendC::MicroAPI;
    uint32_t length = width - r;
    MaskReg full = UpdateMask<float>(length), active, pivot;
    RegTensor<int32_t> positions;
    Adds(positions, lanes, int32_t(r), full);
    if constexpr (Reverse)
        Compares<int32_t, AscendC::CMPMODE::LT>(active, positions, int32_t(k), full);
    else
        Compares<int32_t, AscendC::CMPMODE::GT>(active, positions, int32_t(k), full);
    Compares<int32_t, AscendC::CMPMODE::EQ>(pivot, positions, int32_t(k), full);
    RegTensor<float> coefficient, product, updated;
    if constexpr (TransposedTri != Reverse)
        LoadAlign(coefficient, tri + k * pitch + r);
    else
    {
        RegTensor<uint32_t> offsets;
        Muls((RegTensor<int32_t> &)offsets, lanes, int32_t(pitch), full);
        Gather(coefficient, tri + r * pitch + k, offsets, active);
    }
    if constexpr (AccumulateDot)
    {
        FusedMulDstAdd(coefficient, solved, dot, active);
        Select(dot, coefficient, dot, active);
    }
    else
    {
        Mul(product, coefficient, solved, active);
        Sub(updated, values, product, active);
        Select(values, updated, values, active);
    }
    Select(values, solved, values, pivot);
}

// The complete RHS stays in registers throughout a triangular solve.
template <bool Reverse, bool TransposedTri, bool AccumulateDot = false>
__simd_vf__ void VectorTriSolve(__ubuf__ float *rhs, __ubuf__ float *tri, uint32_t width, uint32_t pitch)
{
    using namespace AscendC::MicroAPI;
    RegTensor<int32_t> lanes;
    Arange(lanes, int32_t(0));
    RegTensor<float> values0, values1, dot0, dot1;
    LoadAlign(values0, rhs);
    if (width > 64) LoadAlign(values1, rhs + 64);
    if constexpr (AccumulateDot)
    {
        Duplicate(dot0, 0.f);
        Duplicate(dot1, 0.f);
    }
    for (uint32_t step = 0; step < width; ++step)
    {
        uint32_t k = Reverse ? width - 1 - step : step;
        RegTensor<float> solved, diagonal;
        RegTensor<uint32_t> selected;
        Duplicate(selected, k % 64);
        if (k < 64)
            Gather(solved, values0, selected);
        else
            Gather(solved, values1, selected);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(diagonal, tri + k * pitch + k);
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        if constexpr (AccumulateDot)
        {
            // Subtract the dot product once from the original RHS entry.
            RegTensor<float> sum;
            if (k < 64)
                Gather(sum, dot0, selected);
            else
                Gather(sum, dot1, selected);
            Sub(solved, solved, sum, all);
        }
        Div(solved, solved, diagonal, all);
        UpdateTriSegment<Reverse, TransposedTri, AccumulateDot>(tri, values0, solved, dot0, lanes, width, pitch, k, 0);
        if (width > 64)
            UpdateTriSegment<Reverse, TransposedTri, AccumulateDot>(tri, values1, solved, dot1, lanes, width, pitch, k,
                                                                    64);
    }
    uint32_t length0 = width;
    MaskReg first = UpdateMask<float>(length0);
    StoreAlign(rhs, values0, first);
    if (width > 64)
    {
        uint32_t length1 = width - 64;
        MaskReg second = UpdateMask<float>(length1);
        StoreAlign(rhs + 64, values1, second);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename FloatReg, typename IntReg, typename VectorMask>
__simd_callee__ void CheckPanelPivot(FloatReg &diag, IntReg &failure, VectorMask &healthy, VectorMask &full, uint32_t k)
{
    using namespace AscendC::MicroAPI;
    MaskReg positive, finite, unfinished, bad;
    RegTensor<int32_t> candidate;
    Compares<float, AscendC::CMPMODE::GT>(positive, diag, 0.f, full);
    Compares<float, AscendC::CMPMODE::LE>(finite, diag, 3.402823466e38F, full);
    And(healthy, positive, finite, full);
    Compares<int32_t, AscendC::CMPMODE::EQ>(unfinished, failure, int32_t(0), full);
    Not(bad, healthy, full);
    And(bad, bad, unfinished, full);
    Duplicate(candidate, int32_t(k + 1));
    Select(failure, candidate, failure, bad);
    And(healthy, healthy, unfinished, full);
}

template <bool Transposed, typename FloatReg, typename IndexReg, typename VectorMask>
__simd_callee__ void UpdatePanelColumn(__ubuf__ float *tile, FloatReg &column, IndexReg &indices, VectorMask &stored,
                                       uint32_t k, uint32_t j)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> current, alpha, product;
    if constexpr (Transposed)
    {
        LoadAlign(current, tile + j * VECTOR_PITCH);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, tile + k * VECTOR_PITCH + j);
    }
    else
    {
        Gather(current, tile + j, indices, stored);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, tile + j * VECTOR_PITCH + k);
    }
    Mul(product, column, alpha, stored);
    Sub(current, current, product, stored);
    if constexpr (Transposed)
        StoreAlign(tile + j * VECTOR_PITCH, current, stored);
    else
        Scatter(tile + j, current, indices, stored);
}

// Keep all pivots in one VF. Status is read by the scalar controller once.
template <bool Transposed>
__simd_vf__ void PanelFactor(__ubuf__ float *tile, __ubuf__ int32_t *status, uint32_t width)
{
    using namespace AscendC::MicroAPI;
    uint32_t validRows = width;
    MaskReg full = UpdateMask<float>(validRows);
    RegTensor<uint32_t> indices;
    Arange((RegTensor<int32_t> &)indices, int32_t(0));
    Muls(indices, indices, uint32_t(VECTOR_PITCH), full);
    RegTensor<int32_t> failure, candidate;
    Duplicate(failure, int32_t(0));
    for (uint32_t k = 0; k < width; ++k)
    {
        RegTensor<float> diag, column, current, alpha, product;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(diag, tile + k * VECTOR_PITCH + k);
        MaskReg healthy;
        CheckPanelPivot(diag, failure, healthy, full, k);
        uint32_t previous = k;
        MaskReg prefix = UpdateMask<float>(previous), active;
        Not(active, prefix, full);
        And(active, active, healthy, full);
        if constexpr (Transposed)
            LoadAlign(column, tile + k * VECTOR_PITCH);
        else
            Gather(column, tile + k, indices, active);
        Sqrt(diag, diag, active);
        Div(column, column, diag, active);
        MaskReg pivot;
        Compares<int32_t, AscendC::CMPMODE::EQ>(pivot, (RegTensor<int32_t> &)indices, int32_t(k * VECTOR_PITCH),
                                                active);
        Select(column, diag, column, pivot);
        if constexpr (Transposed)
            StoreAlign(tile + k * VECTOR_PITCH, column, active);
        else
            Scatter(tile + k, column, indices, active);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint32_t j = k + 1; j < width; ++j)
        {
            uint32_t previousRows = j;
            MaskReg prefixJ = UpdateMask<float>(previousRows), stored;
            Not(stored, prefixJ, full);
            And(stored, stored, healthy, full);
            UpdatePanelColumn<Transposed>(tile, column, indices, stored, k, j);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    uint32_t one = 1;
    MaskReg first = UpdateMask<int32_t>(one);
    StoreAlign(status, failure, first);
}

// Keep all pivots in one VF. Status is read by the scalar controller once.
template <bool Transposed, typename FloatReg, typename IndexReg, typename VectorMask>
__simd_callee__ void UpdatePanelFourRow(__ubuf__ float *tile, IndexReg &indices, VectorMask &stored, FloatReg &column0,
                                        FloatReg &column1, FloatReg &column2, FloatReg &column3, uint32_t base,
                                        uint32_t j)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> current, alpha, product, sum;
    if constexpr (Transposed)
        LoadAlign(current, tile + j * VECTOR_PITCH);
    else
        Gather(current, tile + j, indices, stored);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha,
                                             tile + (Transposed ? base * VECTOR_PITCH + j : j * VECTOR_PITCH + base));
    Mul(sum, column0, alpha, stored);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(
        alpha, tile + (Transposed ? (base + 1) * VECTOR_PITCH + j : j * VECTOR_PITCH + base + 1));
    Mul(product, column1, alpha, stored);
    Add(sum, sum, product, stored);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(
        alpha, tile + (Transposed ? (base + 2) * VECTOR_PITCH + j : j * VECTOR_PITCH + base + 2));
    Mul(product, column2, alpha, stored);
    Add(sum, sum, product, stored);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(
        alpha, tile + (Transposed ? (base + 3) * VECTOR_PITCH + j : j * VECTOR_PITCH + base + 3));
    Mul(product, column3, alpha, stored);
    Add(sum, sum, product, stored);
    // Preserve low-order contributions before updating a larger diagonal.
    Sub(current, current, sum, stored);
    if constexpr (Transposed)
        StoreAlign(tile + j * VECTOR_PITCH, current, stored);
    else
        Scatter(tile + j, current, indices, stored);
}

template <bool Transposed, typename IndexReg, typename VectorMask>
__simd_callee__ void UpdatePanelFour(__ubuf__ float *tile, IndexReg &indices, VectorMask &blockHealthy,
                                     VectorMask &full, uint32_t base, uint32_t end, uint32_t width)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> column0, column1, column2, column3;
    if constexpr (Transposed)
    {
        LoadAlign(column0, tile + base * VECTOR_PITCH);
        LoadAlign(column1, tile + (base + 1) * VECTOR_PITCH);
        LoadAlign(column2, tile + (base + 2) * VECTOR_PITCH);
        LoadAlign(column3, tile + (base + 3) * VECTOR_PITCH);
    }
    else
    {
        Gather(column0, tile + base, indices, full);
        Gather(column1, tile + base + 1, indices, full);
        Gather(column2, tile + base + 2, indices, full);
        Gather(column3, tile + base + 3, indices, full);
    }
    RegTensor<int32_t> lanes;
    Arange(lanes, int32_t(0));
    for (uint32_t j = end; j < width; ++j)
    {
        MaskReg stored;
        Compares<int32_t, AscendC::CMPMODE::GE>(stored, lanes, int32_t(j), blockHealthy);
        UpdatePanelFourRow<Transposed>(tile, indices, stored, column0, column1, column2, column3, base, j);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool Transposed, typename Index, typename Failure, typename VectorMask>
__simd_callee__ void PanelFactorPivot(__ubuf__ float *tile, Index &indices, Failure &failure, VectorMask &healthy,
                                      VectorMask &full, uint32_t k, uint32_t end)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> diag, column;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(diag, tile + k * VECTOR_PITCH + k);
    CheckPanelPivot(diag, failure, healthy, full, k);
    MaskReg active;
    RegTensor<int32_t> lanes;
    Arange(lanes, int32_t(0));
    Compares<int32_t, AscendC::CMPMODE::GE>(active, lanes, int32_t(k), healthy);
    if constexpr (Transposed)
        LoadAlign(column, tile + k * VECTOR_PITCH);
    else
        Gather(column, tile + k, indices, active);
    Sqrt(diag, diag, active);
    Div(column, column, diag, active);
    MaskReg pivotMask;
    Compares<int32_t, AscendC::CMPMODE::EQ>(pivotMask, lanes, int32_t(k), active);
    Select(column, diag, column, pivotMask);
    if constexpr (Transposed)
        StoreAlign(tile + k * VECTOR_PITCH, column, active);
    else
        Scatter(tile + k, column, indices, active);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint32_t j = k + 1; j < end; ++j)
    {
        MaskReg stored;
        Compares<int32_t, AscendC::CMPMODE::GE>(stored, lanes, int32_t(j), healthy);
        UpdatePanelColumn<Transposed>(tile, column, indices, stored, k, j);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool Transposed>
__simd_vf__ void PanelFactorBlock4(__ubuf__ float *tile, __ubuf__ int32_t *status, uint32_t width)
{
    using namespace AscendC::MicroAPI;
    width = PANEL;
    uint32_t validRows = PANEL;
    MaskReg full = UpdateMask<float>(validRows);
    RegTensor<uint32_t> indices;
    Arange((RegTensor<int32_t> &)indices, int32_t(0));
    Muls(indices, indices, uint32_t(VECTOR_PITCH), full);
    RegTensor<int32_t> failure, candidate;
    Duplicate(failure, int32_t(0));
    for (uint32_t base = 0; base < width; base += 4)
    {
        uint32_t end = base + 4;
        MaskReg blockHealthy;
        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            PanelFactorPivot<Transposed>(tile, indices, failure, blockHealthy, full, base + lane, end);
        }
        UpdatePanelFour<Transposed>(tile, indices, blockHealthy, full, base, end, width);
    }
    uint32_t one = 1;
    MaskReg first = UpdateMask<int32_t>(one);
    StoreAlign(status, failure, first);
}

constexpr int TRANSFER_TILE = 128;
constexpr int TRANSFER_PITCH = TRANSFER_TILE + 8;

__global__ __vector__ void PanelVectorKernel(GM_ADDR wAddr, int pitch, int n, int start, int count, GM_ADDR infoAddr)
{
    using namespace AscendC;
    GlobalTensor<float> w;
    GlobalTensor<int> info;
    w.SetGlobalBuffer((__gm__ float *)wAddr);
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (2 * PANEL * VECTOR_PITCH + 64) * sizeof(float));
    LocalTensor<float> tile = buffer.Get<float>();
    int width = n - start < PANEL ? n - start : PANEL;
    for (int q = GetBlockIdx(); q < count; q += GetBlockNum())
    {
        if (info.GetValue(q)) continue;
        int64_t base = int64_t(q) * pitch * pitch + int64_t(start) * pitch + start;
        DataCopyExtParams load{uint16_t(width), uint32_t(width * 4), uint32_t((pitch - width) * 4),
                               uint32_t((VECTOR_PITCH - (width + 7) / 8 * 8) / 8), 0};
        DataCopyPad(tile, w[base], load, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto transposed = tile[PANEL * VECTOR_PITCH];
        auto status = transposed[PANEL * VECTOR_PITCH].ReinterpretCast<int32_t>();
        auto tp = (__ubuf__ float *)tile.GetPhyAddr();
        auto up = (__ubuf__ float *)transposed.GetPhyAddr();
        TransposeHardware(transposed, tile, width, width, VECTOR_PITCH);
        if (width == PANEL)
            PanelFactorBlock4<true>(up, (__ubuf__ int32_t *)status.GetPhyAddr(), width);
        else
            PanelFactor<true>(up, (__ubuf__ int32_t *)status.GetPhyAddr(), width);
        TransposeHardware(tile, transposed, width, width, VECTOR_PITCH);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        int failure = status.GetValue(0);
        if (failure) failure += start;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // The tile is private workspace. Unpack selects completed columns on failure.
        DataCopyPad(
            w[base], tile,
            DataCopyExtParams{uint16_t(width), uint32_t(width * 4), uint32_t((VECTOR_PITCH - (width + 7) / 8 * 8) / 8),
                              uint32_t((pitch - width) * 4), 0});
        status.SetValue(0, failure);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(info[q], status, DataCopyExtParams{1, uint32_t(sizeof(int32_t)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void Panel(float *w, int pitch, int n, int start, int count, int *info, aclrtStream s)
{
    unsigned cores = VectorCores();
    unsigned blocks = unsigned(count) < cores ? unsigned(count) : cores;
    PanelVectorKernel<<<blocks, nullptr, s>>>((GM_ADDR)w, pitch, n, start, count, (GM_ADDR)info);
}

template <bool LowerInRows>
__simd_vf__ void MergeStoredRows(__ubuf__ float *source, __ubuf__ float *destination, uint32_t rows, uint32_t cols,
                                 int rowFirst, int colFirst)
{
    using namespace AscendC::MicroAPI;
    RegTensor<int32_t> lanes, positions;
    Arange(lanes, int32_t(0));
    for (uint32_t r = 0; r < rows; ++r)
    {
        for (uint32_t c = 0; c < cols; c += 64)
        {
            uint32_t remaining = cols - c;
            MaskReg full = UpdateMask<float>(remaining), selected;
            Adds(positions, lanes, int32_t(c), full);
            if constexpr (LowerInRows)
                Compares<int32_t, AscendC::CMPMODE::LE>(selected, positions, int32_t(r) + rowFirst - colFirst, full);
            else
                Compares<int32_t, AscendC::CMPMODE::GE>(selected, positions, int32_t(r) + rowFirst - colFirst, full);
            RegTensor<float> values, original;
            LoadAlign(values, source + r * TRANSFER_PITCH + c);
            LoadAlign(original, destination + r * TRANSFER_PITCH + c);
            Select(values, values, original, selected);
            StoreAlign(destination + r * TRANSFER_PITCH + c, values, full);
        }
    }
}

struct TransferLayout
{
    int rowFirst, colFirst, rows, cols, externalRows, externalCols, lda, pitch, upper;
    int64_t externalOffset, workOffset;
    uint32_t ubColsStride, ubExternalStride;
};
__aicore__ inline void StoreTransferTile(const GlobalFloat &w, const GlobalFloat &ga, const LocalFloat &source,
                                         const LocalFloat &destination, const LocalFloat &original,
                                         const TransferLayout &t)
{
    using namespace AscendC;
    DataCopyPad(
        source, w[t.workOffset],
        DataCopyExtParams{uint16_t(t.rows), uint32_t(t.cols * 4), uint32_t((t.pitch - t.cols) * 4), t.ubColsStride, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    bool diagonal = t.rowFirst == t.colFirst;
    if (diagonal)
        DataCopyPad(t.upper ? destination : original, ga[t.externalOffset],
                    DataCopyExtParams{uint16_t(t.externalRows), uint32_t(t.externalCols * 4),
                                      uint32_t((t.lda - t.externalCols) * 4), t.ubExternalStride, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto sp = (__ubuf__ float *)source.GetPhyAddr();
    auto dp = (__ubuf__ float *)destination.GetPhyAddr();
    if (t.upper && diagonal) MergeStoredRows<true>(sp, dp, t.rows, t.cols, t.rowFirst, t.colFirst);
    if (!t.upper)
    {
        TransposeHardware(destination, source, t.rows, t.cols, TRANSFER_PITCH);
        if (diagonal)
            MergeStoredRows<false>(dp, (__ubuf__ float *)original.GetPhyAddr(), t.externalRows, t.externalCols,
                                   t.colFirst, t.rowFirst);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    auto output = t.upper && !diagonal ? source : !t.upper && diagonal ? original : destination;
    DataCopyPad(ga[t.externalOffset], output,
                DataCopyExtParams{uint16_t(t.externalRows), uint32_t(t.externalCols * 4), t.ubExternalStride,
                                  uint32_t((t.lda - t.externalCols) * 4), 0});
}

__aicore__ inline void LoadTransferTile(const GlobalFloat &w, const GlobalFloat &ga, const LocalFloat &source,
                                        const LocalFloat &destination, const TransferLayout &t)
{
    using namespace AscendC;
    DataCopyPad(source, ga[t.externalOffset],
                DataCopyExtParams{uint16_t(t.externalRows), uint32_t(t.externalCols * 4),
                                  uint32_t((t.lda - t.externalCols) * 4), t.ubExternalStride, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (!t.upper) TransposeHardware(destination, source, t.externalRows, t.externalCols, TRANSFER_PITCH);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    auto output = t.upper ? source : destination;
    DataCopyPad(
        w[t.workOffset], output,
        DataCopyExtParams{uint16_t(t.rows), uint32_t(t.cols * 4), t.ubColsStride, uint32_t((t.pitch - t.cols) * 4), 0});
}

template <bool Store>
__aicore__ inline int TransferColumns(const GlobalInt &info, const LocalInt &status, __gm__ float *pa, int q, int local,
                                      int n)
{
    using namespace AscendC;
    if constexpr (!Store)
    {
        if (local == 0)
        {
            status.SetValue(0, pa ? 0 : -3);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad(info[q], status, DataCopyExtParams{1, 4, 0, 0, 0});
        }
    }
    int columns = n;
    if constexpr (Store)
    {
        int failure = info.GetValue(q);
        if (failure < 0) return -1;
        if (failure > 0) columns = failure - 1;
    }
    return columns;
}

// Shared pack/unpack DMA path. Only tiles intersecting the stored side are used.
template <bool Store>
__global__ __vector__ void MatrixTransferVectorKernel(GM_ADDR aAddr, GM_ADDR aArrayAddr, int firstA, int n, int lda,
                                                      int upper, int count, GM_ADDR wAddr, int pitch, GM_ADDR infoAddr)
{
    using namespace AscendC;
    GlobalTensor<float> w;
    GlobalTensor<int> info;
    w.SetGlobalBuffer((__gm__ float *)wAddr);
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    constexpr int bufferStride = TRANSFER_TILE * TRANSFER_PITCH + 8;
    pipe.InitBuffer(buffer, (3 * bufferStride + 64) * sizeof(float));
    auto source = buffer.Get<float>();
    auto destination = source[bufferStride];
    auto original = destination[bufferStride];
    auto status = original[bufferStride].ReinterpretCast<int32_t>();
    int tiles = (n + TRANSFER_TILE - 1) / TRANSFER_TILE;
    int perMatrix = tiles * tiles;
    for (int64_t task = GetBlockIdx(); task < int64_t(count) * perMatrix; task += GetBlockNum())
    {
        int q = task / perMatrix, local = task % perMatrix;
        int rowFirst = local / tiles * TRANSFER_TILE, colFirst = local % tiles * TRANSFER_TILE;
        if (colFirst > rowFirst) continue;
        auto pa = MatrixPointer(aAddr, aArrayAddr, int64_t(firstA) + q);
        int columns = TransferColumns<Store>(info, status, pa, q, local, n);
        if (columns < 0) continue;
        if (!pa || colFirst >= columns)
        {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            continue;
        }
        int rows = n - rowFirst < TRANSFER_TILE ? n - rowFirst : TRANSFER_TILE;
        int cols = columns - colFirst < TRANSFER_TILE ? columns - colFirst : TRANSFER_TILE;
        int externalRows = upper ? rows : cols, externalCols = upper ? cols : rows;
        int64_t externalOffset = upper ? int64_t(rowFirst) * lda + colFirst : int64_t(colFirst) * lda + rowFirst;
        int64_t workOffset = int64_t(q) * pitch * pitch + int64_t(rowFirst) * pitch + colFirst;
        uint32_t ubColsStride = (TRANSFER_PITCH - (cols + 7) / 8 * 8) / 8;
        uint32_t ubExternalStride = (TRANSFER_PITCH - (externalCols + 7) / 8 * 8) / 8;
        GlobalTensor<float> ga;
        ga.SetGlobalBuffer(pa);
        TransferLayout layout{rowFirst, colFirst, rows,           cols,       externalRows, externalCols,    lda,
                              pitch,    upper,    externalOffset, workOffset, ubColsStride, ubExternalStride};
        if constexpr (Store)
            StoreTransferTile(w, ga, source, destination, original, layout);
        else
            LoadTransferTile(w, ga, source, destination, layout);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void Pack(const MatrixBatch &a, float *w, int pitch, int *info, aclrtStream s)
{
    MatrixTransferVectorKernel<false><<<VectorCores(), nullptr, s>>>(
        (GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld, a.upper, a.count, (GM_ADDR)w, pitch, (GM_ADDR)info);
}

void Unpack(const MatrixBatch &a, const float *w, int pitch, const int *info, aclrtStream s)
{
    MatrixTransferVectorKernel<true><<<VectorCores(), nullptr, s>>>(
        (GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld, a.upper, a.count, (GM_ADDR)w, pitch, (GM_ADDR)info);
}

constexpr int SMALL_PITCH = 40;
constexpr int SMALL_STRIDE = 32 * SMALL_PITCH + 8;

// Four independent matrices share the 64 lanes (16 rows per matrix).
template <bool Transposed, int UbPitch, typename RegI, typename RegMask, typename RegF>
__simd_callee__ void FactorGroupColumn(__ubuf__ float *tile, RegI &bases, RegI &rows, RegMask &boundedHealthy,
                                       RegF &diagonal, RegF &column, uint32_t width, uint32_t k)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>(), active;
    Compares<int32_t, AscendC::CMPMODE::GE>(active, rows, int32_t(k), boundedHealthy);
    RegTensor<int32_t> offsets;
    if constexpr (Transposed)
        Adds(offsets, rows, int32_t(k * UbPitch), full);
    else
    {
        Muls(offsets, rows, int32_t(UbPitch), full);
        Adds(offsets, offsets, int32_t(k), full);
    }
    Add(offsets, offsets, bases, full);
    Gather(column, tile, (RegTensor<uint32_t> &)offsets, active);
    Div(column, column, diagonal, active);
    // Store sqrt(a_kk) directly, rather than rounding a_kk / sqrt(a_kk) again.
    MaskReg pivot;
    Compares<int32_t, AscendC::CMPMODE::EQ>(pivot, rows, int32_t(k), active);
    Select(column, diagonal, column, pivot);
    Scatter(tile, column, (RegTensor<uint32_t> &)offsets, active);
}

template <bool Transposed, int UbPitch, typename RegI, typename RegMask>
__simd_callee__ void FactorGroupStored(RegI &bases, RegI &rows, RegMask &boundedHealthy, RegMask &active, RegI &offsets,
                                       uint32_t j)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    Compares<int32_t, AscendC::CMPMODE::GE>(active, rows, int32_t(j), boundedHealthy);
    if constexpr (Transposed)
        Adds(offsets, rows, int32_t(j * UbPitch), full);
    else
    {
        Muls(offsets, rows, int32_t(UbPitch), full);
        Adds(offsets, offsets, int32_t(j), full);
    }
    Add(offsets, offsets, bases, full);
}

template <bool Transposed, int UbPitch, typename RegI, typename RegMask, typename RegF>
__simd_callee__ void FactorGroupUpdate(__ubuf__ float *tile, RegI &bases, RegI &rows, RegMask &boundedHealthy,
                                       RegF &column, RegF &alpha, uint32_t width, uint32_t j)
{
    using namespace AscendC::MicroAPI;
    MaskReg active;
    RegTensor<int32_t> offsets;
    FactorGroupStored<Transposed, UbPitch>(bases, rows, boundedHealthy, active, offsets, j);
    RegTensor<float> current, product;
    Gather(current, tile, (RegTensor<uint32_t> &)offsets, active);
    Mul(product, column, alpha, active);
    Sub(current, current, product, active);
    Scatter(tile, current, (RegTensor<uint32_t> &)offsets, active);
}

// Load a completed column once and keep it across the rank-four update.
template <bool Transposed, int UbPitch, typename RegI, typename RegMask, typename RegF>
__simd_callee__ void FactorGroupLoad(__ubuf__ float *tile, RegI &bases, RegI &rows, RegMask &mask, RegF &column,
                                     uint32_t k)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<int32_t> offsets;
    if constexpr (Transposed)
        Adds(offsets, rows, int32_t(k * UbPitch), full);
    else
    {
        Muls(offsets, rows, int32_t(UbPitch), full);
        Adds(offsets, offsets, int32_t(k), full);
    }
    Add(offsets, offsets, bases, full);
    Gather(column, tile, (RegTensor<uint32_t> &)offsets, mask);
}

template <bool Transposed, int UbPitch, bool First, typename RegI, typename RegMask, typename RegF>
__simd_callee__ void FactorGroupRank(__ubuf__ float *tile, RegI &bases, RegMask &mask, RegF &sum, RegF &column,
                                     uint32_t k, uint32_t j)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<int32_t> offsets;
    Adds(offsets, bases, int32_t(Transposed ? k * UbPitch + j : j * UbPitch + k), full);
    RegTensor<float> alpha, product;
    Gather(alpha, tile, (RegTensor<uint32_t> &)offsets, mask);
    if constexpr (First)
        Mul(sum, column, alpha, mask);
    else
    {
        Mul(product, column, alpha, mask);
        Add(sum, sum, product, mask);
    }
}

template <bool Transposed, int UbPitch, typename RegI, typename RegMask, typename RegF>
__simd_callee__ void FactorGroupTrailing(__ubuf__ float *tile, RegI &bases, RegI &rows, RegMask &boundedHealthy,
                                         RegF &column0, RegF &column1, RegF &column2, RegF &column3, uint32_t base,
                                         uint32_t end, uint32_t j)
{
    using namespace AscendC::MicroAPI;
    MaskReg active;
    RegTensor<int32_t> offsets;
    FactorGroupStored<Transposed, UbPitch>(bases, rows, boundedHealthy, active, offsets, j);
    RegTensor<float> current, sum;
    Gather(current, tile, (RegTensor<uint32_t> &)offsets, active);
    FactorGroupRank<Transposed, UbPitch, true>(tile, bases, active, sum, column0, base, j);
    if (base + 1 < end) FactorGroupRank<Transposed, UbPitch, false>(tile, bases, active, sum, column1, base + 1, j);
    if (base + 2 < end) FactorGroupRank<Transposed, UbPitch, false>(tile, bases, active, sum, column2, base + 2, j);
    if (base + 3 < end) FactorGroupRank<Transposed, UbPitch, false>(tile, bases, active, sum, column3, base + 3, j);
    Sub(current, current, sum, active);
    Scatter(tile, current, (RegTensor<uint32_t> &)offsets, active);
}

template <bool Transposed, int UbPitch, typename IntReg, typename VectorMask>
__simd_callee__ void FactorGroupPivot(__ubuf__ float *tile, IntReg &bases, IntReg &rows0, IntReg &rows1,
                                      VectorMask &within0, VectorMask &within1, VectorMask &pending, IntReg &failure,
                                      VectorMask &bounded0, VectorMask &bounded1, uint32_t width, uint32_t k,
                                      uint32_t end)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<int32_t> offsets, pivot, candidate;
    RegTensor<float> one;
    Duplicate(one, 1.f);
    Adds(offsets, bases, int32_t(k * (UbPitch + 1)), full);
    RegTensor<float> diagonal, column0, column1, alpha;
    Gather(diagonal, tile, (RegTensor<uint32_t> &)offsets, pending);
    MaskReg positive, finite, healthy, bad, validPivot;
    Duplicate(pivot, int32_t(k));
    Compares<int32_t, AscendC::CMPMODE::LT>(validPivot, pivot, int32_t(width), pending);
    Select(diagonal, diagonal, one, validPivot);
    Compares<float, AscendC::CMPMODE::GT>(positive, diagonal, 0.f, pending);
    Compares<float, AscendC::CMPMODE::LE>(finite, diagonal, 3.402823466e38F, pending);
    And(healthy, positive, finite, pending);
    Not(bad, healthy, pending);
    Duplicate(candidate, int32_t(k + 1));
    Select(failure, candidate, failure, bad);
    pending = healthy;
    Sqrt(diagonal, diagonal, healthy);
    And(bounded0, healthy, within0, full);
    And(bounded1, healthy, within1, full);
    FactorGroupColumn<Transposed, UbPitch>(tile, bases, rows0, bounded0, diagonal, column0, width, k);
    if (width > 16) FactorGroupColumn<Transposed, UbPitch>(tile, bases, rows1, bounded1, diagonal, column1, width, k);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint32_t j = k + 1; j < end; ++j)
    {
        uint32_t index = Transposed ? k * UbPitch + j : j * UbPitch + k;
        Adds(offsets, bases, int32_t(index), full);
        Gather(alpha, tile, (RegTensor<uint32_t> &)offsets, healthy);
        FactorGroupUpdate<Transposed, UbPitch>(tile, bases, rows0, bounded0, column0, alpha, width, j);
        if (width > 16) FactorGroupUpdate<Transposed, UbPitch>(tile, bases, rows1, bounded1, column1, alpha, width, j);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool Transposed, int UbPitch = SMALL_PITCH>
__simd_vf__ void FactorGroup(__ubuf__ float *tile, __ubuf__ int32_t *status, uint32_t width, uint32_t count,
                             uint32_t matrixStride)
{
    using namespace AscendC::MicroAPI;
    MaskReg full = CreateMask<float, MaskPattern::ALL>(), matrices;
    RegTensor<int32_t> lanes, groups, bases, rows0, rows1, fifteen, failure;
    Arange(lanes, int32_t(0));
    ShiftRights(groups, lanes, int16_t(4), full);
    Duplicate(fifteen, int32_t(15));
    And(rows0, lanes, fifteen, full);
    Adds(rows1, rows0, int32_t(16), full);
    Muls(bases, groups, int32_t(matrixStride), full);
    Compares<int32_t, AscendC::CMPMODE::LT>(matrices, groups, int32_t(count), full);
    MaskReg within0, within1;
    Compares<int32_t, AscendC::CMPMODE::LT>(within0, rows0, int32_t(width), matrices);
    Compares<int32_t, AscendC::CMPMODE::LT>(within1, rows1, int32_t(width), matrices);
    Duplicate(failure, int32_t(0));
    MaskReg pending = matrices;
    uint32_t groupCount = (width + 3) / 4;
    for (uint32_t group = 0; group < groupCount; ++group)
    {
        uint32_t base = group * 4, end = base + 4;
        MaskReg bounded0, bounded1;
        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            FactorGroupPivot<Transposed, UbPitch>(tile, bases, rows0, rows1, within0, within1, pending, failure,
                                                  bounded0, bounded1, width, base + lane, end);
        }
        RegTensor<float> c00, c01, c10, c11, c20, c21, c30, c31;
        FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows0, bounded0, c00, base);
        if (width > 16) FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows1, bounded1, c01, base);
        FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows0, bounded0, c10, base + 1);
        if (width > 16) FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows1, bounded1, c11, base + 1);
        FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows0, bounded0, c20, base + 2);
        if (width > 16) FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows1, bounded1, c21, base + 2);
        FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows0, bounded0, c30, base + 3);
        if (width > 16) FactorGroupLoad<Transposed, UbPitch>(tile, bases, rows1, bounded1, c31, base + 3);
        for (uint32_t j = end; j < width; ++j)
        {
            FactorGroupTrailing<Transposed, UbPitch>(tile, bases, rows0, bounded0, c00, c10, c20, c30, base, end, j);
            if (width > 16)
                FactorGroupTrailing<Transposed, UbPitch>(tile, bases, rows1, bounded1, c01, c11, c21, c31, base, end,
                                                         j);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    MaskReg first;
    Compares<int32_t, AscendC::CMPMODE::EQ>(first, rows0, int32_t(0), matrices);
    Scatter(status, failure, (RegTensor<uint32_t> &)groups, first);
}

__aicore__ inline bool LoadSmallMatrices(const LocalFloat &tile, __gm__ float **pointers, GM_ADDR aAddr,
                                         GM_ADDR aArrayAddr, int64_t first, int firstA, int active, int n, int lda,
                                         int ubPitch, int matrixStride)
{
    using namespace AscendC;
    bool contiguous = true;
    for (int q = 0; q < active; ++q)
    {
        pointers[q] = (__gm__ float *)aAddr;
        if (aArrayAddr)
        {
            GlobalTensor<uint64_t> array;
            array.SetGlobalBuffer((__gm__ uint64_t *)aArrayAddr);
            pointers[q] = (__gm__ float *)array.GetValue(firstA + first + q);
        }
        if (!pointers[q] || !pointers[0] || (q && pointers[q] != pointers[0] + int64_t(q) * n * lda))
            contiguous = false;
    }
    uint32_t ubStride = (ubPitch - (n + 7) / 8 * 8) / 8;
    if (contiguous)
    {
        GlobalTensor<float> ga;
        ga.SetGlobalBuffer(pointers[0]);
        DataCopyPad(tile, ga,
                    DataCopyExtParams{uint16_t(active * n), uint32_t(n * 4), uint32_t((lda - n) * 4), ubStride, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    else
    {
        for (int q = 0; q < active; ++q)
        {
            if (!pointers[q]) continue;
            GlobalTensor<float> ga;
            ga.SetGlobalBuffer(pointers[q]);
            DataCopyPad(tile[q * matrixStride], ga,
                        DataCopyExtParams{uint16_t(n), uint32_t(n * 4), uint32_t((lda - n) * 4), ubStride, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    }
    return contiguous;
}
__aicore__ inline void StoreSmallCompleted(const GlobalFloat &ga, const LocalFloat &matrix, int failure, int n, int lda,
                                           int upper, int ubPitch)
{
    using namespace AscendC;
    if (!failure)
    {
        DataCopyPad(ga, matrix,
                    DataCopyExtParams{uint16_t(n), uint32_t(n * 4), uint32_t((ubPitch - (n + 7) / 8 * 8) / 8),
                                      uint32_t((lda - n) * 4), 0});
        return;
    }
    int columns = failure - 1;
    for (int c = 0; c < n; ++c)
    {
        if (upper)
        {
            uint32_t length = c + 1 < columns ? c + 1 : columns;
            if (length)
                DataCopyPad(ga[int64_t(c) * lda], matrix[c * ubPitch], DataCopyExtParams{1, length * 4, 0, 0, 0});
        }
        else if (c < columns)
        {
            DataCopyPad(ga[int64_t(c) * lda], matrix[c * ubPitch], DataCopyExtParams{1, uint32_t(n * 4), 0, 0, 0});
        }
    }
}

__aicore__ inline void StoreSmallMatrices(const LocalFloat &tile, const LocalInt &status, __gm__ float **pointers,
                                          bool contiguous, int active, int n, int lda, int upper, int ubPitch,
                                          int matrixStride, uint32_t ubStride)
{
    using namespace AscendC;
    bool bulkStore = contiguous;
    for (int q = 0; q < active; ++q)
        if (status.GetValue(q)) bulkStore = false;
    if (bulkStore)
    {
        GlobalTensor<float> ga;
        ga.SetGlobalBuffer(pointers[0]);
        DataCopyPad(ga, tile,
                    DataCopyExtParams{uint16_t(active * n), uint32_t(n * 4), ubStride, uint32_t((lda - n) * 4), 0});
    }
    for (int q = 0; !bulkStore && q < active; ++q)
    {
        if (!pointers[q]) continue;
        GlobalTensor<float> ga;
        ga.SetGlobalBuffer(pointers[q]);
        auto matrix = tile[q * matrixStride];
        int failure = status.GetValue(q);
        StoreSmallCompleted(ga, matrix, failure, n, lda, upper, ubPitch);
    }
}

__aicore__ inline void ClearMissingSmallMatrices(const LocalFloat &tile, __gm__ float **pointers, int active,
                                                 int matrixStride)
{
    using namespace AscendC;
    bool missing = false;
    for (int q = 0; q < active; ++q)
    {
        if (!pointers[q])
        {
            Duplicate(tile[q * matrixStride], 0.f, uint32_t(matrixStride));
            missing = true;
        }
    }
    if (missing) PipeBarrier<PIPE_V>();
}

// Matrices up to one panel need no GM workspace or Cube launch.
__global__ __vector__ void SmallFactorVectorKernel(GM_ADDR aAddr, GM_ADDR aArrayAddr, int firstA, int n, int lda,
                                                   int upper, int count, GM_ADDR infoAddr)
{
    using namespace AscendC;
    GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (4 * SMALL_STRIDE + 64) * sizeof(float));
    auto tile = buffer.Get<float>();
    auto status = tile[4 * SMALL_STRIDE].ReinterpretCast<int32_t>();
    int ubPitch = SMALL_PITCH;
    int matrixStride = n * ubPitch;
    for (int64_t first = int64_t(GetBlockIdx()) * 4; first < count; first += int64_t(GetBlockNum()) * 4)
    {
        int active = count - first < 4 ? count - first : 4;
        __gm__ float *pointers[4];
        bool contiguous =
            LoadSmallMatrices(tile, pointers, aAddr, aArrayAddr, first, firstA, active, n, lda, ubPitch, matrixStride);
        uint32_t ubStride = (ubPitch - (n + 7) / 8 * 8) / 8;
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ClearMissingSmallMatrices(tile, pointers, active, matrixStride);
        auto tp = (__ubuf__ float *)tile.GetPhyAddr();
        auto sp = (__ubuf__ int32_t *)status.GetPhyAddr();
        if (upper)
            FactorGroup<false>(tp, sp, n, active, matrixStride);
        else
            FactorGroup<true>(tp, sp, n, active, matrixStride);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int q = 0; q < active; ++q)
        {
            if (!pointers[q]) status.SetValue(q, -3);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(info[first], status, DataCopyExtParams{1, uint32_t(active * 4), 0, 0, 0});
        StoreSmallMatrices(tile, status, pointers, contiguous, active, n, lda, upper, ubPitch, matrixStride, ubStride);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <bool Reverse, bool TransposedTri, int TriPitch, int ColumnPitch, typename VectorMask>
__simd_callee__ void SolveFourRhsPivots(__ubuf__ float *columns, __ubuf__ float *tri, VectorMask &mask, uint32_t base,
                                        uint32_t r)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> x, y, alpha, product;
    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        uint32_t step = base + lane, j = Reverse ? 63 - step : step;
        LoadAlign(x, columns + j * ColumnPitch + r);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, tri + j * (TriPitch + 1));
        Div(x, x, alpha, mask);
        StoreAlign(columns + j * ColumnPitch + r, x, mask);
        for (uint32_t nextLane = lane + 1; nextLane < 4; ++nextLane)
        {
            uint32_t t = Reverse ? 63 - base - nextLane : base + nextLane;
            uint32_t row = Reverse ? j : t, col = Reverse ? t : j;
            uint32_t index = TransposedTri ? col * TriPitch + row : row * TriPitch + col;
            LoadAlign(y, columns + t * ColumnPitch + r);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, tri + index);
            Mul(product, x, alpha, mask);
            Sub(y, y, product, mask);
            StoreAlign(columns + t * ColumnPitch + r, y, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <bool Reverse, bool TransposedTri, int TriPitch, int ColumnPitch, typename VectorMask>
__simd_callee__ void UpdateFourRhsTail(__ubuf__ float *columns, __ubuf__ float *tri, VectorMask &mask, uint32_t base,
                                       uint32_t r)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> y, alpha, product;
    uint32_t j0 = Reverse ? 63 - base : base;
    uint32_t j1 = Reverse ? 62 - base : base + 1;
    uint32_t j2 = Reverse ? 61 - base : base + 2;
    uint32_t j3 = Reverse ? 60 - base : base + 3;
    RegTensor<float> x0, x1, x2, x3;
    LoadAlign(x0, columns + j0 * ColumnPitch + r);
    LoadAlign(x1, columns + j1 * ColumnPitch + r);
    LoadAlign(x2, columns + j2 * ColumnPitch + r);
    LoadAlign(x3, columns + j3 * ColumnPitch + r);
    for (uint32_t next = base + 4; next < 64; ++next)
    {
        uint32_t t = Reverse ? 63 - next : next;
        RegTensor<float> sum;
        LoadAlign(y, columns + t * ColumnPitch + r);
        uint32_t row = Reverse ? j0 : t, col = Reverse ? t : j0;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha,
                                                 tri + (TransposedTri ? col * TriPitch + row : row * TriPitch + col));
        Mul(sum, x0, alpha, mask);
        row = Reverse ? j1 : t;
        col = Reverse ? t : j1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha,
                                                 tri + (TransposedTri ? col * TriPitch + row : row * TriPitch + col));
        Mul(product, x1, alpha, mask);
        Add(sum, sum, product, mask);
        row = Reverse ? j2 : t;
        col = Reverse ? t : j2;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha,
                                                 tri + (TransposedTri ? col * TriPitch + row : row * TriPitch + col));
        Mul(product, x2, alpha, mask);
        Add(sum, sum, product, mask);
        row = Reverse ? j3 : t;
        col = Reverse ? t : j3;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha,
                                                 tri + (TransposedTri ? col * TriPitch + row : row * TriPitch + col));
        Mul(product, x3, alpha, mask);
        Add(sum, sum, product, mask);
        // Round the RHS once per rank-four update, rather than four times.
        Sub(y, y, sum, mask);
        StoreAlign(columns + t * ColumnPitch + r, y, mask);
    }
}

// Four solved RHS vectors share one load/store of each remaining vector.
template <bool Reverse, bool TransposedTri, int TriPitch, int ColumnPitch>
__simd_vf__ void PanelRhsSolveBlock4(__ubuf__ float *columns, __ubuf__ float *tri, uint32_t rowCount)
{
    using namespace AscendC::MicroAPI;
    uint32_t rowGroups = (rowCount + 63) / 64;
    for (uint32_t rowGroup = 0; rowGroup < rowGroups; ++rowGroup)
    {
        uint32_t r = rowGroup * 64, length = rowCount - r;
        MaskReg mask = UpdateMask<float>(length);
        for (uint32_t group = 0; group < 16; ++group)
        {
            uint32_t base = group * 4;
            RegTensor<float> x, y, alpha, product;
            SolveFourRhsPivots<Reverse, TransposedTri, TriPitch, ColumnPitch>(columns, tri, mask, base, r);
            UpdateFourRhsTail<Reverse, TransposedTri, TriPitch, ColumnPitch>(columns, tri, mask, base, r);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
}

// Solve hardware-transposed RHS vectors in UB.
template <bool Reverse, bool TransposedTri, int TriPitch = VECTOR_PITCH, int ColumnPitch = 128>
__simd_vf__ void PanelRhsSolveColumnsSerial(__ubuf__ float *columns, __ubuf__ float *tri, uint32_t rowCount,
                                            uint32_t width)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> x, y, alpha, product;
    for (uint32_t step = 0; step < width; ++step)
    {
        uint32_t j = Reverse ? width - 1 - step : step;
        for (uint32_t r = 0; r < rowCount; r += 64)
        {
            uint32_t activeRows = rowCount - r;
            MaskReg mask = UpdateMask<float>(activeRows);
            LoadAlign(x, columns + j * ColumnPitch + r);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, tri + j * TriPitch + j);
            Div(x, x, alpha, mask);
            StoreAlign(columns + j * ColumnPitch + r, x, mask);
            for (uint32_t next = step + 1; next < width; ++next)
            {
                uint32_t t = Reverse ? width - 1 - next : next;
                uint32_t row = Reverse ? j : t, col = Reverse ? t : j;
                uint32_t index = TransposedTri ? col * TriPitch + row : row * TriPitch + col;
                LoadAlign(y, columns + t * ColumnPitch + r);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, tri + index);
                Mul(product, x, alpha, mask);
                Sub(y, y, product, mask);
                StoreAlign(columns + t * ColumnPitch + r, y, mask);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <bool Reverse, bool TransposedTri, int TriPitch = VECTOR_PITCH, int ColumnPitch = 128>
__aicore__ inline void PanelRhsSolveColumns(__ubuf__ float *columns, __ubuf__ float *tri, uint32_t rowCount,
                                            uint32_t width)
{
    if (width == PANEL)
        PanelRhsSolveBlock4<Reverse, TransposedTri, TriPitch, ColumnPitch>(columns, tri, rowCount);
    else
        PanelRhsSolveColumnsSerial<Reverse, TransposedTri, TriPitch, ColumnPitch>(columns, tri, rowCount, width);
}

template <bool SubtractSource, int SourcePitch>
__simd_vf__ void UpdateDenseTriangle(__ubuf__ float *destination, __ubuf__ float *source, uint32_t width)
{
    using namespace AscendC::MicroAPI;
    uint32_t length = width;
    MaskReg full = UpdateMask<float>(length);
    RegTensor<int32_t> rows;
    Arange(rows, int32_t(0));
    for (uint32_t c = 0; c < width; ++c)
    {
        MaskReg selected;
        Compares<int32_t, AscendC::CMPMODE::GE>(selected, rows, int32_t(c), full);
        RegTensor<float> values;
        LoadAlign(values, source + c * SourcePitch);
        if constexpr (SubtractSource)
        {
            RegTensor<float> current;
            LoadAlign(current, destination + c * DENSE_PITCH);
            Sub(values, current, values, selected);
        }
        StoreAlign(destination + c * DENSE_PITCH, values, selected);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Reuse the validated 72-element panel layout for each resident diagonal block.
__aicore__ inline void FactorCompact(const AscendC::LocalTensor<float> &canonical,
                                     const AscendC::LocalTensor<float> &compact,
                                     const AscendC::LocalTensor<int32_t> &status, int width)
{
    using namespace AscendC;
    uint16_t blocks = uint16_t((width + 7) / 8);
    DataCopy(compact.ReinterpretCast<uint32_t>(), canonical.ReinterpretCast<uint32_t>(),
             DataCopyParams{uint16_t(width), blocks, uint16_t(DENSE_PITCH / 8 - blocks),
                            uint16_t(VECTOR_PITCH / 8 - blocks)});
    PipeBarrier<PIPE_V>();
    auto tp = (__ubuf__ float *)compact.GetPhyAddr();
    auto sp = (__ubuf__ int32_t *)status.GetPhyAddr();
    if (width == PANEL)
        PanelFactorBlock4<true>(tp, sp, width);
    else
        PanelFactor<true>(tp, sp, width);
    // The compact block's unused half is scratch, not matrix output. Preserve the
    // caller's opposite triangle when copying the canonical lower factor back.
    UpdateDenseTriangle<false, VECTOR_PITCH>((__ubuf__ float *)canonical.GetPhyAddr(),
                                             (__ubuf__ float *)compact.GetPhyAddr(), width);
    PipeBarrier<PIPE_V>();
}

struct FactorBuffers
{
    LocalFloat original, transposed, product, compact;
    LocalInt status;
};
struct MediumLayout
{
    int n, lda, upper, finalPhase, width, externalCols, externalRows;
    uint32_t ubStride;
    int productPitch;
};
__aicore__ inline void LoadMediumFactor(const GlobalFloat &ga, GM_ADDR productAddr, int64_t q, const MediumLayout &t,
                                        const FactorBuffers &f)
{
    using namespace AscendC;
    DataCopyPad(f.original, ga,
                DataCopyExtParams{uint16_t(t.externalCols), uint32_t(t.externalRows * 4),
                                  uint32_t((t.lda - t.externalRows) * 4), t.ubStride, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    if (t.finalPhase)
    {
        GlobalTensor<float> gp;
        gp.SetGlobalBuffer((__gm__ float *)productAddr + q * t.productPitch * t.productPitch);
        DataCopyPad(
            f.product, gp,
            DataCopyExtParams{uint16_t(t.width), uint32_t(t.width * 4), uint32_t((t.productPitch - t.width) * 4),
                              uint32_t((VECTOR_PITCH - (t.width + 7) / 8 * 8) / 8), 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
}
__aicore__ inline void StoreMediumFactor(const GlobalFloat &ga, const MediumLayout &t, const FactorBuffers &f,
                                         int failure)
{
    using namespace AscendC;
    if (!failure)
    {
        DataCopyPad(ga, f.original,
                    DataCopyExtParams{uint16_t(t.externalCols), uint32_t(t.externalRows * 4), t.ubStride,
                                      uint32_t((t.lda - t.externalRows) * 4), 0});
    }
    else if (!t.upper)
    {
        int columns = failure - 1;
        if (columns)
            DataCopyPad(ga, f.original,
                        DataCopyExtParams{uint16_t(columns), uint32_t(t.externalRows * 4), t.ubStride,
                                          uint32_t((t.lda - t.externalRows) * 4), 0});
    }
    else
    {
        for (int c = 0; c < t.externalCols; ++c)
        {
            uint32_t length = c + 1 < failure - 1 ? c + 1 : failure - 1;
            if (length)
                DataCopyPad(ga[int64_t(c) * t.lda], f.original[c * DENSE_PITCH],
                            DataCopyExtParams{1, length * 4, 0, 0, 0});
        }
    }
}
__aicore__ inline int FactorMediumResident(const GlobalFloat &ga, GM_ADDR productAddr, int64_t q, const MediumLayout &t,
                                           const FactorBuffers &f)
{
    using namespace AscendC;
    LoadMediumFactor(ga, productAddr, q, t, f);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto canonical = t.upper ? f.transposed : f.original;
    auto cp = (__ubuf__ float *)canonical.GetPhyAddr();
    if (t.upper) TransposeHardware(f.transposed, f.original, t.externalCols, t.externalRows, DENSE_PITCH);
    if (t.finalPhase)
    {
        auto productColumns = t.upper ? f.original : f.transposed;
        TransposeHardware(productColumns, f.product, t.width, t.width, VECTOR_PITCH, DENSE_PITCH);
        UpdateDenseTriangle<true, DENSE_PITCH>(cp, (__ubuf__ float *)productColumns.GetPhyAddr(), t.width);
    }
    int panelWidth = t.width < PANEL ? t.width : PANEL;
    FactorCompact(canonical, f.product, f.status, panelWidth);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    int failure = f.status.GetValue(0);
    if (!t.finalPhase && t.n > PANEL)
    {
        int columns = failure ? failure - 1 : PANEL;
        if (columns)
            PanelRhsSolveColumns<false, true, VECTOR_PITCH, DENSE_PITCH>(
                cp + PANEL, (__ubuf__ float *)f.product.GetPhyAddr(), t.n - PANEL, columns);
    }
    if (t.upper) TransposeHardware(f.original, f.transposed, t.externalRows, t.externalCols, DENSE_PITCH);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    StoreMediumFactor(ga, t, f, failure);
    return failure;
}

__global__ __vector__ void MediumFactorVectorKernel(GM_ADDR aAddr, GM_ADDR aArrayAddr, int firstA, int n, int lda,
                                                    int upper, int count, GM_ADDR infoAddr, int finalPhase,
                                                    GM_ADDR productAddr, int productPitch)
{
    using namespace AscendC;
    GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (2 * 128 * DENSE_PITCH + 64 * VECTOR_PITCH + 64) * sizeof(float));
    auto original = buffer.Get<float>();
    auto transposed = original[128 * DENSE_PITCH];
    auto product = transposed[128 * DENSE_PITCH];
    auto status = product[64 * VECTOR_PITCH].ReinterpretCast<int32_t>();
    int width = finalPhase ? n - PANEL : n;
    int externalCols = !finalPhase && n > PANEL && !upper ? PANEL : width;
    int externalRows = !finalPhase && n > PANEL && upper ? PANEL : width;
    uint32_t ubStride = (DENSE_PITCH - (externalRows + 7) / 8 * 8) / 8;
    for (int64_t q = GetBlockIdx(); q < count; q += GetBlockNum())
    {
        if (finalPhase && info.GetValue(q)) continue;
        auto pa = MatrixPointer(aAddr, aArrayAddr, int64_t(firstA) + q);
        int failure = -3;
        if (pa)
        {
            GlobalTensor<float> ga;
            ga.SetGlobalBuffer(pa + (finalPhase ? int64_t(PANEL) * (lda + 1) : 0));
            MediumLayout layout{n, lda, upper, finalPhase, width, externalCols, externalRows, ubStride, productPitch};
            FactorBuffers resident{original, transposed, product, product, status};
            failure = FactorMediumResident(ga, productAddr, q, layout, resident);
        }
        if (failure > 0 && finalPhase) failure += PANEL;
        status.SetValue(0, failure);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(info[q], status, DataCopyExtParams{1, uint32_t(sizeof(int32_t)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void MediumFactorStage(const MatrixBatch &a, float *product, int productPitch, bool finalPhase, int *info,
                       aclrtStream s)
{
    MediumFactorVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                            a.upper, a.count, (GM_ADDR)info, int(finalPhase),
                                                            (GM_ADDR)product, productPitch);
}

__simd_vf__ void SubtractPanelProduct(__ubuf__ float *tile, __ubuf__ float *product, uint32_t rows, uint32_t columns)
{
    using namespace AscendC::MicroAPI;
    uint32_t segments = (rows + 63) / 64;
    for (uint32_t c = 0; c < columns; ++c)
    {
        for (uint32_t segment = 0; segment < segments; ++segment)
        {
            uint32_t first = segment * 64, length = rows - first;
            MaskReg mask = UpdateMask<float>(length);
            RegTensor<float> value, contribution;
            LoadAlign(value, tile + c * DENSE_PITCH + first);
            LoadAlign(contribution, product + c * DENSE_PITCH + first);
            Sub(value, value, contribution, mask);
            StoreAlign(tile + c * DENSE_PITCH + first, value, mask);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

struct LeftLayout
{
    int n, lda, upper, start, width, nativePitch;
    uint32_t diagStride;
};
__aicore__ inline void StoreLeftDiagonal(const GlobalFloat &ga, const LeftLayout &t, const FactorBuffers &f,
                                         int failure, int completed)
{
    using namespace AscendC;
    if (!failure || !t.upper)
    {
        if (completed)
            DataCopyPad(ga[int64_t(t.start) * (int64_t(t.lda) + 1)], f.original,
                        DataCopyExtParams{uint16_t(completed), uint32_t(t.width * 4), t.diagStride,
                                          uint32_t((t.lda - t.width) * 4), 0});
    }
    else
    {
        for (int c = 0; c < t.width; ++c)
        {
            uint32_t length = c + 1 < completed ? c + 1 : completed;
            if (length)
                DataCopyPad(ga[int64_t(t.start) * (int64_t(t.lda) + 1) + int64_t(c) * t.lda],
                            f.original[c * t.nativePitch], DataCopyExtParams{1, length * 4, 0, 0, 0});
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
__aicore__ inline int FactorLeftDiagonal(const GlobalFloat &ga, const GlobalFloat &gp, const LeftLayout &t,
                                         const FactorBuffers &f)
{
    using namespace AscendC;
    int64_t diagonalOffset = int64_t(t.start) * (int64_t(t.lda) + 1);
    DataCopyPad(
        f.original, ga[diagonalOffset],
        DataCopyExtParams{uint16_t(t.width), uint32_t(t.width * 4), uint32_t((t.lda - t.width) * 4), t.diagStride, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    if (t.start)
        DataCopyPad(f.product, gp,
                    DataCopyExtParams{uint16_t(t.width), uint32_t(t.width * 4), uint32_t((PANEL - t.width) * 4),
                                      uint32_t((VECTOR_PITCH - (t.width + 7) / 8 * 8) / 8), 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto canonical = t.upper ? f.transposed : f.original;
    auto contribution = t.upper ? f.original : f.transposed;
    auto cp = (__ubuf__ float *)canonical.GetPhyAddr();
    if (t.upper) TransposeHardware(f.transposed, f.original, t.width, t.width, VECTOR_PITCH, DENSE_PITCH);
    if (t.start)
    {
        TransposeHardware(contribution, f.product, t.width, t.width, VECTOR_PITCH, DENSE_PITCH);
        UpdateDenseTriangle<true, DENSE_PITCH>(cp, (__ubuf__ float *)contribution.GetPhyAddr(), t.width);
    }
    FactorCompact(canonical, f.compact, f.status, t.width);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    int failure = f.status.GetValue(0);
    int completed = failure ? failure - 1 : t.width;
    if (t.upper) TransposeHardware(f.original, f.transposed, t.width, t.width, DENSE_PITCH, VECTOR_PITCH);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    StoreLeftDiagonal(ga, t, f, failure, completed);
    return failure;
}
__aicore__ inline void SolveLeftRows(const GlobalFloat &ga, const GlobalFloat &gp, const LeftLayout &t,
                                     const FactorBuffers &f, int completed)
{
    using namespace AscendC;
    auto canonical = t.upper ? f.transposed : f.original;
    auto contribution = t.upper ? f.original : f.transposed;
    auto cp = (__ubuf__ float *)canonical.GetPhyAddr();
    if (completed)
    {
        int rowTiles = (t.n - t.start - t.width + 127) / 128;
        for (int tile = 0; tile < rowTiles; ++tile)
        {
            int firstRow = t.start + t.width + tile * 128;
            int rows = t.n - firstRow < 128 ? t.n - firstRow : 128;
            int externalCols = t.upper ? rows : t.width;
            int externalRows = t.upper ? t.width : rows;
            uint32_t nativeStride = (t.nativePitch - (externalRows + 7) / 8 * 8) / 8;
            int64_t offset = t.upper ? int64_t(firstRow) * t.lda + t.start : int64_t(t.start) * t.lda + firstRow;
            DataCopyPad(f.original, ga[offset],
                        DataCopyExtParams{uint16_t(externalCols), uint32_t(externalRows * 4),
                                          uint32_t((t.lda - externalRows) * 4), nativeStride, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            if (t.start)
                DataCopyPad(f.product, gp[int64_t(firstRow - t.start) * PANEL],
                            DataCopyExtParams{uint16_t(rows), uint32_t(t.width * 4), uint32_t((PANEL - t.width) * 4),
                                              uint32_t((VECTOR_PITCH - (t.width + 7) / 8 * 8) / 8), 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (t.upper) TransposeHardware(f.transposed, f.original, rows, t.width, VECTOR_PITCH, DENSE_PITCH);
            if (t.start)
            {
                TransposeHardware(contribution, f.product, rows, t.width, VECTOR_PITCH, DENSE_PITCH);
                SubtractPanelProduct(cp, (__ubuf__ float *)contribution.GetPhyAddr(), rows, t.width);
            }
            PanelRhsSolveColumns<false, true, VECTOR_PITCH, DENSE_PITCH>(cp, (__ubuf__ float *)f.compact.GetPhyAddr(),
                                                                         rows, completed);
            if (t.upper) TransposeHardware(f.original, f.transposed, completed, rows, DENSE_PITCH, VECTOR_PITCH);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            // For a failed pivot, finish only the earlier factor columns.
            DataCopyPad(
                ga[offset], f.original,
                DataCopyExtParams{uint16_t(t.upper ? rows : completed), uint32_t((t.upper ? completed : rows) * 4),
                                  uint32_t((t.nativePitch - ((t.upper ? completed : rows) + 7) / 8 * 8) / 8),
                                  uint32_t((t.lda - (t.upper ? completed : rows)) * 4), 0});
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

// Factor a diagonal panel and solve all rows beneath it in one vector task.
// Only the next selected columns are read/written; the untouched tail stays in A.
__global__ __vector__ void LeftFactorVectorKernel(GM_ADDR aAddr, GM_ADDR aArrayAddr, int firstA, int n, int lda,
                                                  int upper, int count, GM_ADDR infoAddr, int start,
                                                  GM_ADDR productAddr, int productRows)
{
    using namespace AscendC;
    GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    constexpr int originalSize = 128 * VECTOR_PITCH;
    constexpr int canonicalSize = PANEL * DENSE_PITCH;
    pipe.InitBuffer(buffer,
                    (originalSize + canonicalSize + 128 * VECTOR_PITCH + PANEL * VECTOR_PITCH + 64) * sizeof(float));
    auto original = buffer.Get<float>();
    auto transposed = original[originalSize];
    auto product = transposed[canonicalSize];
    auto compact = product[128 * VECTOR_PITCH];
    auto status = compact[PANEL * VECTOR_PITCH].ReinterpretCast<int32_t>();
    int width = n - start < PANEL ? n - start : PANEL;
    int nativePitch = upper ? VECTOR_PITCH : DENSE_PITCH;
    uint32_t diagStride = (nativePitch - (width + 7) / 8 * 8) / 8;
    for (int64_t q = GetBlockIdx(); q < count; q += GetBlockNum())
    {
        if (start && info.GetValue(q)) continue;
        auto pa = MatrixPointer(aAddr, aArrayAddr, int64_t(firstA) + q);
        int failure = -3;
        if (pa)
        {
            GlobalTensor<float> ga, gp;
            ga.SetGlobalBuffer(pa);
            gp.SetGlobalBuffer((__gm__ float *)productAddr + q * productRows * PANEL);
            LeftLayout layout{n, lda, upper, start, width, nativePitch, diagStride};
            FactorBuffers resident{original, transposed, product, compact, status};
            failure = FactorLeftDiagonal(ga, gp, layout, resident);
            SolveLeftRows(ga, gp, layout, resident, failure ? failure - 1 : width);
        }
        if (failure > 0) failure += start;
        status.SetValue(0, failure);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(info[q], status, DataCopyExtParams{1, uint32_t(sizeof(int32_t)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void LeftFactorStage(const MatrixBatch &a, float *product, int productRows, int start, int *info, aclrtStream s)
{
    unsigned cores = VectorCores();
    unsigned blocks = unsigned(a.count) < cores ? unsigned(a.count) : cores;
    LeftFactorVectorKernel<<<blocks, nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld, a.upper,
                                                   a.count, (GM_ADDR)info, start, (GM_ADDR)product, productRows);
}

void SmallFactorVector(const MatrixBatch &a, int *info, aclrtStream s)
{
    if (a.n > 32)
    {
        MediumFactorVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                                a.upper, a.count, (GM_ADDR)info, 0, nullptr, 0);
    }
    else
    {
        SmallFactorVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                               a.upper, a.count, (GM_ADDR)info);
    }
}

__global__ __vector__ void PanelSolveVectorKernel(GM_ADDR wAddr, int pitch, int n, int start, int count,
                                                  GM_ADDR infoAddr)
{
    using namespace AscendC;
    GlobalTensor<float> w;
    GlobalTensor<int> info;
    w.SetGlobalBuffer((__gm__ float *)wAddr);
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (PANEL * VECTOR_PITCH + 128 * VECTOR_PITCH + 128 * PANEL) * sizeof(float));
    auto tri = buffer.Get<float>();
    auto rows = tri[PANEL * VECTOR_PITCH];
    auto columns = rows[128 * VECTOR_PITCH];
    int width = n - start < PANEL ? n - start : PANEL;
    int tiles = (n - start - width + 127) / 128;
    for (int task = GetBlockIdx(); task < count * tiles; task += GetBlockNum())
    {
        int q = task / tiles, first = start + width + task % tiles * 128;
        int status = info.GetValue(q);
        int cols = status > 0 ? status - start - 1 : width;
        if (status < 0 || cols <= 0) continue;
        int rowCount = n - first < 128 ? n - first : 128;
        int64_t base = int64_t(q) * pitch * pitch;
        DataCopyExtParams loadTri{uint16_t(cols), uint32_t(cols * 4), uint32_t((pitch - cols) * 4),
                                  uint32_t((VECTOR_PITCH - (cols + 7) / 8 * 8) / 8), 0};
        DataCopyExtParams loadRows{uint16_t(rowCount), uint32_t(cols * 4), uint32_t((pitch - cols) * 4),
                                   uint32_t((VECTOR_PITCH - (cols + 7) / 8 * 8) / 8), 0};
        DataCopyPad(tri, w[base + int64_t(start) * pitch + start], loadTri,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(rows, w[base + int64_t(first) * pitch + start], loadRows,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        TransposeHardware(columns, rows, rowCount, cols, VECTOR_PITCH, 128);
        PanelRhsSolveColumns<false, false>((__ubuf__ float *)columns.GetPhyAddr(), (__ubuf__ float *)tri.GetPhyAddr(),
                                           rowCount, cols);
        TransposeHardware(rows, columns, cols, rowCount, 128, VECTOR_PITCH);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            w[base + int64_t(first) * pitch + start], rows,
            DataCopyExtParams{uint16_t(rowCount), uint32_t(cols * 4), uint32_t((VECTOR_PITCH - (cols + 7) / 8 * 8) / 8),
                              uint32_t((pitch - cols) * 4), 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void PanelSolve(float *w, int pitch, int n, int start, int count, int *info, aclrtStream s)
{
    PanelSolveVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)w, pitch, n, start, count, (GM_ADDR)info);
}

__aicore__ inline void SolveResidentRhs(const LocalFloat &rows, const LocalFloat &columns, const LocalFloat &tri,
                                        int rowCount, int width, int reverse, int upper)
{
    using namespace AscendC;
    auto rowPtr = (__ubuf__ float *)rows.GetPhyAddr();
    auto colPtr = (__ubuf__ float *)columns.GetPhyAddr();
    auto triPtr = (__ubuf__ float *)tri.GetPhyAddr();
    if (rowCount == 1)
    {
        if (reverse)
        {
            if (upper)
                VectorTriSolve<true, false>(rowPtr, triPtr, width, VECTOR_PITCH);
            else
                VectorTriSolve<true, true>(rowPtr, triPtr, width, VECTOR_PITCH);
        }
        else
        {
            if (upper)
                VectorTriSolve<false, false>(rowPtr, triPtr, width, VECTOR_PITCH);
            else
                VectorTriSolve<false, true>(rowPtr, triPtr, width, VECTOR_PITCH);
        }
    }
    else
    {
        TransposeHardware(columns, rows, rowCount, width, VECTOR_PITCH, 128);
        if (reverse)
        {
            if (upper)
                PanelRhsSolveColumns<true, false>(colPtr, triPtr, rowCount, width);
            else
                PanelRhsSolveColumns<true, true>(colPtr, triPtr, rowCount, width);
        }
        else
        {
            if (upper)
                PanelRhsSolveColumns<false, false>(colPtr, triPtr, rowCount, width);
            else
                PanelRhsSolveColumns<false, true>(colPtr, triPtr, rowCount, width);
        }
        TransposeHardware(rows, columns, width, rowCount, 128, VECTOR_PITCH);
    }
}

__global__ __vector__ void SolvePanelVectorKernel(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR aArrayAddr, GM_ADDR bArrayAddr,
                                                  int firstA, int firstB, int lda, int ldb, int upper, int nrhs,
                                                  int start, int width, int reverse, int count, GM_ADDR infoAddr)
{
    using namespace AscendC;
    GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    if (info.GetValue(0)) return;
    int groups = (nrhs + 127) / 128;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (PANEL * VECTOR_PITCH + 128 * VECTOR_PITCH + 128 * PANEL) * sizeof(float));
    auto tri = buffer.Get<float>();
    auto rows = tri[PANEL * VECTOR_PITCH];
    auto columns = rows[128 * VECTOR_PITCH];
    // Bound launch geometry by hardware; batch size may exceed the grid limit.
    for (int64_t task = GetBlockIdx(); task < int64_t(count) * groups; task += GetBlockNum())
    {
        int q = task / groups, firstRhs = task % groups * 128;
        auto pa = MatrixPointer(aAddr, aArrayAddr, int64_t(firstA) + q);
        auto pb = MatrixPointer(bAddr, bArrayAddr, int64_t(firstB) + q);
        if (!pa || !pb) continue;
        GlobalTensor<float> ga, gb;
        ga.SetGlobalBuffer(pa);
        gb.SetGlobalBuffer(pb);
        int rowCount = nrhs - firstRhs < 128 ? nrhs - firstRhs : 128;
        uint32_t ubStride = (VECTOR_PITCH - (width + 7) / 8 * 8) / 8;
        DataCopyPad(tri, ga[int64_t(start) * lda + start],
                    DataCopyExtParams{uint16_t(width), uint32_t(width * 4), uint32_t((lda - width) * 4), ubStride, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(
            rows, gb[int64_t(firstRhs) * ldb + start],
            DataCopyExtParams{uint16_t(rowCount), uint32_t(width * 4), uint32_t((ldb - width) * 4), ubStride, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SolveResidentRhs(rows, columns, tri, rowCount, width, reverse, upper);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            gb[int64_t(firstRhs) * ldb + start], rows,
            DataCopyExtParams{uint16_t(rowCount), uint32_t(width * 4), ubStride, uint32_t((ldb - width) * 4), 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void SolvePanel(const MatrixBatch &a, const MatrixBatch &b, int nrhs, int start, int width, bool reverse, int *info,
                aclrtStream s)
{
    SolvePanelVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)b.single, (GM_ADDR)a.array,
                                                          (GM_ADDR)b.array, a.first, b.first, a.ld, b.ld, a.upper, nrhs,
                                                          start, width, int(reverse), a.count, (GM_ADDR)info);
}

template <bool Strided>
__simd_vf__ void UpdateRhs(__ubuf__ float *rhs, __ubuf__ float *solved, __ubuf__ float *coefficients, uint32_t rows,
                           uint32_t width)
{
    using namespace AscendC::MicroAPI;
    RegTensor<int32_t> lanes;
    Arange(lanes, int32_t(0));
    for (uint32_t r = 0; r < rows; r += 64)
    {
        uint32_t length = rows - r;
        MaskReg mask = UpdateMask<float>(length);
        RegTensor<float> value, coefficient, alpha, product, sum;
        LoadAlign(value, rhs + r);
        Duplicate(sum, 0.f);
        RegTensor<uint32_t> offsets;
        if constexpr (Strided) Muls((RegTensor<int32_t> &)offsets, lanes, int32_t(VECTOR_PITCH), mask);
        for (uint32_t j = 0; j < width; ++j)
        {
            if constexpr (Strided)
                Gather(coefficient, coefficients + r * VECTOR_PITCH + j, offsets, mask);
            else
                LoadAlign(coefficient, coefficients + j * TRANSFER_PITCH + r);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(alpha, solved + j);
            Mul(product, coefficient, alpha, mask);
            Add(sum, sum, product, mask);
        }
        // Accumulate A_panel*x_panel before subtracting from the RHS. Repeated
        // subtraction from a much larger RHS loses the small products' low bits.
        Sub(value, value, sum, mask);
        StoreAlign(rhs + r, value, mask);
    }
}

__aicore__ inline void LoadRhsUpdate(const GlobalFloat &ga, const GlobalFloat &gb, const LocalFloat &coefficients,
                                     const LocalFloat &rhs, const LocalFloat &solved, int i, int start, int width,
                                     int active, int lda, bool strided)
{
    using namespace AscendC;
    if (strided)
        DataCopyPad(coefficients, ga[int64_t(i) * lda + start],
                    DataCopyExtParams{uint16_t(active), uint32_t(width * 4), uint32_t((lda - width) * 4),
                                      uint32_t((VECTOR_PITCH - (width + 7) / 8 * 8) / 8), 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    else
        DataCopyPad(coefficients, ga[int64_t(start) * lda + i],
                    DataCopyExtParams{uint16_t(width), uint32_t(active * 4), uint32_t((lda - active) * 4),
                                      uint32_t((TRANSFER_PITCH - (active + 7) / 8 * 8) / 8), 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    DataCopyPad(rhs, gb[i], DataCopyExtParams{1, uint32_t(active * 4), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    DataCopyPad(solved, gb[start], DataCopyExtParams{1, uint32_t(width * 4), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
}

__global__ __vector__ void SolveUpdateVectorKernel(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR aArrayAddr, GM_ADDR bArrayAddr,
                                                   int firstA, int firstB, int n, int lda, int ldb, int upper,
                                                   int count, int start, int width, int reverse, GM_ADDR infoAddr)
{
    using namespace AscendC;
    GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    if (info.GetValue(0)) return;
    int rows = reverse ? start : n - start - width;
    int tiles = (rows + TRANSFER_TILE - 1) / TRANSFER_TILE;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    constexpr int coefficientSize = PANEL * TRANSFER_PITCH;
    constexpr int stridedSize = TRANSFER_TILE * VECTOR_PITCH;
    constexpr int capacity = coefficientSize > stridedSize ? coefficientSize : stridedSize;
    pipe.InitBuffer(buffer, (capacity + TRANSFER_TILE + PANEL) * sizeof(float));
    auto coefficients = buffer.Get<float>();
    auto rhs = coefficients[capacity];
    auto solved = rhs[TRANSFER_TILE];
    for (int64_t task = GetBlockIdx(); task < int64_t(count) * tiles; task += GetBlockNum())
    {
        int q = task / tiles, rowFirst = task % tiles * TRANSFER_TILE;
        int active = rows - rowFirst < TRANSFER_TILE ? rows - rowFirst : TRANSFER_TILE;
        int i = rowFirst + (reverse ? 0 : start + width);
        auto pa = MatrixPointer(aAddr, aArrayAddr, int64_t(firstA) + q);
        auto pb = MatrixPointer(bAddr, bArrayAddr, int64_t(firstB) + q);
        if (!pa || !pb) continue;
        GlobalTensor<float> ga, gb;
        ga.SetGlobalBuffer(pa);
        gb.SetGlobalBuffer(pb);
        bool strided = upper != reverse;
        LoadRhsUpdate(ga, gb, coefficients, rhs, solved, i, start, width, active, lda, strided);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto rp = (__ubuf__ float *)rhs.GetPhyAddr();
        auto sp = (__ubuf__ float *)solved.GetPhyAddr();
        auto cp = (__ubuf__ float *)coefficients.GetPhyAddr();
        if (strided)
            UpdateRhs<true>(rp, sp, cp, active, width);
        else
            UpdateRhs<false>(rp, sp, cp, active, width);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(gb[i], rhs, DataCopyExtParams{1, uint32_t(active * 4), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void SolveUpdateVector(const MatrixBatch &a, const MatrixBatch &b, int start, int width, bool reverse, int *info,
                       aclrtStream s)
{
    SolveUpdateVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)b.single, (GM_ADDR)a.array,
                                                           (GM_ADDR)b.array, a.first, b.first, a.n, a.ld, b.ld, a.upper,
                                                           a.count, start, width, int(reverse), (GM_ADDR)info);
}

__global__ __vector__ void SmallSolveVectorKernel(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR aArrayAddr, GM_ADDR bArrayAddr,
                                                  int firstA, int firstB, int n, int lda, int ldb, int upper, int count,
                                                  int nrhs, GM_ADDR infoAddr)
{
    using namespace AscendC;
    const int rhsCount = nrhs > 0 ? nrhs : 1;
    constexpr int pitch = 136;
    GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    if (info.GetValue(0)) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (128 * pitch + 128) * sizeof(float));
    auto tri = buffer.Get<float>();
    auto rhs = tri[128 * pitch];
    for (int64_t task = GetBlockIdx(); task < int64_t(count) * nrhs; task += GetBlockNum())
    {
        int q = task / rhsCount, column = task % rhsCount;
        auto pa = MatrixPointer(aAddr, aArrayAddr, int64_t(firstA) + q);
        auto pb = MatrixPointer(bAddr, bArrayAddr, int64_t(firstB) + q);
        // The preceding info launch validates every pointer before computation.
        if (!pa || !pb) continue;
        GlobalTensor<float> ga, gb;
        ga.SetGlobalBuffer(pa);
        gb.SetGlobalBuffer(pb + int64_t(column) * ldb);
        DataCopyPad(tri, ga,
                    DataCopyExtParams{uint16_t(n), uint32_t(n * 4), uint32_t((lda - n) * 4),
                                      uint32_t((pitch - (n + 7) / 8 * 8) / 8), 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(rhs, gb, DataCopyExtParams{1, uint32_t(n * 4), 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto rp = (__ubuf__ float *)rhs.GetPhyAddr();
        auto tp = (__ubuf__ float *)tri.GetPhyAddr();
        if (upper)
        {
            VectorTriSolve<false, false, true>(rp, tp, n, pitch);
            VectorTriSolve<true, false, true>(rp, tp, n, pitch);
        }
        else
        {
            VectorTriSolve<false, true, true>(rp, tp, n, pitch);
            VectorTriSolve<true, true, true>(rp, tp, n, pitch);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(gb, rhs, DataCopyExtParams{1, uint32_t(n * 4), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

void SmallSolve(const MatrixBatch &a, const MatrixBatch &b, int nrhs, int *info, aclrtStream s)
{
    unsigned cores = VectorCores();
    int64_t tasks = int64_t(a.count) * nrhs;
    unsigned blocks = tasks < cores ? unsigned(tasks) : cores;
    SmallSolveVectorKernel<<<blocks, nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)b.single, (GM_ADDR)a.array,
                                                   (GM_ADDR)b.array, a.first, b.first, a.n, a.ld, b.ld, a.upper,
                                                   a.count, nrhs, (GM_ADDR)info);
}

__simd_vf__ void SubtractTile(__ubuf__ float *x, __ubuf__ float *y, uint32_t rows)
{
    using namespace AscendC::MicroAPI;
    for (uint32_t r = 0; r < rows; ++r)
    {
        for (uint32_t c = 0; c < rows; c += 64)
        {
            uint32_t length = rows - c;
            MaskReg mask = UpdateMask<float>(length);
            RegTensor<float> a, b;
            LoadAlign(a, x + r * TRANSFER_TILE + c);
            LoadAlign(b, y + r * TRANSFER_TILE + c);
            Sub(a, a, b, mask);
            StoreAlign(x + r * TRANSFER_TILE + c, a, mask);
        }
    }
}

__aicore__ inline void UpdateDenseTail(const LocalFloat &x, const LocalFloat &y, const GlobalFloat &w,
                                       const GlobalFloat &p, int q, int pitch, int end, int remaining)
{
    using namespace AscendC;
    int64_t base = int64_t(q) * pitch * pitch;
    uint32_t ubStride = (TRANSFER_TILE - (remaining + 7) / 8 * 8) / 8;
    DataCopyExtParams copy{uint16_t(remaining), uint32_t(remaining * 4), uint32_t((pitch - remaining) * 4), ubStride,
                           0};
    DataCopyPad(x, w[base + int64_t(end) * pitch + end], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
    DataCopyPad(y, p[base], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SubtractTile((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)y.GetPhyAddr(), remaining);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(w[base + int64_t(end) * pitch + end], x,
                DataCopyExtParams{uint16_t(remaining), uint32_t(remaining * 4), ubStride,
                                  uint32_t((pitch - remaining) * 4), 0});
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// Stream contiguous rows through UB. Only the stored lower triangle is updated.
__global__ __vector__ void UpdateVectorKernel(GM_ADDR wAddr, GM_ADDR pAddr, int pitch, int n, int end, int count,
                                              GM_ADDR infoAddr)
{
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 2 * TRANSFER_TILE * TRANSFER_TILE * sizeof(float));
    LocalTensor<float> x = buffer.Get<float>();
    LocalTensor<float> y = x[TRANSFER_TILE * TRANSFER_TILE];
    GlobalTensor<float> w, p;
    GlobalTensor<int> info;
    w.SetGlobalBuffer((__gm__ float *)wAddr);
    p.SetGlobalBuffer((__gm__ float *)pAddr);
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    int remaining = n - end;
    if (remaining <= TRANSFER_TILE)
    {
        // One Cube output tile is dense; its unused upper half is private workspace.
        for (int q = GetBlockIdx(); q < count; q += GetBlockNum())
        {
            if (info.GetValue(q)) continue;
            UpdateDenseTail(x, y, w, p, q, pitch, end, remaining);
        }
        return;
    }
    for (int q = 0; q < count; ++q)
    {
        if (info.GetValue(q)) continue;
        int64_t base = int64_t(q) * pitch * pitch;
        for (int i = end + GetBlockIdx(); i < n; i += GetBlockNum())
        {
            uint32_t length = i - end + 1;
            DataCopyExtParams copy{1, length * 4, 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(x, w[base + int64_t(i) * pitch + end], copy, pad);
            DataCopyPad(y, p[base + int64_t(i - end) * pitch], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Sub(x, x, y, (length + 7) / 8 * 8);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(w[base + int64_t(i) * pitch + end], x, copy);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

void Update(float *w, float *p, int pitch, int n, int end, int count, int *info, aclrtStream s)
{
    UpdateVectorKernel<<<VectorCores(), nullptr, s>>>((GM_ADDR)w, (GM_ADDR)p, pitch, n, end, count, (GM_ADDR)info);
}
}  // namespace Cholesky
