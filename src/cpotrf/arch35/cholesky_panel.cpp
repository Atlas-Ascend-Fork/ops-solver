/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cholesky_device.h"

namespace Cholesky
{
namespace
{
// Balanced triangular tiles keep both matrix layouts coalesced during Schur updates.
// Invert t = row*(row+1)/2, then correct the FP32 square-root rounding at triangle boundaries.
__simt_callee__ inline int PanelTriangleRow(int tile)
{
    int row = static_cast<int>((sqrtf(static_cast<float>(8 * tile + 1)) - 1.0f) * 0.5f);
    if (tile < row * (row + 1) / 2)
    {
        --row;
    }
    if (tile >= (row + 1) * (row + 2) / 2)
    {
        ++row;
    }
    return row;
}

__simt_vf__ __aicore__ __launch_bounds__(PANEL_THREADS) void PanelApplyVF(__gm__ aclsolverFloatComplex* single,
                                                                          __gm__ const uint64_t* array,
                                                                          __gm__ float* work, int n, int lda,
                                                                          bool upper, int offset, int width,
                                                                          __gm__ int* info, int batch, int groups)
{
    int rows = n - offset - width, capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int rowTiles = (rows + WARP_SIZE - 1) / WARP_SIZE, tiles = rowTiles * (rowTiles + 1) / 2;
    int64_t packedStride = static_cast<int64_t>(capacity) * GRAM_COMPONENT_ROWS;
    int group = blockIdx.x % groups, lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    for (int m = blockIdx.x / groups; m < batch; m += gridDim.x / groups)
    {
        if (info[m] != 0)
        {
            continue;
        }
        auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        auto firstColumn = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work + 2 * batch * packedStride) +
                           static_cast<int64_t>(m) * capacity;
        for (int tile = group; tile < tiles; tile += groups)
        {
            int row = PanelTriangleRow(tile);
            int col = tile - row * (row + 1) / 2;
            int r = row * WARP_SIZE + (upper ? warp : lane), c = col * WARP_SIZE + (upper ? lane : warp);
            if (r >= rows || c >= rows || r < c)
            {
                continue;
            }
            auto z = ReadLower(a, lda, offset + width + r, offset + width + c, upper);
            if (!offset)
            {
                int row = width + r, col = width + c;
                z = SubFirstColumn(z, LoadFirstColumn(a, lda, row, upper), LoadFirstColumn(a, lda, col, upper),
                                   LoadFirstColumn(firstColumn, 0, row, false),
                                   LoadFirstColumn(firstColumn, 0, col, false));
            }
            if (r == c)
            {
                z.imag = 0;
            }
            WriteLower(a, lda, offset + width + r, offset + width + c, upper, z);
        }
    }
}
__simt_callee__ inline aclsolverFloatComplex ReadPanelDiagonal(__ubuf__ aclsolverFloatComplex* diagonal, int r, int c,
                                                               bool upper)
{
    auto value = ReadUB(diagonal, upper ? r * SIMD_PANEL_STRIDE + c : c * SIMD_PANEL_STRIDE + r);
    if (upper)
    {
        value.imag = -value.imag;
    }
    return value;
}
__simt_callee__ inline void PackPanelOutput(__gm__ float* work, int m, int batch, int64_t packedStride, int panelRows,
                                            int validRows, int completed, bool upper, int tile, int offset,
                                            __ubuf__ aclsolverFloatComplex* panel,
                                            __ubuf__ aclsolverFloatComplex* rawPanel)
{
    auto real = work + static_cast<int64_t>(m) * packedStride,
         imag = work + static_cast<int64_t>(batch + m) * packedStride;
    // Transpose upper output and pack Cube operands in one visit to each valid row.
    for (int pos = threadIdx.x; pos < panelRows * FACTOR_BLOCK; pos += PANEL_IO_THREADS)
    {
        int r = pos / FACTOR_BLOCK, c = pos % FACTOR_BLOCK;
        bool valid = r < validRows && c < completed;
        auto value = valid ? ReadUB(panel, c * SIMD_PANEL_STRIDE + r) : aclsolverFloatComplex{0, 0};
        if (upper && valid)
        {
            WriteUB(rawPanel, r * SIMD_PANEL_STRIDE + c, {value.real, -value.imag});
        }
        int index = (tile + r) * GRAM_COMPONENT_ROWS + c;
        bool skip = !offset && !c;
        real[index] = skip ? 0 : value.real;
        real[index + FACTOR_BLOCK] = skip ? 0 : value.imag;
        imag[index] = skip ? 0 : value.imag;
        imag[index + FACTOR_BLOCK] = skip ? 0 : -value.real;
    }
    return;
}

__simt_callee__ inline void CompensatePanel(int validRows, int panelRows, int completed, bool upper, int capacity,
                                            int width, int tile, __gm__ aclsolverFloatComplex* firstColumn,
                                            __ubuf__ aclsolverFloatComplex* diagonal,
                                            __ubuf__ aclsolverFloatComplex* panel, __ubuf__ aclsolverFloatComplex* low,
                                            __ubuf__ aclsolverFloatComplex* firstLow)
{
    if (threadIdx.x < static_cast<uint32_t>(completed))
    {
        WriteUB(firstLow, threadIdx.x, LoadFirstColumn(firstColumn, 0, threadIdx.x, false));
    }
    asc_syncthreads();
    if (threadIdx.x < SIMD_PANEL_ROWS)
    {
        int r = threadIdx.x;
        aclsolverFloatComplex remainder{0, 0};
        if (r < validRows)
        {
            auto value = NormalizeFirstColumn(ReadUB(panel, r), diagonal[0].real, firstLow[0].real, remainder);
            WriteUB(panel, r, value);
            WriteLower(firstColumn, capacity, width + tile + r, 0, false, remainder);
        }
        WriteUB(low, r, remainder);
    }
    asc_syncthreads();
    int rowShift = 0;
    while ((1 << rowShift) < panelRows)
    {
        ++rowShift;
    }
    int rowSlots = 1 << rowShift;
    for (int pos = threadIdx.x; pos < completed * rowSlots; pos += PANEL_IO_THREADS)
    {
        int r = pos & (rowSlots - 1), c = pos >> rowShift;
        if (r >= validRows || !c)
        {
            continue;
        }
        auto value = SubFirstColumn(ReadUB(panel, c * SIMD_PANEL_STRIDE + r), ReadUB(panel, r),
                                    ReadPanelDiagonal(diagonal, c, 0, upper), ReadUB(low, r), ReadUB(firstLow, c));
        WriteUB(panel, c * SIMD_PANEL_STRIDE + r, value);
    }
}

__simt_callee__ inline void TransposePanelInput(int validRows, int completed, __ubuf__ aclsolverFloatComplex* rawPanel,
                                                __ubuf__ aclsolverFloatComplex* panel)
{
    for (int pos = threadIdx.x; pos < validRows * FACTOR_BLOCK; pos += PANEL_IO_THREADS)
    {
        int r = pos / FACTOR_BLOCK, c = pos % FACTOR_BLOCK;
        auto value = c < completed ? ReadUB(rawPanel, r * SIMD_PANEL_STRIDE + c) : aclsolverFloatComplex{0, 0};
        value.imag = -value.imag;
        WriteUB(panel, c * SIMD_PANEL_STRIDE + r, value);
    }
}

__simt_callee__ inline void CachePanelDiagonal(__ubuf__ aclsolverFloatComplex* diagonal, int completed)
{
    if (threadIdx.x < static_cast<uint32_t>(completed))
    {
        int c = threadIdx.x, index = c * (SIMD_PANEL_STRIDE + 1);
        auto value = ReadUB(diagonal, index);
        value.imag = 1.0f / value.real;
        WriteUB(diagonal, index, value);
    }
}

// Retain the same compensated column order while panels distribute rows across AIV cores.
__simt_callee__ inline void UpdatePanelDiagonals(__gm__ float* work, int batch, int m, int capacity,
                                                 int64_t packedStride, int offset, int width, int tile, int validRows,
                                                 __ubuf__ aclsolverFloatComplex* panel)
{
    auto firstColumn = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work + 2 * batch * packedStride);
    auto saved = reinterpret_cast<__gm__ float*>(firstColumn + static_cast<int64_t>(batch) * capacity) +
                 static_cast<int64_t>(m) * capacity * FACTOR_DIAGONAL_COMPONENTS;
    for (int r = threadIdx.x; r < validRows; r += PANEL_IO_THREADS)
    {
        int row = offset + width + tile + r;
        FloatPair state{saved[FACTOR_DIAGONAL_COMPONENTS * row], saved[FACTOR_DIAGONAL_COMPONENTS * row + 1]};
        auto low = !offset ? LoadFirstColumn(firstColumn + static_cast<int64_t>(m) * capacity, 0, row, false)
                           : aclsolverFloatComplex{0, 0};
        for (int c = 0; c < width; ++c)
        {
            UpdateScalarDiagonal(state, ReadUB(panel, c * SIMD_PANEL_STRIDE + r), low, !offset && !c);
        }
        saved[FACTOR_DIAGONAL_COMPONENTS * row] = state.high;
        saved[FACTOR_DIAGONAL_COMPONENTS * row + 1] = state.low;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(PANEL_IO_THREADS) void PanelIOVF(
    __gm__ float* work, int n, int lda, bool upper, int offset, int width, __gm__ int* info, int batch, int m, int tile,
    __ubuf__ aclsolverFloatComplex* diagonal, __ubuf__ aclsolverFloatComplex* panel,
    __ubuf__ aclsolverFloatComplex* low, __ubuf__ int* completedInfo, __ubuf__ aclsolverFloatComplex* rawDiagonal,
    __ubuf__ aclsolverFloatComplex* rawPanel, bool output, bool retained)
{
    int rows = n - offset - width, padded = (rows + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT,
        capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int validRows = rows - tile < SIMD_PANEL_ROWS ? rows - tile : SIMD_PANEL_ROWS;
    int panelRows = padded - tile < SIMD_PANEL_ROWS ? padded - tile : SIMD_PANEL_ROWS;
    int64_t packedStride = static_cast<int64_t>(capacity) * GRAM_COMPONENT_ROWS;
    auto firstColumn = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work + 2 * batch * packedStride) +
                       static_cast<int64_t>(m) * capacity;
    int completed = info[m] ? info[m] - offset - 1 : width;
    completed = completed < 0 ? 0 : (completed < width ? completed : width);
    if (!threadIdx.x)
    {
        completedInfo[0] = completed;
    }
    if (output)
    {
        if (retained && !info[m])
        {
            UpdatePanelDiagonals(work, batch, m, capacity, packedStride, offset, width, tile, validRows, panel);
        }
        PackPanelOutput(work, m, batch, packedStride, panelRows, validRows, completed, upper, tile, offset, panel,
                        rawPanel);
        return;
    }
    auto firstLow = diagonal;
    diagonal = rawDiagonal;
    CachePanelDiagonal(diagonal, completed);
    if (upper)
    {
        TransposePanelInput(validRows, completed, rawPanel, panel);
    }
    // Lower DMA input already has the canonical layout. Clear only its padded rows.
    for (int pos = threadIdx.x; pos < (panelRows - validRows) * FACTOR_BLOCK; pos += PANEL_IO_THREADS)
    {
        int r = validRows + pos / FACTOR_BLOCK, c = pos % FACTOR_BLOCK;
        WriteUB(panel, c * SIMD_PANEL_STRIDE + r, {0, 0});
    }
    asc_syncthreads();
    if (!offset && completed)
    {
        CompensatePanel(validRows, panelRows, completed, upper, capacity, width, tile, firstColumn, diagonal, panel,
                        low, firstLow);
    }
    asc_syncthreads();
}

// Reuse each solved column for two RHS columns while keeping vector registers named.
template <typename WholeReg, typename WholeMask>
__simd_callee__ inline void NormalizePanelColumn(WholeReg& ar, WholeReg& ai, __ubuf__ float* panel,
                                                 __ubuf__ float* diagonal, uint32_t column, uint32_t completed,
                                                 WholeMask& mask)
{
    if (column >= completed)
    {
        return;
    }
    WholeReg inverse, v0, v1;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
        inverse, diagonal + 2 * (column * (SIMD_PANEL_STRIDE + 1)) + 1);
    AscendC::Reg::Mul(ar, ar, inverse, mask);
    AscendC::Reg::Mul(ai, ai, inverse, mask);
    AscendC::Reg::Interleave(v0, v1, ar, ai);
    AscendC::Reg::StoreAlign(panel + column * (2 * SIMD_PANEL_STRIDE), v0, mask);
    AscendC::Reg::StoreAlign(panel + column * (2 * SIMD_PANEL_STRIDE) + SIMD_FLOAT_LANES, v1, mask);
}

template <bool UPPER, typename WholeReg, typename WholeMask>
__simd_callee__ inline void SubtractPanelColumn(WholeReg& ar, WholeReg& ai, WholeReg& pr, WholeReg& negativePr,
                                                WholeReg& negativePi, __ubuf__ float* diagonal, uint32_t source,
                                                uint32_t target, WholeMask& mask)
{
    WholeReg cr, ci;
    uint32_t element = UPPER ? target * SIMD_PANEL_STRIDE + source : source * SIMD_PANEL_STRIDE + target;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(cr, diagonal + 2 * element);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(ci, diagonal + 2 * element + 1);
    if constexpr (UPPER)
    {
        AscendC::Reg::Neg(ci, ci, mask);
    }
    AscendC::Reg::MulAddDst(ar, negativePr, cr, mask);
    AscendC::Reg::MulAddDst(ar, negativePi, ci, mask);
    AscendC::Reg::MulAddDst(ai, pr, ci, mask);
    AscendC::Reg::MulAddDst(ai, negativePi, cr, mask);
}

// Reuse solved register columns before reading and writing the next panel.
template <bool UPPER, typename R, typename M>
__simd_callee__ inline void UpdatePanelTrailing(__ubuf__ float* panel, __ubuf__ float* diagonal, uint16_t c,
                                                uint16_t start, R& re0, R& nre0, R& nim0, R& re1, R& nre1, R& nim1,
                                                R& re2, R& nre2, R& nim2, M& mask)
{
    AscendC::Reg::RegTensor<float> ar, ai, v0, v1;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(ar, ai,
                                                                            panel + c * (2 * SIMD_PANEL_STRIDE));
    SubtractPanelColumn<UPPER>(ar, ai, re0, nre0, nim0, diagonal, start + 0, c, mask);
    SubtractPanelColumn<UPPER>(ar, ai, re1, nre1, nim1, diagonal, start + 1, c, mask);
    SubtractPanelColumn<UPPER>(ar, ai, re2, nre2, nim2, diagonal, start + 2, c, mask);
    AscendC::Reg::Interleave(v0, v1, ar, ai);
    AscendC::Reg::StoreAlign(panel + c * (2 * SIMD_PANEL_STRIDE), v0, mask);
    AscendC::Reg::StoreAlign(panel + c * (2 * SIMD_PANEL_STRIDE) + SIMD_FLOAT_LANES, v1, mask);
}

template <bool UPPER, typename R, typename M>
__simd_callee__ inline void BeginPanelStrip(__ubuf__ float* panel, __ubuf__ float* diagonal, uint16_t start,
                                            uint32_t completed, R& re0, R& im0, R& nre0, R& nim0, R& re1, R& im1,
                                            R& re2, R& im2, M& mask)
{
    NormalizePanelColumn(re0, im0, panel, diagonal, start + 0, completed, mask);
    AscendC::Reg::Neg(nre0, re0, mask);
    AscendC::Reg::Neg(nim0, im0, mask);
    if (start + 1 < completed)
    {
        SubtractPanelColumn<UPPER>(re1, im1, re0, nre0, nim0, diagonal, start + 0, start + 1, mask);
    }
    if (start + 2 < completed)
    {
        SubtractPanelColumn<UPPER>(re2, im2, re0, nre0, nim0, diagonal, start + 0, start + 2, mask);
    }
}

template <bool UPPER>
__simd_vf__ inline void PanelSolveVF(__ubuf__ float* panel, __ubuf__ float* diagonal, uint32_t completed,
                                     uint32_t first)
{
    uint32_t count = SIMD_FLOAT_LANES;
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    for (uint16_t start = first; start < completed; start += ORDINARY_PANEL_STRIP)
    {
        AscendC::Reg::RegTensor<float> re0, im0, re1, im1, re2, im2;
        LoadComplexColumn(re0, im0, panel, start + 0, completed, 2 * SIMD_PANEL_STRIDE);

        LoadComplexColumn(re1, im1, panel, start + 1, completed, 2 * SIMD_PANEL_STRIDE);

        LoadComplexColumn(re2, im2, panel, start + 2, completed, 2 * SIMD_PANEL_STRIDE);

        AscendC::Reg::RegTensor<float> nre0, nim0;
        BeginPanelStrip<UPPER>(panel, diagonal, start, completed, re0, im0, nre0, nim0, re1, im1, re2, im2, mask);
        NormalizePanelColumn(re1, im1, panel, diagonal, start + 1, completed, mask);
        AscendC::Reg::RegTensor<float> nre1, nim1;
        AscendC::Reg::Neg(nre1, re1, mask);
        AscendC::Reg::Neg(nim1, im1, mask);
        if (start + 2 < completed)
        {
            SubtractPanelColumn<UPPER>(re2, im2, re1, nre1, nim1, diagonal, start + 1, start + 2, mask);
        }
        NormalizePanelColumn(re2, im2, panel, diagonal, start + 2, completed, mask);
        AscendC::Reg::RegTensor<float> nre2, nim2;
        AscendC::Reg::Neg(nre2, re2, mask);
        AscendC::Reg::Neg(nim2, im2, mask);
        for (uint16_t c = start + ORDINARY_PANEL_STRIP; c < completed; ++c)
        {
            UpdatePanelTrailing<UPPER>(panel, diagonal, c, start, re0, nre0, nim0, re1, nre1, nim1, re2, nre2, nim2,
                                       mask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void StorePanel(bool upper, int completed, int validRows, int lda, int64_t first,
                                  AscendC::LocalTensor<float>& rawPanelLocal, AscendC::LocalTensor<float>& panelLocal,
                                  AscendC::GlobalTensor<float>& global)
{
    uint32_t outputElements = upper ? completed : validRows;
    uint16_t outputBlocks = upper ? validRows : completed;
    AscendC::DataCopyExtParams copyOut{
        outputBlocks, outputElements * COMPLEX_BYTES,
        static_cast<uint32_t>((SIMD_PANEL_STRIDE - (outputElements + DMA_COMPLEX_ALIGNMENT - 1) /
                                                       DMA_COMPLEX_ALIGNMENT * DMA_COMPLEX_ALIGNMENT) /
                              DMA_COMPLEX_ALIGNMENT),
        (static_cast<int64_t>(lda) - outputElements) * COMPLEX_BYTES, 0};
    auto output = upper ? rawPanelLocal : panelLocal;
    AscendC::DataCopyPad(global[first * 2], output, copyOut);
}

__aicore__ inline int64_t LoadPanel(bool upper, int validRows, int width, int lda, int offset, int tile,
                                    AscendC::LocalTensor<float>& rawPanelLocal, AscendC::LocalTensor<float>& panelLocal,
                                    AscendC::GlobalTensor<float>& global, AscendC::DataCopyPadExtParams<float>& padding)
{
    uint16_t blocks = upper ? validRows : width;
    uint32_t elements = upper ? width : validRows;
    int64_t first = upper ? static_cast<int64_t>(offset + width + tile) * lda + offset
                          : static_cast<int64_t>(offset) * lda + offset + width + tile;
    AscendC::DataCopyExtParams panelCopy{
        blocks, elements * COMPLEX_BYTES, (static_cast<int64_t>(lda) - elements) * COMPLEX_BYTES,
        static_cast<uint32_t>((SIMD_PANEL_STRIDE -
                               (elements + DMA_COMPLEX_ALIGNMENT - 1) / DMA_COMPLEX_ALIGNMENT * DMA_COMPLEX_ALIGNMENT) /
                              DMA_COMPLEX_ALIGNMENT),
        0};
    auto loadTarget = upper ? rawPanelLocal : panelLocal;
    AscendC::DataCopyPad(loadTarget, global[first * 2], panelCopy, padding);
    return first;
}

// Keep a short panel's real and imaginary lanes interleaved to use the whole vector.
template <typename I, typename M>
__simd_callee__ inline void InterleavedPanelIndices(I& indices, M& odd, M& mask)
{
    I positions, one, parity;
    AscendC::Reg::Arange(positions, 0);
    AscendC::Reg::Duplicate(one, 1);
    AscendC::Reg::And(parity, positions, one, mask);
    AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::EQ>(odd, parity, 1, mask);
    AscendC::Reg::Xor(indices, positions, one, mask);
}

template <typename R>
__simd_callee__ inline void LoadInterleavedPanel(R& value, __ubuf__ float* panel, uint32_t column, uint32_t completed)
{
    if (column < completed)
    {
        AscendC::Reg::LoadAlign(value, panel + column * (COMPLEX_COMPONENTS * SIMD_PANEL_STRIDE));
    }
    else
    {
        AscendC::Reg::Duplicate(value, 0.0f);
    }
}

template <typename R, typename I, typename M>
__simd_callee__ inline void NormalizeInterleavedPanel(R& value, R& negative, R& rotated, __ubuf__ float* panel,
                                                      __ubuf__ float* diagonal, uint32_t column, uint32_t completed,
                                                      I& index, M& odd, M& mask)
{
    if (column >= completed)
    {
        return;
    }
    R inverse, flipped;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
        inverse, diagonal + COMPLEX_COMPONENTS * column * (SIMD_PANEL_STRIDE + 1) + 1);
    AscendC::Reg::Mul(value, value, inverse, mask);
    AscendC::Reg::StoreAlign(panel + column * (COMPLEX_COMPONENTS * SIMD_PANEL_STRIDE), value, mask);
    // negative=[-re,-im] and rotated=[-im,+re] subtract the conjugated complex product.
    AscendC::Reg::Neg(negative, value, mask);
    AscendC::Reg::Gather(rotated, negative, index);
    AscendC::Reg::Neg(flipped, rotated, mask);
    AscendC::Reg::Select(rotated, flipped, rotated, odd);
}

template <bool UPPER, typename R, typename M>
__simd_callee__ inline void SubtractInterleavedPanel(R& value, R& negative, R& rotated, __ubuf__ float* diagonal,
                                                     uint32_t source, uint32_t target, M& mask)
{
    R real, imag;
    uint32_t index = UPPER ? target * SIMD_PANEL_STRIDE + source : source * SIMD_PANEL_STRIDE + target;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(real, diagonal + COMPLEX_COMPONENTS * index);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(imag,
                                                                         diagonal + COMPLEX_COMPONENTS * index + 1);
    if constexpr (UPPER)
    {
        AscendC::Reg::Neg(imag, imag, mask);
    }
    AscendC::Reg::MulAddDst(value, negative, real, mask);
    AscendC::Reg::MulAddDst(value, rotated, imag, mask);
}

template <bool UPPER, typename R, typename I, typename M>
__simd_callee__ inline void BeginInterleavedPanel(__ubuf__ float* panel, __ubuf__ float* diagonal, uint32_t start,
                                                  uint32_t completed, R& c0, R& c1, R& c2, R& c3, R& n0, R& t0, R& n1,
                                                  R& t1, I& index, M& odd, M& mask)
{
    NormalizeInterleavedPanel(c0, n0, t0, panel, diagonal, start, completed, index, odd, mask);
    if (start + 1 < completed)
    {
        SubtractInterleavedPanel<UPPER>(c1, n0, t0, diagonal, start, start + 1, mask);
    }
    if (start + 2 < completed)
    {
        SubtractInterleavedPanel<UPPER>(c2, n0, t0, diagonal, start, start + 2, mask);
    }
    if (start + 3 < completed)
    {
        SubtractInterleavedPanel<UPPER>(c3, n0, t0, diagonal, start, start + 3, mask);
    }
    NormalizeInterleavedPanel(c1, n1, t1, panel, diagonal, start + 1, completed, index, odd, mask);
    if (start + 2 < completed)
    {
        SubtractInterleavedPanel<UPPER>(c2, n1, t1, diagonal, start + 1, start + 2, mask);
    }
    if (start + 3 < completed)
    {
        SubtractInterleavedPanel<UPPER>(c3, n1, t1, diagonal, start + 1, start + 3, mask);
    }
}

template <bool UPPER, typename R, typename M>
__simd_callee__ inline void UpdateInterleavedTrailing(__ubuf__ float* panel, __ubuf__ float* diagonal, uint32_t column,
                                                      uint32_t start, R& n0, R& t0, R& n1, R& t1, R& n2, R& t2, R& n3,
                                                      R& t3, M& mask)
{
    R value;
    AscendC::Reg::LoadAlign(value, panel + column * (COMPLEX_COMPONENTS * SIMD_PANEL_STRIDE));
    SubtractInterleavedPanel<UPPER>(value, n0, t0, diagonal, start, column, mask);
    SubtractInterleavedPanel<UPPER>(value, n1, t1, diagonal, start + 1, column, mask);
    SubtractInterleavedPanel<UPPER>(value, n2, t2, diagonal, start + 2, column, mask);
    SubtractInterleavedPanel<UPPER>(value, n3, t3, diagonal, start + 3, column, mask);
    AscendC::Reg::StoreAlign(panel + column * (COMPLEX_COMPONENTS * SIMD_PANEL_STRIDE), value, mask);
}

template <bool UPPER>
__simd_vf__ inline void InterleavedPanelSolveVF(__ubuf__ float* panel, __ubuf__ float* diagonal, uint32_t completed,
                                                uint32_t first, uint32_t rows)
{
    uint32_t count = rows * COMPLEX_COMPONENTS;
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg odd;
    AscendC::Reg::RegTensor<int32_t> positions;
    InterleavedPanelIndices(positions, odd, mask);
    auto& index = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(positions);
    for (uint16_t start = first; start < completed; start += PANEL_STRIP)
    {
        AscendC::Reg::RegTensor<float> c0, c1, c2, c3, n0, t0, n1, t1, n2, t2, n3, t3;
        LoadInterleavedPanel(c0, panel, start, completed);
        LoadInterleavedPanel(c1, panel, start + 1, completed);
        LoadInterleavedPanel(c2, panel, start + 2, completed);
        LoadInterleavedPanel(c3, panel, start + 3, completed);
        BeginInterleavedPanel<UPPER>(panel, diagonal, start, completed, c0, c1, c2, c3, n0, t0, n1, t1, index, odd,
                                     mask);
        NormalizeInterleavedPanel(c2, n2, t2, panel, diagonal, start + 2, completed, index, odd, mask);
        if (start + 3 < completed)
        {
            SubtractInterleavedPanel<UPPER>(c3, n2, t2, diagonal, start + 2, start + 3, mask);
        }
        NormalizeInterleavedPanel(c3, n3, t3, panel, diagonal, start + 3, completed, index, odd, mask);
        for (uint16_t column = start + PANEL_STRIP; column < completed; ++column)
        {
            UpdateInterleavedTrailing<UPPER>(panel, diagonal, column, start, n0, t0, n1, t1, n2, t2, n3, t3, mask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

__aicore__ inline int SolvePanelTile(bool upper, int offset, int validRows, AscendC::LocalTensor<uint8_t>& storage,
                                     __ubuf__ aclsolverFloatComplex* panel, __ubuf__ aclsolverFloatComplex* rawDiagonal)
{
    int completed = storage.ReinterpretCast<int>().GetValue(PANEL_INFO_OFFSET_BYTES / sizeof(int));
    if (validRows <= INTERLEAVED_PANEL_ROWS)
    {
        if (upper)
        {
            asc_vf_call<InterleavedPanelSolveVF<true>>(reinterpret_cast<__ubuf__ float*>(panel),
                                                       reinterpret_cast<__ubuf__ float*>(rawDiagonal), completed,
                                                       offset ? 0 : 1, validRows);
        }
        else
        {
            asc_vf_call<InterleavedPanelSolveVF<false>>(reinterpret_cast<__ubuf__ float*>(panel),
                                                        reinterpret_cast<__ubuf__ float*>(rawDiagonal), completed,
                                                        offset ? 0 : 1, validRows);
        }
        return completed;
    }
    if (upper)
    {
        asc_vf_call<PanelSolveVF<true>>(reinterpret_cast<__ubuf__ float*>(panel),
                                        reinterpret_cast<__ubuf__ float*>(rawDiagonal), completed, offset ? 0 : 1);
    }
    else
    {
        asc_vf_call<PanelSolveVF<false>>(reinterpret_cast<__ubuf__ float*>(panel),
                                         reinterpret_cast<__ubuf__ float*>(rawDiagonal), completed, offset ? 0 : 1);
    }
    return completed;
}

__aicore__ inline void ProcessPanelTiles(
    __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* array, __gm__ float* work, int n, int lda, bool upper,
    int offset, int width, __gm__ int* info, int batch, int groups, int core, int cores, int rows, int padded,
    AscendC::LocalTensor<uint8_t>& storage, __ubuf__ aclsolverFloatComplex* diagonal,
    __ubuf__ aclsolverFloatComplex* panel, __ubuf__ aclsolverFloatComplex* low, __ubuf__ int* completedInfo,
    __ubuf__ aclsolverFloatComplex* rawDiagonal, __ubuf__ aclsolverFloatComplex* rawPanel,
    AscendC::LocalTensor<float>& rawDiagonalLocal, AscendC::LocalTensor<float>& rawPanelLocal,
    AscendC::LocalTensor<float>& panelLocal, bool retained)
{
    for (int m = core / groups; m < batch; m += cores / groups)
    {
        auto pointer = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : a;
        AscendC::GlobalTensor<float> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pointer));
        AscendC::DataCopyExtParams diagonalCopy{static_cast<uint16_t>(width),
                                                static_cast<uint32_t>(width * COMPLEX_BYTES),
                                                (static_cast<int64_t>(lda) - width) * COMPLEX_BYTES,
                                                static_cast<uint32_t>(UbPaddingBlocks(SIMD_PANEL_STRIDE, width)), 0};
        AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(rawDiagonalLocal,
                             global[static_cast<int64_t>(offset) * (static_cast<int64_t>(lda) + 1) * 2], diagonalCopy,
                             padding);
        // Diagonal and panel copies are independent; the panel barrier waits for both.
        for (int tile = core % groups * SIMD_PANEL_ROWS; tile < padded; tile += groups * SIMD_PANEL_ROWS)
        {
            int validRows = rows - tile < SIMD_PANEL_ROWS ? rows - tile : SIMD_PANEL_ROWS;
            int64_t first =
                LoadPanel(upper, validRows, width, lda, offset, tile, rawPanelLocal, panelLocal, global, padding);
            AscendC::PipeBarrier<PIPE_ALL>();
            asc_vf_call<PanelIOVF>(dim3(PANEL_IO_THREADS), work, n, lda, upper, offset, width, info, batch, m, tile,
                                   diagonal, panel, low, completedInfo, rawDiagonal, rawPanel, false, retained);
            AscendC::PipeBarrier<PIPE_ALL>();
            int completed = SolvePanelTile(upper, offset, validRows, storage, panel, rawDiagonal);
            AscendC::PipeBarrier<PIPE_ALL>();
            asc_vf_call<PanelIOVF>(dim3(PANEL_IO_THREADS), work, n, lda, upper, offset, width, info, batch, m, tile,
                                   diagonal, panel, low, completedInfo, rawDiagonal, rawPanel, true, retained);
            // Upper output requires the transpose; lower packing only reads the panel.
            if (upper)
            {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            if (completed)
            {
                StorePanel(upper, completed, validRows, lda, first, rawPanelLocal, panelLocal, global);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ inline void PreparePanel(__gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* array, __gm__ float* work,
                                    int n, int lda, bool upper, int offset, int width, __gm__ int* info, int batch,
                                    int groups, bool retained)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, SIMD_PANEL_BYTES);
    auto storage = buffer.Get<uint8_t>();
    auto diagonal = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(storage.GetPhyAddr());
    auto panel = diagonal + PANEL_OFFSET, low = diagonal + PANEL_LOW_OFFSET;
    auto completedInfo = reinterpret_cast<__ubuf__ int*>(low + SIMD_PANEL_ROWS);
    int core = AscendC::GetBlockIdx(), cores = AscendC::GetBlockNum() * AscendC::GetTaskRatio();
    int rows = n - offset - width, padded = (rows + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    auto rawDiagonal =
        reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(storage.GetPhyAddr() + PANEL_RAW_DIAGONAL_BYTES);
    auto rawPanel = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(storage.GetPhyAddr() + PANEL_RAW_PANEL_BYTES);
    auto local = storage.ReinterpretCast<float>();
    auto rawDiagonalLocal = local[PANEL_RAW_DIAGONAL_BYTES / sizeof(float)];
    auto rawPanelLocal = local[PANEL_RAW_PANEL_BYTES / sizeof(float)];
    auto panelLocal = local[PANEL_OFFSET * COMPLEX_COMPONENTS];
    ProcessPanelTiles(a, array, work, n, lda, upper, offset, width, info, batch, groups, core, cores, rows, padded,
                      storage, diagonal, panel, low, completedInfo, rawDiagonal, rawPanel, rawDiagonalLocal,
                      rawPanelLocal, panelLocal, retained);
}

__simt_callee__ inline aclsolverFloatComplex ReadPackedDiagonal(__ubuf__ aclsolverFloatComplex* diagonal, int matrix,
                                                                int row, int column, bool upper)
{
    auto z = ReadUB(diagonal + matrix * PACKED_DIAGONAL_STRIDE,
                    upper ? row * SIMD_PANEL_STRIDE + column : column * SIMD_PANEL_STRIDE + row);
    if (upper && row != column)
    {
        z.imag = -z.imag;
    }
    return z;
}

__simt_callee__ inline void PreparePackedMetadata(int64_t flat, int active, int64_t total, int rows, int matrixShift,
                                                  int firstMatrix, int width, int offset, const __gm__ int* info,
                                                  __ubuf__ aclsolverFloatComplex* diagonal, __ubuf__ uint32_t* metadata)
{
    int matrices =
        static_cast<int>((flat + active - 1 < total ? flat + active - 1 : total - 1) >> matrixShift) - firstMatrix + 1;
    for (int p = threadIdx.x; p < matrices * width; p += PANEL_IO_THREADS)
    {
        int m = p / width, c = p % width;
        int index = m * PACKED_DIAGONAL_STRIDE + c * (SIMD_PANEL_STRIDE + 1);
        auto z = ReadUB(diagonal, index);
        z.imag = 1.0f / z.real;
        WriteUB(diagonal, index, z);
    }
    if (threadIdx.x < SIMD_FLOAT_LANES)
    {
        int r = threadIdx.x;
        int row = r & ((1 << matrixShift) - 1);
        bool valid = r < active && flat + r < total && row < rows;
        int m = valid ? (flat + r) >> matrixShift : firstMatrix;
        int completed = valid ? (info[m] ? info[m] - offset - 1 : width) : 0;
        completed = completed < 0 ? 0 : (completed < width ? completed : width);
        metadata[r] = static_cast<uint32_t>((m - firstMatrix) * PACKED_DIAGONAL_STRIDE * 2);
        metadata[METADATA_COMPLETED_OFFSET + r] = completed;
        metadata[METADATA_BASE_OFFSET + r] = static_cast<uint32_t>(((r >> matrixShift) << matrixShift) * 2);
        metadata[METADATA_ROW_OFFSET + r] = static_cast<uint32_t>(row);
    }
    asc_syncthreads();
}

__simt_callee__ inline void CopyPackedPanel(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array,
                                            int64_t flat, int active, int64_t total, int rows, int matrixShift,
                                            int firstMatrix, int rowShift, int colShift, int rowSlots, int colSlots,
                                            int lda, int offset, int width, bool upper, bool output,
                                            __ubuf__ aclsolverFloatComplex* panel, __ubuf__ uint32_t* metadata)
{
    for (int p = threadIdx.x; p < colSlots * rowSlots; p += PANEL_IO_THREADS)
    {
        int r = upper ? p >> colShift : p & (rowSlots - 1), c = upper ? p & (colSlots - 1) : p >> rowShift;
        int row = r & ((1 << matrixShift) - 1);
        bool valid = r < active && flat + r < total && row < rows;
        int m = valid ? (flat + r) >> matrixShift : firstMatrix;
        auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        if (!output)
        {
            auto z = valid && c < metadata[METADATA_COMPLETED_OFFSET + r]
                         ? ReadLower(a, lda, offset + width + row, offset + c, upper)
                         : aclsolverFloatComplex{0, 0};
            WriteUB(panel, c * SIMD_PANEL_STRIDE + r, z);
        }
        else if (valid)
        {
            auto z = ReadUB(panel, c * SIMD_PANEL_STRIDE + r);
            if (c < metadata[METADATA_COMPLETED_OFFSET + r])
            {
                WriteLower(a, lda, offset + width + row, offset + c, upper, z);
            }
        }
    }
}

__simt_callee__ inline void StorePackedTail(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array,
                                            int64_t flat, int active, int64_t total, int rows, int matrixShift,
                                            int rowShift, int rowSlots, const __gm__ int* info, int lda, int offset,
                                            int width, bool upper, __ubuf__ aclsolverFloatComplex* tail)
{
    for (int p = threadIdx.x; p < rows * rowSlots; p += PANEL_IO_THREADS)
    {
        int r = p & (rowSlots - 1), c = p >> rowShift;
        if (r >= active || flat + r >= total)
        {
            continue;
        }
        int m = (flat + r) >> matrixShift, row = r & ((1 << matrixShift) - 1);
        if (row >= rows || row < c || info[m])
        {
            continue;
        }
        auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        auto z = ReadUB(tail, c * SIMD_PANEL_STRIDE + r);
        if (row == c)
        {
            z.imag = 0;
        }
        WriteLower(a, lda, offset + width + row, offset + width + c, upper, z);
    }
    return;
}

__simt_callee__ inline void CompensatePackedPanel(__gm__ float* work, int batch, int64_t stride, int capacity,
                                                  int64_t flat, int active, int64_t total, int matrixShift,
                                                  int firstMatrix, int rowShift, int rowSlots, int colSlots, int width,
                                                  bool upper, __ubuf__ aclsolverFloatComplex* diagonal,
                                                  __ubuf__ aclsolverFloatComplex* panel,
                                                  __ubuf__ aclsolverFloatComplex* low, __ubuf__ uint32_t* metadata)
{
    if (threadIdx.x < SIMD_FLOAT_LANES)
    {
        int r = threadIdx.x;
        aclsolverFloatComplex remainder{0, 0};
        if (r < active && flat + r < total && metadata[METADATA_COMPLETED_OFFSET + r])
        {
            int m = (flat + r) >> matrixShift, row = r & ((1 << matrixShift) - 1);
            auto firstColumn = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work + 2 * batch * stride) +
                               static_cast<int64_t>(m) * capacity;
            auto d = ReadPackedDiagonal(diagonal, m - firstMatrix, 0, 0, upper);
            auto value = NormalizeFirstColumn(ReadUB(panel, r), d.real, firstColumn[0].real, remainder);
            WriteUB(panel, r, value);
            WriteLower(firstColumn, capacity, width + row, 0, false, remainder);
        }
        WriteUB(low, r, remainder);
    }
    asc_syncthreads();
    for (int p = threadIdx.x; p < colSlots * rowSlots; p += PANEL_IO_THREADS)
    {
        int r = p & (rowSlots - 1), c = p >> rowShift;
        if (r >= active || flat + r >= total || !c || c >= metadata[METADATA_COMPLETED_OFFSET + r])
        {
            continue;
        }
        int m = (flat + r) >> matrixShift;
        auto firstColumn = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work + 2 * batch * stride) +
                           static_cast<int64_t>(m) * capacity;
        auto coefficient = ReadPackedDiagonal(diagonal, m - firstMatrix, c, 0, upper);
        auto value = SubFirstColumn(ReadUB(panel, c * SIMD_PANEL_STRIDE + r), ReadUB(panel, r), coefficient,
                                    ReadUB(low, r), LoadFirstColumn(firstColumn, 0, c, false));
        WriteUB(panel, c * SIMD_PANEL_STRIDE + r, value);
    }
}

__simt_callee__ inline void LoadPackedTail(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array,
                                           __gm__ float* work, int batch, int64_t stride, int capacity, int64_t flat,
                                           int active, int64_t total, int rows, int matrixShift, int firstMatrix,
                                           int rowShift, int rowSlots, const __gm__ int* info, int lda, int offset,
                                           int width, bool upper, __ubuf__ aclsolverFloatComplex* panel,
                                           __ubuf__ aclsolverFloatComplex* tail)
{
    for (int p = threadIdx.x; p < rows * rowSlots; p += PANEL_IO_THREADS)
    {
        int r = p & (rowSlots - 1), c = p >> rowShift;
        aclsolverFloatComplex z{0, 0};
        if (r < active && flat + r < total)
        {
            int m = (flat + r) >> matrixShift, row = r & ((1 << matrixShift) - 1);
            auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
            if (row < rows && row >= c && !info[m])
            {
                z = ReadLower(a, lda, offset + width + row, offset + width + c, upper);
                if (row == c)
                {
                    z.imag = 0;
                }
                if (!offset)
                {
                    auto firstColumn = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work + 2 * batch * stride) +
                                       static_cast<int64_t>(m) * capacity;
                    auto lowRow = LoadFirstColumn(firstColumn, 0, width + row, false);
                    auto lowCol = LoadFirstColumn(firstColumn, 0, width + c, false);
                    z = SubFirstColumn(z, ReadUB(panel, r), ReadUB(panel, ((m - firstMatrix) << matrixShift) + c),
                                       lowRow, lowCol);
                }
            }
        }
        WriteUB(tail, c * SIMD_PANEL_STRIDE + r, z);
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ __launch_bounds__(PANEL_IO_THREADS) void PackedPanelIOVF(
    __gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, __gm__ float* work, int n, int lda, bool upper,
    int offset, int width, const __gm__ int* info, int batch, int64_t flat, int active,
    __ubuf__ aclsolverFloatComplex* diagonal, __ubuf__ aclsolverFloatComplex* panel,
    __ubuf__ aclsolverFloatComplex* low, __ubuf__ uint32_t* metadata, __ubuf__ aclsolverFloatComplex* tail, bool output)
{
    int rows = n - offset - width, capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int matrixShift = PackedRowShift(rows);
    int64_t total = static_cast<int64_t>(batch) << matrixShift,
            stride = static_cast<int64_t>(capacity) * GRAM_COMPONENT_ROWS;
    int firstMatrix = flat >> matrixShift;
    int rowShift = 0, colShift = 0;
    while ((1 << rowShift) < active)
    {
        ++rowShift;
    }
    while ((1 << colShift) < width)
    {
        ++colShift;
    }
    int rowSlots = 1 << rowShift, colSlots = 1 << colShift;
    if (!output)
    {
        PreparePackedMetadata(flat, active, total, rows, matrixShift, firstMatrix, width, offset, info, diagonal,
                              metadata);
    }
    CopyPackedPanel(single, array, flat, active, total, rows, matrixShift, firstMatrix, rowShift, colShift, rowSlots,
                    colSlots, lda, offset, width, upper, output, panel, metadata);
    if (output)
    {
        StorePackedTail(single, array, flat, active, total, rows, matrixShift, rowShift, rowSlots, info, lda, offset,
                        width, upper, tail);
        return;
    }
    asc_syncthreads();
    if (!offset)
    {
        CompensatePackedPanel(work, batch, stride, capacity, flat, active, total, matrixShift, firstMatrix, rowShift,
                              rowSlots, colSlots, width, upper, diagonal, panel, low, metadata);
    }
    asc_syncthreads();
    LoadPackedTail(single, array, work, batch, stride, capacity, flat, active, total, rows, matrixShift, firstMatrix,
                   rowShift, rowSlots, info, lda, offset, width, upper, panel, tail);
}

template <bool UPPER, typename R, typename I, typename M>
__simd_callee__ inline void NormalizePackedColumn(R& ar, R& ai, __ubuf__ float* panel, __ubuf__ float* diagonal,
                                                  uint32_t column, uint32_t completed, M& mask, I& offsets, I& counts,
                                                  M& storeMask)
{
    if (column >= completed)
    {
        return;
    }
    R inverse, v0, v1;
    I index;
    M selected;
    AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(selected, counts, column, mask);
    AscendC::Reg::Adds(index, offsets, static_cast<uint32_t>(2 * (column * (SIMD_PANEL_STRIDE + 1)) + 1), mask);
    AscendC::Reg::Gather<float>(inverse, diagonal, index, selected);
    AscendC::Reg::Mul(ar, ar, inverse, selected);
    AscendC::Reg::Mul(ai, ai, inverse, selected);
    AscendC::Reg::Interleave(v0, v1, ar, ai);
    AscendC::Reg::StoreAlign(panel + column * (2 * SIMD_PANEL_STRIDE), v0, storeMask);
    AscendC::Reg::StoreAlign(panel + column * (2 * SIMD_PANEL_STRIDE) + SIMD_FLOAT_LANES, v1, storeMask);
}

template <bool UPPER, typename R, typename I, typename M>
__simd_callee__ inline void SubtractPackedColumn(R& ar, R& ai, R& pr, R& negativePr, R& negativePi,
                                                 __ubuf__ float* diagonal, uint32_t source, uint32_t target, M& mask,
                                                 I& offsets, I& counts)
{
    R cr, ci;
    I index;
    M selected;
    AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(selected, counts, target, mask);
    uint32_t element = UPPER ? target * SIMD_PANEL_STRIDE + source : source * SIMD_PANEL_STRIDE + target;
    AscendC::Reg::Adds(index, offsets, element * 2, mask);
    AscendC::Reg::Gather<float>(cr, diagonal, index, selected);
    AscendC::Reg::Adds(index, index, 1u, mask);
    AscendC::Reg::Gather<float>(ci, diagonal, index, selected);
    if constexpr (UPPER)
    {
        AscendC::Reg::Neg(ci, ci, mask);
    }
    AscendC::Reg::MulAddDst(ar, negativePr, cr, selected);
    AscendC::Reg::MulAddDst(ar, negativePi, ci, selected);
    AscendC::Reg::MulAddDst(ai, pr, ci, selected);
    AscendC::Reg::MulAddDst(ai, negativePi, cr, selected);
}

// Reuse solved register columns before reading and writing the next panel.
template <bool UPPER, typename R, typename I, typename M>
__simd_callee__ inline void UpdatePackedTrailing(__ubuf__ float* panel, __ubuf__ float* diagonal, uint16_t c,
                                                 uint16_t start, R& re0, R& nre0, R& nim0, R& re1, R& nre1, R& nim1,
                                                 R& re2, R& nre2, R& nim2, R& re3, R& nre3, R& nim3, M& mask,
                                                 I& offsets, I& counts, M& storeMask)
{
    AscendC::Reg::RegTensor<float> ar, ai, v0, v1;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(ar, ai,
                                                                            panel + c * (2 * SIMD_PANEL_STRIDE));
    SubtractPackedColumn<UPPER>(ar, ai, re0, nre0, nim0, diagonal, start + 0, c, mask, offsets, counts);
    SubtractPackedColumn<UPPER>(ar, ai, re1, nre1, nim1, diagonal, start + 1, c, mask, offsets, counts);
    SubtractPackedColumn<UPPER>(ar, ai, re2, nre2, nim2, diagonal, start + 2, c, mask, offsets, counts);
    SubtractPackedColumn<UPPER>(ar, ai, re3, nre3, nim3, diagonal, start + 3, c, mask, offsets, counts);
    AscendC::Reg::Interleave(v0, v1, ar, ai);
    AscendC::Reg::StoreAlign(panel + c * (2 * SIMD_PANEL_STRIDE), v0, storeMask);
    AscendC::Reg::StoreAlign(panel + c * (2 * SIMD_PANEL_STRIDE) + SIMD_FLOAT_LANES, v1, storeMask);
}

template <bool UPPER, typename R, typename I, typename M>
__simd_callee__ inline void BeginPackedStrip(__ubuf__ float* panel, __ubuf__ float* diagonal, uint16_t start,
                                             uint32_t completed, R& re0, R& im0, R& nre0, R& nim0, R& re1, R& im1,
                                             R& nre1, R& nim1, R& re2, R& im2, R& re3, R& im3, M& mask, I& offsets,
                                             I& counts, M& storeMask)
{
    NormalizePackedColumn<UPPER>(re0, im0, panel, diagonal, start + 0, completed, mask, offsets, counts, storeMask);
    AscendC::Reg::Neg(nre0, re0, mask);
    AscendC::Reg::Neg(nim0, im0, mask);
    if (start + 1 < completed)
    {
        SubtractPackedColumn<UPPER>(re1, im1, re0, nre0, nim0, diagonal, start + 0, start + 1, mask, offsets, counts);
    }
    if (start + 2 < completed)
    {
        SubtractPackedColumn<UPPER>(re2, im2, re0, nre0, nim0, diagonal, start + 0, start + 2, mask, offsets, counts);
    }
    if (start + 3 < completed)
    {
        SubtractPackedColumn<UPPER>(re3, im3, re0, nre0, nim0, diagonal, start + 0, start + 3, mask, offsets, counts);
    }
    NormalizePackedColumn<UPPER>(re1, im1, panel, diagonal, start + 1, completed, mask, offsets, counts, storeMask);

    AscendC::Reg::Neg(nre1, re1, mask);
    AscendC::Reg::Neg(nim1, im1, mask);
    if (start + 2 < completed)
    {
        SubtractPackedColumn<UPPER>(re2, im2, re1, nre1, nim1, diagonal, start + 1, start + 2, mask, offsets, counts);
    }
    if (start + 3 < completed)
    {
        SubtractPackedColumn<UPPER>(re3, im3, re1, nre1, nim1, diagonal, start + 1, start + 3, mask, offsets, counts);
    }
}

template <bool UPPER>
__simd_vf__ inline void PackedPanelSolveVF(__ubuf__ float* panel, __ubuf__ float* diagonal, uint32_t completed,
                                           uint32_t first, __ubuf__ uint32_t* metadata, uint32_t active)
{
    uint32_t count = active, fullCount = SIMD_FLOAT_LANES;
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    auto storeMask = AscendC::Reg::UpdateMask<float>(fullCount);
    AscendC::Reg::RegTensor<uint32_t> offsets, counts;
    AscendC::Reg::LoadAlign(offsets, metadata);
    AscendC::Reg::LoadAlign(counts, metadata + METADATA_COMPLETED_OFFSET);
    for (uint16_t start = first; start < completed; start += PANEL_STRIP)
    {
        AscendC::Reg::RegTensor<float> re0, im0, re1, im1, re2, im2, re3, im3;
        LoadComplexColumn(re0, im0, panel, start + 0, completed, 2 * SIMD_PANEL_STRIDE);

        LoadComplexColumn(re1, im1, panel, start + 1, completed, 2 * SIMD_PANEL_STRIDE);

        LoadComplexColumn(re2, im2, panel, start + 2, completed, 2 * SIMD_PANEL_STRIDE);

        LoadComplexColumn(re3, im3, panel, start + 3, completed, 2 * SIMD_PANEL_STRIDE);
        AscendC::Reg::RegTensor<float> nre0, nim0, nre1, nim1;
        BeginPackedStrip<UPPER>(panel, diagonal, start, completed, re0, im0, nre0, nim0, re1, im1, nre1, nim1, re2, im2,
                                re3, im3, mask, offsets, counts, storeMask);
        AscendC::Reg::RegTensor<float> nre2, nim2, nre3, nim3;
        NormalizePackedColumn<UPPER>(re2, im2, panel, diagonal, start + 2, completed, mask, offsets, counts, storeMask);

        AscendC::Reg::Neg(nre2, re2, mask);
        AscendC::Reg::Neg(nim2, im2, mask);
        if (start + 3 < completed)
        {
            SubtractPackedColumn<UPPER>(re3, im3, re2, nre2, nim2, diagonal, start + 2, start + 3, mask, offsets,
                                        counts);
        }
        NormalizePackedColumn<UPPER>(re3, im3, panel, diagonal, start + 3, completed, mask, offsets, counts, storeMask);

        AscendC::Reg::Neg(nre3, re3, mask);
        AscendC::Reg::Neg(nim3, im3, mask);
        for (uint16_t c = start + PANEL_STRIP; c < completed; ++c)
        {
            UpdatePackedTrailing<UPPER>(panel, diagonal, c, start, re0, nre0, nim0, re1, nre1, nim1, re2, nre2, nim2,
                                        re3, nre3, nim3, mask, offsets, counts, storeMask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

// Update a short trailing triangle while its solved panel is still in UB.
__simd_vf__ inline void PackedPanelSchurVF(__ubuf__ float* panel, __ubuf__ float* tail, __ubuf__ uint32_t* metadata,
                                           uint32_t rows, uint32_t width, uint32_t first, uint32_t active)
{
    uint32_t count = active, fullCount = SIMD_FLOAT_LANES;
    auto mask = AscendC::Reg::UpdateMask<float>(count);
    auto storeMask = AscendC::Reg::UpdateMask<float>(fullCount);
    AscendC::Reg::RegTensor<uint32_t> bases, rowIndex, counts, index;
    AscendC::Reg::LoadAlign(bases, metadata + METADATA_BASE_OFFSET);
    AscendC::Reg::LoadAlign(rowIndex, metadata + METADATA_ROW_OFFSET);
    AscendC::Reg::LoadAlign(counts, metadata + METADATA_COMPLETED_OFFSET);
    AscendC::Reg::MaskReg valid, triangle, selected;
    AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(valid, counts, width, mask);
    for (uint16_t col = 0; col < rows; ++col)
    {
        AscendC::Reg::RegTensor<float> ar, ai, v0, v1;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(ar, ai,
                                                                                tail + col * (2 * SIMD_PANEL_STRIDE));
        AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GE>(triangle, rowIndex, col, mask);
        AscendC::Reg::And(selected, valid, triangle, mask);
        for (uint16_t k = first; k < width; ++k)
        {
            AscendC::Reg::RegTensor<float> pr, pi, cr, ci, ncr, nci;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
                pr, pi, panel + k * (2 * SIMD_PANEL_STRIDE));
            AscendC::Reg::Adds(index, bases, static_cast<uint32_t>(k * 2 * SIMD_PANEL_STRIDE + 2 * col), mask);
            AscendC::Reg::Gather<float>(cr, panel, index, selected);
            AscendC::Reg::Adds(index, index, 1u, mask);
            AscendC::Reg::Gather<float>(ci, panel, index, selected);
            AscendC::Reg::Neg(ncr, cr, mask);
            AscendC::Reg::Neg(nci, ci, mask);
            AscendC::Reg::MulAddDst(ar, pr, ncr, selected);
            AscendC::Reg::MulAddDst(ai, pr, ci, selected);
            AscendC::Reg::MulAddDst(ar, pi, nci, selected);
            AscendC::Reg::MulAddDst(ai, pi, ncr, selected);
        }
        AscendC::Reg::Interleave(v0, v1, ar, ai);
        AscendC::Reg::StoreAlign(tail + col * (2 * SIMD_PANEL_STRIDE), v0, storeMask);
        AscendC::Reg::StoreAlign(tail + col * (2 * SIMD_PANEL_STRIDE) + SIMD_FLOAT_LANES, v1, storeMask);
    }
}

__aicore__ inline void LoadPackedDiagonal(__gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* array, int m,
                                          int first, int width, int lda, int offset,
                                          AscendC::LocalTensor<uint8_t>& storage)
{
    auto pointer = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : a;
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pointer));
    AscendC::DataCopyExtParams copy{static_cast<uint16_t>(width), static_cast<uint32_t>(width * COMPLEX_BYTES),
                                    (static_cast<int64_t>(lda) - width) * COMPLEX_BYTES,
                                    static_cast<uint32_t>(UbPaddingBlocks(SIMD_PANEL_STRIDE, width)), 0};
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    auto local = storage.ReinterpretCast<float>()[static_cast<int64_t>(m - first) * PACKED_DIAGONAL_STRIDE * 2];
    AscendC::DataCopyPad(local, global[static_cast<int64_t>(offset) * (static_cast<int64_t>(lda) + 1) * 2], copy,
                         padding);
}

__aicore__ inline void PreparePackedPanel(__gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* array,
                                          __gm__ float* work, int n, int lda, bool upper, int offset, int width,
                                          __gm__ int* info, int batch)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, PACKED_PANEL_BYTES);
    auto storage = buffer.Get<uint8_t>();
    auto diagonal = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(storage.GetPhyAddr());
    auto panel = diagonal + PACKED_PANEL_OFFSET, low = diagonal + PACKED_LOW_OFFSET;
    auto metadata = reinterpret_cast<__ubuf__ uint32_t*>(low + SIMD_FLOAT_LANES);
    auto tail = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(storage.GetPhyAddr() + PACKED_TAIL_BYTES);
    int core = AscendC::GetBlockIdx(), cores = AscendC::GetBlockNum() * AscendC::GetTaskRatio();
    int rows = n - offset - width;
    int matrixShift = PackedRowShift(rows);
    int64_t total = static_cast<int64_t>(batch) << matrixShift;
    int active = (1 << matrixShift) * PACKED_MATRICES_PER_AIV;
    for (int64_t flat = static_cast<int64_t>(core) * active; flat < total; flat += static_cast<int64_t>(cores) * active)
    {
        int first = flat >> matrixShift,
            last = (flat + active - 1 < total ? flat + active - 1 : total - 1) >> matrixShift;
        for (int m = first; m <= last; ++m)
        {
            LoadPackedDiagonal(a, array, m, first, width, lda, offset, storage);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<PackedPanelIOVF>(dim3(PANEL_IO_THREADS), a, array, work, n, lda, upper, offset, width, info, batch,
                                     flat, active, diagonal, panel, low, metadata, tail, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (upper)
        {
            asc_vf_call<PackedPanelSolveVF<true>>(reinterpret_cast<__ubuf__ float*>(panel),
                                                  reinterpret_cast<__ubuf__ float*>(diagonal), width, offset ? 0 : 1,
                                                  metadata, active);
        }
        else
        {
            asc_vf_call<PackedPanelSolveVF<false>>(reinterpret_cast<__ubuf__ float*>(panel),
                                                   reinterpret_cast<__ubuf__ float*>(diagonal), width, offset ? 0 : 1,
                                                   metadata, active);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<PackedPanelSchurVF>(reinterpret_cast<__ubuf__ float*>(panel),
                                        reinterpret_cast<__ubuf__ float*>(tail), metadata, rows, width, offset ? 0 : 1,
                                        active);
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<PackedPanelIOVF>(dim3(PANEL_IO_THREADS), a, array, work, n, lda, upper, offset, width, info, batch,
                                     flat, active, diagonal, panel, low, metadata, tail, true);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <bool FIRST = false>
__global__ __aicore__ __vector__ void PanelKernel(__gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* array,
                                                  __gm__ float* work, int n, int lda, bool upper, int offset, int width,
                                                  __gm__ int* info, int batch, int groups, bool retained)
{
    if constexpr (FIRST)
    {
        asc_vf_call<PanelApplyVF>(dim3(PANEL_THREADS), a, array, work, n, lda, upper, offset, width, info, batch,
                                  groups);
    }
    else
    {
        int rows = n - offset - width;
        if (batch >= BATCH_THRESHOLD && rows <= PACKED_TAIL_ROWS)
        {
            PreparePackedPanel(a, array, work, n, lda, upper, offset, width, info, batch);
        }
        else
        {
            PreparePanel(a, array, work, n, lda, upper, offset, width, info, batch, groups, retained);
        }
    }
}
}  // namespace
void LaunchPanel(aclrtStream stream, bool first, int blocks, size_t sharedBytes, aclsolverFloatComplex* a,
                 const uint64_t* array, float* work, int n, int lda, bool upper, int offset, int width, int* info,
                 int batch, int groups)
{
    bool retained = UseFactorDiagonalState(batch);
    if (first)
    {
        PanelKernel<true>
            <<<blocks, nullptr, stream>>>(a, array, work, n, lda, upper, offset, width, info, batch, groups, retained);
    }
    else
    {
        PanelKernel<false><<<blocks, sharedBytes, stream>>>(a, array, work, n, lda, upper, offset, width, info, batch,
                                                            groups, retained);
    }
}
}  // namespace Cholesky
