/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define K_MAX_SHAPE_DIM 0
#include "../../utils/cholesky/cholesky.h"
#include "../../utils/cholesky/cholesky_update.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "tiling/platform/platform_ascendc.h"

namespace Cholesky
{
namespace
{
using Arch = Catlass::Arch::Ascend950;
// Full FP32 products and accumulation: HF32 is deliberately disabled.
using Policy = Catlass::Gemm::MmadPingpong<Arch, true, false, GEMM_ACCUMULATOR_STAGES, false, GEMM_INPUT_STAGES,
                                           GEMM_INPUT_STAGES, GEMM_INPUT_STAGES, GEMM_INPUT_STAGES>;
using Copy = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<Arch, float, Catlass::layout::RowMajor, float,
                                                        Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor,
                                                        void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using Schedule = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<GEMM_SWIZZLE_LOG, 0>;
__aicore__ inline uint32_t TriangleRow(uint32_t tile, uint32_t dimension)
{
    uint32_t row = 0, end = dimension;
    while (row + 1 < end)
    {
        uint32_t middle = (row + end) / 2;
        if (middle * (middle + 1) / 2 <= tile)
        {
            row = middle;
        }
        else
        {
            end = middle;
        }
    }
    return row;
}

__aicore__ inline void LoadGramCorrection(AscendC::LocalTensor<float>& first, AscendC::GlobalTensor<float>& a,
                                          __gm__ float* packed, int capacity, int batch, int m, int offset, int minor,
                                          int major, uint32_t count, int columns,
                                          AscendC::DataCopyPadExtParams<float>& padding)
{
    AscendC::GlobalTensor<float> low;
    int64_t stride = static_cast<int64_t>(capacity) * GRAM_COMPONENT_ROWS;
    low.SetGlobalBuffer(packed + 2 * static_cast<int64_t>(batch) * stride + 2 * static_cast<int64_t>(m) * capacity);
    AscendC::DataCopyExtParams rowCopy{1, count * COMPLEX_BYTES, 0, 0, 0};
    AscendC::DataCopyExtParams colCopy{1, static_cast<uint32_t>(columns * COMPLEX_BYTES), 0, 0, 0};
    AscendC::DataCopyPad(first, a[2 * (offset + minor)], rowCopy, padding);
    AscendC::DataCopyPad(first[SCHUR_ROW_LOW], low[2 * (offset + minor)], rowCopy, padding);
    AscendC::DataCopyPad(first[SCHUR_COLUMN_HIGH], a[2 * (offset + major)], colCopy, padding);
    AscendC::DataCopyPad(first[SCHUR_COLUMN_LOW], low[2 * (offset + major)], colCopy, padding);
}

template <uint32_t TILE>
__aicore__ inline void ApplyGramTile(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array,
                                     __gm__ float* packed, int capacity, int batch, int m, int offset, int minor,
                                     int major, uint32_t count, int columns, int lda, bool upper, uint32_t row,
                                     uint32_t col, uint32_t sub, int half, uint32_t iteration,
                                     AscendC::LocalTensor<float>& matrix, AscendC::LocalTensor<float>& rr,
                                     AscendC::LocalTensor<float>& ir, AscendC::LocalTensor<float>& first,
                                     AscendC::GlobalTensor<float>& a)
{
    auto pointer = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pointer));
    int64_t matrixOffset = 2 * (static_cast<int64_t>(offset + major) * lda + offset + minor);
    AscendC::DataCopyExtParams matrixCopy{static_cast<uint16_t>(columns), count * COMPLEX_BYTES,
                                          (static_cast<int64_t>(lda) - count) * COMPLEX_BYTES,
                                          static_cast<uint32_t>(UbPaddingBlocks(TILE, count)), 0};
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(matrix, a[matrixOffset], matrixCopy, padding);
    bool compensated = offset == FACTOR_BLOCK && !upper;
    if (compensated)
    {
        LoadGramCorrection(first, a, packed, capacity, batch, m, offset, minor, major, count, columns, padding);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    ApplySchurRectangleVF<TILE>(
        reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(rr.GetPhyAddr()) + (iteration % SCHUR_BUFFER_COUNT) * (TILE * TILE / 2),
        reinterpret_cast<__ubuf__ float*>(ir.GetPhyAddr()) + (iteration % SCHUR_BUFFER_COUNT) * (TILE * TILE / 2),
        count, columns, row == col, upper, sub * half, reinterpret_cast<__ubuf__ float*>(first.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(first.GetPhyAddr()) + SCHUR_COLUMN_HIGH, compensated);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams copyOut{static_cast<uint16_t>(columns), count * COMPLEX_BYTES, matrixCopy.dstStride,
                                       (static_cast<int64_t>(lda) - count) * COMPLEX_BYTES, 0};
    AscendC::DataCopyPad(a[matrixOffset], matrix, copyOut);
}

template <uint32_t TILE>
__aicore__ inline void ApplyGram(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int rows,
                                 int capacity, int batch, bool upper, int n, int lda, int offset,
                                 const __gm__ int* info, __gm__ float* packed)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixBuffer, realBuffer, imagBuffer;
    pipe.InitBuffer(matrixBuffer, TILE * TILE * sizeof(float));
    pipe.InitBuffer(realBuffer, TILE * TILE * sizeof(float));
    pipe.InitBuffer(imagBuffer, TILE * TILE * sizeof(float));
    AscendC::TBuf<AscendC::TPosition::VECCALC> firstBuffer;
    pipe.InitBuffer(firstBuffer, SCHUR_CORRECTION_FLOATS * sizeof(float));
    auto first = firstBuffer.Get<float>();
    auto matrix = matrixBuffer.Get<float>(), rr = realBuffer.Get<float>(), ir = imagBuffer.Get<float>();
    uint32_t dimension = (rows + TILE - 1) / TILE, tiles = dimension * (dimension + 1) / 2;
    uint32_t subblocks = AscendC::GetSubBlockNum(), sub = AscendC::GetBlockIdx() % subblocks;
    uint32_t core = AscendC::GetBlockIdx() / subblocks;
    int actualRows = n - offset;
    uint32_t iteration = 0;
    AscendC::GlobalTensor<float> a;
    for (uint64_t job = core; job < static_cast<uint64_t>(tiles) * batch; job += AscendC::GetBlockNum())
    {
        uint32_t m = job / tiles, tile = job % tiles;
        uint32_t row = TriangleRow(tile, dimension);
        uint32_t col = tile - row * (row + 1) / 2;
        int majorBlock = (upper ? row : col) * TILE, minor = (upper ? col : row) * TILE;
        int half = (rows - majorBlock < TILE ? rows - majorBlock : TILE) / 2;
        int major = majorBlock + sub * half;
        int columns = actualRows - major < half ? actualRows - major : half;
        uint32_t count = actualRows - minor < TILE ? actualRows - minor : TILE;
        uint16_t ready = (iteration % SCHUR_BUFFER_COUNT) * FLAGS_PER_SCHUR_BUFFER;
        AscendC::CrossCoreWaitFlag<CROSS_CORE_MODE, PIPE_V>(ready);
        if (columns > 0 && count > 0 && info[m] == 0)
        {
            ApplyGramTile<TILE>(single, array, packed, capacity, batch, m, offset, minor, major, count, columns, lda,
                                upper, row, col, sub, half, iteration, matrix, rr, ir, first, a);
        }
        // Empty tiles must also finish the ready wait before acknowledging reuse.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<CROSS_CORE_MODE, PIPE_MTE3>(ready + 1);
        ++iteration;
    }
}

template <uint32_t TILE, typename Resource, typename BlockMmad, typename LA, typename LB>
__aicore__ inline void GramProductPhase(GM_ADDR real, GM_ADDR imag, uint32_t matrix, int64_t packedStride, int phase,
                                        uint32_t iteration, Resource& resource, BlockMmad& blockMmad, LA& la, LB& lb,
                                        Catlass::GemmCoord& coord, Catlass::GemmCoord& actual)
{
    AscendC::GlobalTensor<float> ga, gb;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(phase ? imag : real) + matrix * packedStride);
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(real) + matrix * packedStride);
    auto ta = tla::MakeTensor(ga, la, Catlass::Arch::PositionGM{});
    auto tb = tla::MakeTensor(gb, lb, Catlass::Arch::PositionGM{});
    uint32_t ubOffset = TILE * TILE * sizeof(float) + static_cast<uint32_t>(phase) * TILE * TILE * sizeof(float) +
                        (iteration % SCHUR_BUFFER_COUNT) * TILE * TILE * sizeof(float) / SCHUR_BUFFER_COUNT;
    auto product = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    auto layout = tla::MakeLayout(tla::MakeShape(actual.m(), actual.n()), tla::MakeStride(TILE, tla::Int<1>{}));
    auto tc = tla::MakeTensor(product, layout, Catlass::Arch::PositionUB{});
    auto leftTile =
        tla::GetTile(ta, tla::MakeCoord(coord.m() * TILE, 0u), tla::MakeShape(actual.m(), GRAM_COMPONENT_ROWS));
    auto rightTile =
        tla::GetTile(tb, tla::MakeCoord(0u, coord.n() * TILE), tla::MakeShape(GRAM_COMPONENT_ROWS, actual.n()));
    blockMmad(leftTile, rightTile, tc, actual);
}

__aicore__ inline void WaitGramBuffer(uint32_t iteration)
{
    if (iteration >= SCHUR_BUFFER_COUNT)
    {
        uint16_t free = (iteration % SCHUR_BUFFER_COUNT) * FLAGS_PER_SCHUR_BUFFER + 1;
        AscendC::CrossCoreWaitFlag<CROSS_CORE_MODE, PIPE_FIX>(free);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_MODE, PIPE_FIX>(free + SECOND_AIV_FLAG_OFFSET);
    }
}

__aicore__ inline void DrainGramBuffers(uint32_t iteration)
{
    for (uint32_t i = iteration > SCHUR_BUFFER_COUNT ? iteration - SCHUR_BUFFER_COUNT : 0; i < iteration; ++i)
    {
        uint16_t free = (i % SCHUR_BUFFER_COUNT) * FLAGS_PER_SCHUR_BUFFER + 1;
        AscendC::CrossCoreWaitFlag<CROSS_CORE_MODE, PIPE_FIX>(free);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_MODE, PIPE_FIX>(free + SECOND_AIV_FLAG_OFFSET);
    }
}

template <uint32_t TILE>
__global__ __aicore__ void RealGram(GM_ADDR real, GM_ADDR imag, int rows, int capacity, int batch, bool upper,
                                    __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* array, int n, int lda,
                                    int offset, const __gm__ int* info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV
    {
        ApplyGram<TILE>(a, array, rows, capacity, batch, upper, n, lda, offset, info,
                        reinterpret_cast<__gm__ float*>(real));
        return;
    }
    uint32_t m = static_cast<uint32_t>(rows);
    auto la = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, GRAM_COMPONENT_ROWS);
    auto lb = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(GRAM_COMPONENT_ROWS, m);
    Catlass::GemmCoord shape{m, m, GRAM_COMPONENT_ROWS};
    Schedule schedule(shape, Catlass::MakeCoord(TILE, TILE));
    uint32_t dimension = (m + TILE - 1) / TILE;
    uint32_t tiles = dimension * (dimension + 1) / 2;
    Catlass::Arch::Resource<Arch> resource;
    using GramMmad = Catlass::Gemm::Block::BlockMmadTla<
        Policy, tla::Shape<tla::Int<TILE>, tla::Int<TILE>, tla::Int<GRAM_COMPONENT_ROWS>>,
        tla::Shape<tla::Int<TILE>, tla::Int<TILE>, tla::Int<SCHUR_REDUCTION_TILE>>, float, float, float, void, Copy>;
    GramMmad blockMmad(resource);
    uint32_t iteration = 0;
    for (uint64_t job = AscendC::GetBlockIdx(); job < static_cast<uint64_t>(tiles) * batch;
         job += AscendC::GetBlockNum())
    {
        WaitGramBuffer(iteration);
        uint32_t matrix = job / tiles;
        uint32_t tile = job % tiles;
        uint32_t row = TriangleRow(tile, dimension);
        uint32_t col = tile - row * (row + 1) / 2;
        Catlass::GemmCoord coord = upper ? Catlass::GemmCoord{row, col, 0} : Catlass::GemmCoord{col, row, 0};
        auto actual = schedule.GetActualBlockShape(coord);
        int64_t packedStride = static_cast<int64_t>(capacity) * GRAM_COMPONENT_ROWS;
        for (int phase = 0; phase < COMPLEX_COMPONENTS; ++phase)
        {
            GramProductPhase<TILE>(real, imag, matrix, packedStride, phase, iteration, resource, blockMmad, la, lb,
                                   coord, actual);
        }
        uint16_t ready = (iteration % SCHUR_BUFFER_COUNT) * FLAGS_PER_SCHUR_BUFFER;
        AscendC::CrossCoreSetFlag<CROSS_CORE_MODE, PIPE_FIX>(ready);
        AscendC::CrossCoreSetFlag<CROSS_CORE_MODE, PIPE_FIX>(ready + SECOND_AIV_FLAG_OFFSET);
        ++iteration;
    }
    DrainGramBuffers(iteration);
    AscendC::PipeBarrier<PIPE_ALL>();
}
// Direct complex rectangle products use the factor's original interleaved storage.
template <bool COLUMN>
__aicore__ inline auto ProductLayout(uint32_t rows, uint32_t cols, int lda)
{
    if constexpr (COLUMN)
    {
        return tla::MakeLayout(tla::MakeShape(rows, cols),
                               tla::MakeStride(tla::Int<1>{}, static_cast<int64_t>(2) * lda));
    }
    else
    {
        return tla::MakeLayout(tla::MakeShape(rows, cols),
                               tla::MakeStride(static_cast<int64_t>(2) * lda, tla::Int<1>{}));
    }
}

template <bool COLUMN>
__aicore__ inline __gm__ aclsolverFloatComplex* SolveProductOperand(GM_ADDR single, __gm__ const uint64_t* array,
                                                                    GM_ADDR work, uint32_t matrix, int n, int nrhs,
                                                                    int lda, int capacity, int rhsPad, int batch,
                                                                    int offset, int width, bool backward, bool diagonal)
{
    auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[matrix])
                   : reinterpret_cast<__gm__ aclsolverFloatComplex*>(single);
    int first = backward ? 0 : offset + width;
    if (diagonal)
    {
        int block = SolveBlockSize(nrhs, batch);
        int panels = (n + block - 1) / block;
        auto inverse = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work) +
                       static_cast<int64_t>(NATIVE_RHS_ROWS + COMPLEX_COMPONENTS * capacity) * rhsPad * batch;
        int64_t tile = static_cast<int64_t>(matrix) * panels + offset / block;
        a = inverse + (tile + (backward ? static_cast<int64_t>(batch) * panels : 0)) * block * block;
    }
    else
    {
        a += COLUMN ? static_cast<int64_t>(offset) * lda + first : static_cast<int64_t>(first) * lda + offset;
    }
    return a;
}

template <typename BlockMmad, typename LA, typename LC, typename RightTile>
__aicore__ inline void SolveProductPhase(__gm__ float* base, int phase, int batch, uint32_t matrix, int64_t bStride,
                                         int64_t cStride, uint32_t k, LA& la, LC& lc, BlockMmad& blockMmad,
                                         RightTile& rightTile, Catlass::GemmCoord& coord, Catlass::GemmCoord& actual)
{
    AscendC::GlobalTensor<float> gb, gc;
    gb.SetGlobalBuffer(base + (static_cast<int64_t>(phase) * batch + matrix) * bStride);
    gc.SetGlobalBuffer(base + 2 * batch * bStride + (static_cast<int64_t>(phase) * batch + matrix) * cStride);
    auto ta = tla::MakeTensor(gb, la, Catlass::Arch::PositionGM{});
    auto tc = tla::MakeTensor(gc, lc, Catlass::Arch::PositionGM{});
    auto leftTile = tla::GetTile(ta, tla::MakeCoord(coord.m() * SCHUR_TILE, 0u), tla::MakeShape(actual.m(), k));
    auto resultTile = tla::GetTile(tc, tla::MakeCoord(coord.m() * SCHUR_TILE, coord.n() * SCHUR_TILE),
                                   tla::MakeShape(actual.m(), actual.n()));
    blockMmad(leftTile, rightTile, resultTile, actual);
}

template <bool COLUMN, typename BlockMmad, typename LA, typename LB, typename LC>
__aicore__ inline void SolveProductJob(uint64_t job, uint32_t tiles, GM_ADDR single, __gm__ const uint64_t* array,
                                       GM_ADDR work, int n, int nrhs, int lda, int capacity, int rhsPad, int batch,
                                       int offset, int width, bool backward, bool diagonal, Schedule& schedule,
                                       BlockMmad& blockMmad, LA& la, LB& lb, LC& lc, uint32_t k, int64_t bStride,
                                       int64_t cStride)
{
    uint32_t matrix = job / tiles;
    auto coord = schedule.GetBlockCoord(job % tiles);
    auto actual = schedule.GetActualBlockShape(coord);
    auto a = SolveProductOperand<COLUMN>(single, array, work, matrix, n, nrhs, lda, capacity, rhsPad, batch, offset,
                                         width, backward, diagonal);
    AscendC::GlobalTensor<float> ga;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    auto tb = tla::MakeTensor(ga, lb, Catlass::Arch::PositionGM{});
    auto rightTile = tla::GetTile(tb, tla::MakeCoord(0u, coord.n() * SCHUR_TILE), tla::MakeShape(k, actual.n()));
    auto base = reinterpret_cast<__gm__ float*>(work);
    for (int phase = 0; phase < COMPLEX_COMPONENTS; ++phase)
    {
        SolveProductPhase(base, phase, batch, matrix, bStride, cStride, k, la, lc, blockMmad, rightTile, coord, actual);
    }
}

template <bool COLUMN>
__global__ __aicore__ __cube__ void RealSolveProduct(GM_ADDR single, __gm__ const uint64_t* array, GM_ADDR work, int n,
                                                     int nrhs, int lda, int batch, int offset, int width, bool backward,
                                                     const __gm__ int* check, bool diagonal, int activeRhs)
{
    if (check && *check != 0)
    {
        return;
    }
    using BLayout = std::conditional_t<COLUMN, Catlass::layout::RowMajor, Catlass::layout::ColumnMajor>;
    // Compute the transpose so that each RHS has contiguous output rows.
    using ProductCopy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, float, Catlass::layout::ColumnMajor, float,
                                                               BLayout, float, Catlass::layout::RowMajor, void>;
    using ProductMmad = Catlass::Gemm::Block::BlockMmadTla<
        Policy, tla::Shape<tla::Int<SCHUR_TILE>, tla::Int<SCHUR_TILE>, tla::Int<SOLVE_REDUCTION_TILE>>,
        tla::Shape<tla::Int<SCHUR_TILE>, tla::Int<SCHUR_TILE>, tla::Int<SOLVE_L0_REDUCTION_TILE>>, float, float, float,
        void, ProductCopy>;
    int rows = diagonal ? width : (backward ? offset : n - offset - width);
    uint32_t m = COLUMN ? 2 * rows : rows;
    uint32_t k = COLUMN ? width : 2 * width;
    uint32_t rhsPad = (nrhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int64_t bStride = static_cast<int64_t>(NATIVE_RHS_ROWS) * rhsPad;
    int64_t cStride = static_cast<int64_t>(2) * capacity * rhsPad;
    auto la = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(rhsPad, k);
    auto lb = ProductLayout<!COLUMN>(k, m, diagonal ? SolveBlockSize(nrhs, batch) : lda);
    auto lc =
        tla::MakeLayout(tla::MakeShape(rhsPad, m), tla::MakeStride(static_cast<int64_t>(2) * capacity, tla::Int<1>{}));
    Catlass::GemmCoord shape{static_cast<uint32_t>((activeRhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT),
                             m, k};
    Schedule schedule(shape, Catlass::MakeCoord(static_cast<uint32_t>(SCHUR_TILE), static_cast<uint32_t>(SCHUR_TILE)));
    uint32_t tiles = schedule.GetCoreLoops();
    Catlass::Arch::Resource<Arch> resource;
    ProductMmad blockMmad(resource);
    for (uint64_t job = AscendC::GetBlockIdx(); job < static_cast<uint64_t>(tiles) * batch;
         job += AscendC::GetBlockNum())
    {
        SolveProductJob<COLUMN>(job, tiles, single, array, work, n, nrhs, lda, capacity, rhsPad, batch, offset, width,
                                backward, diagonal, schedule, blockMmad, la, lb, lc, k, bStride, cStride);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
}  // namespace

void Schur(aclrtStream stream, int rows, float* real, float* imag, int capacity, int batch, bool upper, int n,
           aclsolverFloatComplex* a, aclsolverFloatComplex** array, int lda, int offset, const int* info)
{
    uint32_t cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t tile = batch >= BATCH_THRESHOLD ? BATCH_SCHUR_TILE : SCHUR_TILE;
    uint32_t dimension = (rows + tile - 1) / tile;
    uint32_t tiles = dimension * (dimension + 1) / 2 * batch;
    uint32_t blocks = cores < tiles ? cores : tiles;
    if (batch >= BATCH_THRESHOLD)
    {
        RealGram<BATCH_SCHUR_TILE><<<blocks, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(real), reinterpret_cast<uint8_t*>(imag), rows, capacity, batch, upper, a,
            reinterpret_cast<const uint64_t*>(array), n, lda, offset, info);
    }
    else
    {
        RealGram<SCHUR_TILE><<<blocks, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(real), reinterpret_cast<uint8_t*>(imag), rows, capacity, batch, upper, a,
            reinterpret_cast<const uint64_t*>(array), n, lda, offset, info);
    }
}
void SolveProduct(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                  const aclsolverFloatComplex* const* array, int lda, float* work, int batch, int offset, int width,
                  bool backward, const int* check, bool diagonal, int activeRhs)
{
    if (activeRhs < 0)
    {
        activeRhs = nrhs;
    }
    int rows = diagonal ? width : (backward ? offset : n - offset - width);
    bool column = diagonal || upper == backward;
    int m = column ? 2 * rows : rows;
    uint64_t tiles =
        static_cast<uint64_t>((m + SCHUR_TILE - 1) / SCHUR_TILE) *
        (((activeRhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT + SCHUR_TILE - 1) / SCHUR_TILE) * batch;
    uint32_t cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t blocks = tiles < cores ? tiles : cores;
    auto single = reinterpret_cast<uint8_t*>(const_cast<aclsolverFloatComplex*>(a));
    auto pointers = reinterpret_cast<const uint64_t*>(array);
    auto scratch = reinterpret_cast<uint8_t*>(work);
    if (column)
    {
        RealSolveProduct<true><<<blocks, nullptr, stream>>>(single, pointers, scratch, n, nrhs, lda, batch, offset,
                                                            width, backward, check, diagonal, activeRhs);
    }
    else
    {
        RealSolveProduct<false><<<blocks, nullptr, stream>>>(single, pointers, scratch, n, nrhs, lda, batch, offset,
                                                             width, backward, check, diagonal, activeRhs);
    }
}
}  // namespace Cholesky
