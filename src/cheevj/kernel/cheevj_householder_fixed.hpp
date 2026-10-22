/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cheevj_householder_fixed.hpp
 * \brief Fixed-shape Householder panel implementations selected by the Cheevj dispatcher.
 */

#ifndef CHEEVJ_C64_HOUSEHOLDER_FIXED_HPP
#define CHEEVJ_C64_HOUSEHOLDER_FIXED_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace Cheevj
{

// N=512 persistent branch.

constexpr int CHEEVJ_HH_PERSISTENT_N = 512;
constexpr int CHEEVJ_HH_PERSISTENT_B = 4;
constexpr int CHEEVJ_HH_PERSISTENT_LD = 32;
constexpr int CHEEVJ_HH_PERSISTENT_ROW_WORKERS = 32;
constexpr int CHEEVJ_HH_PERSISTENT_PARTICIPANTS = CHEEVJ_HH_PERSISTENT_ROW_WORKERS + 1;
constexpr int CHEEVJ_HH_PERSISTENT_ROWS = 16;
constexpr int CHEEVJ_HH_PERSISTENT_SEGMENT = 64;
constexpr int CHEEVJ_HH_PERSISTENT_SEGMENTS = CHEEVJ_HH_PERSISTENT_N / CHEEVJ_HH_PERSISTENT_SEGMENT;
constexpr int CHEEVJ_HH_PERSISTENT_SLAB = CHEEVJ_HH_PERSISTENT_ROWS * CHEEVJ_HH_PERSISTENT_N;
constexpr int CHEEVJ_HH_PERSISTENT_ACCUM = CHEEVJ_HH_PERSISTENT_ROWS * CHEEVJ_HH_PERSISTENT_SEGMENT;
constexpr int CHEEVJ_HH_PERSISTENT_SCATTER = CHEEVJ_HH_PERSISTENT_N * 8;
constexpr int CHEEVJ_HH_PERSISTENT_BARRIER_INTS = CHEEVJ_HH_PERSISTENT_PARTICIPANTS * 8;

// Float workspace layout.  The software-barrier workspace is a separate
// int32 allocation and is intentionally not aliased with these panel arrays.
constexpr int CHEEVJ_HH_PERSISTENT_WS_V_REAL = 0;
constexpr int CHEEVJ_HH_PERSISTENT_WS_V_IMAG =
    CHEEVJ_HH_PERSISTENT_WS_V_REAL + CHEEVJ_HH_PERSISTENT_B * CHEEVJ_HH_PERSISTENT_N;
constexpr int CHEEVJ_HH_PERSISTENT_WS_W_REAL =
    CHEEVJ_HH_PERSISTENT_WS_V_IMAG + CHEEVJ_HH_PERSISTENT_B * CHEEVJ_HH_PERSISTENT_N;
constexpr int CHEEVJ_HH_PERSISTENT_WS_W_IMAG =
    CHEEVJ_HH_PERSISTENT_WS_W_REAL + CHEEVJ_HH_PERSISTENT_B * CHEEVJ_HH_PERSISTENT_N;
constexpr int CHEEVJ_HH_PERSISTENT_WS_Y_REAL =
    CHEEVJ_HH_PERSISTENT_WS_W_IMAG + CHEEVJ_HH_PERSISTENT_B * CHEEVJ_HH_PERSISTENT_N;
constexpr int CHEEVJ_HH_PERSISTENT_WS_Y_IMAG = CHEEVJ_HH_PERSISTENT_WS_Y_REAL + CHEEVJ_HH_PERSISTENT_N;
constexpr int CHEEVJ_HH_PERSISTENT_WS_FLOATS = CHEEVJ_HH_PERSISTENT_WS_Y_IMAG + CHEEVJ_HH_PERSISTENT_N;

__aicore__ inline float CheevjHouseholderSqrt(float value, const AscendC::LocalTensor<float> &staging)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    AscendC::Duplicate(staging, value, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sqrt(staging, staging, 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    return staging.GetValue(0);
}

template <int Segment, int Segments>
__aicore__ inline float CheevjHouseholderReduce(const AscendC::LocalTensor<float> &source,
                                                const AscendC::LocalTensor<float> &partials,
                                                const AscendC::LocalTensor<float> &result)
{
    AscendC::WholeReduceSum<float, true>(partials, source, Segment, static_cast<int32_t>(Segments), 1, 1, Segment / 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::WholeReduceSum<float, true>(result, partials, Segments, static_cast<int32_t>(1), 1, 1, Segments);
    AscendC::PipeBarrier<PIPE_ALL>();
    return result.GetValue(0);
}

__aicore__ inline float CheevjPersistentReduce512(const AscendC::LocalTensor<float> &source,
                                                  const AscendC::LocalTensor<float> &partials,
                                                  const AscendC::LocalTensor<float> &result)
{
    return CheevjHouseholderReduce<CHEEVJ_HH_PERSISTENT_SEGMENT, CHEEVJ_HH_PERSISTENT_SEGMENTS>(source, partials,
                                                                                                result);
}

__aicore__ inline void CheevjIgnoreHouseholderInitArgs(AscendC::TPipe *pipe, GM_ADDR matrixReal, GM_ADDR matrixImag,
                                                       GM_ADDR panelVReal, GM_ADDR panelVImag, GM_ADDR panelWReal,
                                                       GM_ADDR panelWImag, GM_ADDR wHReal, GM_ADDR wHImag,
                                                       GM_ADDR wHImagNeg, GM_ADDR diagonal, GM_ADDR offDiagonal,
                                                       GM_ADDR tauReal, GM_ADDR tauImag, GM_ADDR panelWorkspace,
                                                       GM_ADDR barrierWorkspace)
{
    (void)pipe;
    (void)matrixReal;
    (void)matrixImag;
    (void)panelVReal;
    (void)panelVImag;
    (void)panelWReal;
    (void)panelWImag;
    (void)wHReal;
    (void)wHImag;
    (void)wHImagNeg;
    (void)diagonal;
    (void)offDiagonal;
    (void)tauReal;
    (void)tauImag;
    (void)panelWorkspace;
    (void)barrierWorkspace;
}

class CheevjHouseholderNb4Persistent
{
   public:
#include "cheevj_householder_init.inc"

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != CHEEVJ_HH_PERSISTENT_PARTICIPANTS || block < 0 ||
            block >= CHEEVJ_HH_PERSISTENT_PARTICIPANTS)
        {
            return;
        }

        auto matrixReal = matrixRealBuf.Get<float>();
        auto matrixImag = matrixImagBuf.Get<float>();
        auto barrierLocal = barrierBuf.Get<int32_t>();
        PrepareProcessBlock(block, matrixReal, matrixImag, barrierLocal);
        for (int step = 0; step < CHEEVJ_HH_PERSISTENT_B; ++step)
        {
            ProcessPanelStep(block, step, matrixReal, matrixImag, barrierLocal);
        }
#endif
    }

   private:
#include "cheevj_householder_n512_impl_part1.inc"
#include "cheevj_householder_n512_impl_part2.inc"
};

// N=1024 streamed branch.

constexpr int CHEEVJ_HH_N1024_STREAMED_N = 1024;
constexpr int CHEEVJ_HH_N1024_STREAMED_B = 8;
constexpr int CHEEVJ_HH_N1024_STREAMED_LD = 32;
constexpr int CHEEVJ_HH_N1024_STREAMED_ROW_WORKERS = 16;
constexpr int CHEEVJ_HH_N1024_STREAMED_PARTICIPANTS = CHEEVJ_HH_N1024_STREAMED_ROW_WORKERS + 1;
constexpr int CHEEVJ_HH_N1024_STREAMED_ROWS = 16;
constexpr int CHEEVJ_HH_N1024_STREAMED_WAVES = 4;
constexpr int CHEEVJ_HH_N1024_STREAMED_SEGMENT = 64;
constexpr int CHEEVJ_HH_N1024_STREAMED_SEGMENTS = CHEEVJ_HH_N1024_STREAMED_N / CHEEVJ_HH_N1024_STREAMED_SEGMENT;
constexpr int CHEEVJ_HH_N1024_STREAMED_SLAB = CHEEVJ_HH_N1024_STREAMED_ROWS * CHEEVJ_HH_N1024_STREAMED_N;
constexpr int CHEEVJ_HH_N1024_STREAMED_ACCUM = CHEEVJ_HH_N1024_STREAMED_ROWS * CHEEVJ_HH_N1024_STREAMED_SEGMENT;
// One sparse-write chunk represents eight dense values in eight 32-byte
// source blocks.  Chunking avoids the 32 KiB staging planes used by n=512.
constexpr int CHEEVJ_HH_N1024_STREAMED_SCATTER_VALUES = 8;
constexpr int CHEEVJ_HH_N1024_STREAMED_SCATTER = CHEEVJ_HH_N1024_STREAMED_SCATTER_VALUES * 8;
constexpr int CHEEVJ_HH_N1024_STREAMED_FULL_SCATTER = CHEEVJ_HH_N1024_STREAMED_N * 8;
constexpr int CHEEVJ_HH_N1024_STREAMED_BARRIER_INTS = CHEEVJ_HH_N1024_STREAMED_PARTICIPANTS * 16;

// Float workspace layout.  The software-barrier workspace is a separate
// int32 allocation and is intentionally not aliased with these panel arrays.
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_V_REAL = 0;
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_V_IMAG =
    CHEEVJ_HH_N1024_STREAMED_WS_V_REAL + CHEEVJ_HH_N1024_STREAMED_B * CHEEVJ_HH_N1024_STREAMED_N;
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_W_REAL =
    CHEEVJ_HH_N1024_STREAMED_WS_V_IMAG + CHEEVJ_HH_N1024_STREAMED_B * CHEEVJ_HH_N1024_STREAMED_N;
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_W_IMAG =
    CHEEVJ_HH_N1024_STREAMED_WS_W_REAL + CHEEVJ_HH_N1024_STREAMED_B * CHEEVJ_HH_N1024_STREAMED_N;
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_Y_REAL =
    CHEEVJ_HH_N1024_STREAMED_WS_W_IMAG + CHEEVJ_HH_N1024_STREAMED_B * CHEEVJ_HH_N1024_STREAMED_N;
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_Y_IMAG = CHEEVJ_HH_N1024_STREAMED_WS_Y_REAL + CHEEVJ_HH_N1024_STREAMED_N;
constexpr int CHEEVJ_HH_N1024_STREAMED_WS_FLOATS = CHEEVJ_HH_N1024_STREAMED_WS_Y_IMAG + CHEEVJ_HH_N1024_STREAMED_N;

constexpr int CHEEVJ_HH_N1024_STREAMED_LOCAL_BYTES =
    2 * CHEEVJ_HH_N1024_STREAMED_SLAB * sizeof(float) + 3 * CHEEVJ_HH_N1024_STREAMED_N * sizeof(float) +
    2 * CHEEVJ_HH_N1024_STREAMED_ACCUM * sizeof(float) + CHEEVJ_HH_N1024_STREAMED_SEGMENT * sizeof(float) +
    4 * CHEEVJ_HH_N1024_STREAMED_N * sizeof(float) + 2 * CHEEVJ_HH_N1024_STREAMED_SCATTER * sizeof(float) +
    CHEEVJ_HH_N1024_STREAMED_BARRIER_INTS * sizeof(int32_t);
static_assert(CHEEVJ_HH_N1024_STREAMED_WAVES * CHEEVJ_HH_N1024_STREAMED_ROW_WORKERS * CHEEVJ_HH_N1024_STREAMED_ROWS ==
                  CHEEVJ_HH_N1024_STREAMED_N,
              "streamed row waves must cover n exactly");
static_assert(CHEEVJ_HH_N1024_STREAMED_LOCAL_BYTES <= 192 * 1024,
              "n=1024 streamed panel exceeds the C220 AIV UB budget");

__aicore__ inline float CheevjN1024Reduce(const AscendC::LocalTensor<float> &source,
                                          const AscendC::LocalTensor<float> &partials,
                                          const AscendC::LocalTensor<float> &result)
{
    return CheevjHouseholderReduce<CHEEVJ_HH_N1024_STREAMED_SEGMENT, CHEEVJ_HH_N1024_STREAMED_SEGMENTS>(
        source, partials, result);
}

__aicore__ inline float CheevjN1024ReduceActive(const AscendC::LocalTensor<float> &source,
                                                const AscendC::LocalTensor<float> &partials,
                                                const AscendC::LocalTensor<float> &result, int active)
{
    (void)active;
    return CheevjN1024Reduce(source, partials, result);
}

class CheevjHouseholderN1024Nb4Streamed
{
   public:
#include "cheevj_householder_init.inc"

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != CHEEVJ_HH_N1024_STREAMED_PARTICIPANTS || block < 0 ||
            block >= CHEEVJ_HH_N1024_STREAMED_PARTICIPANTS)
        {
            return;
        }

        auto matrixReal = matrixRealBuf.Get<float>();
        auto matrixImag = matrixImagBuf.Get<float>();
        auto barrierLocal = barrierBuf.Get<int32_t>();
        AscendC::Duplicate(barrierLocal, static_cast<int32_t>(0), CHEEVJ_HH_N1024_STREAMED_BARRIER_INTS);
        AscendC::PipeBarrier<PIPE_ALL>();

        if (block == 0)
        {
            PrepareScatterIndex();
        }

        for (int step = 0; step < CHEEVJ_HH_N1024_STREAMED_B; ++step)
        {
            if (block == 0)
            {
                BuildReflector(step);
            }

            // Barrier 1/2: all 33 AIVs observe the published v.
            PanelBarrier(barrierLocal);
            if (block != 0)
            {
                LoadReflectorForGemv(step);
                for (int wave = 0; wave < CHEEVJ_HH_N1024_STREAMED_WAVES; ++wave)
                {
                    const int rowBegin =
                        (wave * CHEEVJ_HH_N1024_STREAMED_ROW_WORKERS + block - 1) * CHEEVJ_HH_N1024_STREAMED_ROWS;
                    if (rowBegin >= activeN)
                    {
                        continue;
                    }
                    LoadSlab(matrixReal, matrixRealGlobal, rowBegin * CHEEVJ_HH_N1024_STREAMED_N);
                    LoadSlab(matrixImag, matrixImagGlobal, rowBegin * CHEEVJ_HH_N1024_STREAMED_N);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    ComputeAndStoreRowGemv(matrixReal, matrixImag, rowBegin);
                }
            }

            // Barrier 2/2: AIV0 observes all 32 disjoint 16-row y writes.
            PanelBarrier(barrierLocal);
            if (block == 0)
            {
                CorrectAndFormW(step);
            }
        }
#endif
    }

   private:
#include "cheevj_householder_n1024_impl_part1.inc"
#include "cheevj_householder_n1024_impl_part2.inc"
};

// N=2048 streamed branch.

constexpr int CHEEVJ_HH_N2048_STREAMED_N = 2048;
constexpr int CHEEVJ_HH_N2048_STREAMED_B = 16;
constexpr int CHEEVJ_HH_N2048_STREAMED_LD = 32;
constexpr int CHEEVJ_HH_N2048_STREAMED_ROW_WORKERS = 32;
constexpr int CHEEVJ_HH_N2048_STREAMED_PARTICIPANTS = CHEEVJ_HH_N2048_STREAMED_ROW_WORKERS + 1;
constexpr int CHEEVJ_HH_N2048_STREAMED_ROWS = 4;
constexpr int CHEEVJ_HH_N2048_STREAMED_WAVES = 16;
constexpr int CHEEVJ_HH_N2048_STREAMED_SEGMENT = 64;
constexpr int CHEEVJ_HH_N2048_STREAMED_SEGMENTS = CHEEVJ_HH_N2048_STREAMED_N / CHEEVJ_HH_N2048_STREAMED_SEGMENT;
constexpr int CHEEVJ_HH_N2048_STREAMED_SLAB = CHEEVJ_HH_N2048_STREAMED_ROWS * CHEEVJ_HH_N2048_STREAMED_N;
// Row workers need only ROWS*SEGMENT lanes for their batched reduction, but
// AIV0 reuses both buffers as full-width vector temporaries while constructing
// and correcting a reflector.  Allocating only the worker footprint makes the
// two 2048-lane products overlap adjacent UB buffers and corrupts the norm
// before the first HER2K update.
constexpr int CHEEVJ_HH_N2048_STREAMED_ACCUM = CHEEVJ_HH_N2048_STREAMED_N;
// Stage 256 dense values per sparse write. This keeps the panel kernel within
// the UB budget while avoiding hundreds of four-byte DMA blocks per column.
constexpr int CHEEVJ_HH_N2048_STREAMED_SCATTER_VALUES = 256;
constexpr int CHEEVJ_HH_N2048_STREAMED_SCATTER = CHEEVJ_HH_N2048_STREAMED_SCATTER_VALUES * 8;
constexpr int CHEEVJ_HH_N2048_STREAMED_BARRIER_INTS = CHEEVJ_HH_N2048_STREAMED_PARTICIPANTS * 16;

// Float workspace layout.  The software-barrier workspace is a separate
// int32 allocation and is intentionally not aliased with these panel arrays.
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_V_REAL = 0;
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_V_IMAG =
    CHEEVJ_HH_N2048_STREAMED_WS_V_REAL + CHEEVJ_HH_N2048_STREAMED_B * CHEEVJ_HH_N2048_STREAMED_N;
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_W_REAL =
    CHEEVJ_HH_N2048_STREAMED_WS_V_IMAG + CHEEVJ_HH_N2048_STREAMED_B * CHEEVJ_HH_N2048_STREAMED_N;
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_W_IMAG =
    CHEEVJ_HH_N2048_STREAMED_WS_W_REAL + CHEEVJ_HH_N2048_STREAMED_B * CHEEVJ_HH_N2048_STREAMED_N;
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_Y_REAL =
    CHEEVJ_HH_N2048_STREAMED_WS_W_IMAG + CHEEVJ_HH_N2048_STREAMED_B * CHEEVJ_HH_N2048_STREAMED_N;
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_Y_IMAG = CHEEVJ_HH_N2048_STREAMED_WS_Y_REAL + CHEEVJ_HH_N2048_STREAMED_N;
constexpr int CHEEVJ_HH_N2048_STREAMED_WS_FLOATS = CHEEVJ_HH_N2048_STREAMED_WS_Y_IMAG + CHEEVJ_HH_N2048_STREAMED_N;

constexpr int CHEEVJ_HH_N2048_STREAMED_LOCAL_BYTES =
    2 * CHEEVJ_HH_N2048_STREAMED_SLAB * sizeof(float) + 3 * CHEEVJ_HH_N2048_STREAMED_N * sizeof(float) +
    2 * CHEEVJ_HH_N2048_STREAMED_ACCUM * sizeof(float) + CHEEVJ_HH_N2048_STREAMED_SEGMENT * sizeof(float) +
    4 * CHEEVJ_HH_N2048_STREAMED_N * sizeof(float) + 2 * CHEEVJ_HH_N2048_STREAMED_SCATTER * sizeof(float) +
    CHEEVJ_HH_N2048_STREAMED_BARRIER_INTS * sizeof(int32_t);
static_assert(CHEEVJ_HH_N2048_STREAMED_WAVES * CHEEVJ_HH_N2048_STREAMED_ROW_WORKERS * CHEEVJ_HH_N2048_STREAMED_ROWS ==
                  CHEEVJ_HH_N2048_STREAMED_N,
              "streamed row waves must cover n exactly");
static_assert((CHEEVJ_HH_N2048_STREAMED_WAVES & 1) == 0, "paired GM-store waves require an even wave count");
static_assert(CHEEVJ_HH_N2048_STREAMED_LOCAL_BYTES <= 192 * 1024,
              "n=2048 streamed panel exceeds the C220 AIV UB budget");

__aicore__ inline float CheevjN2048Reduce(const AscendC::LocalTensor<float> &source,
                                          const AscendC::LocalTensor<float> &partials,
                                          const AscendC::LocalTensor<float> &result)
{
    return CheevjHouseholderReduce<CHEEVJ_HH_N2048_STREAMED_SEGMENT, CHEEVJ_HH_N2048_STREAMED_SEGMENTS>(
        source, partials, result);
}

class CheevjHouseholderN2048Nb4Streamed
{
   public:
#include "cheevj_householder_init.inc"

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != CHEEVJ_HH_N2048_STREAMED_PARTICIPANTS || block < 0 ||
            block >= CHEEVJ_HH_N2048_STREAMED_PARTICIPANTS)
        {
            return;
        }

        auto matrixReal = matrixRealBuf.Get<float>();
        auto matrixImag = matrixImagBuf.Get<float>();
        auto barrierLocal = barrierBuf.Get<int32_t>();
        AscendC::Duplicate(barrierLocal, static_cast<int32_t>(0), CHEEVJ_HH_N2048_STREAMED_BARRIER_INTS);
        AscendC::PipeBarrier<PIPE_ALL>();

        if (block == 0)
        {
            PrepareScatterIndex();
        }

        for (int step = 0; step < CHEEVJ_HH_N2048_STREAMED_B; ++step)
        {
            ProcessPanelStep(block, step, matrixReal, matrixImag, barrierLocal);
        }
#endif
    }

   private:
#include "cheevj_householder_n2048_impl_part1.inc"
#include "cheevj_householder_n2048_impl_part2.inc"
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_HOUSEHOLDER_FIXED_HPP
