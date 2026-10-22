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
 * \file cheevj_tridiag_vectors.hpp
 * \brief Fixed-shape tridiagonal eigenvector implementations selected by the Cheevj dispatcher.
 */

#ifndef CHEEVJ_C64_TRIDIAG_VECTORS_HPP
#define CHEEVJ_C64_TRIDIAG_VECTORS_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#include "cheevj_tridiag_vectors_reset.hpp"

namespace Cheevj
{

// N=512 tridiagonal eigenvector branch.

constexpr int CHEEVJ_TRIDIAG512_N = 512;
constexpr int CHEEVJ_TRIDIAG512_ROW_WORKERS = 32;
constexpr int CHEEVJ_TRIDIAG512_ROWS_PER_WORKER = 16;
constexpr int CHEEVJ_TRIDIAG512_PARTICIPANTS = CHEEVJ_TRIDIAG512_ROW_WORKERS + 1;
constexpr int CHEEVJ_TRIDIAG512_LOCAL_Q = CHEEVJ_TRIDIAG512_N * CHEEVJ_TRIDIAG512_ROWS_PER_WORKER;
constexpr int CHEEVJ_TRIDIAG512_GATHER_TILE = 64;
constexpr int CHEEVJ_TRIDIAG512_BARRIER_INTS = CHEEVJ_TRIDIAG512_PARTICIPANTS * 8;
constexpr int CHEEVJ_TRIDIAG512_COMMAND_INTS = 32;
constexpr int CHEEVJ_TRIDIAG512_MAX_ITERATIONS = 256;
constexpr float CHEEVJ_TRIDIAG512_EPSILON = 1.1920928955078125e-7f;
constexpr float CHEEVJ_TRIDIAG512_SAFE_MIN = 1.0e-30f;

enum CheevjTridiag512Command : int32_t
{
    CHEEVJ_TRIDIAG512_ROTATE = 1,
    CHEEVJ_TRIDIAG512_SWAP = 2,
    CHEEVJ_TRIDIAG512_STOP = 3,
};

using CheevjTridiag512VectorsReset =
    CheevjTridiagVectorsReset<CHEEVJ_TRIDIAG512_BARRIER_INTS, CHEEVJ_TRIDIAG512_COMMAND_INTS>;

template <int MatrixN, int RowsPerWorker>
class CheevjTridiagFixedVectors
{
   private:
#include "cheevj_tridiag_vectors_fixed_impl.inc"
};

using CheevjTridiag512Vectors = CheevjTridiagFixedVectors<CHEEVJ_TRIDIAG512_N, CHEEVJ_TRIDIAG512_ROWS_PER_WORKER>;

constexpr int CHEEVJ_TRIDIAG1024_N = 1024;
constexpr int CHEEVJ_TRIDIAG1024_ROW_WORKERS = 32;
constexpr int CHEEVJ_TRIDIAG1024_ROWS_PER_WORKER = 32;
constexpr int CHEEVJ_TRIDIAG1024_PARTICIPANTS = CHEEVJ_TRIDIAG1024_ROW_WORKERS + 1;
constexpr int CHEEVJ_TRIDIAG1024_LOCAL_Q = CHEEVJ_TRIDIAG1024_N * CHEEVJ_TRIDIAG1024_ROWS_PER_WORKER;
constexpr int CHEEVJ_TRIDIAG1024_GATHER_TILE = 64;
constexpr int CHEEVJ_TRIDIAG1024_BARRIER_INTS = CHEEVJ_TRIDIAG1024_PARTICIPANTS * 8;
constexpr int CHEEVJ_TRIDIAG1024_COMMAND_INTS = 32;
constexpr int CHEEVJ_TRIDIAG1024_MAX_ITERATIONS = 256;
constexpr float CHEEVJ_TRIDIAG1024_EPSILON = 1.1920928955078125e-7f;
constexpr float CHEEVJ_TRIDIAG1024_SAFE_MIN = 1.0e-30f;
constexpr int CHEEVJ_TRIDIAG1024_UB_BYTES = (CHEEVJ_TRIDIAG1024_LOCAL_Q + 5 * CHEEVJ_TRIDIAG1024_N +
                                             CHEEVJ_TRIDIAG1024_COMMAND_INTS + CHEEVJ_TRIDIAG1024_BARRIER_INTS) *
                                            sizeof(int32_t);
static_assert(CHEEVJ_TRIDIAG1024_UB_BYTES <= 192 * 1024, "n=1024 persistent tridiagonal Q slab exceeds 910B UB");
static_assert(CHEEVJ_TRIDIAG1024_ROW_WORKERS * CHEEVJ_TRIDIAG1024_ROWS_PER_WORKER == CHEEVJ_TRIDIAG1024_N,
              "worker row slabs must cover Q exactly once");
static_assert(CHEEVJ_TRIDIAG1024_N % CHEEVJ_TRIDIAG1024_GATHER_TILE == 0,
              "row-major store requires complete Gather tiles");

enum CheevjTridiag1024Command : int32_t
{
    CHEEVJ_TRIDIAG1024_ROTATE = 1,
    CHEEVJ_TRIDIAG1024_SWAP = 2,
    CHEEVJ_TRIDIAG1024_STOP = 3,
};

using CheevjTridiag1024VectorsReset =
    CheevjTridiagVectorsReset<CHEEVJ_TRIDIAG1024_BARRIER_INTS, CHEEVJ_TRIDIAG1024_COMMAND_INTS>;

using CheevjTridiag1024Vectors = CheevjTridiagFixedVectors<CHEEVJ_TRIDIAG1024_N, CHEEVJ_TRIDIAG1024_ROWS_PER_WORKER>;

// N=2048 tridiagonal eigenvector branch.

constexpr int CHEEVJ_TRIDIAG2048_N = 2048;
constexpr int CHEEVJ_TRIDIAG2048_ROW_WORKERS = 32;
constexpr int CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER = 16;
constexpr int CHEEVJ_TRIDIAG2048_ROW_WAVES = 4;
constexpr int CHEEVJ_TRIDIAG2048_ROWS_PER_WAVE = CHEEVJ_TRIDIAG2048_ROW_WORKERS * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER;
constexpr int CHEEVJ_TRIDIAG2048_PARTICIPANTS = CHEEVJ_TRIDIAG2048_ROW_WORKERS + 1;
constexpr int CHEEVJ_TRIDIAG2048_LOCAL_Q = CHEEVJ_TRIDIAG2048_N * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER;
constexpr int CHEEVJ_TRIDIAG2048_GATHER_TILE = 64;
constexpr int CHEEVJ_TRIDIAG2048_BARRIER_INTS = CHEEVJ_TRIDIAG2048_PARTICIPANTS * 8;
constexpr int CHEEVJ_TRIDIAG2048_COMMAND_INTS = 32;
constexpr int CHEEVJ_TRIDIAG2048_MAX_ITERATIONS = 256;
constexpr float CHEEVJ_TRIDIAG2048_EPSILON = 1.1920928955078125e-7f;
constexpr float CHEEVJ_TRIDIAG2048_SAFE_MIN = 1.0e-30f;
constexpr int CHEEVJ_TRIDIAG2048_UB_BYTES = (CHEEVJ_TRIDIAG2048_LOCAL_Q + 5 * CHEEVJ_TRIDIAG2048_N +
                                             CHEEVJ_TRIDIAG2048_COMMAND_INTS + CHEEVJ_TRIDIAG2048_BARRIER_INTS) *
                                            sizeof(int32_t);
static_assert(CHEEVJ_TRIDIAG2048_UB_BYTES <= 192 * 1024, "n=2048 persistent tridiagonal Q slab exceeds 910B UB");
static_assert(CHEEVJ_TRIDIAG2048_UB_BYTES == 173216, "n=2048 UB accounting changed; re-audit before launch");
static_assert(CHEEVJ_TRIDIAG2048_ROW_WAVES * CHEEVJ_TRIDIAG2048_ROWS_PER_WAVE == CHEEVJ_TRIDIAG2048_N,
              "device waves must cover all Q rows exactly once");
static_assert(CHEEVJ_TRIDIAG2048_ROWS_PER_WAVE == 512, "each device wave must own one 512-row band");
static_assert(CHEEVJ_TRIDIAG2048_N % CHEEVJ_TRIDIAG2048_GATHER_TILE == 0,
              "row-major store requires complete Gather tiles");
static_assert(CHEEVJ_TRIDIAG2048_GATHER_TILE <= 64, "C220 Q transpose must not issue long-count Gather");

enum CheevjTridiag2048Command : int32_t
{
    CHEEVJ_TRIDIAG2048_ROTATE = 1,
    CHEEVJ_TRIDIAG2048_SWAP = 2,
    CHEEVJ_TRIDIAG2048_STOP = 3,
};

using CheevjTridiag2048VectorsReset =
    CheevjTridiagVectorsReset<CHEEVJ_TRIDIAG2048_BARRIER_INTS, CHEEVJ_TRIDIAG2048_COMMAND_INTS>;

class CheevjTridiag2048Vectors
{
   public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR eigenvalues,
                                GM_ADDR eigenvectors, GM_ADDR info, GM_ADDR commandWorkspace, GM_ADDR barrierWorkspace,
                                int wave)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        diagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(diagonal));
        offDiagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(offDiagonal));
        eigenvaluesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvalues));
        eigenvectorsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvectors));
        infoGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(info));
        commandGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commandWorkspace));
        barrierGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(barrierWorkspace));
        rowWave = wave;

        pipe->InitBuffer(diagonalBuf, CHEEVJ_TRIDIAG2048_N * sizeof(float));
        pipe->InitBuffer(offDiagonalBuf, CHEEVJ_TRIDIAG2048_N * sizeof(float));
        pipe->InitBuffer(localQBuf, CHEEVJ_TRIDIAG2048_LOCAL_Q * sizeof(float));
        pipe->InitBuffer(scratch0Buf, CHEEVJ_TRIDIAG2048_N * sizeof(float));
        pipe->InitBuffer(scratch1Buf, CHEEVJ_TRIDIAG2048_N * sizeof(float));
        pipe->InitBuffer(gatherIndexBuf, CHEEVJ_TRIDIAG2048_N * sizeof(uint32_t));
        pipe->InitBuffer(commandBuf, CHEEVJ_TRIDIAG2048_COMMAND_INTS * sizeof(int32_t));
        pipe->InitBuffer(barrierBuf, CHEEVJ_TRIDIAG2048_BARRIER_INTS * sizeof(int32_t));
#else
        (void)pipe;
        (void)diagonal;
        (void)offDiagonal;
        (void)eigenvalues;
        (void)eigenvectors;
        (void)info;
        (void)commandWorkspace;
        (void)barrierWorkspace;
        (void)wave;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != CHEEVJ_TRIDIAG2048_PARTICIPANTS || block < 0 ||
            block >= CHEEVJ_TRIDIAG2048_PARTICIPANTS || rowWave < 0 || rowWave >= CHEEVJ_TRIDIAG2048_ROW_WAVES)
        {
            return;
        }

        const int waveRowBegin = rowWave * CHEEVJ_TRIDIAG2048_ROWS_PER_WAVE;

        auto barrierLocal = barrierBuf.Get<int32_t>();
        AscendC::Duplicate(barrierLocal, static_cast<int32_t>(0), CHEEVJ_TRIDIAG2048_BARRIER_INTS);
        AscendC::PipeBarrier<PIPE_ALL>();

        float scale = 0.0f;
        if (block == 0)
        {
            LoadTridiagonal();
            scale = ScaleTridiagonal();
        }
        else
        {
            InitializeWorkerIdentity(waveRowBegin + (block - 1) * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        }

        // Nobody may publish or consume the first rotation until all worker
        // slabs contain identity and the controller has scaled d/e.
        CommandBarrier(barrierLocal);
        if (block == 0)
        {
            ControllerProcess(scale, barrierLocal);
        }
        else
        {
            WorkerProcess(waveRowBegin + (block - 1) * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER, barrierLocal);
        }
#endif
    }

   private:
#include "cheevj_tridiag_vectors_n2048_impl.inc"
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_TRIDIAG_VECTORS_HPP
