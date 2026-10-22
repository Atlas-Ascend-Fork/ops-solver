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
 * \file cheevj_jobzv_assembly.hpp
 * \brief Fixed-shape eigenvector archive and backtransform implementations.
 */

#ifndef CHEEVJ_C64_JOBZV_ASSEMBLY_HPP
#define CHEEVJ_C64_JOBZV_ASSEMBLY_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#include "cheevj_jobzv_archive.hpp"
#include "cheevj_jobzv_streamed.hpp"

namespace Cheevj
{

// N=512 archive and backtransform branch.

constexpr int CHEEVJ_E2E_N512_JOBZV_N = 512;
constexpr int CHEEVJ_E2E_N512_JOBZV_PANEL = 4;
constexpr int CHEEVJ_E2E_N512_JOBZV_PANEL_LD = 32;
constexpr int CHEEVJ_E2E_N512_JOBZV_WORKERS = 32;
constexpr int CHEEVJ_E2E_N512_JOBZV_COLUMNS_PER_WORKER = 16;
constexpr int CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX = CHEEVJ_E2E_N512_JOBZV_N * CHEEVJ_E2E_N512_JOBZV_COLUMNS_PER_WORKER;
constexpr int CHEEVJ_E2E_N512_JOBZV_GATHER_TILE = 256;
constexpr int CHEEVJ_E2E_N512_JOBZV_STRIDED_STAGE = CHEEVJ_E2E_N512_JOBZV_GATHER_TILE * 8;
constexpr int CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES = 64;
constexpr int CHEEVJ_E2E_N512_JOBZV_REDUCTION_REPEATS = CHEEVJ_E2E_N512_JOBZV_N / CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES;
constexpr int CHEEVJ_E2E_N512_JOBZV_ARCHIVE_UB_BYTES =
    (CHEEVJ_E2E_N512_JOBZV_STRIDED_STAGE + CHEEVJ_E2E_N512_JOBZV_N + 2 * CHEEVJ_E2E_N512_JOBZV_GATHER_TILE) *
    sizeof(float);
constexpr int CHEEVJ_E2E_N512_JOBZV_BACKTRANSFORM_UB_BYTES =
    (2 * CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX + CHEEVJ_E2E_N512_JOBZV_STRIDED_STAGE + CHEEVJ_E2E_N512_JOBZV_GATHER_TILE +
     7 * CHEEVJ_E2E_N512_JOBZV_N + CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES) *
        sizeof(float) +
    8 * sizeof(int32_t);
static_assert(CHEEVJ_E2E_N512_JOBZV_ARCHIVE_UB_BYTES <= 192 * 1024, "n512 reflector archive exceeds C220 UB");
static_assert(CHEEVJ_E2E_N512_JOBZV_BACKTRANSFORM_UB_BYTES <= 192 * 1024, "n512 reverse backtransform exceeds C220 UB");

using CheevjE2EN512ArchivePanelReflectors =
    CheevjArchivePanelReflectors<CHEEVJ_E2E_N512_JOBZV_N, CHEEVJ_E2E_N512_JOBZV_PANEL, CHEEVJ_E2E_N512_JOBZV_PANEL_LD,
                                 CHEEVJ_E2E_N512_JOBZV_GATHER_TILE>;

class CheevjE2EN512JobzVBacktransform
{
   public:
#include "cheevj_jobzv_init.inc"

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != CHEEVJ_E2E_N512_JOBZV_WORKERS || block < 0 ||
            block >= CHEEVJ_E2E_N512_JOBZV_WORKERS || LoadInfo() != 0)
        {
            return;
        }

        const int columnBegin = block * CHEEVJ_E2E_N512_JOBZV_COLUMNS_PER_WORKER;
        auto localReal = localRealBuf.Get<float>();
        auto localImag = localImagBuf.Get<float>();
        auto tauReal = tauRealBuf.Get<float>();
        auto tauImag = tauImagBuf.Get<float>();

        LoadRowMajorColumns(columnBegin, localReal);
        AscendC::Duplicate(localImag, 0.0f, CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX);
        CopyIn(tauReal, tauRealGlobal, CHEEVJ_E2E_N512_JOBZV_N);
        CopyIn(tauImag, tauImagGlobal, CHEEVJ_E2E_N512_JOBZV_N);
        AscendC::PipeBarrier<PIPE_ALL>();

        // If T=Q^H*A*Q with Q=H(0)...H(510), the original eigenvectors are
        // Q*Z, so left application starts at the rightmost reflector.
        for (int step = CHEEVJ_E2E_N512_JOBZV_N - 2; step >= 0; --step)
        {
            const float tauR = tauReal.GetValue(step);
            const float tauI = tauImag.GetValue(step);
            if (tauR == 0.0f && tauI == 0.0f)
            {
                continue;
            }
            LoadReflector(step);
            for (int localColumn = 0; localColumn < CHEEVJ_E2E_N512_JOBZV_COLUMNS_PER_WORKER; ++localColumn)
            {
                auto vectorReal = localReal[localColumn * CHEEVJ_E2E_N512_JOBZV_N];
                auto vectorImag = localImag[localColumn * CHEEVJ_E2E_N512_JOBZV_N];
                if (step >= 256)
                {
                    ApplyReflectorRange<256, 256>(tauR, tauI, vectorReal, vectorImag);
                }
                else
                {
                    ApplyReflectorRange<0, 512>(tauR, tauI, vectorReal, vectorImag);
                }
            }
        }

        const int outputOffset = columnBegin * CHEEVJ_E2E_N512_JOBZV_N;
        CopyOut(eigenvectorRealGlobal, outputOffset, localReal, CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX);
        CopyOut(eigenvectorImagGlobal, outputOffset, localImag, CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX);
#endif
    }

   private:
#include "cheevj_jobzv_common.inc"

    AscendC::TBuf<AscendC::TPosition::VECCALC> localRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localImagBuf;
#include "cheevj_jobzv_buffers.inc"

    __aicore__ inline void InitBuffers(AscendC::TPipe *pipe)
    {
        pipe->InitBuffer(localRealBuf, CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX * sizeof(float));
        pipe->InitBuffer(localImagBuf, CHEEVJ_E2E_N512_JOBZV_LOCAL_MATRIX * sizeof(float));
        InitCommonBuffers(pipe, CHEEVJ_E2E_N512_JOBZV_N, CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES);
    }

    __aicore__ inline void LoadRowMajorColumns(int columnBegin, const AscendC::LocalTensor<float> &localReal)
    {
        for (int localColumn = 0; localColumn < CHEEVJ_E2E_N512_JOBZV_COLUMNS_PER_WORKER; ++localColumn)
        {
            const int globalColumn = columnBegin + localColumn;
            CopyIn(localReal[localColumn * CHEEVJ_E2E_N512_JOBZV_N],
                   tridiagonalEigenvectorsGlobal[globalColumn * CHEEVJ_E2E_N512_JOBZV_N], CHEEVJ_E2E_N512_JOBZV_N);
        }
    }

    __aicore__ inline void LoadReflector(int step)
    {
        auto reflectorReal = reflectorRealBuf.Get<float>();
        auto reflectorImag = reflectorImagBuf.Get<float>();
        const int offset = step * CHEEVJ_E2E_N512_JOBZV_N;
        CopyIn(reflectorReal, reflectorRealGlobal[offset], CHEEVJ_E2E_N512_JOBZV_N);
        CopyIn(reflectorImag, reflectorImagGlobal[offset], CHEEVJ_E2E_N512_JOBZV_N);
    }

    template <int Count>
    __aicore__ inline float ReduceRange(const AscendC::LocalTensor<float> &source)
    {
        static_assert(Count % CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES == 0,
                      "reflector reduction range must be lane aligned");
        constexpr int repeats = Count / CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES;
        auto reduction = reductionBuf.Get<float>();
        AscendC::WholeReduceSum<float, true>(reduction, source, CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES,
                                             static_cast<int32_t>(repeats), 1, 1,
                                             CHEEVJ_E2E_N512_JOBZV_REDUCTION_LANES / 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WholeReduceSum<float, true>(reduction[repeats], reduction, repeats, static_cast<int32_t>(1), 1, 1,
                                             repeats);
        AscendC::PipeBarrier<PIPE_ALL>();
        return reduction.GetValue(repeats);
    }

    template <int Offset, int Count>
    __aicore__ inline void ApplyReflectorRange(float tauR, float tauI, const AscendC::LocalTensor<float> &vectorReal,
                                               const AscendC::LocalTensor<float> &vectorImag)
    {
        auto reflectorReal = reflectorRealBuf.Get<float>()[Offset];
        auto reflectorImag = reflectorImagBuf.Get<float>()[Offset];
        auto activeReal = vectorReal[Offset];
        auto activeImag = vectorImag[Offset];
        auto scratch0 = scratch0Buf.Get<float>();
        auto scratch1 = scratch1Buf.Get<float>();
        auto scratch2 = scratch2Buf.Get<float>();

        AscendC::Mul(scratch0, reflectorReal, activeReal, Count);
        AscendC::Mul(scratch1, reflectorImag, activeImag, Count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch0, scratch0, scratch1, Count);
        AscendC::PipeBarrier<PIPE_V>();
        const float dotReal = ReduceRange<Count>(scratch0);

        AscendC::Mul(scratch1, reflectorReal, activeImag, Count);
        AscendC::Mul(scratch2, reflectorImag, activeReal, Count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(scratch1, scratch1, scratch2, Count);
        AscendC::PipeBarrier<PIPE_V>();
        const float dotImag = ReduceRange<Count>(scratch1);

        const float factorReal = tauR * dotReal - tauI * dotImag;
        const float factorImag = tauR * dotImag + tauI * dotReal;

        // y <- y-v*(tau*dot).
        AscendC::Muls(scratch0, reflectorReal, factorReal, Count);
        AscendC::Muls(scratch1, reflectorImag, factorImag, Count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(scratch0, scratch0, scratch1, Count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(activeReal, activeReal, scratch0, Count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(scratch1, reflectorReal, factorImag, Count);
        AscendC::Muls(scratch2, reflectorImag, factorReal, Count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch1, scratch1, scratch2, Count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(activeImag, activeImag, scratch1, Count);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// N=1024 archive and backtransform branch.

constexpr int CHEEVJ_E2E_N1024_JOBZV_N = 1024;
constexpr int CHEEVJ_E2E_N1024_JOBZV_PANEL = 8;
constexpr int CHEEVJ_E2E_N1024_JOBZV_PANEL_LD = 32;
constexpr int CHEEVJ_E2E_N1024_JOBZV_WORKERS = 32;
constexpr int CHEEVJ_E2E_N1024_JOBZV_COLUMNS_PER_WAVE = 16;
constexpr int CHEEVJ_E2E_N1024_JOBZV_WAVES = 2;
constexpr int CHEEVJ_E2E_N1024_JOBZV_LOCAL_MATRIX = CHEEVJ_E2E_N1024_JOBZV_N * CHEEVJ_E2E_N1024_JOBZV_COLUMNS_PER_WAVE;
constexpr int CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE = 64;
constexpr int CHEEVJ_E2E_N1024_JOBZV_STRIDED_STAGE = CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE * 8;
constexpr int CHEEVJ_E2E_N1024_JOBZV_REDUCTION_LANES = 64;
constexpr int CHEEVJ_E2E_N1024_JOBZV_REDUCTION_REPEATS =
    CHEEVJ_E2E_N1024_JOBZV_N / CHEEVJ_E2E_N1024_JOBZV_REDUCTION_LANES;
constexpr int CHEEVJ_E2E_N1024_JOBZV_ARCHIVE_UB_BYTES =
    (CHEEVJ_E2E_N1024_JOBZV_STRIDED_STAGE + CHEEVJ_E2E_N1024_JOBZV_N + 2 * CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE) *
    sizeof(float);
constexpr int CHEEVJ_E2E_N1024_JOBZV_BACKTRANSFORM_UB_BYTES =
    (2 * CHEEVJ_E2E_N1024_JOBZV_LOCAL_MATRIX + CHEEVJ_E2E_N1024_JOBZV_STRIDED_STAGE +
     CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE + 7 * CHEEVJ_E2E_N1024_JOBZV_N + CHEEVJ_E2E_N1024_JOBZV_REDUCTION_LANES) *
        sizeof(float) +
    8 * sizeof(int32_t);
static_assert(CHEEVJ_E2E_N1024_JOBZV_WORKERS * CHEEVJ_E2E_N1024_JOBZV_COLUMNS_PER_WAVE * CHEEVJ_E2E_N1024_JOBZV_WAVES ==
                  CHEEVJ_E2E_N1024_JOBZV_N,
              "n1024 backtransform waves must cover all columns");
static_assert(CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE <= 64, "C220 Gather paths must remain at most 64 lanes");
static_assert(CHEEVJ_E2E_N1024_JOBZV_ARCHIVE_UB_BYTES <= 192 * 1024, "n1024 reflector archive exceeds C220 UB");
static_assert(CHEEVJ_E2E_N1024_JOBZV_BACKTRANSFORM_UB_BYTES <= 192 * 1024,
              "n1024 reverse backtransform exceeds C220 UB");

using CheevjE2EN1024ArchivePanelReflectors =
    CheevjArchivePanelReflectors<CHEEVJ_E2E_N1024_JOBZV_N, CHEEVJ_E2E_N1024_JOBZV_PANEL,
                                 CHEEVJ_E2E_N1024_JOBZV_PANEL_LD, CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE>;

using CheevjE2EN1024JobzVBacktransform =
    CheevjStreamedJobzVBacktransform<CHEEVJ_E2E_N1024_JOBZV_N, CHEEVJ_E2E_N1024_JOBZV_WORKERS,
                                     CHEEVJ_E2E_N1024_JOBZV_COLUMNS_PER_WAVE, CHEEVJ_E2E_N1024_JOBZV_WAVES,
                                     CHEEVJ_E2E_N1024_JOBZV_LOCAL_MATRIX, CHEEVJ_E2E_N1024_JOBZV_GATHER_TILE,
                                     CHEEVJ_E2E_N1024_JOBZV_REDUCTION_LANES>;

// N=2048 archive and backtransform branch.

constexpr int CHEEVJ_E2E_N2048_JOBZV_N = 2048;
constexpr int CHEEVJ_E2E_N2048_JOBZV_PANEL = 16;
constexpr int CHEEVJ_E2E_N2048_JOBZV_PANEL_LD = 32;
constexpr int CHEEVJ_E2E_N2048_JOBZV_WORKERS = 32;
constexpr int CHEEVJ_E2E_N2048_JOBZV_COLUMNS_PER_WAVE = 8;
constexpr int CHEEVJ_E2E_N2048_JOBZV_WAVES = 8;
constexpr int CHEEVJ_E2E_N2048_JOBZV_LOCAL_MATRIX = CHEEVJ_E2E_N2048_JOBZV_N * CHEEVJ_E2E_N2048_JOBZV_COLUMNS_PER_WAVE;
constexpr int CHEEVJ_E2E_N2048_JOBZV_LOCAL_GUARD = 64;
constexpr int CHEEVJ_E2E_N2048_JOBZV_LOCAL_ALLOCATION =
    CHEEVJ_E2E_N2048_JOBZV_LOCAL_MATRIX + CHEEVJ_E2E_N2048_JOBZV_LOCAL_GUARD;
constexpr int CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE = 64;
constexpr int CHEEVJ_E2E_N2048_JOBZV_STRIDED_STAGE = CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE * 8;
constexpr int CHEEVJ_E2E_N2048_JOBZV_REDUCTION_LANES = 64;
constexpr int CHEEVJ_E2E_N2048_JOBZV_REDUCTION_SEGMENTS =
    CHEEVJ_E2E_N2048_JOBZV_N / CHEEVJ_E2E_N2048_JOBZV_REDUCTION_LANES;
constexpr int CHEEVJ_E2E_N2048_JOBZV_ARCHIVE_UB_BYTES =
    (CHEEVJ_E2E_N2048_JOBZV_STRIDED_STAGE + CHEEVJ_E2E_N2048_JOBZV_N + 2 * CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE) *
    sizeof(float);
constexpr int CHEEVJ_E2E_N2048_JOBZV_BACKTRANSFORM_UB_BYTES =
    (2 * CHEEVJ_E2E_N2048_JOBZV_LOCAL_ALLOCATION + CHEEVJ_E2E_N2048_JOBZV_STRIDED_STAGE +
     CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE + 7 * CHEEVJ_E2E_N2048_JOBZV_N + CHEEVJ_E2E_N2048_JOBZV_REDUCTION_LANES) *
        sizeof(float) +
    8 * sizeof(int32_t);
static_assert(CHEEVJ_E2E_N2048_JOBZV_WORKERS * CHEEVJ_E2E_N2048_JOBZV_COLUMNS_PER_WAVE * CHEEVJ_E2E_N2048_JOBZV_WAVES ==
                  CHEEVJ_E2E_N2048_JOBZV_N,
              "n2048 backtransform waves must cover all columns");
static_assert(CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE <= 64, "C220 Gather paths must remain at most 64 lanes");
static_assert(CHEEVJ_E2E_N2048_JOBZV_N % CHEEVJ_E2E_N2048_JOBZV_REDUCTION_LANES == 0,
              "n2048 reflector reductions require complete 64-lane tiles");
static_assert(CHEEVJ_E2E_N2048_JOBZV_ARCHIVE_UB_BYTES <= 192 * 1024, "n2048 reflector archive exceeds C220 UB");
static_assert(CHEEVJ_E2E_N2048_JOBZV_BACKTRANSFORM_UB_BYTES <= 192 * 1024,
              "n2048 reverse backtransform exceeds C220 UB");

using CheevjE2EN2048ArchivePanelReflectors =
    CheevjArchivePanelReflectors<CHEEVJ_E2E_N2048_JOBZV_N, CHEEVJ_E2E_N2048_JOBZV_PANEL,
                                 CHEEVJ_E2E_N2048_JOBZV_PANEL_LD, CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE>;

using CheevjE2EN2048JobzVBacktransform =
    CheevjStreamedJobzVBacktransform<CHEEVJ_E2E_N2048_JOBZV_N, CHEEVJ_E2E_N2048_JOBZV_WORKERS,
                                     CHEEVJ_E2E_N2048_JOBZV_COLUMNS_PER_WAVE, CHEEVJ_E2E_N2048_JOBZV_WAVES,
                                     CHEEVJ_E2E_N2048_JOBZV_LOCAL_ALLOCATION, CHEEVJ_E2E_N2048_JOBZV_GATHER_TILE,
                                     CHEEVJ_E2E_N2048_JOBZV_REDUCTION_LANES>;

}  // namespace Cheevj

#endif  // CHEEVJ_C64_JOBZV_ASSEMBLY_HPP
