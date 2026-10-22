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
 * \file cheevj_tridiag_inverse.hpp
 * \brief SIMD-batched inverse iteration for fixed-size real symmetric tridiagonal eigensystems.
 */

#ifndef CHEEVJ_C64_TRIDIAG_INVERSE_HPP
#define CHEEVJ_C64_TRIDIAG_INVERSE_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace Cheevj
{

constexpr int CHEEVJ_TRIDIAG_INVERSE_MAX_N = 2048;
constexpr int CHEEVJ_TRIDIAG_INVERSE_COLUMNS = 8;
constexpr int CHEEVJ_TRIDIAG_INVERSE_WORKERS = 32;
constexpr int CHEEVJ_TRIDIAG_INVERSE_WAVE_COLUMNS = CHEEVJ_TRIDIAG_INVERSE_COLUMNS * CHEEVJ_TRIDIAG_INVERSE_WORKERS;
constexpr int CHEEVJ_TRIDIAG_INVERSE_COEFFICIENT_BATCH = 8;
constexpr float CHEEVJ_TRIDIAG_INVERSE_SHIFT_SCALE = 1.0e-5f;
constexpr float CHEEVJ_TRIDIAG_INVERSE_PIVOT_BIAS = 1.0e-20f;

class CheevjTridiagInverseIteration
{
   public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR eigenvalues,
                                GM_ADDR eigenvectorsRowMajor, int n, int waveStart)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        diagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(diagonal));
        offDiagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(offDiagonal));
        eigenvaluesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvalues));
        eigenvectorsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvectorsRowMajor));
        matrixOrder = n;
        firstWaveColumn = waveStart;

        pipe->InitBuffer(diagonalBuf, CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(float));
        pipe->InitBuffer(offDiagonalBuf, CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(float));
        pipe->InitBuffer(eigenvaluesBuf, CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(float));
        pipe->InitBuffer(cPrimeBuf, CHEEVJ_TRIDIAG_INVERSE_COLUMNS * CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(float));
        pipe->InitBuffer(workBuf, CHEEVJ_TRIDIAG_INVERSE_COLUMNS * CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(float));
        pipe->InitBuffer(diagonalBroadcastBuf, 64 * sizeof(float));
        pipe->InitBuffer(edgePreviousBroadcastBuf, 64 * sizeof(float));
        pipe->InitBuffer(edgeCurrentBroadcastBuf, 64 * sizeof(float));
        pipe->InitBuffer(shiftBuf, CHEEVJ_TRIDIAG_INVERSE_COLUMNS * sizeof(float));
        pipe->InitBuffer(denominatorBuf, CHEEVJ_TRIDIAG_INVERSE_COLUMNS * sizeof(float));
        pipe->InitBuffer(tempBuf, CHEEVJ_TRIDIAG_INVERSE_COLUMNS * sizeof(float));
        pipe->InitBuffer(normBuf, CHEEVJ_TRIDIAG_INVERSE_COLUMNS * sizeof(float));
        pipe->InitBuffer(columnBuf, CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(float));
        pipe->InitBuffer(gatherIndexBuf, CHEEVJ_TRIDIAG_INVERSE_MAX_N * sizeof(uint32_t));
#else
        (void)pipe;
        (void)diagonal;
        (void)offDiagonal;
        (void)eigenvalues;
        (void)eigenvectorsRowMajor;
        (void)n;
        (void)waveStart;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (matrixOrder <= 0 || matrixOrder > CHEEVJ_TRIDIAG_INVERSE_MAX_N || block < 0 ||
            block >= CHEEVJ_TRIDIAG_INVERSE_WORKERS)
        {
            return;
        }

        const int firstColumn = firstWaveColumn + block * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
        int validColumns = matrixOrder - firstColumn;
        if (validColumns > CHEEVJ_TRIDIAG_INVERSE_COLUMNS)
        {
            validColumns = CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
        }
        if (validColumns <= 0)
        {
            return;
        }

        LoadCoefficients();
        PrepareShifts(firstColumn, validColumns);
        PrepareStartVectors(firstColumn, validColumns);
        // Two refinements close the 512/1024 orthogonality gap.  The 2048
        // backend needs additional refinement for low-projection dense inputs.
        const int iterations = matrixOrder == 2048 ? 5 : 2;
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            SolveShiftedBatch(validColumns);
            NormalizeBatch(validColumns);
        }
        StoreColumns(firstColumn, validColumns);
#endif
    }

   private:
    AscendC::GlobalTensor<float> diagonalGlobal;
    AscendC::GlobalTensor<float> offDiagonalGlobal;
    AscendC::GlobalTensor<float> eigenvaluesGlobal;
    AscendC::GlobalTensor<float> eigenvectorsGlobal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offDiagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> eigenvaluesBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cPrimeBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalBroadcastBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> edgePreviousBroadcastBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> edgeCurrentBroadcastBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shiftBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denominatorBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> columnBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndexBuf;

    int matrixOrder = 0;
    int firstWaveColumn = 0;

    __aicore__ inline float ShiftScale() const
    {
        return matrixOrder >= 1024 ? 1.0e-6f : CHEEVJ_TRIDIAG_INVERSE_SHIFT_SCALE;
    }

    __aicore__ inline void PrepareStartVectors(int firstColumn, int validColumns)
    {
        auto work = workBuf.Get<float>();
        AscendC::Duplicate(work, 0.0f, matrixOrder * CHEEVJ_TRIDIAG_INVERSE_COLUMNS);
        AscendC::PipeBarrier<PIPE_V>();
        if (matrixOrder == 2048)
        {
            // Dense deterministic starts avoid zero projection on localized
            // eigenvectors without requiring host-generated random state.
            for (int row = 0; row < matrixOrder; ++row)
            {
                for (int localColumn = 0; localColumn < validColumns; ++localColumn)
                {
                    uint32_t bits = static_cast<uint32_t>(row + 1) * 2654435761U;
                    bits ^= static_cast<uint32_t>(firstColumn + localColumn + 1) * 2246822519U;
                    bits ^= bits >> 16;
                    work.SetValue(row * CHEEVJ_TRIDIAG_INVERSE_COLUMNS + localColumn, (bits & 1U) == 0U ? 1.0f : -1.0f);
                }
            }
        }
        else
        {
            AscendC::Duplicate(work, 1.0f, validColumns);
            // Retain the shared e0 component while adding an independent
            // direction for eigenvectors with a tiny first-row projection.
            for (int localColumn = 0; localColumn < validColumns; ++localColumn)
            {
                const int seedRow = (firstColumn + localColumn) % matrixOrder;
                if (seedRow != 0)
                {
                    work.SetValue(seedRow * CHEEVJ_TRIDIAG_INVERSE_COLUMNS + localColumn, 1.0f);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadOne(const AscendC::LocalTensor<float> &destination,
                                   const AscendC::GlobalTensor<float> &source, int count)
    {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((8 - count % 8) % 8), 0.0f};
        AscendC::DataCopyPad(destination, source, copy, pad);
    }

    __aicore__ inline void LoadCoefficients()
    {
        LoadOne(diagonalBuf.Get<float>(), diagonalGlobal, matrixOrder);
        LoadOne(offDiagonalBuf.Get<float>(), offDiagonalGlobal, matrixOrder - 1);
        LoadOne(eigenvaluesBuf.Get<float>(), eigenvaluesGlobal, matrixOrder);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareShifts(int firstColumn, int validColumns)
    {
        auto eigenvalues = eigenvaluesBuf.Get<float>();
        auto shift = shiftBuf.Get<float>();
        auto temp = tempBuf.Get<float>();
        AscendC::Adds(shift, eigenvalues[firstColumn], 0.0f, validColumns);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(temp, shift, validColumns);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(temp, temp, 1.0f, validColumns);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(temp, temp, ShiftScale(), validColumns);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(shift, shift, temp, validColumns);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastBatch(int rowBase)
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        auto diagonalBroadcast = diagonalBroadcastBuf.Get<float>();
        auto edgePrevious = edgePreviousBroadcastBuf.Get<float>();
        auto edgeCurrent = edgeCurrentBroadcastBuf.Get<float>();

        AscendC::Brcb(diagonalBroadcast, diagonal[rowBase], 1, {1, 8});
        if (rowBase >= CHEEVJ_TRIDIAG_INVERSE_COEFFICIENT_BATCH)
        {
            AscendC::Brcb(edgePrevious, offDiagonal[rowBase - CHEEVJ_TRIDIAG_INVERSE_COEFFICIENT_BATCH], 1, {1, 8});
        }
        else
        {
            AscendC::Duplicate(edgePrevious, 0.0f, 64);
        }
        if (rowBase + 1 < matrixOrder)
        {
            AscendC::Brcb(edgeCurrent, offDiagonal[rowBase], 1, {1, 8});
        }
        else
        {
            AscendC::Duplicate(edgeCurrent, 0.0f, 64);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BackSubstituteBatch(const AscendC::LocalTensor<float> &cPrime,
                                               const AscendC::LocalTensor<float> &work,
                                               const AscendC::LocalTensor<float> &temp, int validColumns)
    {
        for (int row = matrixOrder - 2; row >= 0; --row)
        {
            const int rowOffset = row * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
            const int nextOffset = (row + 1) * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
            AscendC::Mul(temp, cPrime[rowOffset], work[nextOffset], validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(work[rowOffset], work[rowOffset], temp, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void SolveShiftedBatch(int validColumns)
    {
        auto cPrime = cPrimeBuf.Get<float>();
        auto work = workBuf.Get<float>();
        auto shift = shiftBuf.Get<float>();
        auto denominator = denominatorBuf.Get<float>();
        auto temp = tempBuf.Get<float>();
        auto diagonalBroadcast = diagonalBroadcastBuf.Get<float>();
        auto edgePrevious = edgePreviousBroadcastBuf.Get<float>();
        auto edgeCurrent = edgeCurrentBroadcastBuf.Get<float>();

        for (int rowBase = 0; rowBase < matrixOrder; rowBase += CHEEVJ_TRIDIAG_INVERSE_COEFFICIENT_BATCH)
        {
            BroadcastBatch(rowBase);
            const int rows = matrixOrder - rowBase < CHEEVJ_TRIDIAG_INVERSE_COEFFICIENT_BATCH
                                 ? matrixOrder - rowBase
                                 : CHEEVJ_TRIDIAG_INVERSE_COEFFICIENT_BATCH;
            for (int slot = 0; slot < rows; ++slot)
            {
                const int row = rowBase + slot;
                const int rowOffset = row * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
                AscendC::Sub(denominator, diagonalBroadcast[slot * 8], shift, validColumns);
                AscendC::PipeBarrier<PIPE_V>();
                if (row > 0)
                {
                    const int previousOffset = (row - 1) * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
                    auto previousEdge =
                        slot == 0 ? edgePrevious[56] : edgeCurrent[(slot - 1) * CHEEVJ_TRIDIAG_INVERSE_COLUMNS];
                    AscendC::Mul(temp, previousEdge, cPrime[previousOffset], validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(denominator, denominator, temp, validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(temp, previousEdge, work[previousOffset], validColumns);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(work[rowOffset], work[rowOffset], temp, validColumns);
                }
                AscendC::Adds(denominator, denominator, CHEEVJ_TRIDIAG_INVERSE_PIVOT_BIAS, validColumns);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(work[rowOffset], work[rowOffset], denominator, validColumns);
                if (row + 1 < matrixOrder)
                {
                    AscendC::Div(cPrime[rowOffset], edgeCurrent[slot * 8], denominator, validColumns);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        BackSubstituteBatch(cPrime, work, temp, validColumns);
    }

    __aicore__ inline void NormalizeBatch(int validColumns)
    {
        auto work = workBuf.Get<float>();
        auto norm = normBuf.Get<float>();
        auto temp = tempBuf.Get<float>();
        AscendC::Duplicate(norm, 0.0f, validColumns);
        for (int row = 0; row < matrixOrder; ++row)
        {
            const int rowOffset = row * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
            AscendC::Mul(temp, work[rowOffset], work[rowOffset], validColumns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(norm, norm, temp, validColumns);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Sqrt(norm, norm, validColumns);
        AscendC::PipeBarrier<PIPE_V>();
        for (int row = 0; row < matrixOrder; ++row)
        {
            const int rowOffset = row * CHEEVJ_TRIDIAG_INVERSE_COLUMNS;
            AscendC::Div(work[rowOffset], work[rowOffset], norm, validColumns);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreColumns(int firstColumn, int validColumns)
    {
        auto work = workBuf.Get<float>();
        auto column = columnBuf.Get<float>();
        auto signedIndex = gatherIndexBuf.Get<int32_t>();
        auto index = signedIndex.template ReinterpretCast<uint32_t>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(matrixOrder * sizeof(float)), 0, 0, 0};
        AscendC::CreateVecIndex(signedIndex, 0, static_cast<uint32_t>(matrixOrder));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(5), matrixOrder);
        AscendC::PipeBarrier<PIPE_V>();
        for (int localColumn = 0; localColumn < validColumns; ++localColumn)
        {
            AscendC::Gather(column, work[localColumn], index, 0, matrixOrder);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(eigenvectorsGlobal[(firstColumn + localColumn) * matrixOrder], column, copy);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_TRIDIAG_INVERSE_HPP
