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
 * \file cheevj_stage_block_jacobi16.hpp
 * \brief Stage-wide jb=16 block-Jacobi helper for Cheevj large path.
 *
 * This component serves the large Cheevj path for n>=33 when the
 * workspace layout has enough per-pair U storage:
 *   - local principal problem: at most two 16-column blocks, so <= 32x32 in UB;
 *   - stage schedule: round-robin block pair schedule;
 *   - stage phases: generate and store per-pair U, right multiply A[:,I] * U,
 *     optionally right multiply V[:,I] * U, then explicitly left multiply
 *     U^H * A_right[I,:].
 *
 * The fixed local sweep count and block-0 convergence reduction are part of
 * the validated execution contract. Changing either requires the complete
 * numerical gate to be rerun.
 */

#ifndef CHEEVJ_C64_STAGE_BLOCK_JACOBI16_HPP
#define CHEEVJ_C64_STAGE_BLOCK_JACOBI16_HPP

#include <cstdint>

#include "cheevj_ascendc_symbols.hpp"
#include "cheevj_pair_schedule.hpp"
#include "cheevj_workspace.hpp"
#include "kernel_operator.h"
#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
#include "cheevj_panel_update.hpp"
#endif
#include "complex_vec.hpp"

namespace Cheevj
{

constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK_SIZE = 8;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK16_MIN_N = 641;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK32_MIN_N = 1281;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM = 64;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS =
    CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM * CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_TILE = 128;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_SWEEPS = 8;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_SCRATCH_PLANES = 3;
constexpr int CHEEVJ_STAGE_BLOCK_JACOBI16_MIN_N = 33;
constexpr int CHEEVJ_STAGE_PANEL_PACK_READY_FLAG = 0xA;
constexpr int CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG = 0xB;
constexpr int CHEEVJ_STAGE_PANEL_PACK_READY_FLAG_PONG = 0xC;
constexpr int CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG_PONG = 0xD;
constexpr int CHEEVJ_STAGE_DIRECT_RIGHT_READY_FLAG = 0x6;
constexpr int CHEEVJ_STAGE_DIRECT_RIGHT_DONE_FLAG = 0x7;
constexpr int CHEEVJ_STAGE_PANEL_MIN_N = 256;
constexpr int CHEEVJ_STAGE_PANEL_N_ALIGN = 64;
constexpr float CHEEVJ_STAGE_BLOCK_JACOBI16_REL_TOL = 1.0e-7f;
constexpr float CHEEVJ_STAGE_BLOCK_JACOBI16_MIN_PIVOT = 1.0e-20f;

struct CheevjStageComplex32
{
    float real;
    float imag;
};

using CheevjStageBlockRange16 = CheevjRoundRobinRange;
using CheevjStageBlockPair16 = CheevjRoundRobinPair;

__aicore__ inline CheevjStageComplex32 MakeStageComplex(float real, float imag)
{
    CheevjStageComplex32 value{real, imag};
    return value;
}

__aicore__ inline float StageAbsFloat(float value) { return value >= 0.0f ? value : -value; }

__aicore__ inline float StageMaxFloat(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }

__aicore__ inline float StageSqrtApprox(float value)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits = (bits >> 1) + 0x1fc00000U;
    float root = __builtin_bit_cast(float, bits);
    for (int iter = 0; iter < 4; ++iter)
    {
        root = 0.5f * (root + value / root);
    }
    return root;
}

__aicore__ inline float StageComplexAbs(CheevjStageComplex32 value)
{
    return StageSqrtApprox(value.real * value.real + value.imag * value.imag);
}

__aicore__ inline int StageMod(int value, int divisor)
{
    const int safeDivisor = divisor != 0 ? divisor : 1;
    const int result = value % safeDivisor;
    return result >= 0 ? result : result + divisor;
}

class CheevjStageBlockPairSchedule16 : public CheevjRoundRobinPairSchedule
{
   public:
    __aicore__ inline void Reset(int matrixN, bool needVectors = false)
    {
        const int selectedBlockSize = matrixN >= CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK32_MIN_N
                                          ? 4 * CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK_SIZE
                                          : (matrixN >= CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK16_MIN_N || needVectors
                                                 ? 2 * CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK_SIZE
                                                 : CHEEVJ_STAGE_BLOCK_JACOBI16_BLOCK_SIZE);
        CheevjRoundRobinPairSchedule::Reset(matrixN, selectedBlockSize);
    }
};

class CheevjStageBlockJacobi16
{
   public:
    __aicore__ inline CheevjStageBlockJacobi16()
        : computeVectors(false), maxLocalSweeps(CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_SWEEPS)
    {
    }

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(blockRealBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(blockImagBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(unitaryRealBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(unitaryImagBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(srcRealBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
        pipe->InitBuffer(srcImagBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
        pipe->InitBuffer(accRealBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
        pipe->InitBuffer(accImagBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
        pipe->InitBuffer(tmp0Buf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
        pipe->InitBuffer(tmp1Buf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * 8 * sizeof(uint32_t));
        pipe->InitBuffer(rowStageRealBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * 8 * sizeof(float));
        pipe->InitBuffer(rowStageImagBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * 8 * sizeof(float));
        pipe->InitBuffer(rowStageImagNegBuf, CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * 8 * sizeof(float));
        pipe->InitBuffer(leftRowsRealBuf,
                         CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM * CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
        pipe->InitBuffer(leftRowsImagBuf,
                         CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM * CHEEVJ_STAGE_BLOCK_JACOBI16_TILE * sizeof(float));
    }

    __aicore__ inline void SetWorkspace(const CheevjPlanarWorkspace &planarWorkspace)
    {
        planar = planarWorkspace;
        schedule.Reset(planar.layout.n);
    }

    __aicore__ inline void SetComputeVectors(bool needVectors)
    {
        computeVectors = needVectors;
        schedule.Reset(planar.layout.n, needVectors);
    }

    __aicore__ inline void SetMaxLocalSweeps(int sweeps)
    {
        maxLocalSweeps = sweeps > 0 ? sweeps : CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_SWEEPS;
    }

    __aicore__ inline bool IsSupportedJobzNPath() const
    {
        return planar.layout.n >= CHEEVJ_STAGE_BLOCK_JACOBI16_MIN_N &&
               planar.layout.scratchPlanes >= CHEEVJ_STAGE_BLOCK_JACOBI16_SCRATCH_PLANES &&
               HasUnitaryStorageCapacity() && (!computeVectors || HasVectorPlanes());
    }

    __aicore__ inline bool RequiresScratchPlanes() const
    {
        return planar.layout.scratchPlanes >= CHEEVJ_STAGE_BLOCK_JACOBI16_SCRATCH_PLANES;
    }

#include "cheevj_stage_block_panel_public.inc"
    __aicore__ inline bool ProcessOuterSweeps(int outerSweeps)
    {
        return CheevjProcessOuterSweeps(*this, outerSweeps, IsSupportedJobzNPath());
    }

    __aicore__ inline bool ProcessOuterSweeps(int outerSweeps, bool needVectors)
    {
        SetComputeVectors(needVectors);
        return ProcessOuterSweeps(outerSweeps);
    }

    __aicore__ inline bool ProcessOneSweep()
    {
        bool allLocalConverged = true;
        const int stages = schedule.StageCount();
        for (int stage = 0; stage < stages; ++stage)
        {
            allLocalConverged = ProcessOneStage(stage) && allLocalConverged;
        }
        return allLocalConverged;
    }

    __aicore__ inline bool ProcessOneStage(int stage)
    {
        bool allLocalConverged = true;
        const int pairCount = schedule.PairCountPerStage();
        const int rawWorkerCount = CheevjWorkspaceWorkerCount();
        const int workerCount = rawWorkerCount > 0 ? rawWorkerCount : 1;

        // 910B normally has enough vector cores for all <=16 pairs. The wave loop
        // keeps the helper functional on smaller local simulators without storing U in GM.
        for (int waveStart = 0; waveStart < pairCount; waveStart += workerCount)
        {
            const int wavePairCount = MinInt(workerCount, pairCount - waveStart);
            allLocalConverged = ProcessStageWave(stage, waveStart, wavePairCount) && allLocalConverged;
        }
        return allLocalConverged;
    }

    __aicore__ inline bool ProcessStageWave(int stage, int waveStart, int wavePairCount)
    {
        const int pairIndex = waveStart + GetBlockIdx();
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        LocalTensor<float> blockReal = blockRealBuf.Get<float>();
        LocalTensor<float> blockImag = blockImagBuf.Get<float>();
        LocalTensor<float> unitaryReal = unitaryRealBuf.Get<float>();
        LocalTensor<float> unitaryImag = unitaryImagBuf.Get<float>();

        const bool localConverged =
            SolveStagePair(pairIndex, pair, localDim, blockReal, blockImag, unitaryReal, unitaryImag);
        CheevjWorkspaceSync();
        ApplyStageRightUpdates(stage, waveStart, wavePairCount, unitaryReal, unitaryImag);
        ApplyStageLeftUpdates(stage, waveStart, wavePairCount, unitaryReal, unitaryImag);
        if (pair.valid && localDim > 1 && localDim <= CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM)
        {
            ScatterPrincipalBlock(pair, localDim, blockReal, blockImag);
        }
        CheevjWorkspaceSync();
        return localConverged;
    }

    __aicore__ inline bool SolveStagePair(int pairIndex, const CheevjStageBlockPair16 &pair, int localDim,
                                          LocalTensor<float> blockReal, LocalTensor<float> blockImag,
                                          LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag)
    {
        if (!pair.valid || localDim <= 1 || localDim > CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM)
        {
            return true;
        }
        ExtractPrincipalBlock(pair, blockReal, blockImag);
        InitLocalIdentity(unitaryReal, unitaryImag, localDim);
        const bool localConverged = RunLocalJacobi(blockReal, blockImag, unitaryReal, unitaryImag, localDim);
        StoreUnitaryForPair(pairIndex, unitaryReal, unitaryImag);
        return localConverged;
    }

    __aicore__ inline void ApplyStageRightUpdates(int stage, int waveStart, int wavePairCount,
                                                  LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag)
    {
#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
        const int panelSlotCount = PanelServiceSlotCount();
        const int panelSubBlockCount = GetSubBlockNum() > 0 ? GetSubBlockNum() : 2;
        const int panelSlotIndex = GetBlockIdx() / panelSubBlockCount;
        const int panelLane = GetBlockIdx() - panelSlotIndex * panelSubBlockCount;
        const bool useDirectSegmentedRight = CanUseDirectSegmentedRightPanel();
        if (waveStart == 0 && useDirectSegmentedRight && panelSlotIndex < schedule.PairCountPerStage() && panelLane < 2)
        {
            (void)ProcessDirectRightPanelStageAiv(stage, panelSlotIndex, panelLane);
        }
        else if (waveStart == 0 && panelSlotIndex < panelSlotCount && panelLane < 2)
        {
            (void)ProcessRightPanelStageAiv(stage, panelSlotIndex, panelLane);
        }
#endif
        ApplyRightUnitaryToMatrixColumnsDistributed(stage, waveStart, wavePairCount, unitaryReal, unitaryImag);
        if (!computeVectors)
        {
            CheevjWorkspaceSync();
        }
        if (computeVectors)
        {
#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
            if (!useDirectSegmentedRight && waveStart == 0 && panelSlotIndex < panelSlotCount && panelLane < 2)
            {
                (void)ProcessRightVectorPanelStageAiv(stage, panelSlotIndex, panelLane);
            }
#endif
            ApplyRightUnitaryToVectorColumnsDistributed(stage, waveStart, wavePairCount, unitaryReal, unitaryImag);
            CheevjWorkspaceSync();
        }
    }

    __aicore__ inline void ApplyStageLeftUpdates(int stage, int waveStart, int wavePairCount,
                                                 LocalTensor<float> unitaryReal, LocalTensor<float> unitaryImag)
    {
#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
        const int panelSlotCount = PanelServiceSlotCount();
        const int panelSubBlockCount = GetSubBlockNum() > 0 ? GetSubBlockNum() : 2;
        const int panelSlotIndex = GetBlockIdx() / panelSubBlockCount;
        const int panelLane = GetBlockIdx() - panelSlotIndex * panelSubBlockCount;
        if (waveStart == 0 && panelSlotIndex < panelSlotCount && panelLane < 2)
        {
            (void)ProcessLeftPanelStageAiv(stage, panelSlotIndex, panelLane);
        }
#endif
        ApplyLeftUnitaryRowsDistributed(stage, waveStart, wavePairCount, unitaryReal, unitaryImag);
        CheevjWorkspaceSync();
    }

   private:
#include "cheevj_block_jacobi_buffers.inc"
    TBuf<TPosition::VECCALC> tmp0Buf;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> rowStageRealBuf;
    TBuf<TPosition::VECCALC> rowStageImagBuf;
    TBuf<TPosition::VECCALC> rowStageImagNegBuf;
    TBuf<TPosition::VECCALC> leftRowsRealBuf;
    TBuf<TPosition::VECCALC> leftRowsImagBuf;
    CheevjPlanarWorkspace planar;
    CheevjStageBlockPairSchedule16 schedule;
    bool computeVectors;
    int maxLocalSweeps;

#include "cheevj_stage_block_left_update.inc"
#include "cheevj_stage_block_local_jacobi.inc"
#include "cheevj_stage_block_panel_impl_part1.inc"
#include "cheevj_stage_block_panel_impl_part2.inc"
#include "cheevj_stage_block_right_update.inc"
#include "cheevj_stage_block_storage.inc"
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_STAGE_BLOCK_JACOBI16_HPP
