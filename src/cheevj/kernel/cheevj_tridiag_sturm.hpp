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
 * \file cheevj_tridiag_sturm.hpp
 * \brief Isolated SIMD Sturm-bisection proof of concept for a real symmetric tridiagonal matrix.
 */

#ifndef CHEEVJ_C64_TRIDIAG_STURM_HPP
#define CHEEVJ_C64_TRIDIAG_STURM_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace Cheevj
{

constexpr int CHEEVJ_STURM_MAX_N = 2048;
constexpr int CHEEVJ_STURM_LANES = 64;
constexpr int CHEEVJ_STURM_COEFFICIENT_BATCH = 8;
constexpr int CHEEVJ_STURM_FP32_MANTISSA_ROUNDS = 24;
// This is deliberately much smaller than a normal FP32 pivot.  Subtracting it
// is therefore a no-op after rounding for ordinary pivots, while an exact zero
// becomes a finite negative pivot before the next division.
constexpr float CHEEVJ_STURM_ZERO_PIVOT_BIAS = 1.0e-20f;

// One AIV owns one contiguous 64-eigenvalue interval.  The low/high outputs
// remain part of the private backend API so callers can seed narrower
// intervals without changing the SIMD refinement body.
class CheevjTridiagSturm
{
   public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR eigenvalues,
                                GM_ADDR lowWorkspace, GM_ADDR highWorkspace, int n, float globalLow, float globalHigh,
                                int rounds)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        diagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(diagonal));
        offDiagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(offDiagonal));
        eigenvaluesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvalues));
        lowGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lowWorkspace));
        highGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(highWorkspace));
        matrixOrder = n;
        lowerBound = globalLow;
        upperBound = globalHigh;
        iterationCount = rounds;

        pipe->InitBuffer(diagonalBuf, CHEEVJ_STURM_MAX_N * sizeof(float));
        pipe->InitBuffer(offDiagonalBuf, CHEEVJ_STURM_MAX_N * sizeof(float));
        pipe->InitBuffer(targetBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(lowBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(highBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(lambdaBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(pivotBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(tmpBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(countBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(countIntBuf, CHEEVJ_STURM_LANES * sizeof(int32_t));
        pipe->InitBuffer(signBuf, CHEEVJ_STURM_LANES * sizeof(uint32_t));
        pipe->InitBuffer(diagonalBroadcastBuf, CHEEVJ_STURM_LANES * sizeof(float));
        pipe->InitBuffer(edgeBroadcastBuf, CHEEVJ_STURM_LANES * sizeof(float));
        // Compare consumes a packed predicate.  Reserve a full 256-byte vector
        // so both Compare and tensor/tensor Select see an aligned mask region.
        pipe->InitBuffer(maskBuf, 256);
#else
        (void)pipe;
        (void)diagonal;
        (void)offDiagonal;
        (void)eigenvalues;
        (void)lowWorkspace;
        (void)highWorkspace;
        (void)n;
        (void)globalLow;
        (void)globalHigh;
        (void)rounds;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int firstEigenIndex = static_cast<int>(AscendC::GetBlockIdx()) * CHEEVJ_STURM_LANES;
        if (matrixOrder <= 0 || matrixOrder > CHEEVJ_STURM_MAX_N || firstEigenIndex >= matrixOrder ||
            iterationCount <= 0)
        {
            return;
        }
        const int validLanes = Min(matrixOrder - firstEigenIndex, CHEEVJ_STURM_LANES);

        LoadTridiagonal();
        PrepareLaneState(firstEigenIndex);

        auto low = lowBuf.Get<float>();
        auto high = highBuf.Get<float>();
        auto lambda = lambdaBuf.Get<float>();
        auto target = targetBuf.Get<float>();
        auto count = countBuf.Get<float>();
        auto mask = maskBuf.Get<uint8_t>();

        // More than 24 full-range bisections cannot refine an FP32 endpoint:
        // the midpoint has exhausted its mantissa.  Avoid re-running the O(n)
        // Sturm recurrence after the endpoints have become bit-identical.
        const int effectiveIterations = Min(iterationCount, CHEEVJ_STURM_FP32_MANTISSA_ROUNDS);
        for (int round = 0; round < effectiveIterations; ++round)
        {
            AscendC::Add(lambda, low, high, CHEEVJ_STURM_LANES);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(lambda, lambda, 0.5f, CHEEVJ_STURM_LANES);
            AscendC::PipeBarrier<PIPE_V>();

            CountLessThan(lambda, count);
            // count(lambda) <= k means lambda is still below eigenvalue k.
            AscendC::Compare(mask, count, target, AscendC::CMPMODE::LE, CHEEVJ_STURM_LANES);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(low, mask, lambda, low, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, CHEEVJ_STURM_LANES);
            AscendC::Select(high, mask, high, lambda, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, CHEEVJ_STURM_LANES);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Add(lambda, low, high, CHEEVJ_STURM_LANES);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(lambda, lambda, 0.5f, CHEEVJ_STURM_LANES);
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreResult(firstEigenIndex, validLanes, lambda, low, high);
#endif
    }

   private:
    AscendC::GlobalTensor<float> diagonalGlobal;
    AscendC::GlobalTensor<float> offDiagonalGlobal;
    AscendC::GlobalTensor<float> eigenvaluesGlobal;
    AscendC::GlobalTensor<float> lowGlobal;
    AscendC::GlobalTensor<float> highGlobal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offDiagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> highBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lambdaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pivotBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> countBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> countIntBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> signBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalBroadcastBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> edgeBroadcastBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;

    int matrixOrder = 0;
    int iterationCount = 0;
    float lowerBound = 0.0f;
    float upperBound = 0.0f;

    __aicore__ inline int Min(int lhs, int rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline void LoadTridiagonal()
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        const int alignedOrder =
            (matrixOrder + CHEEVJ_STURM_COEFFICIENT_BATCH - 1) & ~(CHEEVJ_STURM_COEFFICIENT_BATCH - 1);
        const uint8_t rightPadding = static_cast<uint8_t>(alignedOrder - matrixOrder);
        AscendC::DataCopyExtParams diagonalCopy{1, static_cast<uint32_t>(matrixOrder * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> diagonalPad{rightPadding != 0, 0, rightPadding, 0.0f};
        AscendC::DataCopyPad(diagonal, diagonalGlobal, diagonalCopy, diagonalPad);
        if (matrixOrder > 1)
        {
            AscendC::DataCopyExtParams offDiagonalCopy{1, static_cast<uint32_t>((matrixOrder - 1) * sizeof(float)), 0,
                                                       0, 0};
            // Pad one element on the left so offDiagonal[row] is the edge
            // preceding row.  Both coefficient arrays can then be consumed
            // by Brcb from an aligned rowBase without a per-row Gather.
            AscendC::DataCopyPadExtParams<float> offDiagonalPad{true, 1, rightPadding, 0.0f};
            AscendC::DataCopyPad(offDiagonal, offDiagonalGlobal, offDiagonalCopy, offDiagonalPad);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if (matrixOrder > 1)
        {
            // The recurrence uses e_i^2.  Squaring the complete array once
            // replaces one Mul in every row of every bisection round.
            AscendC::Mul(offDiagonal, offDiagonal, offDiagonal, alignedOrder);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareLaneState(int firstEigenIndex)
    {
        auto target = targetBuf.Get<float>();
        auto signedTarget = target.ReinterpretCast<int32_t>();
        auto low = lowBuf.Get<float>();
        auto high = highBuf.Get<float>();

        AscendC::CreateVecIndex(signedTarget, static_cast<int32_t>(firstEigenIndex),
                                static_cast<uint32_t>(CHEEVJ_STURM_LANES));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(target, signedTarget, AscendC::RoundMode::CAST_NONE, CHEEVJ_STURM_LANES);
        AscendC::Duplicate(low, lowerBound, CHEEVJ_STURM_LANES);
        AscendC::Duplicate(high, upperBound, CHEEVJ_STURM_LANES);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastCoefficientBatch(int rowBase)
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonalSquared = offDiagonalBuf.Get<float>();
        auto diagonalBroadcast = diagonalBroadcastBuf.Get<float>();
        auto edgeBroadcast = edgeBroadcastBuf.Get<float>();
        // One Brcb turns eight adjacent coefficients into eight aligned
        // 32-byte blocks.  A level-0 binary instruction can broadcast any
        // one of those blocks over all 64 lanes with srcBlkStride=0.
        AscendC::Brcb(diagonalBroadcast, diagonal[rowBase], 1, {1, 8});
        AscendC::Brcb(edgeBroadcast, offDiagonalSquared[rowBase], 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulatePivotSign(const AscendC::LocalTensor<float> &pivot,
                                               const AscendC::LocalTensor<int32_t> &count)
    {
        auto pivotBits = pivot.ReinterpretCast<uint32_t>();
        auto signBits = signBuf.Get<uint32_t>();
        auto sign = signBits.ReinterpretCast<int32_t>();
        // IEEE-754 puts the sign in bit 31.  Extracting it directly replaces
        // Duplicate + Compare + Select + Muls + Adds in the old count path.
        AscendC::ShiftRight<uint32_t, false>(signBits, pivotBits, static_cast<uint32_t>(31), AscendC::MASK_PLACEHOLDER,
                                             1, {1, 1, 8, 8}, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<int32_t, false>(count, count, sign, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    }

    __aicore__ inline void FinishPivot(const AscendC::LocalTensor<float> &pivot,
                                       const AscendC::LocalTensor<int32_t> &count)
    {
        AscendC::Adds<float, false>(pivot, pivot, -CHEEVJ_STURM_ZERO_PIVOT_BIAS, AscendC::MASK_PLACEHOLDER, 1,
                                    {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AccumulatePivotSign(pivot, count);
    }

    __aicore__ inline void ProcessFirstRow(const AscendC::LocalTensor<float> &lambda,
                                           const AscendC::LocalTensor<int32_t> &count)
    {
        auto pivot = pivotBuf.Get<float>();
        auto diagonalBroadcast = diagonalBroadcastBuf.Get<float>();
        AscendC::Sub<float, false>(pivot, diagonalBroadcast, lambda, AscendC::MASK_PLACEHOLDER, 1, {1, 0, 1, 8, 0, 8});
        AscendC::PipeBarrier<PIPE_V>();
        FinishPivot(pivot, count);
    }

    __aicore__ inline void ProcessRegularRow(const AscendC::LocalTensor<float> &lambda,
                                             const AscendC::LocalTensor<int32_t> &count, int coefficientSlot)
    {
        auto pivot = pivotBuf.Get<float>();
        auto quotient = tmpBuf.Get<float>();
        auto diagonalBroadcast = diagonalBroadcastBuf.Get<float>();
        auto edgeBroadcast = edgeBroadcastBuf.Get<float>();
        const int coefficientOffset = coefficientSlot * CHEEVJ_STURM_COEFFICIENT_BATCH;

        AscendC::Div<float, false>(quotient, edgeBroadcast[coefficientOffset], pivot, AscendC::MASK_PLACEHOLDER, 1,
                                   {1, 0, 1, 8, 0, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(pivot, diagonalBroadcast[coefficientOffset], lambda, AscendC::MASK_PLACEHOLDER, 1,
                                   {1, 0, 1, 8, 0, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(pivot, pivot, quotient, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        FinishPivot(pivot, count);
    }

    __aicore__ inline void CountLessThanFixed512(const AscendC::LocalTensor<float> &lambda,
                                                 const AscendC::LocalTensor<int32_t> &count)
    {
        BroadcastCoefficientBatch(0);
        ProcessFirstRow(lambda, count);
#pragma unroll
        for (int slot = 1; slot < CHEEVJ_STURM_COEFFICIENT_BATCH; ++slot)
        {
            ProcessRegularRow(lambda, count, slot);
        }
        for (int rowBase = CHEEVJ_STURM_COEFFICIENT_BATCH; rowBase < 512; rowBase += CHEEVJ_STURM_COEFFICIENT_BATCH)
        {
            BroadcastCoefficientBatch(rowBase);
#pragma unroll
            for (int slot = 0; slot < CHEEVJ_STURM_COEFFICIENT_BATCH; ++slot)
            {
                ProcessRegularRow(lambda, count, slot);
            }
        }
    }

    __aicore__ inline void CountLessThanGeneric(const AscendC::LocalTensor<float> &lambda,
                                                const AscendC::LocalTensor<int32_t> &count)
    {
        BroadcastCoefficientBatch(0);
        ProcessFirstRow(lambda, count);
        int row = 1;
        for (; row < matrixOrder && row < CHEEVJ_STURM_COEFFICIENT_BATCH; ++row)
        {
            ProcessRegularRow(lambda, count, row);
        }
        for (int rowBase = CHEEVJ_STURM_COEFFICIENT_BATCH; rowBase < matrixOrder;
             rowBase += CHEEVJ_STURM_COEFFICIENT_BATCH)
        {
            BroadcastCoefficientBatch(rowBase);
            const int rowsThisBatch = Min(matrixOrder - rowBase, CHEEVJ_STURM_COEFFICIENT_BATCH);
            for (int slot = 0; slot < rowsThisBatch; ++slot)
            {
                ProcessRegularRow(lambda, count, slot);
            }
        }
    }

    __aicore__ inline void CountLessThan(const AscendC::LocalTensor<float> &lambda,
                                         const AscendC::LocalTensor<float> &count)
    {
        auto countInt = countIntBuf.Get<int32_t>();
        AscendC::Duplicate(countInt, static_cast<int32_t>(0), CHEEVJ_STURM_LANES);
        AscendC::PipeBarrier<PIPE_V>();

        // All hot instructions process exactly one FP32 vector.  Setting the
        // Normal mask once avoids the Counter-mode setup/restore emitted by
        // every calCount overload, which dominated the measured scalar pipe.
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(CHEEVJ_STURM_LANES);
        if (matrixOrder == 512)
        {
            CountLessThanFixed512(lambda, countInt);
        }
        else
        {
            CountLessThanGeneric(lambda, countInt);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, int32_t, false>(count, countInt, AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER,
                                             1, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ResetMask();
    }

    __aicore__ inline void StoreResult(int firstEigenIndex, int validLanes,
                                       const AscendC::LocalTensor<float> &eigenvalues,
                                       const AscendC::LocalTensor<float> &low, const AscendC::LocalTensor<float> &high)
    {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(validLanes * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(eigenvaluesGlobal[firstEigenIndex], eigenvalues, copy);
        AscendC::DataCopyPad(lowGlobal[firstEigenIndex], low, copy);
        AscendC::DataCopyPad(highGlobal[firstEigenIndex], high, copy);
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_TRIDIAG_STURM_HPP
