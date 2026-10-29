/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

namespace RealLuKernel
{
__aicore__ inline void SmallInverse::ComputeDiagonalReciprocals(LocalTensor<float> diagonalValues)
{
    int count = n * lanes;
    for (int k = 0; k < n; ++k)
    {
        Adds(diagonalValues[k * lanes], L(k, k), 0.0f, lanes);
    }
    PipeBarrier<PIPE_V>();
    Abs(tmp, diagonalValues, count);
    PipeBarrier<PIPE_V>();
    ReduceMin(tmp[rowStride], tmp, tmp, count);
    Fence<HardEvent::V_S>();
    float least = tmp[rowStride].GetValue(0);
    Fence<HardEvent::S_V>();
    Abs(tmp, diagonalValues, count);
    PipeBarrier<PIPE_V>();
    ReduceMax(tmp[rowStride], tmp, tmp, count);
    Fence<HardEvent::V_S>();
    float greatest = tmp[rowStride].GetValue(0);
    Fence<HardEvent::S_V>();
    Duplicate(tmp, 1.0f, count);
    PipeBarrier<PIPE_V>();
#if (__NPU_ARCH__ == 3510)
    if (least >= 1.1754943508222875e-38f && greatest <= 1.0e37f)
    {
        Div<float, NORMAL_DIVIDE>(tmp, tmp, diagonalValues, count);
    }
    else
    {
        Divide(tmp, tmp, diagonalValues, count);
    }
#else
    Divide(tmp, tmp, diagonalValues, count);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SmallInverse::DiagonalReciprocals()
{
    // Compact input/output staging is only eight matrices. Reuse the
    // dead range flags/permutation workspace for the complete diagonal.
    auto diagonalValues = compact ? pivIndex.ReinterpretCast<float>() : raw;
    ComputeDiagonalReciprocals(diagonalValues);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (n == 4 || n == 8)
    {
        return;
    }
#endif
    if (!compact)
    {
        Duplicate(inv, 0.0f, n * n * lanes);
        PipeBarrier<PIPE_V>();
    }
    // In compact mode only U's diagonal changes; original strict-lower
    // coefficients remain live until their lower column is cached.
    for (int k = 0; k < n; ++k)
    {
        Adds(X(k, k), tmp[k * lanes], 0.0f, lanes);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SmallInverse::ReciprocalPair()
{
#if (__NPU_ARCH__ == 3510)
    // The fast exact division variant is valid when both the denominator
    // and its reciprocal are normal. Retain IEEE division for extremes.
    Abs(tmp, L(0, 0), lanes);
    Abs(tmp[lanes], L(1, 1), lanes);
    PipeBarrier<PIPE_V>();
    ReduceMin(tmp[rowStride], tmp, tmp, 2 * lanes);
    Fence<HardEvent::V_S>();
    float least = tmp[rowStride].GetValue(0);
    Fence<HardEvent::S_V>();
    Abs(tmp, L(0, 0), lanes);
    Abs(tmp[lanes], L(1, 1), lanes);
    PipeBarrier<PIPE_V>();
    ReduceMax(tmp[rowStride], tmp, tmp, 2 * lanes);
    Fence<HardEvent::V_S>();
    float greatest = tmp[rowStride].GetValue(0);
    Fence<HardEvent::S_V>();
    Adds(raw, L(0, 0), 0.0f, lanes);
    Adds(raw[lanes], L(1, 1), 0.0f, lanes);
    Duplicate(tmp, 1.0f, 2 * lanes);
    PipeBarrier<PIPE_V>();
    if (least >= 1.1754943508222875e-38f && greatest <= 1.0e37f)
    {
        Div<float, NORMAL_DIVIDE>(tmp, tmp, raw, 2 * lanes);
    }
    else
    {
        Divide(tmp, tmp, raw, 2 * lanes);
    }
#else
    Adds(raw, L(0, 0), 0.0f, lanes);
    Adds(raw[lanes], L(1, 1), 0.0f, lanes);
    Duplicate(tmp, 1.0f, 2 * lanes);
    PipeBarrier<PIPE_V>();
    Divide(tmp, tmp, raw, 2 * lanes);
#endif
    PipeBarrier<PIPE_V>();
    Adds(X(0, 0), tmp, 0.0f, lanes);
    Adds(X(1, 1), tmp[lanes], 0.0f, lanes);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SmallInverse::InverseTwo()
{
    // STRTI2 followed by the descending-column SGEMV used by LAPACK
    // SGETRI. Avoid the extra rounding from solving individual RHSs.
    ReciprocalPair();
    Mul(X(0, 1), L(0, 1), X(0, 0), lanes);
    Muls(X(1, 0), L(1, 0), -1.0f, lanes);
    PipeBarrier<PIPE_V>();
    Mul(X(0, 1), X(0, 1), X(1, 1), lanes);
    Mul(X(1, 0), X(1, 0), X(1, 1), lanes);
    PipeBarrier<PIPE_V>();
    Muls(X(0, 1), X(0, 1), -1.0f, lanes);
    PipeBarrier<PIPE_V>();
    Mul(tmp, X(0, 1), L(1, 0), lanes);
    PipeBarrier<PIPE_V>();
    Sub(X(0, 0), X(0, 0), tmp, lanes);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, piv, 2.0f, CMPMODE::EQ, lanes);
    PipeBarrier<PIPE_V>();
    for (int row = 0; row < 2; ++row)
    {
        Adds(tmp, X(row, 0), 0.0f, lanes);
        PipeBarrier<PIPE_V>();
        Select(X(row, 0), mask, X(row, 1), X(row, 0), SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        Select(X(row, 1), mask, tmp, X(row, 1), SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        PipeBarrier<PIPE_V>();
    }
}

#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
__aicore__ inline void SmallInverse::GenericUpperInverse()
{
    for (int j = 1; j < n; ++j)
    {
        for (int row = 0; row < j; ++row)
        {
            Mul(X(row, j), L(row, j), X(row, row), lanes);
            PipeBarrier<PIPE_V>();
            for (int k = row + 1; k < j; ++k)
            {
                MulAddDst(X(row, j), L(k, j), X(row, k), lanes);
                PipeBarrier<PIPE_V>();
            }
            Mul(X(row, j), X(row, j), X(j, j), lanes);
            PipeBarrier<PIPE_V>();
            Muls(X(row, j), X(row, j), -1.0f, lanes);
            PipeBarrier<PIPE_V>();
        }
    }
}

#endif
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
__aicore__ inline void SmallInverse::GenericLowerInverse()
{
    // SGEMV accumulates its dot before the final non-fused subtraction.
    for (int j = n - 2; j >= 0; --j)
    {
        for (int row = 0; row < n; ++row)
        {
            Duplicate(tmp, 0.0f, lanes);
            PipeBarrier<PIPE_V>();
            for (int k = j + 1; k < n; ++k)
            {
                MulAddDst(tmp, L(k, j), X(row, k), lanes);
                PipeBarrier<PIPE_V>();
            }
            Sub(X(row, j), X(row, j), tmp, lanes);
            PipeBarrier<PIPE_V>();
        }
    }
}

#endif
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
__aicore__ inline void SmallInverse::GenericInversePermutation()
{
    if (n > 2)
    {
        for (int k = n - 1; k >= 0; --k)
        {
            for (int column = k + 1; column < n; ++column)
            {
                CompareScalar(mask, piv[k * lanes], static_cast<float>(column + 1), CMPMODE::EQ, lanes);
                PipeBarrier<PIPE_V>();
                for (int row = 0; row < n; ++row)
                {
                    Adds(tmp, X(row, k), 0.0f, lanes);
                    PipeBarrier<PIPE_V>();
                    Select(X(row, k), mask, X(row, column), X(row, k), SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
                    Select(X(row, column), mask, tmp, X(row, column), SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }
}

#endif
__aicore__ inline void SmallInverse::LapackInverse(bool identityPivots)
{
    DiagonalReciprocals();
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    Fence<HardEvent::V_S>();
    if (compact)
    {
        if (!InplaceUpperLow())
        {
            InplaceUpperHigh();
        }
        InplaceLowerInverse();
    }
    else if (!SeparateInverseLow() && !SeparateInverseMiddle() && !SeparateInverseHigh())
    {
        SmallLapackColumns(lu, inv, tmp, n, lanes);
    }
    Fence<HardEvent::V_S>();
#else
    GenericUpperInverse();
    GenericLowerInverse();
    GenericInversePermutation();
#endif
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::SaveRefinementInput()
{
    // raw otherwise becomes scratch only for the first row during
    // DiagonalReciprocals. pivIndex is unused until final permutation.
    Adds(raw, lu, 0.0f, n * n * lanes);
    Adds(pivIndex.ReinterpretCast<float>(), lu, 0.0f, rowStride);
    PipeBarrier<PIPE_V>();
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::InitialRefinement()
{
    auto original = reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr());
    auto inverse = reinterpret_cast<__ubuf__ float *>(inv.GetPhyAddr());
    auto residual = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto pivotValues = reinterpret_cast<__ubuf__ float *>(piv.GetPhyAddr());
    auto scratch = reinterpret_cast<__ubuf__ float *>(tmp.GetPhyAddr());
    for (int k = n - 1; k >= 0; --k)
    {
        for (int wave = 0; wave < lanes; wave += 64)
        {
            uint32_t valid = lanes - wave < 64 ? lanes - wave : 64;
            SmallInversePivotVF(inverse + wave, pivotValues + wave, n, lanes, k, valid);
        }
    }
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t valid = lanes - wave < 64 ? lanes - wave : 64;
        SmallInverseResidualVF(original + wave, inverse + wave, residual + wave, n, lanes, valid);
    }
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t valid = lanes - wave < 64 ? lanes - wave : 64;
        SmallInverseCorrectionVF(residual + wave, inverse + wave, residual + wave, scratch + wave, n, lanes, valid);
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline bool SmallInverse::RefinementNeedsCorrection()
{
    ReduceMax(tmp, tmp[2 * rowStride + 2 * lanes], tmp[rowStride], lanes);
    Fence<HardEvent::V_S>();
    bool needsCorrection = tmp.GetValue(0) > 0.0f;
    Fence<HardEvent::S_V>();
    return needsCorrection;
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
template <bool fusedResidual>
__aicore__ inline void SmallInverse::ScoreRefinement()
{
    auto original = reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr());
    auto inverse = reinterpret_cast<__ubuf__ float *>(inv.GetPhyAddr());
    auto residual = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto flags = reinterpret_cast<__ubuf__ float *>(tmp.GetPhyAddr()) + 2 * rowStride + 2 * lanes;
    auto infoValues = reinterpret_cast<__ubuf__ int32_t *>(infos.GetPhyAddr());
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t valid = lanes - wave < 64 ? lanes - wave : 64;
        SmallInverseSelectVF<false, fusedResidual>(original + wave, inverse + wave, residual + wave, flags + wave,
                                                   infoValues + wave, n, lanes, valid);
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
template <bool fusedResidual>
__aicore__ inline void SmallInverse::DampedRefinement(float step)
{
    auto original = reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr());
    auto inverse = reinterpret_cast<__ubuf__ float *>(inv.GetPhyAddr());
    auto residual = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
    auto scratch = reinterpret_cast<__ubuf__ float *>(tmp.GetPhyAddr());
    auto flags = scratch + 2 * rowStride + 2 * lanes;
    auto infoValues = reinterpret_cast<__ubuf__ int32_t *>(infos.GetPhyAddr());
    for (int wave = 0; wave < lanes; wave += 64)
    {
        uint32_t valid = lanes - wave < 64 ? lanes - wave : 64;
        SmallInverseDampedVF<fusedResidual>(original + wave, inverse + wave, residual + wave, scratch + wave, n, lanes,
                                            valid, step);
        SmallInverseSelectVF<true, fusedResidual>(original + wave, inverse + wave, residual + wave, flags + wave,
                                                  infoValues + wave, n, lanes, valid);
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::RefineInverse()
{
    Adds(raw, pivIndex.ReinterpretCast<float>(), 0.0f, rowStride);
    PipeBarrier<PIPE_V>();
    // Form C=X*P before refining, retaining the original A in raw.
    InitialRefinement();
    Adds(inv, lu, 0.0f, n * n * lanes);
    PipeBarrier<PIPE_V>();
    ScoreRefinement();
    // Retain full and half Newton steps and the original acceptance order.
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        if (!RefinementNeedsCorrection())
        {
            break;
        }
        DampedRefinement(iteration == 0 ? 1.0f : 0.5f);
    }
    if (n == 3)
    {
        ScoreRefinement<true>();
        if (RefinementNeedsCorrection())
        {
            DampedRefinement<true>(0.5f);
        }
    }
}

#endif

__aicore__ inline void SmallInverse::PreparePivots(GM_ADDR pivot, int first, int count, bool factor)
{
    if (factor)
    {
        Factor();
    }
    else if (pivot != nullptr)
    {
        GlobalTensor<int> p;
        p.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivot));
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * n * sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int> pad{true, 0, static_cast<uint8_t>(Align(count * n) - count * n), 0};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(raw.ReinterpretCast<int>(), p[static_cast<int64_t>(first) * n], copy, pad);
        Fence<HardEvent::MTE2_V>();
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
        SmallPackedPivotToSoAVF(reinterpret_cast<__ubuf__ int32_t *>(piv.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ int32_t *>(raw.GetPhyAddr()), n, lanes, count);
#else
        Gather(piv.ReinterpretCast<int>(), raw.ReinterpretCast<int>(), pivIndex, 0, n * lanes);
#endif
        PipeBarrier<PIPE_V>();
        Cast(piv, piv.ReinterpretCast<int>(), RoundMode::CAST_NONE, n * lanes);
        PipeBarrier<PIPE_V>();
    }
    else
    {
        for (int k = 0; k < n; ++k)
        {
            Duplicate(piv[k * lanes], static_cast<float>(k + 1), lanes);
        }
    }
}

__aicore__ inline void SmallInverse::ClassifySingular()
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    Fence<HardEvent::V_S>();
    SmallSingularInfoVF(reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ int32_t *>(infos.GetPhyAddr()), n, lanes, lanes);
#else
    auto state = tmp[2 * rowStride + 2 * lanes];
    Duplicate(state, 0.0f, lanes);
    PipeBarrier<PIPE_V>();
    for (int k = 0; k < n; ++k)
    {
        CompareScalar(mask, L(k, k), 0.0f, CMPMODE::EQ, lanes);
        CompareScalar(mask[maskBytes], state, 0.0f, CMPMODE::EQ, lanes);
        Duplicate(tmp[2 * rowStride + lanes], static_cast<float>(k + 1), lanes);
        PipeBarrier<PIPE_V>();
        And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
            mask[maskBytes].ReinterpretCast<uint16_t>(), maskBytes / sizeof(uint16_t));
        PipeBarrier<PIPE_V>();
        Select(state, mask, tmp[2 * rowStride + lanes], state, SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        PipeBarrier<PIPE_V>();
    }
    Cast(infos, state, RoundMode::CAST_RINT, lanes);
#endif
}

__aicore__ inline void SmallInverse::Run(GM_ADDR input, GM_ADDR output, GM_ADDR pivot, GM_ADDR info, int first,
                                         int count, bool factor, bool identityPivots, bool arrays, int lda, int ldc)
{
    GlobalTensor<float> a, c;
    GlobalTensor<int> result;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    LoadGroup(input, a, first, count, arrays, lda);
    PipeBarrier<PIPE_V>();
    Duplicate(infos, 0, lanes);
    bool refine = false;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    refine = factor && n >= 3 && n <= 8;
    if (refine)
    {
        SaveRefinementInput();
    }
#endif
    PreparePivots(pivot, first, count, factor);
    ClassifySingular();
    PipeBarrier<PIPE_V>();
    if (n == 2)
    {
        InverseTwo();
    }
    else
    {
        LapackInverse(identityPivots);
    }
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (refine)
    {
        RefineInverse();
    }
#endif
    StoreGroup(output, c, first, count, arrays, ldc, arrays || lda != 0, identityPivots, refine);
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams infoCopy{1, static_cast<uint32_t>(count * sizeof(int)), 0, 0, 0};
    DataCopyPad(result[first], infos, infoCopy);
    Fence<HardEvent::MTE3_S>();
}
}  // namespace RealLuKernel
