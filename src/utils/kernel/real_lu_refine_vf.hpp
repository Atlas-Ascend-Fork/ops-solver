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
#include "real_lu_factor_vf.hpp"

namespace RealLuKernel
{
// Reverse-order column swaps form C=X*P after inverting P*A.
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallInversePivotVF(__ubuf__ float *inverse,
                                                                                     __ubuf__ float *pivots,
                                                                                     uint16_t order, uint16_t pitch,
                                                                                     uint16_t k, uint32_t count)
{
    MicroAPI::RegTensor<float> pivot, first, second;
    MicroAPI::RegTensor<int32_t> pivotColumn;
    MicroAPI::RegTensor<uint32_t> lane, index;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(pivot, pivots + k * pitch, 1u, valid);
    MicroAPI::Cast<int32_t, float, SMALL_FACTOR_PIVOT_CAST>(pivotColumn, pivot, valid);
    MicroAPI::Adds(pivotColumn, pivotColumn, -1, valid);
    MicroAPI::Muls(index, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(pivotColumn), static_cast<uint32_t>(pitch),
                   valid);
    MicroAPI::Add(index, index, lane, valid);
    for (uint16_t row = 0; row < order; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(first, inverse + (row * order + k) * pitch,
                                                                            1u, valid);
        MicroAPI::Gather(second, inverse + row * order * pitch, index, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
        MicroAPI::Scatter(inverse + row * order * pitch, first, index, valid);
        MicroAPI::StoreAlign(inverse + (row * order + k) * pitch, second, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

template <bool selectCandidate, bool fusedResidual>
__simd_callee__ inline void SmallResidualProducts(__ubuf__ float *original, __ubuf__ float *inverse,
                                                  __ubuf__ float *candidate, uint16_t order, uint16_t pitch,
                                                  uint16_t row, uint16_t column, MicroAPI::MaskReg &valid,
                                                  MicroAPI::RegTensor<float> &left, MicroAPI::RegTensor<float> &right,
                                                  MicroAPI::RegTensor<float> &other,
                                                  MicroAPI::RegTensor<float> &product, MicroAPI::RegTensor<float> &dot,
                                                  MicroAPI::RegTensor<float> &candidateDot)
{
    for (uint16_t k = 0; k < order; ++k)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, inverse + (row * order + k) * pitch,
                                                                            1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            right, original + (k * order + column) * pitch, 1u, valid);
        if constexpr (fusedResidual)
        {
            MicroAPI::MulAddDst<float>(dot, left, right, valid);
        }
        else
        {
            MicroAPI::Mul<float>(product, left, right, valid);
            MicroAPI::Add<float>(dot, dot, product, valid);
        }
        if constexpr (selectCandidate)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                other, candidate + (row * order + k) * pitch, 1u, valid);
            if constexpr (fusedResidual)
            {
                MicroAPI::MulAddDst<float>(candidateDot, other, right, valid);
            }
            else
            {
                MicroAPI::Mul<float>(product, other, right, valid);
                MicroAPI::Add<float>(candidateDot, candidateDot, product, valid);
            }
        }
    }
}

template <bool selectCandidate>
__simd_callee__ inline void SmallAccumulateResidualRow(
    __ubuf__ float *original, __ubuf__ float *inverse, __ubuf__ float *candidate, uint32_t entryOffset, bool diagonal,
    MicroAPI::MaskReg &valid, MicroAPI::RegTensor<float> &dot, MicroAPI::RegTensor<float> &candidateDot,
    MicroAPI::RegTensor<float> &sum, MicroAPI::RegTensor<float> &candidateSum, MicroAPI::RegTensor<float> &aSum,
    MicroAPI::RegTensor<float> &xSum, MicroAPI::RegTensor<float> &cSum)
{
    MicroAPI::RegTensor<float> target, left, absolute;
    MicroAPI::Duplicate<float>(target, 0.0f, valid);
    if (diagonal)
    {
        MicroAPI::Duplicate<float>(target, 1.0f, valid);
    }
    MicroAPI::Sub<float>(dot, target, dot, valid);
    MicroAPI::Abs<float>(absolute, dot, valid);
    MicroAPI::Add<float>(sum, sum, absolute, valid);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, original + entryOffset, 1u, valid);
    MicroAPI::Abs<float>(absolute, left, valid);
    MicroAPI::Add<float>(aSum, aSum, absolute, valid);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, inverse + entryOffset, 1u, valid);
    MicroAPI::Abs<float>(absolute, left, valid);
    MicroAPI::Add<float>(xSum, xSum, absolute, valid);
    if constexpr (selectCandidate)
    {
        MicroAPI::Sub<float>(candidateDot, target, candidateDot, valid);
        MicroAPI::Abs<float>(absolute, candidateDot, valid);
        MicroAPI::Add<float>(candidateSum, candidateSum, absolute, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, candidate + entryOffset, 1u, valid);
        MicroAPI::Abs<float>(absolute, left, valid);
        MicroAPI::Add<float>(cSum, cSum, absolute, valid);
    }
}

template <bool selectCandidate, bool fusedResidual>
__simd_callee__ inline void SmallResidualColumn(__ubuf__ float *original, __ubuf__ float *inverse,
                                                __ubuf__ float *candidate, uint16_t order, uint16_t pitch,
                                                uint16_t column, MicroAPI::MaskReg &valid,
                                                MicroAPI::RegTensor<float> &score,
                                                MicroAPI::RegTensor<float> &candidateScore,
                                                MicroAPI::RegTensor<float> &aNorm, MicroAPI::RegTensor<float> &xNorm,
                                                MicroAPI::RegTensor<float> &cNorm)
{
    MicroAPI::RegTensor<float> left, right, other, product, dot, candidateDot;
    MicroAPI::RegTensor<float> sum, candidateSum, aSum, xSum, cSum;

    MicroAPI::Duplicate<float>(sum, 0.0f, valid);
    MicroAPI::Duplicate<float>(candidateSum, 0.0f, valid);
    MicroAPI::Duplicate<float>(aSum, 0.0f, valid);
    MicroAPI::Duplicate<float>(xSum, 0.0f, valid);
    MicroAPI::Duplicate<float>(cSum, 0.0f, valid);
    for (uint16_t row = 0; row < order; ++row)
    {
        MicroAPI::Duplicate<float>(dot, 0.0f, valid);
        MicroAPI::Duplicate<float>(candidateDot, 0.0f, valid);
        SmallResidualProducts<selectCandidate, fusedResidual>(original, inverse, candidate, order, pitch, row, column,
                                                              valid, left, right, other, product, dot, candidateDot);
        SmallAccumulateResidualRow<selectCandidate>(original, inverse, candidate, (row * order + column) * pitch,
                                                    row == column, valid, dot, candidateDot, sum, candidateSum, aSum,
                                                    xSum, cSum);
    }
    MicroAPI::Max<float>(score, score, sum, valid);
    MicroAPI::Max<float>(aNorm, aNorm, aSum, valid);
    MicroAPI::Max<float>(xNorm, xNorm, xSum, valid);
    if constexpr (selectCandidate)
    {
        MicroAPI::Max<float>(candidateScore, candidateScore, candidateSum, valid);
        MicroAPI::Max<float>(cNorm, cNorm, cSum, valid);
    }
}

// The residual uses rounded FLOAT32 products and sums. Fused accumulation is
// used for the correction, which is a separate matrix multiplication.
template <bool fusedResidual = false>
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallInverseDampedVF(
    __ubuf__ float *original, __ubuf__ float *inverse, __ubuf__ float *candidate, __ubuf__ float *scratch,
    uint16_t order, uint16_t pitch, uint32_t count, float step)
{
    MicroAPI::RegTensor<float> left, right, other, product, dot, candidateDot, value;
    auto valid = MicroAPI::UpdateMask<float>(count);
    for (uint16_t row = 0; row < order; ++row)
    {
        for (uint16_t column = 0; column < order; ++column)
        {
            MicroAPI::Duplicate<float>(dot, 0.0f, valid);
            SmallResidualProducts<false, fusedResidual>(original, inverse, inverse, order, pitch, row, column, valid,
                                                        left, right, other, product, dot, candidateDot);
            MicroAPI::Duplicate<float>(value, 0.0f, valid);
            if (row == column)
            {
                MicroAPI::Duplicate<float>(value, 1.0f, valid);
            }
            MicroAPI::Sub<float>(value, value, dot, valid);
            MicroAPI::StoreAlign(scratch + column * pitch, value, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t column = 0; column < order; ++column)
        {
            SmallRefinementCorrectionDot(scratch, inverse, order, pitch, column, valid, dot);
            MicroAPI::Muls<float>(dot, dot, step, valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                value, inverse + (row * order + column) * pitch, 1u, valid);
            MicroAPI::Add<float>(value, value, dot, valid);
            MicroAPI::StoreAlign(candidate + (row * order + column) * pitch, value, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <bool selectCandidate, bool fusedResidual = false>
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallInverseSelectVF(
    __ubuf__ float *original, __ubuf__ float *inverse, __ubuf__ float *candidate, __ubuf__ float *flags,
    __ubuf__ int32_t *infos, uint16_t order, uint16_t pitch, uint32_t count)
{
    MicroAPI::RegTensor<float> left, right, target;
    MicroAPI::RegTensor<float> score, candidateScore, aNorm, xNorm, cNorm;
    MicroAPI::RegTensor<int32_t> info;
    MicroAPI::MaskReg better, nonsingular;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::Duplicate<float>(score, 0.0f, valid);
    MicroAPI::Duplicate<float>(candidateScore, 0.0f, valid);
    MicroAPI::Duplicate<float>(aNorm, 0.0f, valid);
    MicroAPI::Duplicate<float>(xNorm, 0.0f, valid);
    MicroAPI::Duplicate<float>(cNorm, 0.0f, valid);
    for (uint16_t column = 0; column < order; ++column)
    {
        SmallResidualColumn<selectCandidate, fusedResidual>(original, inverse, candidate, order, pitch, column, valid,
                                                            score, candidateScore, aNorm, xNorm, cNorm);
    }
    MicroAPI::LoadAlign<int32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(info, infos, 1u, valid);
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(nonsingular, info, 0, valid);
    if constexpr (selectCandidate)
    {
        MicroAPI::Compare<float, CMPMODE::LT>(better, candidateScore, score, valid);
        MicroAPI::MaskAnd(better, better, nonsingular, valid);
        MicroAPI::Select<float>(score, candidateScore, score, better);
        MicroAPI::Select<float>(xNorm, cNorm, xNorm, better);
        for (uint16_t entry = 0; entry < order * order; ++entry)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, inverse + entry * pitch, 1u,
                                                                                valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(right, candidate + entry * pitch, 1u,
                                                                                valid);
            MicroAPI::Select<float>(left, right, left, better);
            MicroAPI::StoreAlign(inverse + entry * pitch, left, valid);
        }
    }
    // Stop below a FLOAT32 backward-error target with a rounding margin.
    MicroAPI::Mul<float>(target, aNorm, xNorm, valid);
    MicroAPI::Muls<float>(target, target, static_cast<float>(order) * 2.2351741790771484e-8f, valid);
    MicroAPI::Compare<float, CMPMODE::GT>(better, score, target, valid);
    MicroAPI::MaskAnd(better, better, nonsingular, valid);
    MicroAPI::Duplicate<float>(left, 1.0f, valid);
    MicroAPI::Duplicate<float>(right, 0.0f, valid);
    MicroAPI::Select<float>(left, left, right, better);
    MicroAPI::StoreAlign(flags, left, valid);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// Keep the first zero diagonal in registers instead of synchronizing each
// comparison through the shared vector pipeline.
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallSingularInfoVF(__ubuf__ float *lu,
                                                                                     __ubuf__ int32_t *info,
                                                                                     uint16_t order, uint16_t pitch,
                                                                                     uint32_t count)
{
    MicroAPI::RegTensor<float> diagonal;
    MicroAPI::RegTensor<int32_t> state, firstZero;
    MicroAPI::MaskReg zero, unset, replace;
    uint16_t waves = (count + 63) / 64;
    for (uint16_t wave = 0; wave < waves; ++wave)
    {
        auto valid = MicroAPI::UpdateMask<int32_t>(count);
        uint32_t lane = static_cast<uint32_t>(wave) * 64;
        MicroAPI::Duplicate<int32_t>(state, 0, valid);
        for (uint16_t k = 0; k < order; ++k)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                diagonal, lu + (k * order + k) * pitch + lane, 1u, valid);
            MicroAPI::CompareScalar<float, CMPMODE::EQ>(zero, diagonal, 0.0f, valid);
            MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(unset, state, 0, valid);
            MicroAPI::MaskAnd(replace, zero, unset, valid);
            MicroAPI::Duplicate<int32_t>(firstZero, static_cast<int32_t>(k + 1), valid);
            MicroAPI::Select<int32_t>(state, firstZero, state, replace);
        }
        MicroAPI::StoreAlign(info + lane, state, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// A compensated FLOAT32 dot retains the product and addition roundoff when
// forming I-X*A. The correction is evaluated into a separate existing buffer
// so every element uses the same uncorrected X.
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallInverseResidualVF(__ubuf__ float *original,
                                                                                        __ubuf__ float *inverse,
                                                                                        __ubuf__ float *residual,
                                                                                        uint16_t order, uint16_t pitch,
                                                                                        uint32_t count)
{
    MicroAPI::RegTensor<float> left, right, product, productError, high, low, sum, part, error, extra;
    auto valid = MicroAPI::UpdateMask<float>(count);
    for (uint16_t row = 0; row < order; ++row)
    {
        for (uint16_t column = 0; column < order; ++column)
        {
            MicroAPI::Duplicate<float>(high, 0.0f, valid);
            MicroAPI::Duplicate<float>(low, 0.0f, valid);
            for (uint16_t k = 0; k < order; ++k)
            {
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    left, inverse + (row * order + k) * pitch, 1u, valid);
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    right, original + (k * order + column) * pitch, 1u, valid);
                MicroAPI::Mul<float>(product, left, right, valid);
                MicroAPI::Neg<float>(productError, product, valid);
                MicroAPI::MulAddDst<float>(productError, left, right, valid);
                // TwoSum(high, product), followed by its product roundoff.
                MicroAPI::Add<float>(sum, high, product, valid);
                MicroAPI::Sub<float>(part, sum, high, valid);
                MicroAPI::Sub<float>(extra, sum, part, valid);
                MicroAPI::Sub<float>(error, high, extra, valid);
                MicroAPI::Sub<float>(extra, product, part, valid);
                MicroAPI::Add<float>(error, error, extra, valid);
                MicroAPI::Add<float>(error, error, productError, valid);
                MicroAPI::Add<float>(low, low, error, valid);
                MicroAPI::Adds<float>(high, sum, 0.0f, valid);
            }
            MicroAPI::Duplicate<float>(sum, 0.0f, valid);
            if (row == column)
            {
                MicroAPI::Duplicate<float>(sum, 1.0f, valid);
            }
            MicroAPI::Sub<float>(sum, sum, high, valid);
            MicroAPI::Sub<float>(sum, sum, low, valid);
            MicroAPI::StoreAlign(residual + (row * order + column) * pitch, sum, valid);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallInverseCorrectionVF(
    __ubuf__ float *residual, __ubuf__ float *inverse, __ubuf__ float *corrected, __ubuf__ float *scratch,
    uint16_t order, uint16_t pitch, uint32_t count)
{
    MicroAPI::RegTensor<float> left, right, correction, value;
    auto valid = MicroAPI::UpdateMask<float>(count);
    for (uint16_t row = 0; row < order; ++row)
    {
        // Cache a complete residual row before overwriting it with X+R*X.
        for (uint16_t k = 0; k < order; ++k)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                left, residual + (row * order + k) * pitch, 1u, valid);
            MicroAPI::StoreAlign(scratch + k * pitch, left, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t column = 0; column < order; ++column)
        {
            SmallRefinementCorrectionDot(scratch, inverse, order, pitch, column, valid, correction);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                value, inverse + (row * order + column) * pitch, 1u, valid);
            MicroAPI::Add<float>(value, value, correction, valid);
            MicroAPI::StoreAlign(corrected + (row * order + column) * pitch, value, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// The column is a CPU loop parameter, so every VF loop limit is independent
// of VF induction variables. This avoids dependent-loop compiler failures and
// keeps only a few register vectors live for arbitrary orders up to 32.
__no_simd_vf_fusion__ __simd_vf__ inline void SmallUpperColumnVF(__ubuf__ float *lu, __ubuf__ float *x,
                                                                 __ubuf__ float *reciprocals, uint16_t order,
                                                                 uint16_t pitch, uint16_t column, uint32_t matrices)
{
    MicroAPI::RegTensor<float> coefficient, other, target, diagonal, negative;
    uint16_t waves = (matrices + 63) / 64;
    for (uint16_t wave = 0; wave < waves; ++wave)
    {
        auto valid = MicroAPI::UpdateMask<float>(matrices);
        uint32_t lane = static_cast<uint32_t>(wave) * 64;
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            diagonal, reciprocals + column * pitch + lane, 1u, valid);
        MicroAPI::Neg<float>(negative, diagonal, valid);
        // Other rows do not consume this column. Keep one complete target
        // in registers, retaining its first Mul and ascending FMA order.
        for (uint16_t row = 0; row < column; ++row)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                coefficient, lu + (row * order + column) * pitch + lane, 1u, valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                other, x + (row * order + row) * pitch + lane, 1u, valid);
            MicroAPI::Mul<float>(target, coefficient, other, valid);
            for (uint16_t k = static_cast<uint16_t>(row + 1); k < column; ++k)
            {
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    coefficient, lu + (k * order + column) * pitch + lane, 1u, valid);
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    other, x + (row * order + k) * pitch + lane, 1u, valid);
                MicroAPI::MulAddDst<float>(target, coefficient, other, valid);
            }
            MicroAPI::Mul<float>(target, target, negative, valid);
            MicroAPI::StoreAlign(x + (row * order + column) * pitch + lane, target, valid);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// A VF handles exactly one wave. Dispatch subsequent waves on the CPU so
// the lower row loops have independent invocation state for every 64 matrices.
__simd_callee__ inline void SmallLowerLastRow(__ubuf__ float *coefficients, uint32_t coefficientStride,
                                              __ubuf__ float *x, uint16_t order, uint16_t pitch, uint16_t column,
                                              MicroAPI::MaskReg &valid)
{
    MicroAPI::RegTensor<float> coefficient, other0, dot0, target0;
    uint16_t begin = column + 1;

    uint16_t row = order - 1;
    MicroAPI::Duplicate<float>(dot0, 0.0f, valid);
    for (uint16_t k = begin; k < order; ++k)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            coefficient, coefficients + k * coefficientStride, 1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(other0, x + (row * order + k) * pitch, 1u,
                                                                            valid);
        MicroAPI::MulAddDst<float>(dot0, coefficient, other0, valid);
    }
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(target0, x + (row * order + column) * pitch, 1u,
                                                                        valid);
    MicroAPI::Sub<float>(target0, target0, dot0, valid);
    MicroAPI::StoreAlign(x + (row * order + column) * pitch, target0, valid);
}

__no_simd_vf_fusion__ __simd_vf__ inline void SmallLowerColumnVF(__ubuf__ float *lu, __ubuf__ float *x, uint16_t order,
                                                                 uint16_t pitch, uint16_t column, uint32_t count)
{
    MicroAPI::RegTensor<float> coefficient, other0, other1, dot0, dot1, target0, target1;
    uint16_t begin = column + 1;
    auto valid = MicroAPI::UpdateMask<float>(count);
    uint16_t rowPairs = order / 2;
    for (uint16_t pair = 0; pair < rowPairs; ++pair)
    {
        uint16_t row = pair * 2;
        MicroAPI::Duplicate<float>(dot0, 0.0f, valid);
        MicroAPI::Duplicate<float>(dot1, 0.0f, valid);
        for (uint16_t k = begin; k < order; ++k)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                coefficient, lu + (k * order + column) * pitch, 1u, valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(other0, x + (row * order + k) * pitch,
                                                                                1u, valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                other1, x + ((row + 1) * order + k) * pitch, 1u, valid);
            MicroAPI::MulAddDst<float>(dot0, coefficient, other0, valid);
            MicroAPI::MulAddDst<float>(dot1, coefficient, other1, valid);
        }
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(target0, x + (row * order + column) * pitch,
                                                                            1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target1, x + ((row + 1) * order + column) * pitch, 1u, valid);
        MicroAPI::Sub<float>(target0, target0, dot0, valid);
        MicroAPI::Sub<float>(target1, target1, dot1, valid);
        MicroAPI::StoreAlign(x + (row * order + column) * pitch, target0, valid);
        MicroAPI::StoreAlign(x + ((row + 1) * order + column) * pitch, target1, valid);
    }
    if (order % 2 != 0)
    {
        SmallLowerLastRow(lu + column * pitch, order * pitch, x, order, pitch, column, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// All upper rows have completed before this routine starts. Cache original
// L(:,column) before any current-column target is cleared or overwritten.
__simd_callee__ inline void SmallCacheLowerColumn(__ubuf__ float *matrix, __ubuf__ float *work, uint16_t order,
                                                  uint16_t pitch, uint16_t column, MicroAPI::MaskReg &valid)
{
    MicroAPI::RegTensor<float> coefficient, zero;
    MicroAPI::Duplicate<float>(zero, 0.0f, valid);
    uint16_t begin = column + 1;
    for (uint16_t k = begin; k < order; ++k)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            coefficient, matrix + (k * order + column) * pitch, 1u, valid);
        MicroAPI::StoreAlign(work + k * pitch, coefficient, valid);
    }
    // Clearing aliases the original factor loads. Complete all such loads
    // before issuing independent stores of zero to the same column.
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
    for (uint16_t k = begin; k < order; ++k)
    {
        MicroAPI::StoreAlign(matrix + (k * order + column) * pitch, zero, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallInplaceLowerColumnVF(
    __ubuf__ float *matrix, __ubuf__ float *work, uint16_t order, uint16_t pitch, uint16_t column, uint32_t count)
{
    MicroAPI::RegTensor<float> coefficient, other0, other1, dot0, dot1, target0, target1;
    auto valid = MicroAPI::UpdateMask<float>(count);
    uint16_t begin = column + 1;
    SmallCacheLowerColumn(matrix, work, order, pitch, column, valid);
    uint16_t pairedRows = order - order % 2;
    for (uint16_t row = 0; row < pairedRows; row += 2)
    {
        MicroAPI::Duplicate<float>(dot0, 0.0f, valid);
        MicroAPI::Duplicate<float>(dot1, 0.0f, valid);
        for (uint16_t k = begin; k < order; ++k)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(coefficient, work + k * pitch, 1u,
                                                                                valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                other0, matrix + (row * order + k) * pitch, 1u, valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                other1, matrix + ((row + 1) * order + k) * pitch, 1u, valid);
            MicroAPI::MulAddDst<float>(dot0, coefficient, other0, valid);
            MicroAPI::MulAddDst<float>(dot1, coefficient, other1, valid);
        }
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target0, matrix + (row * order + column) * pitch, 1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target1, matrix + ((row + 1) * order + column) * pitch, 1u, valid);
        MicroAPI::Sub<float>(target0, target0, dot0, valid);
        MicroAPI::Sub<float>(target1, target1, dot1, valid);
        MicroAPI::StoreAlign(matrix + (row * order + column) * pitch, target0, valid);
        MicroAPI::StoreAlign(matrix + ((row + 1) * order + column) * pitch, target1, valid);
    }
    if (order % 2 != 0)
    {
        SmallLowerLastRow(work, pitch, matrix, order, pitch, column, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__aicore__ inline void SmallLapackColumns(LocalTensor<float> factors, LocalTensor<float> inverse,
                                          LocalTensor<float> reciprocals, int order, int matrices)
{
    auto lu = reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr());
    auto x = reinterpret_cast<__ubuf__ float *>(inverse.GetPhyAddr());
    auto r = reinterpret_cast<__ubuf__ float *>(reciprocals.GetPhyAddr());
    for (uint16_t column = 1; column < order; ++column)
    {
        SmallUpperColumnVF(lu, x, r, order, matrices, column, matrices);
    }
    for (int column = order - 2; column >= 0; --column)
    {
        for (int lane = 0; lane < matrices; lane += 64)
        {
            uint32_t count = matrices - lane < 64 ? matrices - lane : 64;
            SmallLowerColumnVF(lu + lane, x + lane, order, matrices, column, count);
        }
    }
}
}  // namespace RealLuKernel
