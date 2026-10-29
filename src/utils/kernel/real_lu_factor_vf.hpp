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
static constexpr MicroAPI::DivSpecificMode SMALL_FACTOR_DIVISION = {MicroAPI::MaskMergeMode::ZEROING, false,
                                                                    DivAlgo::PRECISION_0ULP_FTZ_FALSE};
// Public 0ULP normal-range algorithm. Call only after the current pivot
// column has passed the conservative, per-value finite nonzero range gate.
static constexpr MicroAPI::DivSpecificMode SMALL_FACTOR_BOUNDED_DIVISION = {MicroAPI::MaskMergeMode::ZEROING, false,
                                                                            DivAlgo::PRECISION_0ULP_FTZ_TRUE};
static constexpr float SMALL_DIVISION_LOWER_BOUND = 2.3283064365386963e-10f;  // 2^-32
static constexpr float SMALL_DIVISION_UPPER_BOUND = 4294967296.0f;            // 2^32

static constexpr MicroAPI::CastTrait SMALL_FACTOR_PIVOT_CAST = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

// Each lane owns one matrix. k is a scalar caller parameter, so all VF loop
// limits are independent of VF induction variables.
__simd_callee__ inline void SmallInitializePivotLabels(MicroAPI::RegTensor<float> &pivot,
                                                       MicroAPI::RegTensor<float> &label, uint16_t k,
                                                       MicroAPI::MaskReg &valid)
{
    MicroAPI::Duplicate<float>(pivot, static_cast<float>(k + 1), valid);
    MicroAPI::Duplicate<float>(label, static_cast<float>(k + 1), valid);
}

__simd_callee__ inline void SmallReplacePivot(MicroAPI::RegTensor<float> &best, MicroAPI::RegTensor<float> &current,
                                              MicroAPI::RegTensor<float> &pivot, MicroAPI::RegTensor<float> &label,
                                              MicroAPI::MaskReg &valid)
{
    MicroAPI::MaskReg replace;
    MicroAPI::Adds<float>(label, label, 1.0f, valid);
    MicroAPI::Compare<float, CMPMODE::GT>(replace, current, best, valid);
    MicroAPI::Select<float>(best, current, best, replace);
    MicroAPI::Select<float>(pivot, label, pivot, replace);
}

__simd_callee__ inline void SmallChoosePivotCore(__ubuf__ float *lu, __ubuf__ float *pivots, uint16_t n, uint16_t pitch,
                                                 uint16_t k, MicroAPI::MaskReg &valid, MicroAPI::RegTensor<float> &best)
{
    MicroAPI::RegTensor<float> value, current, pivot, label;
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, lu + (k * n + k) * pitch, 1u, valid);
    MicroAPI::Abs<float>(best, value, valid);
    SmallInitializePivotLabels(pivot, label, k, valid);
    for (uint16_t row = static_cast<uint16_t>(k + 1); row < n; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, lu + (row * n + k) * pitch, 1u,
                                                                            valid);
        MicroAPI::Abs<float>(current, value, valid);
        SmallReplacePivot(best, current, pivot, label, valid);
    }
    MicroAPI::StoreAlign(pivots + k * pitch, pivot, valid);
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallChoosePivotVF(__ubuf__ float *lu,
                                                                                    __ubuf__ float *pivots, uint16_t n,
                                                                                    uint16_t pitch, uint16_t k,
                                                                                    uint32_t count)
{
    MicroAPI::RegTensor<float> best;
    auto valid = MicroAPI::UpdateMask<float>(count);
    SmallChoosePivotCore(lu, pivots, n, pitch, k, valid, best);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// Left-looking n3..8 gate; pivot search and tie rules are unchanged.
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallChoosePivotNormalVF(__ubuf__ float *lu,
                                                                                          __ubuf__ float *pivots,
                                                                                          __ubuf__ float *normalFlags,
                                                                                          uint16_t n, uint16_t pitch,
                                                                                          uint16_t k, uint32_t count)
{
    MicroAPI::RegTensor<float> value, best, current;
    MicroAPI::MaskReg normal;
    auto valid = MicroAPI::UpdateMask<float>(count);
    SmallChoosePivotCore(lu, pivots, n, pitch, k, valid, best);
    // This is the original scaling predicate on the selected pivot. Inf is
    // normal under that predicate; NaN, tiny values and signed zero are not.
    MicroAPI::CompareScalar<float, CMPMODE::GE>(normal, best, 1.1754943508222875e-38f, valid);
    MicroAPI::Duplicate<float>(value, 1.0f, valid);
    MicroAPI::Duplicate<float>(current, 0.0f, valid);
    MicroAPI::Select<float>(value, value, current, normal);
    MicroAPI::StoreAlign(normalFlags, value, valid);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// Retain the original pivot order and strict first-maximum tie behavior.
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallChoosePivotRangeVF(__ubuf__ float *lu,
                                                                                         __ubuf__ float *pivots,
                                                                                         __ubuf__ float *safeFlags,
                                                                                         uint16_t n, uint16_t pitch,
                                                                                         uint16_t k, uint32_t count)
{
    MicroAPI::RegTensor<float> value, best, current, pivot, label;
    MicroAPI::MaskReg safe, bounded;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, lu + (k * n + k) * pitch, 1u, valid);
    MicroAPI::Abs<float>(best, value, valid);
    MicroAPI::CompareScalar<float, CMPMODE::GE>(safe, best, SMALL_DIVISION_LOWER_BOUND, valid);
    MicroAPI::CompareScalar<float, CMPMODE::LE>(bounded, best, SMALL_DIVISION_UPPER_BOUND, valid);
    MicroAPI::MaskAnd(safe, safe, bounded, valid);
    SmallInitializePivotLabels(pivot, label, k, valid);
    for (uint16_t row = static_cast<uint16_t>(k + 1); row < n; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, lu + (row * n + k) * pitch, 1u,
                                                                            valid);
        MicroAPI::Abs<float>(current, value, valid);
        MicroAPI::CompareScalar<float, CMPMODE::GE>(bounded, current, SMALL_DIVISION_LOWER_BOUND, valid);
        MicroAPI::MaskAnd(safe, safe, bounded, valid);
        MicroAPI::CompareScalar<float, CMPMODE::LE>(bounded, current, SMALL_DIVISION_UPPER_BOUND, valid);
        MicroAPI::MaskAnd(safe, safe, bounded, valid);
        SmallReplacePivot(best, current, pivot, label, valid);
    }
    MicroAPI::StoreAlign(pivots + k * pitch, pivot, valid);
    // Zero, subnormal, NaN, infinity, and finite out-of-range operands all
    // reject the entire group. Even inactive identity padding may reject a
    // tail group; correctness takes precedence over this optional fast path.
    MicroAPI::Duplicate<float>(value, 1.0f, valid);
    MicroAPI::Duplicate<float>(label, 0.0f, valid);
    MicroAPI::Select<float>(value, value, label, safe);
    MicroAPI::StoreAlign(safeFlags, value, valid);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallSwapPivotVF(__ubuf__ float *lu,
                                                                                  __ubuf__ float *pivots, uint16_t n,
                                                                                  uint16_t pitch, uint16_t k,
                                                                                  uint32_t count, uint16_t columns)
{
    MicroAPI::RegTensor<float> pivot, first, second;
    MicroAPI::RegTensor<int32_t> pivotRow;
    MicroAPI::RegTensor<uint32_t> lane, index;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(pivot, pivots + k * pitch, 1u, valid);
    MicroAPI::Cast<int32_t, float, SMALL_FACTOR_PIVOT_CAST>(pivotRow, pivot, valid);
    MicroAPI::Adds(pivotRow, pivotRow, -1, valid);
    MicroAPI::Muls(index, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(pivotRow), static_cast<uint32_t>(n) * pitch,
                   valid);
    MicroAPI::Add(index, index, lane, valid);
    for (uint16_t col = 0; col < columns; ++col)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(first, lu + (k * n + col) * pitch, 1u,
                                                                            valid);
        MicroAPI::Gather(second, lu + col * pitch, index, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
        MicroAPI::Scatter(lu + col * pitch, first, index, valid);
        MicroAPI::StoreAlign(lu + (k * n + col) * pitch, second, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

// Left-looking partial pivoting finalizes one column at a time. All preceding
// products are accumulated before the rounded subtraction from that column.
// Use fused accumulation for the upper triangular solve. Keep the current
// pivot column's Schur update and its zero-pivot arithmetic unchanged.
__simd_callee__ inline void SmallApplyColumnPivots(__ubuf__ float *lu, __ubuf__ float *pivots, uint16_t n,
                                                   uint16_t pitch, uint16_t column, MicroAPI::MaskReg &valid)
{
    MicroAPI::RegTensor<float> pivot, first, second;
    MicroAPI::RegTensor<int32_t> pivotRow;
    MicroAPI::RegTensor<uint32_t> lane, index;
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    for (uint16_t k = 0; k < column; ++k)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(pivot, pivots + k * pitch, 1u, valid);
        MicroAPI::Cast<int32_t, float, SMALL_FACTOR_PIVOT_CAST>(pivotRow, pivot, valid);
        MicroAPI::Adds(pivotRow, pivotRow, -1, valid);
        MicroAPI::Muls(index, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(pivotRow),
                       static_cast<uint32_t>(n) * pitch, valid);
        MicroAPI::Add(index, index, lane, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(first, lu + (k * n + column) * pitch, 1u,
                                                                            valid);
        MicroAPI::Gather(second, lu + column * pitch, index, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
        MicroAPI::Scatter(lu + column * pitch, first, index, valid);
        MicroAPI::StoreAlign(lu + (k * n + column) * pitch, second, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallLeftColumnVF(__ubuf__ float *lu,
                                                                                   __ubuf__ float *pivots, uint16_t n,
                                                                                   uint16_t pitch, uint16_t column,
                                                                                   uint32_t count)
{
    MicroAPI::RegTensor<float> first, left, right, product, dot;
    auto valid = MicroAPI::UpdateMask<float>(count);
    SmallApplyColumnPivots(lu, pivots, n, pitch, column, valid);
    for (uint16_t row = 1; row < n; ++row)
    {
        MicroAPI::Duplicate<float>(dot, 0.0f, valid);
        for (uint16_t k = 0; k < column; ++k)
        {
            if (k < row)
            {
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, lu + (row * n + k) * pitch,
                                                                                    1u, valid);
                MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    right, lu + (k * n + column) * pitch, 1u, valid);
                if (row < column)
                {
                    MicroAPI::MulAddDst<float>(dot, left, right, valid);
                }
                else
                {
                    MicroAPI::Mul<float>(product, left, right, valid);
                    MicroAPI::Add<float>(dot, dot, product, valid);
                }
            }
        }
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(first, lu + (row * n + column) * pitch, 1u,
                                                                            valid);
        MicroAPI::Sub<float>(first, first, dot, valid);
        MicroAPI::StoreAlign(lu + (row * n + column) * pitch, first, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallScaleColumnVF(__ubuf__ float *lu, uint16_t n,
                                                                                    uint16_t pitch, uint16_t column,
                                                                                    uint32_t count)
{
    MicroAPI::RegTensor<float> diagonal, reciprocal, numerator, quotient, product, absolute;
    MicroAPI::MaskReg normal, nonzero;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(diagonal, lu + (column * n + column) * pitch,
                                                                        1u, valid);
    MicroAPI::Duplicate<float>(reciprocal, 1.0f, valid);
    MicroAPI::Div<float, &SMALL_FACTOR_DIVISION>(reciprocal, reciprocal, diagonal, valid);
    MicroAPI::Abs<float>(absolute, diagonal, valid);
    MicroAPI::CompareScalar<float, CMPMODE::GE>(normal, absolute, 1.1754943508222875e-38f, valid);
    MicroAPI::CompareScalar<float, CMPMODE::NE>(nonzero, diagonal, 0.0f, valid);
    for (uint16_t row = static_cast<uint16_t>(column + 1); row < n; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(numerator, lu + (row * n + column) * pitch,
                                                                            1u, valid);
        MicroAPI::Div<float, &SMALL_FACTOR_DIVISION>(quotient, numerator, diagonal, valid);
        MicroAPI::Mul<float>(product, numerator, reciprocal, valid);
        MicroAPI::Select<float>(quotient, product, quotient, normal);
        MicroAPI::Select<float>(quotient, quotient, numerator, nonzero);
        MicroAPI::StoreAlign(lu + (row * n + column) * pitch, quotient, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// The caller proved every selected pivot uses the original normal mask.
// Each discarded row division is removed; the selected reciprocal and Mul
// remain exactly the old IEEE-0ULP/no-FTZ operations, including Inf and -0.
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallScaleNormalGroupVF(__ubuf__ float *lu, uint16_t n,
                                                                                         uint16_t pitch,
                                                                                         uint16_t column,
                                                                                         uint32_t count)
{
    MicroAPI::RegTensor<float> diagonal, reciprocal, numerator, product;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(diagonal, lu + (column * n + column) * pitch,
                                                                        1u, valid);
    MicroAPI::Duplicate<float>(reciprocal, 1.0f, valid);
    MicroAPI::Div<float, &SMALL_FACTOR_DIVISION>(reciprocal, reciprocal, diagonal, valid);
    for (uint16_t row = static_cast<uint16_t>(column + 1); row < n; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(numerator, lu + (row * n + column) * pitch,
                                                                            1u, valid);
        MicroAPI::Mul<float>(product, numerator, reciprocal, valid);
        MicroAPI::StoreAlign(lu + (row * n + column) * pitch, product, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <bool bounded>
__simd_callee__ inline void SmallEliminatePivotCore(__ubuf__ float *lu, uint16_t n, uint16_t pitch, uint16_t k,
                                                    uint32_t count)
{
    MicroAPI::RegTensor<float> diagonal, numerator, coefficient, upper, target, product, reciprocal, absolute;
    MicroAPI::MaskReg normalPivot;
    auto valid = MicroAPI::UpdateMask<float>(count);
    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(diagonal, lu + (k * n + k) * pitch, 1u, valid);
    for (uint16_t row = static_cast<uint16_t>(k + 1); row < n; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(numerator, lu + (row * n + k) * pitch, 1u,
                                                                            valid);
        if constexpr (bounded)
        {
            MicroAPI::Div<float, &SMALL_FACTOR_BOUNDED_DIVISION>(coefficient, numerator, diagonal, valid);
        }
        else
        {
            MicroAPI::Div<float, &SMALL_FACTOR_DIVISION>(coefficient, numerator, diagonal, valid);
        }
        if (n == 2)
        {
            // LAPACK SGETRF2 scales a normal pivot column by its rounded
            // reciprocal. Its SFMIN fallback divides tiny pivots directly.
            MicroAPI::Duplicate<float>(reciprocal, 1.0f, valid);
            MicroAPI::Div<float, &SMALL_FACTOR_DIVISION>(reciprocal, reciprocal, diagonal, valid);
            MicroAPI::Mul<float>(product, numerator, reciprocal, valid);
            MicroAPI::Abs<float>(absolute, diagonal, valid);
            MicroAPI::CompareScalar<float, CMPMODE::GE>(normalPivot, absolute, 1.1754943508222875e-38f, valid);
            MicroAPI::Select<float>(coefficient, product, coefficient, normalPivot);
        }
        MicroAPI::StoreAlign(lu + (row * n + k) * pitch, coefficient, valid);
        for (uint16_t col = static_cast<uint16_t>(k + 1); col < n; ++col)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(upper, lu + (k * n + col) * pitch, 1u,
                                                                                valid);
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(target, lu + (row * n + col) * pitch,
                                                                                1u, valid);
            MicroAPI::Mul<float>(product, coefficient, upper, valid);
            MicroAPI::Sub<float>(target, target, product, valid);
            MicroAPI::StoreAlign(lu + (row * n + col) * pitch, target, valid);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallEliminatePivotVF(__ubuf__ float *lu, uint16_t n,
                                                                                       uint16_t pitch, uint16_t k,
                                                                                       uint32_t count)
{
    SmallEliminatePivotCore<false>(lu, n, pitch, k, count);
}

// n9..32 only. The caller selects one specialization for the complete group;
// the false specialization keeps the original IEEE division and arithmetic.
template <bool bounded>
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void SmallEliminatePivotRangeVF(__ubuf__ float *lu,
                                                                                            uint16_t n, uint16_t pitch,
                                                                                            uint16_t k, uint32_t count)
{
    SmallEliminatePivotCore<bounded>(lu, n, pitch, k, count);
}

__simd_callee__ inline void SmallRefinementCorrectionDot(__ubuf__ float *scratch, __ubuf__ float *inverse,
                                                         uint16_t order, uint16_t pitch, uint16_t column,
                                                         MicroAPI::MaskReg &valid, MicroAPI::RegTensor<float> &dot)
{
    MicroAPI::RegTensor<float> left, right;
    MicroAPI::Duplicate<float>(dot, 0.0f, valid);
    for (uint16_t k = 0; k < order; ++k)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(left, scratch + k * pitch, 1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            right, inverse + (k * order + column) * pitch, 1u, valid);
        MicroAPI::MulAddDst<float>(dot, left, right, valid);
    }
}

// MicroAPI Gather/Scatter offsets are elements. Generate one register index
// per wave instead of retaining two full UB transpose-index tables.
__simd_callee__ inline void SmallPackedLaneIndices(MicroAPI::RegTensor<uint32_t> &index, uint16_t stride)
{
    auto full = MicroAPI::CreateMask<uint32_t>();
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(index), 0);
    MicroAPI::Muls(index, index, static_cast<uint32_t>(stride), full);
}

template <bool identity>
__simd_callee__ inline void SmallStorePackedCell(__ubuf__ float *packed, __ubuf__ float *soa,
                                                 __ubuf__ uint32_t *permutation, uint16_t n, uint16_t pitch,
                                                 uint16_t packedStride, uint16_t columnStride, uint32_t first,
                                                 uint16_t row, uint16_t col, MicroAPI::RegTensor<uint32_t> &base,
                                                 MicroAPI::MaskReg &valid)
{
    MicroAPI::RegTensor<uint32_t> mapped, index;
    MicroAPI::RegTensor<float> value;

    MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(value, soa + (row * n + col) * pitch + first,
                                                                        1u, valid);
    if constexpr (identity)
    {
        MicroAPI::Scatter(packed + first * packedStride + col * columnStride + row, value, base, valid);
    }
    else
    {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            mapped, permutation + col * pitch + first, 1u, valid);
        MicroAPI::Muls(mapped, mapped, static_cast<uint32_t>(columnStride), valid);
        MicroAPI::Adds(mapped, mapped, static_cast<uint32_t>(row), valid);
        MicroAPI::Add(index, base, mapped, valid);
        MicroAPI::Scatter(packed + first * packedStride, value, index, valid);
    }
}

template <bool toPacked, bool identity>
__simd_callee__ inline void SmallConvertPackedLayout(__ubuf__ float *packed, __ubuf__ float *soa,
                                                     __ubuf__ uint32_t *permutation, uint16_t n, uint16_t pitch,
                                                     uint16_t packedStride, uint16_t columnStride, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> base;
    MicroAPI::RegTensor<float> value;
    SmallPackedLaneIndices(base, packedStride);
    uint16_t waves = (count + 63) / 64;
    for (uint16_t wave = 0; wave < waves; ++wave)
    {
        auto valid = MicroAPI::UpdateMask<float>(count);
        uint32_t first = static_cast<uint32_t>(wave) * 64;
        for (uint16_t row = 0; row < n; ++row)
        {
            for (uint16_t col = 0; col < n; ++col)
            {
                if constexpr (toPacked)
                {
                    SmallStorePackedCell<identity>(packed, soa, permutation, n, pitch, packedStride, columnStride,
                                                   first, row, col, base, valid);
                }
                else
                {
                    MicroAPI::Gather(value, packed + first * packedStride + col * columnStride + row, base, valid);
                    MicroAPI::StoreAlign(soa + (row * n + col) * pitch + first, value, valid);
                }
            }
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__no_simd_vf_fusion__ __simd_vf__ inline void SmallPackedToSoAVF(__ubuf__ float *soa, __ubuf__ float *packed,
                                                                 uint16_t n, uint16_t pitch, uint16_t packedStride,
                                                                 uint16_t columnStride, uint32_t count)
{
    SmallConvertPackedLayout<false, true>(packed, soa, nullptr, n, pitch, packedStride, columnStride, count);
}

__no_simd_vf_fusion__ __simd_vf__ inline void SmallPackedPivotToSoAVF(__ubuf__ int32_t *soa, __ubuf__ int32_t *packed,
                                                                      uint16_t n, uint16_t pitch, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> index;
    MicroAPI::RegTensor<int32_t> value;
    SmallPackedLaneIndices(index, n);
    uint16_t waves = (count + 63) / 64;
    for (uint16_t wave = 0; wave < waves; ++wave)
    {
        auto valid = MicroAPI::UpdateMask<int32_t>(count);
        uint32_t first = static_cast<uint32_t>(wave) * 64;
        for (uint16_t col = 0; col < n; ++col)
        {
            MicroAPI::Gather(value, packed + first * n + col, index, valid);
            MicroAPI::StoreAlign(soa + col * pitch + first, value, valid);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

// Forward swaps produce the source-column -> final-destination map. This is
// the inverse of the reverse column swaps at the end of LAPACK SGETRI.
__no_simd_vf_fusion__ __simd_vf__ inline void SmallColumnPermutationVF(__ubuf__ uint32_t *permutation,
                                                                       __ubuf__ float *pivots, uint16_t n,
                                                                       uint16_t pitch, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> label, first, second, index, lanes;
    MicroAPI::RegTensor<int32_t> pivotInteger;
    MicroAPI::RegTensor<float> pivot;
    static constexpr MicroAPI::CastTrait cast = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lanes), 0);
    uint16_t waves = (count + 63) / 64;
    for (uint16_t wave = 0; wave < waves; ++wave)
    {
        auto valid = MicroAPI::UpdateMask<uint32_t>(count);
        uint32_t begin = static_cast<uint32_t>(wave) * 64;
        for (uint16_t col = 0; col < n; ++col)
        {
            MicroAPI::Duplicate(label, static_cast<uint32_t>(col), valid);
            MicroAPI::StoreAlign(permutation + col * pitch + begin, label, valid);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t k = 0; k < n; ++k)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(pivot, pivots + k * pitch + begin, 1u,
                                                                                valid);
            MicroAPI::Cast<int32_t, float, cast>(pivotInteger, pivot, valid);
            MicroAPI::Adds(pivotInteger, pivotInteger, -1, valid);
            MicroAPI::Muls(index, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(pivotInteger),
                           static_cast<uint32_t>(pitch), valid);
            MicroAPI::Add(index, index, lanes, valid);
            MicroAPI::LoadAlign<uint32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                first, permutation + k * pitch + begin, 1u, valid);
            MicroAPI::Gather(second, permutation + begin, index, valid);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::Scatter(permutation + begin, first, index, valid);
            MicroAPI::StoreAlign(permutation + k * pitch + begin, second, valid);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
    }
}

template <bool identity>
__no_simd_vf_fusion__ __simd_vf__ inline void SmallSoAToPackedVF(__ubuf__ float *packed, __ubuf__ float *soa,
                                                                 __ubuf__ uint32_t *permutation, uint16_t n,
                                                                 uint16_t pitch, uint16_t packedStride,
                                                                 uint16_t columnStride, uint32_t count)
{
    SmallConvertPackedLayout<true, identity>(packed, soa, permutation, n, pitch, packedStride, columnStride, count);
}

// Unused lanes are independent identity matrices, so they cannot force an
// otherwise normal reciprocal group onto the denormal/zero division path.
__no_simd_vf_fusion__ __simd_vf__ inline void SmallCompleteDiagonalVF(__ubuf__ float *lu, uint16_t n, uint16_t pitch,
                                                                      uint32_t count)
{
    MicroAPI::RegTensor<float> diagonal, one;
    auto full = MicroAPI::CreateMask<float>();
    MicroAPI::Duplicate<float>(one, 1.0f, full);
    uint32_t pitchRemaining = pitch;
    uint16_t waves = (pitch + 63) / 64;
    for (uint16_t wave = 0; wave < waves; ++wave)
    {
        auto stored = MicroAPI::UpdateMask<float>(pitchRemaining);
        auto active = MicroAPI::UpdateMask<float>(count);
        uint32_t first = static_cast<uint32_t>(wave) * 64;
        for (uint16_t k = 0; k < n; ++k)
        {
            MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                diagonal, lu + (k * n + k) * pitch + first, 1u, stored);
            MicroAPI::Select<float>(diagonal, diagonal, one, active);
            MicroAPI::StoreAlign(lu + (k * n + k) * pitch + first, diagonal, stored);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}
}  // namespace RealLuKernel
