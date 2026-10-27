/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cholesky_device.h"

namespace Cholesky
{
namespace
{
__simt_callee__ inline float RefineScalarRoot(float pivot, float& inverse)
{
    float root = sqrtf(pivot);
    root += RootRemainder(pivot, root);
    if (root == 0)
    {
        inverse = INFINITY;
        return root;
    }
    inverse = 1.0f / root;
    if (root > 0 && root < FINITE_ROOT_BOUND)
    {
        inverse = fmaf(inverse, fmaf(-root, inverse, 1.0f), inverse);
    }
    return root;
}

// A warp factors eight columns at a time, keeping the working panel in registers.
template <bool COMPENSATED, bool UPPER>
__simt_callee__ inline void NormalizeSmallColumn(__gm__ aclsolverFloatComplex* a,
                                                 __gm__ aclsolverFloatComplex* firstColumn, int lda, int offset, int m,
                                                 int firstStride, int lane, int n, int c, int k, float p, float d,
                                                 float inverse, float (&re)[FACTOR_STRIP], float (&im)[FACTOR_STRIP],
                                                 aclsolverFloatComplex& firstLow, FloatPair& diagonalState)
{
    if (lane >= c && lane < n)
    {
        aclsolverFloatComplex value{lane == c ? d : re[k] * inverse, lane == c ? 0 : im[k] * inverse};
        if constexpr (COMPENSATED)
        {
            if (!c)
            {
                float rootLow = RootRemainder(p, d);
                firstLow = {rootLow, 0};
                if (lane)
                {
                    value = NormalizeFirstColumn({re[k], im[k]}, d, rootLow, firstLow);
                }
                WriteLower(firstColumn + static_cast<int64_t>(m) * firstStride, firstStride, lane, 0, false, firstLow);
            }
        }
        re[k] = value.real;
        im[k] = value.imag;
        if (lane > c)
        {
            UpdateScalarDiagonal(diagonalState, {re[k], im[k]}, firstLow, COMPENSATED && c == 0);
        }
        WriteLower(a, lda, offset + lane, offset + c, UPPER, {re[k], im[k]});
    }
}

template <bool COMPENSATED, bool UPPER>
__simt_callee__ inline void UpdateSmallTrailing(__gm__ aclsolverFloatComplex* a, int lda, int offset, int lane, int n,
                                                int first, int shuffleBase, float (&re)[FACTOR_STRIP],
                                                float (&im)[FACTOR_STRIP], aclsolverFloatComplex firstLow)
{
    for (int c = first + FACTOR_STRIP; c < n; ++c)
    {
        bool selected = lane >= c && lane < n;
        auto z = selected ? ReadLower(a, lda, offset + lane, offset + c, UPPER) : aclsolverFloatComplex{0, 0};
#pragma unroll
        for (int k = 0; k < FACTOR_STRIP; ++k)
        {
            float cr = asc_shfl(re[k], shuffleBase + c), ci = asc_shfl(im[k], shuffleBase + c);
            if (COMPENSATED && !first && !k)
            {
                aclsolverFloatComplex low{asc_shfl(firstLow.real, shuffleBase + c),
                                          asc_shfl(firstLow.imag, shuffleBase + c)};
                if (selected)
                {
                    z = SubFirstColumn(z, {re[k], im[k]}, {cr, ci}, firstLow, low);
                }
                continue;
            }
            if (selected)
            {
                z = SubProduct(z, {re[k], im[k]}, {cr, -ci});
            }
        }
        if (selected)
        {
            WriteLower(a, lda, offset + lane, offset + c, UPPER, z);
        }
    }
}

template <bool COMPENSATED>
__simt_callee__ inline void UpdateSmallStrip(int k, int c, int n, int lane, int first, int shuffleBase,
                                             float (&re)[FACTOR_STRIP], float (&im)[FACTOR_STRIP],
                                             aclsolverFloatComplex firstLow)
{
    float lr = re[k], li = im[k];
#pragma unroll
    for (int j = k + 1; j < FACTOR_STRIP; ++j)
    {
        float jr = asc_shfl(lr, shuffleBase + first + j), ji = asc_shfl(li, shuffleBase + first + j);
        aclsolverFloatComplex low{0, 0};
        if constexpr (COMPENSATED)
        {
            if (!c)
            {
                low = {asc_shfl(firstLow.real, shuffleBase + j), asc_shfl(firstLow.imag, shuffleBase + j)};
            }
        }
        if (lane >= first + j && lane < n)
        {
            auto z = COMPENSATED && !c ? SubFirstColumn({re[j], im[j]}, {lr, li}, {jr, ji}, firstLow, low)
                                       : SubProduct({re[j], im[j]}, {lr, li}, {jr, -ji});
            re[j] = z.real;
            im[j] = z.imag;
        }
    }
}

template <bool UPPER>
__simt_callee__ inline void LoadSmallStrip(__gm__ aclsolverFloatComplex* a, int lda, int offset, int lane, int n,
                                           int first, float (&re)[FACTOR_STRIP], float (&im)[FACTOR_STRIP])
{
#pragma unroll
    for (int j = 0; j < FACTOR_STRIP; ++j)
    {
        int c = first + j;
        auto z =
            lane < n && c <= lane ? ReadLower(a, lda, offset + lane, offset + c, UPPER) : aclsolverFloatComplex{0, 0};
        re[j] = z.real;
        im[j] = z.imag;
    }
}

template <bool COMPENSATED, bool UPPER>
__simt_callee__ inline int FactorSmallPivot(__gm__ aclsolverFloatComplex* a, __gm__ aclsolverFloatComplex* firstColumn,
                                            int lda, int offset, int m, int firstStride, int lane, int n, int c, int k,
                                            int first, int shuffleBase, float (&re)[FACTOR_STRIP],
                                            float (&im)[FACTOR_STRIP], aclsolverFloatComplex& firstLow,
                                            FloatPair& diagonalState)
{
    float original = asc_shfl(re[k], shuffleBase + c);
    float corrected = asc_shfl(diagonalState.high + diagonalState.low, shuffleBase + c);
    float p = fabsf(original) < FINITE_ROOT_BOUND ? corrected : original;
    if (p <= 0)
    {
        return c + 1;
    }
    float inverse = 0;
    float d = RefineScalarRoot(p, inverse);
    NormalizeSmallColumn<COMPENSATED, UPPER>(a, firstColumn, lda, offset, m, firstStride, lane, n, c, k, p, d, inverse,
                                             re, im, firstLow, diagonalState);
    UpdateSmallStrip<COMPENSATED>(k, c, n, lane, first, shuffleBase, re, im, firstLow);
    return 0;
}

template <int VF_THREADS, int LANES, bool UPPER, bool COMPENSATED = false>
__simt_vf__ __aicore__ __launch_bounds__(VF_THREADS) void FactorSmallVF(__gm__ aclsolverFloatComplex* single,
                                                                        __gm__ const uint64_t* array, int n, int lda,
                                                                        __gm__ int* info, int batch, int offset,
                                                                        __gm__ aclsolverFloatComplex* firstColumn,
                                                                        int firstStride)
{
    int lane = threadIdx.x % LANES, group = threadIdx.x / LANES;
    int shuffleBase = (threadIdx.x % WARP_SIZE) / LANES * LANES;
    int groups = VF_THREADS / LANES;
    for (int m = blockIdx.x * groups + group; m < batch; m += gridDim.x * groups)
    {
        if (offset && info[m] != 0)
        {
            continue;
        }
        auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        FloatPair diagonalState{lane < n ? ReadLower(a, lda, offset + lane, offset + lane, UPPER).real : 0, 0};
        int failure = 0;
        aclsolverFloatComplex firstLow{0, 0};
        for (int first = 0; first < n; first += FACTOR_STRIP)
        {
            float re[FACTOR_STRIP], im[FACTOR_STRIP];
            LoadSmallStrip<UPPER>(a, lda, offset, lane, n, first, re, im);
#pragma unroll
            for (int k = 0; k < FACTOR_STRIP; ++k)
            {
                int c = first + k;
                if (c >= n || failure)
                {
                    continue;
                }
                failure = FactorSmallPivot<COMPENSATED, UPPER>(a, firstColumn, lda, offset, m, firstStride, lane, n, c,
                                                               k, first, shuffleBase, re, im, firstLow, diagonalState);
            }
            if (failure)
            {
                break;
            }
            UpdateSmallTrailing<COMPENSATED, UPPER>(a, lda, offset, lane, n, first, shuffleBase, re, im, firstLow);
        }
        if (lane == 0)
        {
            info[m] = failure ? offset + failure : 0;
        }
    }
}

template <typename WholeMask, typename WholeIndex>
__simd_callee__ inline void WholeTrailingMask(uint32_t column, WholeMask& components, WholeMask& active,
                                              WholeMask& selected, WholeIndex& index)
{
    uint32_t count = column;
    auto prefix = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::Not(selected, prefix, components);
    AscendC::Reg::And(selected, selected, active, components);
    AscendC::Reg::Duplicate(index, column);
}

template <typename R, typename M>
__simd_callee__ inline void CompensateWholeColumn(R& real, R& imag, R& lowReal, R& lowImag, R& pivot, R& diagonal,
                                                  R& zero, M& selected, M& pivotMask, M& components)
{
    R sourceReal, sourceImag, rootLow, negativeRoot, twiceRoot, negativeLow;
    AscendC::Reg::Move(sourceReal, real);
    AscendC::Reg::Move(sourceImag, imag);
    AscendC::Reg::Div(real, real, diagonal, selected);
    AscendC::Reg::Div(imag, imag, diagonal, selected);
    AscendC::Reg::Select(real, diagonal, real, pivotMask);
    AscendC::Reg::Move(rootLow, pivot);
    AscendC::Reg::Neg(negativeRoot, diagonal, components);
    AscendC::Reg::MulAddDst(rootLow, negativeRoot, diagonal, components);
    AscendC::Reg::Add(twiceRoot, diagonal, diagonal, components);
    AscendC::Reg::Div(rootLow, rootLow, twiceRoot, components);
    // Nonfinite roots have no rounding remainder.
    AscendC::Reg::MaskReg finiteRoot;
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(finiteRoot, diagonal, FINITE_ROOT_BOUND, components);
    AscendC::Reg::Select(rootLow, rootLow, zero, finiteRoot);
    AscendC::Reg::Neg(negativeLow, rootLow, components);
    AscendC::Reg::Move(lowReal, sourceReal);
    AscendC::Reg::Move(lowImag, sourceImag);
    AscendC::Reg::MulAddDst(lowReal, real, negativeRoot, components);
    AscendC::Reg::MulAddDst(lowImag, imag, negativeRoot, components);
    AscendC::Reg::MulAddDst(lowReal, real, negativeLow, components);
    AscendC::Reg::MulAddDst(lowImag, imag, negativeLow, components);
    AscendC::Reg::Div(lowReal, lowReal, diagonal, components);
    AscendC::Reg::Div(lowImag, lowImag, diagonal, components);
    AscendC::Reg::Select(lowReal, lowReal, zero, finiteRoot);
    AscendC::Reg::Select(lowImag, lowImag, zero, finiteRoot);
    AscendC::Reg::Select(lowReal, rootLow, lowReal, pivotMask);
    AscendC::Reg::Select(lowImag, zero, lowImag, pivotMask);
}

template <int k, bool COMPENSATED, typename R, typename M>
__simd_callee__ inline void StoreWholeColumn(__ubuf__ float* matrix, __ubuf__ float* low, uint32_t column,
                                             uint32_t offset, uint32_t firstColumn, R& real, R& imag, R& lowReal,
                                             R& lowImag, R& zero, R& v0, R& v1, M& pivotMask, M& selected,
                                             M& components, M& storeFirst, M& storeSecond)
{
    // Clear the diagonal after normalization, including nonfinite pivots.
    AscendC::Reg::Select(imag, zero, imag, pivotMask);
    AscendC::Reg::Interleave(v0, v1, real, imag);
    AscendC::Reg::Interleave<float>(storeFirst, storeSecond, selected, selected);
    AscendC::Reg::StoreAlign(matrix + column * WHOLE_FLOAT_STRIDE, v0, storeFirst);
    AscendC::Reg::StoreAlign(matrix + column * WHOLE_FLOAT_STRIDE + SIMD_FLOAT_LANES, v1, storeSecond);
    if constexpr (k == 0 && COMPENSATED)
    {
        if (!offset && !firstColumn)
        {
            AscendC::Reg::Interleave(v0, v1, lowReal, lowImag);
            AscendC::Reg::Interleave<float>(storeFirst, storeSecond, components, components);
            AscendC::Reg::StoreAlign(low, v0, storeFirst);
            AscendC::Reg::StoreAlign(low + SIMD_FLOAT_LANES, v1, storeSecond);
        }
    }
}

// Keep a compensated diagonal separately from the rounded trailing matrix.
// Off-diagonal updates retain their existing strip and Cube implementations.
template <typename R, typename M>
__simd_callee__ inline void LoadFactorDiagonal(__ubuf__ float* matrix, uint32_t stride, R& high, R& low, M& components)
{
    AscendC::Reg::RegTensor<int32_t> positions;
    AscendC::Reg::Arange(positions, 0);
    AscendC::Reg::Muls(positions, positions, static_cast<int32_t>(stride), components);
    // Index generation uses signed lanes; Gather requires the same nonnegative indices as uint32_t.
    auto& indices = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(positions);
    AscendC::Reg::Gather(high, matrix, indices, components);
    AscendC::Reg::Duplicate(low, 0.0f);
}

template <typename R, typename M>
__simd_callee__ inline void AddDiagonalProduct(R& high, R& low, R& left, R& right, M& selected)
{
    R product, remainder, sum, part, error;
    AscendC::Reg::Mul(product, left, right, selected);
    AscendC::Reg::Neg(remainder, product, selected);
    AscendC::Reg::MulAddDst(remainder, left, right, selected);
    AscendC::Reg::Add(sum, high, product, selected);
    AscendC::Reg::Sub(part, sum, high, selected);
    AscendC::Reg::Sub(error, sum, part, selected);
    AscendC::Reg::Sub(error, high, error, selected);
    AscendC::Reg::Sub(part, product, part, selected);
    AscendC::Reg::Add(error, error, part, selected);
    AscendC::Reg::Add(error, error, remainder, selected);
    AscendC::Reg::Add(low, low, error, selected);
    AscendC::Reg::Select(high, sum, high, selected);
}

template <typename R, typename I, typename M>
__simd_callee__ inline void SelectFactorPivot(R& pivot, R& original, R& high, R& low, I& index, M& components)
{
    R corrected, magnitude;
    AscendC::Reg::MaskReg finite;
    AscendC::Reg::Gather(pivot, original, index);
    AscendC::Reg::Abs(magnitude, pivot, components);
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(finite, magnitude, FINITE_ROOT_BOUND, components);
    AscendC::Reg::Add(corrected, high, low, components);
    AscendC::Reg::Gather(magnitude, corrected, index);
    AscendC::Reg::Select(pivot, magnitude, pivot, finite);
}

template <typename R, typename M>
__simd_callee__ inline void UpdateFactorDiagonal(R& high, R& low, R& real, R& imag, R& lowReal, R& lowImag,
                                                 uint32_t column, M& active, M& components, bool firstColumn)
{
    uint32_t through = column + 1;
    auto inclusive = AscendC::Reg::UpdateMask<float>(through);
    AscendC::Reg::MaskReg selected;
    AscendC::Reg::Not(selected, inclusive, components);
    AscendC::Reg::And(selected, selected, active, components);
    if (!firstColumn)
    {
        // Compensated norm summation avoids rounding every rank-one diagonal update.
        // The cancellation-prone first column retains its product remainders below.
        R energy, change, sum, error;
        AscendC::Reg::Mul(energy, real, real, selected);
        AscendC::Reg::MulAddDst(energy, imag, imag, selected);
        AscendC::Reg::Sub(change, low, energy, selected);
        AscendC::Reg::Add(sum, high, change, selected);
        AscendC::Reg::Sub(error, sum, high, selected);
        AscendC::Reg::Sub(low, change, error, selected);
        AscendC::Reg::Select(high, sum, high, selected);
        return;
    }
    R negative;
    AscendC::Reg::Neg(negative, real, selected);
    AddDiagonalProduct(high, low, negative, real, selected);
    AscendC::Reg::Neg(negative, imag, selected);
    AddDiagonalProduct(high, low, negative, imag, selected);
    AscendC::Reg::Add(negative, real, real, selected);
    AscendC::Reg::Neg(negative, negative, selected);
    AddDiagonalProduct(high, low, negative, lowReal, selected);
    AscendC::Reg::Add(negative, imag, imag, selected);
    AscendC::Reg::Neg(negative, negative, selected);
    AddDiagonalProduct(high, low, negative, lowImag, selected);
    AscendC::Reg::Neg(negative, lowReal, selected);
    AddDiagonalProduct(high, low, negative, lowReal, selected);
    AscendC::Reg::Neg(negative, lowImag, selected);
    AddDiagonalProduct(high, low, negative, lowImag, selected);
}

// Correct the vector square root and reciprocal before normalizing a column.
// Keep zero and nonfinite pivots on the original exceptional-value path.
template <typename R, typename M>
__simd_callee__ inline void RefineFactorRoot(R& diagonal, R& inverse, R& pivot, M& components)
{
    R error, auxiliary;
    AscendC::Reg::MaskReg finite, positive;
    AscendC::Reg::Sqrt(diagonal, pivot, components);
    AscendC::Reg::Duplicate(auxiliary, 1.0f);
    AscendC::Reg::Div(inverse, auxiliary, diagonal, components);
    AscendC::Reg::Move(error, pivot);
    AscendC::Reg::Neg(auxiliary, diagonal, components);
    AscendC::Reg::MulAddDst(error, auxiliary, diagonal, components);
    // Reuse the reciprocal for the Newton square-root correction and normalization.
    AscendC::Reg::Mul(error, error, inverse, components);
    AscendC::Reg::Muls(error, error, 0.5f, components);
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(finite, diagonal, FINITE_ROOT_BOUND, components);
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(positive, diagonal, 0.0f, components);
    AscendC::Reg::And(finite, finite, positive, components);
    AscendC::Reg::Duplicate(auxiliary, 0.0f);
    AscendC::Reg::Select(error, error, auxiliary, finite);
    AscendC::Reg::Add(diagonal, diagonal, error, components);
    AscendC::Reg::Duplicate(error, 1.0f);
    AscendC::Reg::Neg(auxiliary, diagonal, components);
    AscendC::Reg::MulAddDst(error, auxiliary, inverse, components);
    AscendC::Reg::Move(auxiliary, inverse);
    AscendC::Reg::MulAddDst(auxiliary, inverse, error, components);
    AscendC::Reg::Select(inverse, auxiliary, inverse, finite);
}

template <int k, bool COMPENSATED = true, typename WholeReg, typename WholeIndex, typename WholeMask>
__simd_callee__ inline void WholeNormalizeColumn(WholeReg& real, WholeReg& imag, WholeReg& lowReal, WholeReg& lowImag,
                                                 WholeReg& diagonalHigh, WholeReg& diagonalLow, WholeIndex& index,
                                                 WholeIndex& failure, WholeIndex& failureIndex, WholeMask& active,
                                                 WholeMask& components, __ubuf__ float* matrix, __ubuf__ float* low,
                                                 uint32_t firstColumn, uint32_t rows, uint32_t offset)
{
    uint32_t column = firstColumn + k;
    if (column < rows)
    {
        AscendC::Reg::RegTensor<float> pivot, diagonal, inverse, zero, v0, v1;
        AscendC::Reg::Duplicate(index, column);
        SelectFactorPivot(pivot, real, diagonalHigh, diagonalLow, index, components);
        AscendC::Reg::MaskReg positive, bad, next;
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(bad, pivot, 0.0f, components);
        AscendC::Reg::Not(positive, bad, components);
        AscendC::Reg::And(bad, bad, active, components);
        AscendC::Reg::Duplicate(failureIndex, offset + column + 1);
        AscendC::Reg::Select(failure, failureIndex, failure, bad);
        AscendC::Reg::And(next, active, positive, components);
        AscendC::Reg::Move(active, next);
        RefineFactorRoot(diagonal, inverse, pivot, components);
        uint32_t before = column, through = column + 1;
        auto prefix = AscendC::Reg::UpdateMask<float>(before);
        auto inclusive = AscendC::Reg::UpdateMask<float>(through);
        AscendC::Reg::MaskReg selected, pivotMask, storeFirst, storeSecond;
        AscendC::Reg::Not(selected, prefix, components);
        AscendC::Reg::And(selected, selected, active, components);
        AscendC::Reg::Xor(pivotMask, prefix, inclusive, components);
        AscendC::Reg::Duplicate(zero, 0.0f);
        if (COMPENSATED && offset == 0 && firstColumn == 0 && k == 0)
        {
            CompensateWholeColumn(real, imag, lowReal, lowImag, pivot, diagonal, zero, selected, pivotMask, components);
        }
        else
        {
            AscendC::Reg::Mul(real, real, inverse, selected);
            AscendC::Reg::Mul(imag, imag, inverse, selected);
            AscendC::Reg::Select(real, diagonal, real, pivotMask);
        }
        StoreWholeColumn<k, COMPENSATED>(matrix, low, column, offset, firstColumn, real, imag, lowReal, lowImag, zero,
                                         v0, v1, pivotMask, selected, components, storeFirst, storeSecond);
        UpdateFactorDiagonal(diagonalHigh, diagonalLow, real, imag, lowReal, lowImag, column, active, components,
                             COMPENSATED && offset == 0 && firstColumn == 0 && k == 0);
    }
}

template <int k, bool COMPENSATED = true, typename WholeReg, typename WholeIndex, typename WholeMask>
__simd_callee__ inline void WholeSubtractColumn(WholeReg& ar, WholeReg& ai, WholeReg& real, WholeReg& imag,
                                                WholeReg& lowReal, WholeReg& lowImag, WholeIndex& index,
                                                WholeMask& components, WholeMask& selected, uint32_t firstColumn,
                                                uint32_t offset)
{
    AscendC::Reg::RegTensor<float> cr, ci, negativeCr, negativeCi;
    AscendC::Reg::Gather(cr, real, index);
    AscendC::Reg::Gather(ci, imag, index);
    AscendC::Reg::Neg(negativeCr, cr, components);
    AscendC::Reg::Neg(negativeCi, ci, components);
    AscendC::Reg::MulAddDst(ar, real, negativeCr, selected);
    AscendC::Reg::MulAddDst(ar, imag, negativeCi, selected);
    AscendC::Reg::MulAddDst(ai, real, ci, selected);
    AscendC::Reg::MulAddDst(ai, imag, negativeCr, selected);
    if (COMPENSATED && offset == 0 && firstColumn == 0 && k == 0)
    {
        AscendC::Reg::RegTensor<float> lr, li, negativeLr, negativeLi;
        AscendC::Reg::Gather(lr, lowReal, index);
        AscendC::Reg::Gather(li, lowImag, index);
        AscendC::Reg::Neg(negativeLr, lr, components);
        AscendC::Reg::Neg(negativeLi, li, components);
        AscendC::Reg::MulAddDst(ar, lowReal, negativeCr, selected);
        AscendC::Reg::MulAddDst(ar, lowImag, negativeCi, selected);
        AscendC::Reg::MulAddDst(ai, lowReal, ci, selected);
        AscendC::Reg::MulAddDst(ai, lowImag, negativeCr, selected);
        AscendC::Reg::MulAddDst(ar, real, negativeLr, selected);
        AscendC::Reg::MulAddDst(ar, imag, negativeLi, selected);
        AscendC::Reg::MulAddDst(ai, real, li, selected);
        AscendC::Reg::MulAddDst(ai, imag, negativeLr, selected);
        AscendC::Reg::MulAddDst(ar, lowReal, negativeLr, selected);
        AscendC::Reg::MulAddDst(ar, lowImag, negativeLi, selected);
        AscendC::Reg::MulAddDst(ai, lowReal, li, selected);
        AscendC::Reg::MulAddDst(ai, lowImag, negativeLr, selected);
    }
}

// Isolate first-column compensation so its temporaries do not spill the ordinary strip loop.
template <typename R>
__simd_callee__ inline void FactorWholeFirstVF(__ubuf__ float* matrix, __ubuf__ float* low, __ubuf__ uint32_t* info,
                                               uint32_t rows, R& diagonalHigh, R& diagonalLow)
{
    uint32_t count = rows, one = 1;
    auto components = AscendC::Reg::UpdateMask<float>(count), active = components;
    AscendC::Reg::RegTensor<uint32_t> failure, index, failureIndex;
    AscendC::Reg::RegTensor<float> real, imag, lowReal, lowImag;
    AscendC::Reg::Duplicate(failure, 0u);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(real, imag, matrix);
    WholeNormalizeColumn<0>(real, imag, lowReal, lowImag, diagonalHigh, diagonalLow, index, failure, failureIndex,
                            active, components, matrix, low, 0, rows, 0);
    for (uint16_t column = 1; column < rows; ++column)
    {
        AscendC::Reg::RegTensor<float> ar, ai, v0, v1;
        AscendC::Reg::MaskReg selected, storeFirst, storeSecond;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(ar, ai,
                                                                                matrix + column * WHOLE_FLOAT_STRIDE);
        WholeTrailingMask(column, components, active, selected, index);
        WholeSubtractColumn<0>(ar, ai, real, imag, lowReal, lowImag, index, components, selected, 0, 0);
        AscendC::Reg::Interleave(v0, v1, ar, ai);
        AscendC::Reg::Interleave<float>(storeFirst, storeSecond, selected, selected);
        AscendC::Reg::StoreAlign(matrix + column * WHOLE_FLOAT_STRIDE, v0, storeFirst);
        AscendC::Reg::StoreAlign(matrix + column * WHOLE_FLOAT_STRIDE + SIMD_FLOAT_LANES, v1, storeSecond);
    }
    auto first = AscendC::Reg::UpdateMask<uint32_t>(one);
    AscendC::Reg::StoreAlign(info, failure, first);
}

// Normalized UB coefficients stay positive; retain source signs once for each strip.
template <typename R, typename M>
__simd_callee__ inline void SignedWholeSubtract(R& ar, R& ai, R& real, R& negativeReal, R& negativeImag,
                                                __ubuf__ float* matrix, uint32_t source, uint32_t target, M& mask)
{
    R cr, ci;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
        cr, matrix + source * WHOLE_FLOAT_STRIDE + 2 * target);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
        ci, matrix + source * WHOLE_FLOAT_STRIDE + 2 * target + 1);
    AscendC::Reg::MulAddDst(ar, negativeReal, cr, mask);
    AscendC::Reg::MulAddDst(ar, negativeImag, ci, mask);
    AscendC::Reg::MulAddDst(ai, real, ci, mask);
    AscendC::Reg::MulAddDst(ai, negativeImag, cr, mask);
}
template <typename M>
__simd_callee__ inline void SignedWholeMask(uint32_t column, M& components, M& active, M& selected)
{
    uint32_t count = column;
    auto prefix = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::Not(selected, prefix, components);
    AscendC::Reg::And(selected, selected, active, components);
}

// Intra-strip coefficients are still in registers, so they need no UB store-to-load fence.
template <typename R, typename I, typename M>
__simd_callee__ inline void RegisterWholeSubtract(R& ar, R& ai, R& real, R& negativeReal, R& negativeImag,
                                                  uint32_t target, I& index, M& components, M& selected)
{
    R cr, ci;
    AscendC::Reg::Duplicate(index, target);
    AscendC::Reg::Gather(cr, real, index);
    AscendC::Reg::Gather(ci, negativeImag, index);
    AscendC::Reg::Neg(ci, ci, components);
    AscendC::Reg::MulAddDst(ar, negativeReal, cr, selected);
    AscendC::Reg::MulAddDst(ar, negativeImag, ci, selected);
    AscendC::Reg::MulAddDst(ai, real, ci, selected);
    AscendC::Reg::MulAddDst(ai, negativeImag, cr, selected);
}

// Four source columns use three registers each; the UB coefficient plane preserves ordinary signs.
template <typename R, typename M>
__simd_callee__ inline void UpdateWholeTrailing(__ubuf__ float* matrix, uint32_t firstColumn, uint32_t rows, R& re0,
                                                R& nre0, R& im0, R& re1, R& nre1, R& im1, R& re2, R& nre2, R& im2,
                                                R& re3, R& nre3, R& im3, M& components, M& active, M& selected)
{
    for (uint16_t c = firstColumn + WHOLE_FACTOR_STRIP; c < rows; ++c)
    {
        AscendC::Reg::RegTensor<float> ar, ai, v0, v1;
        AscendC::Reg::MaskReg storeFirst, storeSecond;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(ar, ai,
                                                                                matrix + c * WHOLE_FLOAT_STRIDE);
        SignedWholeMask(c, components, active, selected);
        SignedWholeSubtract(ar, ai, re0, nre0, im0, matrix, firstColumn + 0, c, selected);
        SignedWholeSubtract(ar, ai, re1, nre1, im1, matrix, firstColumn + 1, c, selected);
        SignedWholeSubtract(ar, ai, re2, nre2, im2, matrix, firstColumn + 2, c, selected);
        SignedWholeSubtract(ar, ai, re3, nre3, im3, matrix, firstColumn + 3, c, selected);
        AscendC::Reg::Interleave(v0, v1, ar, ai);
        AscendC::Reg::Interleave<float>(storeFirst, storeSecond, selected, selected);
        AscendC::Reg::StoreAlign(matrix + c * WHOLE_FLOAT_STRIDE, v0, storeFirst);
        AscendC::Reg::StoreAlign(matrix + c * WHOLE_FLOAT_STRIDE + SIMD_FLOAT_LANES, v1, storeSecond);
    }
}

template <typename R, typename I, typename M>
__simd_callee__ inline void FactorWholeStage0(__ubuf__ float* matrix, __ubuf__ float* low, uint32_t firstColumn,
                                              uint32_t rows, uint32_t offset, R& lowReal, R& lowImag, R& diagonalHigh,
                                              R& diagonalLow, I& index, I& failure, I& failureIndex, M& active,
                                              M& components, M& selected, R& re0, R& im0, R& nre0, R& re1, R& im1,
                                              R& nre1)
{
    WholeNormalizeColumn<0, false>(re0, im0, lowReal, lowImag, diagonalHigh, diagonalLow, index, failure, failureIndex,
                                   active, components, matrix, low, firstColumn, rows, offset);

    AscendC::Reg::Neg(nre0, re0, components);
    AscendC::Reg::Neg(im0, im0, components);
    if (firstColumn + 1 < rows)
    {
        SignedWholeMask(firstColumn + 1, components, active, selected);
        RegisterWholeSubtract(re1, im1, re0, nre0, im0, firstColumn + 1, index, components, selected);
    }
}

template <typename R, typename I, typename M>
__simd_callee__ inline void FactorWholeStage1(__ubuf__ float* matrix, __ubuf__ float* low, uint32_t firstColumn,
                                              uint32_t rows, uint32_t offset, R& lowReal, R& lowImag, R& diagonalHigh,
                                              R& diagonalLow, I& index, I& failure, I& failureIndex, M& active,
                                              M& components, M& selected, R& re0, R& im0, R& nre0, R& re1, R& im1,
                                              R& nre1, R& re2, R& im2, R& nre2)
{
    WholeNormalizeColumn<1, false>(re1, im1, lowReal, lowImag, diagonalHigh, diagonalLow, index, failure, failureIndex,
                                   active, components, matrix, low, firstColumn, rows, offset);

    AscendC::Reg::Neg(nre1, re1, components);
    AscendC::Reg::Neg(im1, im1, components);
    if (firstColumn + 2 < rows)
    {
        SignedWholeMask(firstColumn + 2, components, active, selected);
        RegisterWholeSubtract(re2, im2, re0, nre0, im0, firstColumn + 2, index, components, selected);
        RegisterWholeSubtract(re2, im2, re1, nre1, im1, firstColumn + 2, index, components, selected);
    }
}

template <typename R, typename I, typename M>
__simd_callee__ inline void FactorWholeStage2(__ubuf__ float* matrix, __ubuf__ float* low, uint32_t firstColumn,
                                              uint32_t rows, uint32_t offset, R& lowReal, R& lowImag, R& diagonalHigh,
                                              R& diagonalLow, I& index, I& failure, I& failureIndex, M& active,
                                              M& components, M& selected, R& re0, R& im0, R& nre0, R& re1, R& im1,
                                              R& nre1, R& re2, R& im2, R& nre2, R& re3, R& im3, R& nre3)
{
    WholeNormalizeColumn<2, false>(re2, im2, lowReal, lowImag, diagonalHigh, diagonalLow, index, failure, failureIndex,
                                   active, components, matrix, low, firstColumn, rows, offset);

    AscendC::Reg::Neg(nre2, re2, components);
    AscendC::Reg::Neg(im2, im2, components);
    if (firstColumn + 3 < rows)
    {
        SignedWholeMask(firstColumn + 3, components, active, selected);
        RegisterWholeSubtract(re3, im3, re0, nre0, im0, firstColumn + 3, index, components, selected);
        RegisterWholeSubtract(re3, im3, re1, nre1, im1, firstColumn + 3, index, components, selected);
        RegisterWholeSubtract(re3, im3, re2, nre2, im2, firstColumn + 3, index, components, selected);
    }
}

template <typename R, typename I, typename M>
__simd_callee__ inline void FactorWholeStage3(__ubuf__ float* matrix, __ubuf__ float* low, uint32_t firstColumn,
                                              uint32_t rows, uint32_t offset, R& lowReal, R& lowImag, R& diagonalHigh,
                                              R& diagonalLow, I& index, I& failure, I& failureIndex, M& active,
                                              M& components, R& re3, R& im3, R& nre3)
{
    WholeNormalizeColumn<3, false>(re3, im3, lowReal, lowImag, diagonalHigh, diagonalLow, index, failure, failureIndex,
                                   active, components, matrix, low, firstColumn, rows, offset);
    AscendC::Reg::Neg(nre3, re3, components);
    AscendC::Reg::Neg(im3, im3, components);
}

template <typename R>
__simd_callee__ inline void FactorWholeVF(__ubuf__ float* matrix, __ubuf__ float* low, __ubuf__ uint32_t* info,
                                          uint32_t rows, uint32_t offset, uint32_t firstStart, R& diagonalHigh,
                                          R& diagonalLow)
{
    uint32_t count = rows, one = 1;
    auto components = AscendC::Reg::UpdateMask<float>(count), active = components;
    AscendC::Reg::RegTensor<uint32_t> failure, index, failureIndex;
    AscendC::Reg::RegTensor<float> lowReal, lowImag;
    if (firstStart)
    {
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::LoadDist::DIST_BRC_B32>(failure, info);
        AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(active, failure, 0u, components);
    }
    else
    {
        AscendC::Reg::Duplicate(failure, 0u);
    }
    for (uint16_t firstColumn = firstStart; firstColumn < rows; firstColumn += WHOLE_FACTOR_STRIP)
    {
        AscendC::Reg::MaskReg selected;
        AscendC::Reg::RegTensor<float> re0, im0;
        LoadComplexColumn(re0, im0, matrix, firstColumn + 0, rows, WHOLE_FLOAT_STRIDE);
        AscendC::Reg::RegTensor<float> re1, im1;
        LoadComplexColumn(re1, im1, matrix, firstColumn + 1, rows, WHOLE_FLOAT_STRIDE);
        AscendC::Reg::RegTensor<float> re2, im2;
        LoadComplexColumn(re2, im2, matrix, firstColumn + 2, rows, WHOLE_FLOAT_STRIDE);
        AscendC::Reg::RegTensor<float> re3, im3;
        LoadComplexColumn(re3, im3, matrix, firstColumn + 3, rows, WHOLE_FLOAT_STRIDE);
        AscendC::Reg::RegTensor<float> nre0, nre1, nre2, nre3;
        FactorWholeStage0(matrix, low, firstColumn, rows, offset, lowReal, lowImag, diagonalHigh, diagonalLow, index,
                          failure, failureIndex, active, components, selected, re0, im0, nre0, re1, im1, nre1);
        FactorWholeStage1(matrix, low, firstColumn, rows, offset, lowReal, lowImag, diagonalHigh, diagonalLow, index,
                          failure, failureIndex, active, components, selected, re0, im0, nre0, re1, im1, nre1, re2, im2,
                          nre2);
        FactorWholeStage2(matrix, low, firstColumn, rows, offset, lowReal, lowImag, diagonalHigh, diagonalLow, index,
                          failure, failureIndex, active, components, selected, re0, im0, nre0, re1, im1, nre1, re2, im2,
                          nre2, re3, im3, nre3);
        FactorWholeStage3(matrix, low, firstColumn, rows, offset, lowReal, lowImag, diagonalHigh, diagonalLow, index,
                          failure, failureIndex, active, components, re3, im3, nre3);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        UpdateWholeTrailing(matrix, firstColumn, rows, re0, nre0, im0, re1, nre1, im1, re2, nre2, im2, re3, nre3, im3,
                            components, active, selected);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
    auto first = AscendC::Reg::UpdateMask<uint32_t>(one);
    AscendC::Reg::StoreAlign(info, failure, first);
}

// First-column and ordinary factorization share one VF invocation and reuse their registers.
__simd_vf__ inline void FactorWholeCombinedVF(__ubuf__ float* matrix, __ubuf__ float* low, __ubuf__ uint32_t* info,
                                              uint32_t rows, uint32_t offset)
{
    uint32_t count = rows;
    auto components = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::RegTensor<float> diagonalHigh, diagonalLow;
    LoadFactorDiagonal(matrix, WHOLE_FLOAT_STRIDE + COMPLEX_COMPONENTS, diagonalHigh, diagonalLow, components);
    if (!offset)
    {
        FactorWholeFirstVF(matrix, low, info, rows, diagonalHigh, diagonalLow);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
    FactorWholeVF(matrix, low, info, rows, offset, offset ? 0 : 1, diagonalHigh, diagonalLow);
}

// DMA handles GM transfers; SIMT transposes only the selected UB triangle.
__simt_vf__ __aicore__ __launch_bounds__(THREADS) void WholeTransposeVF(__ubuf__ aclsolverFloatComplex* upperMatrix,
                                                                        __ubuf__ aclsolverFloatComplex* lowerMatrix,
                                                                        int n, bool output)
{
    for (int pos = threadIdx.x; pos < n * n; pos += THREADS)
    {
        int r = pos / n, c = pos % n;
        if (r < c)
        {
            continue;
        }
        if (output)
        {
            auto z = ReadUB(lowerMatrix, c * SIMD_PANEL_STRIDE + r);
            WriteUB(upperMatrix, r * SIMD_PANEL_STRIDE + c, {z.real, -z.imag});
        }
        else
        {
            auto z = ReadUB(upperMatrix, r * SIMD_PANEL_STRIDE + c);
            WriteUB(lowerMatrix, c * SIMD_PANEL_STRIDE + r, {z.real, -z.imag});
        }
    }
}

__aicore__ inline void StoreWholeTile(int n, int lda, bool upper, __ubuf__ aclsolverFloatComplex* address,
                                      AscendC::LocalTensor<float>& matrix, AscendC::LocalTensor<float>& upperMatrix,
                                      AscendC::GlobalTensor<float>& a, AscendC::DataCopyExtParams& copy)
{
    AscendC::DataCopyExtParams copyOut{static_cast<uint16_t>(n), static_cast<uint32_t>(n * COMPLEX_BYTES),
                                       copy.dstStride, (static_cast<int64_t>(lda) - n) * COMPLEX_BYTES, 0};
    if (upper)
    {
        asc_vf_call<WholeTransposeVF>(dim3(THREADS),
                                      reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(upperMatrix.GetPhyAddr()),
                                      address, n, true);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(a, upperMatrix, copyOut);
    }
    else
    {
        AscendC::DataCopyPad(a, matrix, copyOut);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void FactorWholeTile(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n,
                                       int lda, bool upper, int m, int offset,
                                       __gm__ aclsolverFloatComplex* firstColumn, int firstStride,
                                       AscendC::LocalTensor<float>& matrix, AscendC::LocalTensor<float>& upperMatrix,
                                       AscendC::LocalTensor<uint32_t>& localInfo, AscendC::LocalTensor<float>& low,
                                       AscendC::GlobalTensor<uint32_t>& infos, AscendC::DataCopyExtParams& infoCopy)
{
    auto pointer = (array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single) +
                   static_cast<int64_t>(offset) * (static_cast<int64_t>(lda) + 1);
    AscendC::GlobalTensor<float> a;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pointer));
    AscendC::DataCopyExtParams copy{static_cast<uint16_t>(n), static_cast<uint32_t>(n * COMPLEX_BYTES),
                                    (static_cast<int64_t>(lda) - n) * COMPLEX_BYTES,
                                    static_cast<uint32_t>(UbPaddingBlocks(SIMD_PANEL_STRIDE, n)), 0};
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    auto address = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(matrix.GetPhyAddr());
    if (upper)
    {
        AscendC::DataCopyPad(upperMatrix, a, copy, padding);
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<WholeTransposeVF>(dim3(THREADS),
                                      reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(upperMatrix.GetPhyAddr()),
                                      address, n, false);
    }
    else
    {
        AscendC::DataCopyPad(matrix, a, copy, padding);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    asc_vf_call<FactorWholeCombinedVF>(reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(low.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ uint32_t*>(localInfo.GetPhyAddr()), n, offset);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(infos[m], localInfo, infoCopy);
    if (!offset && firstColumn)
    {
        AscendC::GlobalTensor<float> first;
        first.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstColumn + static_cast<int64_t>(m) * firstStride));
        AscendC::DataCopyExtParams copyLow{1, static_cast<uint32_t>(n * COMPLEX_BYTES), 0, 0, 0};
        AscendC::DataCopyPad(first, low, copyLow);
    }
    StoreWholeTile(n, lda, upper, address, matrix, upperMatrix, a, copy);
}

__aicore__ inline void WholeUBFactor(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n, int lda,
                                     bool upper, __gm__ int* info, int batch, int offset,
                                     __gm__ aclsolverFloatComplex* firstColumn, int firstStride)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixBuffer, infoBuffer, lowBuffer, upperBuffer;
    pipe.InitBuffer(matrixBuffer, WHOLE_MATRIX_BYTES);
    pipe.InitBuffer(upperBuffer, WHOLE_MATRIX_BYTES);
    pipe.InitBuffer(infoBuffer, DMA_ALIGNMENT_BYTES);
    pipe.InitBuffer(lowBuffer, FACTOR_BLOCK * sizeof(aclsolverFloatComplex));
    auto matrix = matrixBuffer.Get<float>();
    auto upperMatrix = upperBuffer.Get<float>();
    auto localInfo = infoBuffer.Get<uint32_t>();
    auto low = lowBuffer.Get<float>();
    AscendC::GlobalTensor<uint32_t> infos;
    infos.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(info));
    AscendC::DataCopyExtParams infoCopy{1, sizeof(uint32_t), 0, 0, 0};
    int core = AscendC::GetBlockIdx(), cores = AscendC::GetBlockNum() * AscendC::GetTaskRatio();
    for (int m = core; m < batch; m += cores)
    {
        if (offset)
        {
            AscendC::DataCopyPadExtParams<uint32_t> paddingInfo{false, 0, 0, 0};
            AscendC::DataCopyPad(localInfo, infos[m], infoCopy, paddingInfo);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (localInfo.GetValue(0))
            {
                continue;
            }
        }
        FactorWholeTile(single, array, n, lda, upper, m, offset, firstColumn, firstStride, matrix, upperMatrix,
                        localInfo, low, infos, infoCopy);
    }
}

// Lower diagonal tiles have contiguous columns and fit entirely in UB.
template <typename R, typename I, typename M>
__simd_callee__ inline void UpdateLowerTrailing(__ubuf__ float* matrix, uint16_t k, uint32_t rows, R& real, R& imag,
                                                R& unused, I& index, M& components, M& active, M& selected,
                                                M& storeMask, M& unusedMask)
{
    for (uint16_t col = k + 1; col < rows; ++col)
    {
        AscendC::Reg::RegTensor<float> a, ar, ai, cr, ci, negativeCr, negativeCi;
        AscendC::Reg::Duplicate(index, static_cast<uint32_t>(col));
        AscendC::Reg::Gather(cr, real, index);
        AscendC::Reg::Gather(ci, imag, index);
        AscendC::Reg::Neg(negativeCr, cr, components);
        AscendC::Reg::Neg(negativeCi, ci, components);
        uint32_t count = col;
        auto prefixRows = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::Not(selected, prefixRows, components);
        AscendC::Reg::And(selected, selected, active, components);
        AscendC::Reg::LoadAlign(a, matrix + col * FACTOR_BLOCK);
        AscendC::Reg::DeInterleave(ar, ai, a, a);
        AscendC::Reg::MulAddDst(ar, real, negativeCr, selected);
        AscendC::Reg::MulAddDst(ar, imag, negativeCi, selected);
        AscendC::Reg::MulAddDst(ai, real, ci, selected);
        AscendC::Reg::MulAddDst(ai, imag, negativeCr, selected);
        AscendC::Reg::Interleave(a, unused, ar, ai);
        AscendC::Reg::Interleave<float>(storeMask, unusedMask, selected, selected);
        AscendC::Reg::StoreAlign(matrix + col * FACTOR_BLOCK, a, storeMask);
    }
}

__simd_vf__ inline void FactorLowerVF(__ubuf__ float* matrix, __ubuf__ uint32_t* info, uint32_t rows, uint32_t offset)
{
    uint32_t componentCount = rows, one = 1;
    auto components = AscendC::Reg::UpdateMask<float>(componentCount);
    auto active = components;
    AscendC::Reg::RegTensor<float> diagonalHigh, diagonalLow;
    LoadFactorDiagonal(matrix, FACTOR_BLOCK + COMPLEX_COMPONENTS, diagonalHigh, diagonalLow, components);
    AscendC::Reg::RegTensor<uint32_t> failure, index, failureIndex;
    AscendC::Reg::Duplicate(failure, 0u);
    for (uint16_t k = 0; k < rows; ++k)
    {
        AscendC::Reg::RegTensor<float> value, real, imag, unused, pivot, diagonal, inverse, zero;
        AscendC::Reg::LoadAlign(value, matrix + k * FACTOR_BLOCK);
        AscendC::Reg::DeInterleave(real, imag, value, value);
        AscendC::Reg::Duplicate(index, static_cast<uint32_t>(k));
        SelectFactorPivot(pivot, real, diagonalHigh, diagonalLow, index, components);
        AscendC::Reg::MaskReg positive, bad, next;
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(bad, pivot, 0.0f, components);
        AscendC::Reg::Not(positive, bad, components);
        AscendC::Reg::And(bad, bad, active, components);
        AscendC::Reg::Duplicate(failureIndex, offset + k + 1);
        AscendC::Reg::Select(failure, failureIndex, failure, bad);
        AscendC::Reg::And(next, active, positive, components);
        AscendC::Reg::Move(active, next);
        RefineFactorRoot(diagonal, inverse, pivot, components);
        uint32_t before = k, through = k + 1;
        auto prefix = AscendC::Reg::UpdateMask<float>(before);
        auto inclusive = AscendC::Reg::UpdateMask<float>(through);
        AscendC::Reg::MaskReg selected, pivotMask, storeMask, unusedMask;
        AscendC::Reg::Not(selected, prefix, components);
        AscendC::Reg::And(selected, selected, active, components);
        AscendC::Reg::Xor(pivotMask, prefix, inclusive, components);
        AscendC::Reg::Mul(real, real, inverse, selected);
        AscendC::Reg::Mul(imag, imag, inverse, selected);
        AscendC::Reg::Select(real, diagonal, real, pivotMask);
        AscendC::Reg::Duplicate(zero, 0.0f);
        AscendC::Reg::Select(imag, zero, imag, pivotMask);
        UpdateFactorDiagonal(diagonalHigh, diagonalLow, real, imag, zero, zero, k, active, components, false);
        AscendC::Reg::Interleave(value, unused, real, imag);
        AscendC::Reg::Interleave<float>(storeMask, unusedMask, selected, selected);
        AscendC::Reg::StoreAlign(matrix + k * FACTOR_BLOCK, value, storeMask);
        UpdateLowerTrailing(matrix, k, rows, real, imag, unused, index, components, active, selected, storeMask,
                            unusedMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
    auto first = AscendC::Reg::UpdateMask<uint32_t>(one);
    AscendC::Reg::StoreAlign(info, failure, first);
}

__aicore__ inline void FactorLower(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n, int lda,
                                   __gm__ int* info, int batch, int offset)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixBuffer, infoBuffer;
    pipe.InitBuffer(matrixBuffer, REGISTER_FACTOR_ORDER * COMPLEX_COMPONENTS * REGISTER_FACTOR_ORDER * sizeof(float));
    pipe.InitBuffer(infoBuffer, DMA_ALIGNMENT_BYTES);
    auto matrix = matrixBuffer.Get<float>();
    auto localInfo = infoBuffer.Get<uint32_t>();
    int block = AscendC::GetBlockIdx(), grid = AscendC::GetBlockNum() * AscendC::GetTaskRatio();
    for (int m = block; m < batch; m += grid)
    {
        if (offset && info[m] != 0)
        {
            continue;
        }
        if (!n)
        {
            info[m] = 0;
            continue;
        }
        auto pointer = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        AscendC::GlobalTensor<float> a;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pointer));
        int64_t first = 2 * (static_cast<int64_t>(offset) * lda + offset);
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(n), static_cast<uint32_t>(n * COMPLEX_BYTES),
                                        (static_cast<int64_t>(lda) - n) * COMPLEX_BYTES,
                                        static_cast<uint32_t>(UbPaddingBlocks(WARP_SIZE, n)), 0};
        AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(matrix, a[first], copy, padding);
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<FactorLowerVF>(reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ uint32_t*>(localInfo.GetPhyAddr()), n, offset);
        AscendC::PipeBarrier<PIPE_ALL>();
        info[m] = localInfo.GetValue(0);
        AscendC::DataCopyExtParams copyOut{static_cast<uint16_t>(n), static_cast<uint32_t>(n * COMPLEX_BYTES),
                                           copy.dstStride, (static_cast<int64_t>(lda) - n) * COMPLEX_BYTES, 0};
        AscendC::DataCopyPad(a[first], matrix, copyOut);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Subgroups keep a complete diagonal tile in registers without intermediate GM updates.
template <int WIDTH>
__simt_callee__ inline void NormalizeDiagonalColumn(__gm__ aclsolverFloatComplex* a,
                                                    __gm__ aclsolverFloatComplex* firstColumn, int lda, int offset,
                                                    int m, int firstStride, int lane, int n, int k, bool upper,
                                                    float pivot, float diagonal, float inverse, float (&re)[WIDTH],
                                                    float (&im)[WIDTH], aclsolverFloatComplex& firstLow,
                                                    FloatPair& diagonalState)
{
    if (lane >= k && lane < n)
    {
        if (firstColumn && k == 0)
        {
            float rootLow = RootRemainder(pivot, diagonal);
            firstLow = {rootLow, 0};
            aclsolverFloatComplex value{diagonal, 0};
            if (lane != k)
            {
                value = NormalizeFirstColumn({re[k], im[k]}, diagonal, rootLow, firstLow);
            }
            re[k] = value.real;
            im[k] = value.imag;
            WriteLower(firstColumn + static_cast<int64_t>(m) * firstStride, firstStride, lane, 0, false, firstLow);
        }
        else
        {
            re[k] = lane == k ? diagonal : re[k] * inverse;
            im[k] = lane == k ? 0 : im[k] * inverse;
        }
        if (lane > k)
        {
            UpdateScalarDiagonal(diagonalState, {re[k], im[k]}, firstLow, firstColumn && k == 0);
        }
        WriteLower(a, lda, offset + lane, offset + k, upper, {re[k], im[k]});
    }
}

template <int WIDTH>
__simt_callee__ inline void UpdateDiagonalTrailing(int k, int n, int lane, int shuffleBase,
                                                   __gm__ aclsolverFloatComplex* firstColumn, float (&re)[WIDTH],
                                                   float (&im)[WIDTH], aclsolverFloatComplex firstLow)
{
#pragma unroll
    for (int c = k + 1; c < WIDTH; ++c)
    {
        float pr = asc_shfl(re[k], shuffleBase + c), pi = asc_shfl(im[k], shuffleBase + c);
        float lowReal = 0, lowImag = 0;
        if (firstColumn && !k)
        {
            lowReal = asc_shfl(firstLow.real, shuffleBase + c);
            lowImag = asc_shfl(firstLow.imag, shuffleBase + c);
        }
        if (lane >= c && lane < n)
        {
            auto z = firstColumn && k == 0
                         ? SubFirstColumn({re[c], im[c]}, {re[k], im[k]}, {pr, pi}, firstLow, {lowReal, lowImag})
                         : SubProduct({re[c], im[c]}, {re[k], im[k]}, {pr, -pi});
            re[c] = z.real;
            im[c] = z.imag;
        }
    }
}

template <int WIDTH, int VF_THREADS>
__simt_vf__ __aicore__ __launch_bounds__(VF_THREADS) void FactorDiagonalVF(
    __gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n, int lda, bool upper, __gm__ int* info,
    int batch, int offset, __gm__ aclsolverFloatComplex* firstColumn, int firstStride)
{
    int lane = threadIdx.x % WIDTH, group = threadIdx.x / WIDTH;
    int shuffleBase = (threadIdx.x % WARP_SIZE) / WIDTH * WIDTH;
    int groups = VF_THREADS / WIDTH;
    for (int m = blockIdx.x * groups + group; m < batch; m += gridDim.x * groups)
    {
        if (offset && info[m] != 0)
        {
            continue;
        }
        auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        float re[WIDTH], im[WIDTH];
#pragma unroll
        for (int c = 0; c < WIDTH; ++c)
        {
            auto z = lane < n && c <= lane ? ReadLower(a, lda, offset + lane, offset + c, upper)
                                           : aclsolverFloatComplex{0, 0};
            re[c] = z.real;
            im[c] = z.imag;
        }
        FloatPair diagonalState{lane < n ? ReadLower(a, lda, offset + lane, offset + lane, upper).real : 0, 0};
        int failure = 0;
        aclsolverFloatComplex firstLow{0, 0};
#pragma unroll
        for (int k = 0; k < WIDTH; ++k)
        {
            if (k < n && !failure)
            {
                float original = asc_shfl(re[k], shuffleBase + k);
                float corrected = asc_shfl(diagonalState.high + diagonalState.low, shuffleBase + k);
                float pivot = fabsf(original) < FINITE_ROOT_BOUND ? corrected : original;
                if (pivot <= 0)
                {
                    failure = k + 1;
                    continue;
                }
                float inverse = 0;
                float diagonal = RefineScalarRoot(pivot, inverse);
                NormalizeDiagonalColumn(a, firstColumn, lda, offset, m, firstStride, lane, n, k, upper, pivot, diagonal,
                                        inverse, re, im, firstLow, diagonalState);
                UpdateDiagonalTrailing(k, n, lane, shuffleBase, firstColumn, re, im, firstLow);
            }
        }
        if (lane == 0)
        {
            info[m] = failure ? offset + failure : 0;
        }
    }
}

// Dispatch the same factorization kernels for both storage modes.
template <int LANES, bool COMPENSATED = false>
__aicore__ inline void DispatchSmallFactor(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n,
                                           int lda, bool upper, __gm__ int* info, int batch, int offset,
                                           __gm__ aclsolverFloatComplex* firstColumn, int firstStride)
{
    if (upper)
    {
        asc_vf_call<FactorSmallVF<BATCH_FACTOR_THREADS, LANES, true, COMPENSATED>>(
            dim3(BATCH_FACTOR_THREADS), single, array, n, lda, info, batch, offset, firstColumn, firstStride);
    }
    else
    {
        asc_vf_call<FactorSmallVF<BATCH_FACTOR_THREADS, LANES, false, COMPENSATED>>(
            dim3(BATCH_FACTOR_THREADS), single, array, n, lda, info, batch, offset, firstColumn, firstStride);
    }
}

// Load a short group before its dependent compensated additions to overlap memory latency.
__simt_callee__ inline void SubtractFactorColumns(FloatPair& diagonal, const __gm__ aclsolverFloatComplex* a,
                                                  const __gm__ aclsolverFloatComplex* firstColumn, int lda, int row,
                                                  int begin, int offset, bool upper)
{
    for (int start = begin; start < offset; start += DIAGONAL_PREFETCH_COLUMNS)
    {
        aclsolverFloatComplex values[DIAGONAL_PREFETCH_COLUMNS];
#pragma unroll
        for (int part = 0; part < DIAGONAL_PREFETCH_COLUMNS; ++part)
        {
            values[part] =
                start + part < offset ? ReadLower(a, lda, row, start + part, upper, true) : aclsolverFloatComplex{0, 0};
        }
#pragma unroll
        for (int part = 0; part < DIAGONAL_PREFETCH_COLUMNS; ++part)
        {
            if (start + part < offset)
            {
                bool first = start == 0 && part == 0;
                auto low = first ? ReadLower(firstColumn, 0, row, 0, false, true) : aclsolverFloatComplex{0, 0};
                UpdateScalarDiagonal(diagonal, values[part], low, first);
            }
        }
    }
}

// Recover each new block's diagonal from the original input rather than a rounded Schur update.
__simt_vf__ __aicore__ __launch_bounds__(BATCH_FACTOR_THREADS) void PrepareFactorDiagonalVF(
    __gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n, int lda, bool upper, __gm__ int* info,
    int batch, int offset, __gm__ aclsolverFloatComplex* firstColumn, int firstStride, __gm__ float* originalDiagonal,
    int fullOrder, int groups, bool whole, bool retained)
{
    int lanes = blockDim.x / groups, lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
    // Whole tiles retain m % gridDim.x ownership while warps prepare several matrices concurrently.
    int first = whole ? blockIdx.x + group * gridDim.x : blockIdx.x * groups + group;
    for (int m = first; m < batch; m += gridDim.x * groups)
    {
        if (offset && info[m] != 0)
        {
            continue;
        }
        auto a = array ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(array[m]) : single;
        int components = retained ? FACTOR_DIAGONAL_COMPONENTS : 1;
        auto saved = originalDiagonal + static_cast<int64_t>(m) * firstStride * components;
        int count = offset ? n : fullOrder - n;
        for (int local = lane; local < count; local += lanes)
        {
            int row = offset ? offset + local : n + local;
            float current = ReadLower(a, lda, row, row, upper).real;
            if (!offset)
            {
                saved[components * row] = current;
                if (components > 1)
                {
                    saved[components * row + 1] = 0;
                }
                continue;
            }
            FloatPair diagonal{saved[components * row], components > 1 ? saved[components * row + 1] : 0};
            if (!retained)
            {
                SubtractFactorColumns(diagonal, a, firstColumn + static_cast<int64_t>(m) * firstStride, lda, row, 0,
                                      offset, upper);
            }
            else if (batch >= BATCH_THRESHOLD && n <= PACKED_TAIL_ROWS)
            {
                // The packed final panel updates its triangle directly, without retaining the last block.
                SubtractFactorColumns(diagonal, a, firstColumn + static_cast<int64_t>(m) * firstStride, lda, row,
                                      offset - FACTOR_BLOCK, offset, upper);
            }
            if (fabsf(current) < FINITE_ROOT_BOUND)
            {
                WriteLower(a, lda, row, row, upper, {diagonal.high + diagonal.low, 0});
            }
        }
    }
    asc_threadfence();
}

__aicore__ inline void PrepareFactorDiagonal(bool whole, __gm__ aclsolverFloatComplex* single,
                                             __gm__ const uint64_t* array, int n, int lda, bool upper, __gm__ int* info,
                                             int batch, int offset, __gm__ aclsolverFloatComplex* firstColumn,
                                             int firstStride, __gm__ float* originalDiagonal, int fullOrder,
                                             bool retained)
{
    int threads = batch < BATCH_THRESHOLD ? THREADS : BATCH_FACTOR_THREADS;
    int groups = 1;
    if (whole && batch >= BATCH_THRESHOLD)
    {
        groups = threads / WARP_SIZE;
    }
    else if (!whole && (batch >= BATCH_THRESHOLD || upper || (firstColumn && !offset)))
    {
        int lanes = batch >= BATCH_THRESHOLD && n <= HALF_WARP_LANES ? HALF_WARP_LANES : WARP_SIZE;
        groups = threads / lanes;
    }
    // Match the following factor stage's matrix ownership; no cross-AIV synchronization is needed.
    asc_vf_call<PrepareFactorDiagonalVF>(dim3(threads), single, array, n, lda, upper, info, batch, offset, firstColumn,
                                         firstStride, originalDiagonal, fullOrder, groups, whole, retained);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool WHOLE = false>
__global__ __aicore__ __vector__ void FactorSmall(__gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array,
                                                  int n, int lda, bool upper, __gm__ int* info, int batch, int offset,
                                                  __gm__ aclsolverFloatComplex* firstColumn, int firstStride,
                                                  __gm__ float* originalDiagonal, int fullOrder, bool retained)
{
    if (originalDiagonal && (offset || fullOrder > n))
    {
        PrepareFactorDiagonal(WHOLE, single, array, n, lda, upper, info, batch, offset, firstColumn, firstStride,
                              originalDiagonal, fullOrder, retained);
    }
    if constexpr (WHOLE)
    {
        WholeUBFactor(single, array, n, lda, upper, info, batch, offset, firstColumn, firstStride);
    }
    else if (firstColumn && !offset)
    {
        if (batch < BATCH_THRESHOLD)
        {
            asc_vf_call<FactorDiagonalVF<WARP_SIZE, THREADS>>(dim3(THREADS), single, array, n, lda, upper, info, batch,
                                                              offset, firstColumn, firstStride);
        }
        else
        {
            DispatchSmallFactor<WARP_SIZE, true>(single, array, n, lda, upper, info, batch, offset, firstColumn,
                                                 firstStride);
        }
    }
    else if (batch < BATCH_THRESHOLD && !upper)
    {
        FactorLower(single, array, n, lda, info, batch, offset);
    }
    else if (batch < BATCH_THRESHOLD)
    {
        asc_vf_call<FactorDiagonalVF<WARP_SIZE, THREADS>>(dim3(THREADS), single, array, n, lda, upper, info, batch,
                                                          offset, nullptr, 0);
    }
    else if (n <= WARP_SIZE / 2)
    {
        DispatchSmallFactor<WARP_SIZE / 2>(single, array, n, lda, upper, info, batch, offset, nullptr, 0);
    }
    else
    {
        DispatchSmallFactor<WARP_SIZE>(single, array, n, lda, upper, info, batch, offset, nullptr, 0);
    }
}
}  // namespace
void LaunchSmallFactor(aclrtStream stream, bool whole, int blocks, aclsolverFloatComplex* a, const uint64_t* array,
                       int n, int lda, bool upper, int* info, int batch, int offset, aclsolverFloatComplex* firstColumn,
                       int firstStride, float* originalDiagonal, int fullOrder)
{
    bool retained = UseFactorDiagonalState(batch);
    if (whole)
    {
        FactorSmall<true><<<blocks, WHOLE_UB_BYTES, stream>>>(a, array, n, lda, upper, info, batch, offset, firstColumn,
                                                              firstStride, originalDiagonal, fullOrder, retained);
    }
    else
    {
        FactorSmall<false><<<blocks, nullptr, stream>>>(a, array, n, lda, upper, info, batch, offset, firstColumn,
                                                        firstStride, originalDiagonal, fullOrder, retained);
    }
}
}  // namespace Cholesky
