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
#include <cmath>

#include "cholesky.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/vector_functions.h"

namespace Cholesky
{
struct FloatPair
{
    float high;
    float low;
};
__simt_callee__ inline FloatPair PairSum(float a, float b)
{
    float high = a + b, part = high - a;
    return {high, (a - (high - part)) + (b - part)};
}
__simt_callee__ inline void PairAddProduct(FloatPair& value, float a, float b)
{
    float product = a * b, remainder = fmaf(a, b, -product);
    auto sum = PairSum(value.high, product);
    value = PairSum(sum.high, (value.low + remainder) + sum.low);
}
__simt_callee__ inline aclsolverFloatComplex DividePair(aclsolverFloatComplex high, aclsolverFloatComplex& low,
                                                        float diagonal)
{
    if (diagonal == 0)
    {
        low = {NAN, NAN};
        return low;
    }
    float re = high.real / diagonal, im = high.imag / diagonal;
    auto real = PairSum(re, (fmaf(-re, diagonal, high.real) + low.real) / diagonal);
    auto imag = PairSum(im, (fmaf(-im, diagonal, high.imag) + low.imag) / diagonal);
    low = {real.low, imag.low};
    return {real.high, imag.high};
}
__simt_callee__ inline aclsolverFloatComplex SubPairProduct(aclsolverFloatComplex high, aclsolverFloatComplex& low,
                                                            aclsolverFloatComplex coefficient,
                                                            aclsolverFloatComplex pivot, aclsolverFloatComplex pivotLow)
{
    FloatPair real{high.real, low.real}, imag{high.imag, low.imag};
    PairAddProduct(real, -coefficient.real, pivot.real);
    PairAddProduct(real, coefficient.imag, pivot.imag);
    PairAddProduct(imag, -coefficient.real, pivot.imag);
    PairAddProduct(imag, -coefficient.imag, pivot.real);
    real.low = fmaf(-coefficient.real, pivotLow.real, real.low);
    real.low = fmaf(coefficient.imag, pivotLow.imag, real.low);
    imag.low = fmaf(-coefficient.real, pivotLow.imag, imag.low);
    imag.low = fmaf(-coefficient.imag, pivotLow.real, imag.low);
    real = PairSum(real.high, real.low);
    imag = PairSum(imag.high, imag.low);
    low = {real.low, imag.low};
    return {real.high, imag.high};
}
__simt_callee__ inline aclsolverFloatComplex SubProduct(aclsolverFloatComplex a, aclsolverFloatComplex b,
                                                        aclsolverFloatComplex c)
{
    a.real = fmaf(-b.real, c.real, a.real);
    a.real = fmaf(b.imag, c.imag, a.real);
    a.imag = fmaf(-b.real, c.imag, a.imag);
    a.imag = fmaf(-b.imag, c.real, a.imag);
    return a;
}
// Retain the first column's rounding remainder before the large rank-one cancellation.
// Subsequent panels use the existing FP32 Cube update on the smaller Schur complement.
__simt_callee__ inline float RootRemainder(float input, float root)
{
    return root > 0 && root < FINITE_ROOT_BOUND ? fmaf(-root, root, input) / (root + root) : 0;
}
__simt_callee__ inline float DivisionRemainder(float input, float quotient, float root, float rootLow)
{
    return root > 0 && root < FINITE_ROOT_BOUND ? (fmaf(-quotient, root, input) - quotient * rootLow) / root : 0;
}
__simt_callee__ inline aclsolverFloatComplex NormalizeFirstColumn(aclsolverFloatComplex input, float root,
                                                                  float rootLow, aclsolverFloatComplex& low)
{
    if (root == 0)
    {
        low = {0, 0};
        return {NAN, NAN};
    }
    aclsolverFloatComplex value{input.real / root, input.imag / root};
    low = {DivisionRemainder(input.real, value.real, root, rootLow),
           DivisionRemainder(input.imag, value.imag, root, rootLow)};
    return value;
}
__simt_callee__ inline aclsolverFloatComplex SubFirstColumn(aclsolverFloatComplex a, aclsolverFloatComplex left,
                                                            aclsolverFloatComplex right, aclsolverFloatComplex leftLow,
                                                            aclsolverFloatComplex rightLow)
{
    right.imag = -right.imag;
    rightLow.imag = -rightLow.imag;
    a = SubProduct(a, left, right);
    a = SubProduct(a, leftLow, right);
    a = SubProduct(a, left, rightLow);
    return SubProduct(a, leftLow, rightLow);
}
__simt_callee__ inline aclsolverFloatComplex LoadFirstColumn(const __gm__ aclsolverFloatComplex* a, int lda, int row,
                                                             bool upper)
{
    int64_t index = upper ? static_cast<int64_t>(row) * lda : row;
    auto value = asc_ldca(reinterpret_cast<__gm__ float2*>(const_cast<__gm__ aclsolverFloatComplex*>(a) + index));
    return {value.x, upper ? -value.y : value.y};
}
// Reuse first-column row vectors across all physical columns of a Schur rectangle.
template <bool LOW_ROWS, bool LOW_COLUMNS, typename R, typename M>
__simd_callee__ inline void SchurFirstCorrection(R& left, R& right, R& highReal, R& highImag, R& lowReal, R& lowImag,
                                                 __ubuf__ float* firstColumns, uint16_t col, M& mask)
{
    R cr, ci, negativeCr, negativeCi;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
        cr, firstColumns + (LOW_COLUMNS ? SCHUR_COLUMN_LOW_RELATIVE : 0) + 2 * col);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
        ci, firstColumns + (LOW_COLUMNS ? SCHUR_COLUMN_LOW_RELATIVE : 0) + 2 * col + 1);
    AscendC::Reg::Neg(negativeCr, cr, mask);
    AscendC::Reg::Neg(negativeCi, ci, mask);
    if constexpr (LOW_ROWS)
    {
        AscendC::Reg::MulAddDst(left, lowReal, negativeCr, mask);
        AscendC::Reg::MulAddDst(left, lowImag, negativeCi, mask);
        AscendC::Reg::MulAddDst(right, lowReal, ci, mask);
        AscendC::Reg::MulAddDst(right, lowImag, negativeCr, mask);
    }
    else
    {
        AscendC::Reg::MulAddDst(left, highReal, negativeCr, mask);
        AscendC::Reg::MulAddDst(left, highImag, negativeCi, mask);
        AscendC::Reg::MulAddDst(right, highReal, ci, mask);
        AscendC::Reg::MulAddDst(right, highImag, negativeCr, mask);
    }
}

template <typename R>
__simd_callee__ inline void LoadComplexColumn(R& real, R& imag, __ubuf__ float* matrix, uint32_t column,
                                              uint32_t columns, uint32_t stride)
{
    if (column < columns)
    {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(real, imag, matrix + column * stride);
    }
    else
    {
        AscendC::Reg::Duplicate(real, 0.0f);
        AscendC::Reg::Duplicate(imag, 0.0f);
    }
}

// Each AIV updates its physical columns with all SIMD_FLOAT_LANES FP32 vector lanes.
template <typename R, typename M>
__simd_callee__ inline void MaskSchurDiagonal(R& right, M& storeComponents, M& componentMask, uint16_t col,
                                              uint32_t columnOffset, uint32_t base, uint32_t count, bool upper)
{
    uint32_t column = col + columnOffset;
    int boundary = static_cast<int>(column) - static_cast<int>(base) + (upper ? 1 : 0);
    uint32_t prefixCount = boundary <= 0                             ? 0
                           : static_cast<uint32_t>(boundary) < count ? static_cast<uint32_t>(boundary)
                                                                     : count;
    auto prefix = AscendC::Reg::UpdateMask<float>(prefixCount);
    if (upper)
    {
        storeComponents = prefix;
    }
    else
    {
        AscendC::Reg::Not(storeComponents, prefix, componentMask);
    }
    if (column >= base && column < base + count)
    {
        uint32_t before = column - base, through = before + 1;
        auto beforeMask = AscendC::Reg::UpdateMask<float>(before);
        auto throughMask = AscendC::Reg::UpdateMask<float>(through);
        AscendC::Reg::MaskReg diagonalMask;
        AscendC::Reg::Xor(diagonalMask, beforeMask, throughMask, componentMask);
        AscendC::Reg::RegTensor<float> zero;
        AscendC::Reg::Duplicate(zero, 0.0f);
        AscendC::Reg::Select(right, zero, right, diagonalMask);
    }
}

template <typename R, typename M>
__simd_callee__ inline void CorrectSchurColumn(R& left, R& right, R& highReal, R& highImag, R& lowReal, R& lowImag,
                                               __ubuf__ float* firstColumns, uint16_t col, M& componentMask)
{
    SchurFirstCorrection<false, false>(left, right, highReal, highImag, lowReal, lowImag, firstColumns, col,
                                       componentMask);
    SchurFirstCorrection<true, false>(left, right, highReal, highImag, lowReal, lowImag, firstColumns, col,
                                      componentMask);
    SchurFirstCorrection<false, true>(left, right, highReal, highImag, lowReal, lowImag, firstColumns, col,
                                      componentMask);
    SchurFirstCorrection<true, true>(left, right, highReal, highImag, lowReal, lowImag, firstColumns, col,
                                     componentMask);
}

template <uint32_t TILE = SCHUR_TILE>
__simd_vf__ inline void ApplySchurRectangleVF(__ubuf__ float* matrix, __ubuf__ float* real, __ubuf__ float* imag,
                                              uint32_t rows, uint16_t columns, bool diagonal, bool upper,
                                              uint32_t columnOffset, __ubuf__ float* firstRows,
                                              __ubuf__ float* firstColumns, bool compensated)
{
    for (uint16_t part = 0; part < TILE / SIMD_FLOAT_LANES && SIMD_FLOAT_LANES * part < rows; ++part)
    {
        uint32_t base = SIMD_FLOAT_LANES * part,
                 count = rows - base < SIMD_FLOAT_LANES ? rows - base : SIMD_FLOAT_LANES;
        uint32_t componentCount = count;
        auto componentMask = AscendC::Reg::UpdateMask<float>(componentCount);
        AscendC::Reg::RegTensor<float> highReal, highImag, lowReal, lowImag;
        if (compensated)
        {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(highReal, highImag,
                                                                                    firstRows + 2 * base);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
                lowReal, lowImag, firstRows + SCHUR_ROW_LOW + 2 * base);
        }
        for (uint16_t col = 0; col < columns; ++col)
        {
            AscendC::Reg::RegTensor<float> left, right, v0, v1, rr, ir;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
                left, right, matrix + col * (2 * TILE) + 2 * base);
            if (compensated)
            {
                CorrectSchurColumn(left, right, highReal, highImag, lowReal, lowImag, firstColumns, col, componentMask);
            }
            AscendC::Reg::LoadAlign(rr, real + col * TILE + base);
            AscendC::Reg::LoadAlign(ir, imag + col * TILE + base);
            AscendC::Reg::Sub(left, left, rr, componentMask);
            AscendC::Reg::Add(right, right, ir, componentMask);
            auto storeComponents = componentMask;
            if (diagonal)
            {
                MaskSchurDiagonal(right, storeComponents, componentMask, col, columnOffset, base, count, upper);
            }
            AscendC::Reg::Interleave(v0, v1, left, right);
            AscendC::Reg::MaskReg storeFirst, storeSecond;
            AscendC::Reg::Interleave<float>(storeFirst, storeSecond, storeComponents, storeComponents);
            AscendC::Reg::StoreAlign(matrix + col * (2 * TILE) + 2 * base, v0, storeFirst);
            AscendC::Reg::StoreAlign(matrix + col * (2 * TILE) + 2 * base + SIMD_FLOAT_LANES, v1, storeSecond);
        }
    }
}
}  // namespace Cholesky
