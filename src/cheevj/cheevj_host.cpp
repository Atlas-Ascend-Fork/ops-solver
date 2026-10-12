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
 * \file cheevj_host.cpp
 * \brief Host launcher for the complex Hermitian Jacobi eigensolver.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

#include "../utils/assert.h"
#include "../utils/gm_addr.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "cheevj_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"

// Keep Cheevj independent of the repository-wide CHECK_ACLRT macro shape.
// The upstream helper gained a caller-supplied cleanup argument after this
// implementation was developed; a private wrapper avoids changing the error
// contract of every other solver while the Cheevj paths are normalized.
// CHEEVJ_CHECK_ACLRT runs the caller-supplied cleanup before returning so
// device buffers acquired before the failure are released (issue #121).
#define CHEEVJ_CHECK_ACLRT(function, cleanup_action)                                   \
    do                                                                                 \
    {                                                                                  \
        const aclError cheevjAclStatus = (function);                                   \
        if (cheevjAclStatus != ACL_SUCCESS)                                            \
        {                                                                              \
            std::cerr << "Cheevj ACL runtime error at " << __FILE__ << ':' << __LINE__ \
                      << " (error code: " << cheevjAclStatus << ')' << std::endl;      \
            cleanup_action;                                                            \
            return cheevjAclStatus;                                                    \
        }                                                                              \
    } while (0)

namespace
{

using Complex = std::complex<float>;
using DoubleComplex = std::complex<double>;

static constexpr int64_t BASE_BLOCK_ELENUM = 16;
static constexpr int64_t COL_ALIGNED_ELENUM = 128;
static constexpr int64_t CHEEVJ_BLOCK_SIZE = 32;
static constexpr int64_t CHEEVJ_TILE_M = 512;
static constexpr int64_t CHEEVJ_WORKSPACE_PLANES_N = 6;
static constexpr int64_t CHEEVJ_WORKSPACE_PLANES_V = 10;
static constexpr int64_t CHEEVJ_MAX_SWEEPS = 32;
static constexpr int64_t CHEEVJ_FIXED_N512 = 512;
static constexpr int64_t CHEEVJ_FIXED_N1024 = 1024;
static constexpr int64_t CHEEVJ_FIXED_N2048 = 2048;
static constexpr int64_t CHEEVJ_FIXED_PANEL_WIDTH = 32;
static constexpr int64_t CHEEVJ_FIXED_PANEL_SIZE = 16;
static constexpr int64_t CHEEVJ_FIXED_PANEL_PARTICIPANTS = 33;
// host 回退路径 OpenMP 并行阈值/线程数：低于对应阈值时并行开销大于收益，
// 退化为串行；线程数按典型开发环境核数设置。
static constexpr int64_t HOST_OMP_PARALLEL_MIN_N = 256;
static constexpr int64_t HOST_OMP_PARALLEL_MIN_ACTIVE = 128;
static constexpr int HOST_OMP_THREAD_NUM = 16;

static constexpr int ORTHOGONALITY_PROBE_COUNT = 16;

struct CheevjTilingData
{
    uint32_t n;
    uint32_t lda;
    uint32_t jobz;
    uint32_t uplo;
    uint32_t isDiagonal;
    uint32_t strideN;
    uint32_t workM;
    uint32_t blockSize;
    uint32_t tileM;
    uint32_t workspacePlanes;
    uint32_t maxSweeps;
};

bool IsVectorMode(int32_t jobz)
{
    return jobz == ACLSOLVER_EIG_MODE_VECTOR || jobz == static_cast<int32_t>('V') || jobz == static_cast<int32_t>('v');
}

bool IsNoVectorMode(int32_t jobz)
{
    return jobz == ACLSOLVER_EIG_MODE_NOVECTOR || jobz == static_cast<int32_t>('N') ||
           jobz == static_cast<int32_t>('n');
}

bool IsLowerMode(int32_t uplo)
{
    // 仅接受公开枚举与 'L'/'l'：不引入 121/122（PLASMA/MAGMA 数值编码）——
    // 若误收易与 'y'/'z' 字符值混淆，且历史代码 122=Lower 的映射与 PLASMA/
    // MAGMA（121=Lower/122=Upper）方向相反，遵循该惯例的调用方会被按错误
    // 三角形读取而静默出错（issue #155）。如需兼容该编码须整体修正映射并
    // 在公开头文件文档化，当前不引入。
    return uplo == ACLSOLVER_FILL_MODE_LOWER || uplo == static_cast<int32_t>('L') || uplo == static_cast<int32_t>('l');
}

bool IsUpperMode(int32_t uplo)
{
    return uplo == ACLSOLVER_FILL_MODE_UPPER || uplo == static_cast<int32_t>('U') || uplo == static_cast<int32_t>('u');
}

int64_t AlignUp(int64_t value, int64_t align) { return align > 0 ? (value + align - 1) / align * align : value; }

int64_t CompactColumnMajorIndex(int64_t row, int64_t col, int64_t n) { return row + col * n; }

bool IsZeroRange(const Complex* values, size_t count)
{
    static constexpr size_t zeroChunkSize = 256;
    static const std::array<Complex, zeroChunkSize> zeroValues{};
    while (count != 0)
    {
        const size_t chunk = std::min(count, zeroChunkSize);
        if (std::memcmp(values, zeroValues.data(), chunk * sizeof(Complex)) != 0)
        {
            return false;
        }
        values += chunk;
        count -= chunk;
    }
    return true;
}

bool CheckedSquareSize(int64_t n, size_t elementBytes, size_t* elements, size_t* bytes)
{
    if (elementBytes == 0)
    {
        return false;
    }
    const size_t order = static_cast<size_t>(n);
    if (order != 0 && order > std::numeric_limits<size_t>::max() / order)
    {
        return false;
    }
    *elements = order * order;
    if (*elements > std::numeric_limits<size_t>::max() / elementBytes)
    {
        return false;
    }
    *bytes = *elements * elementBytes;
    return true;
}

bool IsFixedShape(int64_t n) { return n == CHEEVJ_FIXED_N512 || n == CHEEVJ_FIXED_N1024 || n == CHEEVJ_FIXED_N2048; }

int64_t NextFixedShape(int64_t n)
{
    if (n < CHEEVJ_FIXED_N512)
    {
        return CHEEVJ_FIXED_N512;
    }
    if (n < CHEEVJ_FIXED_N1024)
    {
        return CHEEVJ_FIXED_N1024;
    }
    if (n < CHEEVJ_FIXED_N2048)
    {
        return CHEEVJ_FIXED_N2048;
    }
    return 0;
}

aclError InvalidParam(int32_t* info, int32_t infoValue, const char* message)
{
    if (info != nullptr)
    {
        *info = infoValue;
    }
    LOG_PRINT("Cheevj invalid parameter: %s\n", message);
    return ACL_ERROR_INVALID_PARAM;
}

Complex LoadHermitianValue(const Complex* a, int64_t lda, int64_t row, int64_t col, bool lower)
{
    if (row == col)
    {
        return Complex(a[row + col * lda].real(), 0.0f);
    }
    if (lower)
    {
        return row > col ? a[row + col * lda] : std::conj(a[col + row * lda]);
    }
    return row < col ? a[row + col * lda] : std::conj(a[col + row * lda]);
}

void PackFullMatrix(const Complex* a, int64_t lda, int64_t n, int32_t uplo, std::vector<Complex>* packed)
{
    packed->assign(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    const bool lower = IsLowerMode(uplo);
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            (*packed)[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))] =
                LoadHermitianValue(a, lda, row, col, lower);
        }
    }
}

bool IsInputDiagonal(const Complex* a, int64_t lda, int64_t n, bool lower)
{
    if (n >= 1024)
    {
        int isDiagonal = 1;
#pragma omp parallel for num_threads(HOST_OMP_THREAD_NUM) schedule(static) reduction(& : isDiagonal)
        for (int64_t col = 0; col < n; ++col)
        {
            const int64_t rowBegin = lower ? col + 1 : 0;
            const int64_t rowEnd = lower ? n : col;
            const size_t count = static_cast<size_t>(rowEnd - rowBegin);
            if (!IsZeroRange(a + rowBegin + col * lda, count))
            {
                isDiagonal = 0;
            }
        }
        return isDiagonal != 0;
    }
    for (int64_t col = 0; col < n; ++col)
    {
        const int64_t rowBegin = lower ? col + 1 : 0;
        const int64_t rowEnd = lower ? n : col;
        const size_t count = static_cast<size_t>(rowEnd - rowBegin);
        if (!IsZeroRange(a + rowBegin + col * lda, count))
        {
            return false;
        }
    }
    return true;
}

bool IsInputTwoByTwoBlockDiagonal(const Complex* a, int64_t lda, int64_t n, bool lower)
{
    if (n >= 1024)
    {
        int isBlockDiagonal = 1;
#pragma omp parallel for num_threads(HOST_OMP_THREAD_NUM) schedule(static) reduction(& : isBlockDiagonal)
        for (int64_t col = 0; col < n; ++col)
        {
            const bool hasStoredBlockMate = lower ? (col % 2 == 0 && col + 1 < n) : (col % 2 != 0);
            const int64_t rowBegin = lower ? col + 1 + static_cast<int64_t>(hasStoredBlockMate) : 0;
            const int64_t rowEnd = lower ? n : col - static_cast<int64_t>(hasStoredBlockMate);
            const size_t count = static_cast<size_t>(rowEnd - rowBegin);
            if (!IsZeroRange(a + rowBegin + col * lda, count))
            {
                isBlockDiagonal = 0;
            }
        }
        return isBlockDiagonal != 0;
    }
    for (int64_t col = 0; col < n; ++col)
    {
        const bool hasStoredBlockMate = lower ? (col % 2 == 0 && col + 1 < n) : (col % 2 != 0);
        const int64_t rowBegin = lower ? col + 1 + static_cast<int64_t>(hasStoredBlockMate) : 0;
        const int64_t rowEnd = lower ? n : col - static_cast<int64_t>(hasStoredBlockMate);
        const size_t count = static_cast<size_t>(rowEnd - rowBegin);
        if (!IsZeroRange(a + rowBegin + col * lda, count))
        {
            return false;
        }
    }
    return true;
}

bool IsInputHermitianTridiagonal(const Complex* a, int64_t lda, int64_t n, bool lower)
{
    if (n < 3)
    {
        return true;
    }
    for (int64_t col = 0; col < n; ++col)
    {
        const int64_t rowBegin = lower ? col + 2 : 0;
        const int64_t rowEnd = lower ? n : std::max<int64_t>(0, col - 1);
        for (int64_t row = rowBegin; row < rowEnd; ++row)
        {
            if (a[row + col * lda] != Complex(0.0f, 0.0f))
            {
                return false;
            }
        }
    }
    return true;
}

float MaximumInputMagnitude(const Complex* a, int64_t lda, int64_t n, bool lower)
{
    // Scan the uplo-selected triangle instead of the diagonal only
    // (issue #176): a Hermitian tridiagonal matrix with zero diagonal and
    // huge off-diagonals (e.g. sub/super diagonals 1e20) has diagonal
    // magnitude 0, so the old diagonal-only scan missed the wide-scale
    // host fallback and the fixed backend silently returned NaN spectra.
    float maximum = 0.0f;
    for (int64_t col = 0; col < n; ++col)
    {
        const int64_t rowBegin = lower ? col : 0;
        const int64_t rowEnd = lower ? n : col + 1;
        for (int64_t row = rowBegin; row < rowEnd; ++row)
        {
            const Complex value = a[row + col * lda];
            const float magnitude = row == col ? std::abs(value.real()) : std::abs(value);
            maximum = std::max(maximum, magnitude);
        }
    }
    return maximum;
}

void ClearInputMatrix(Complex* a, int64_t lda, int64_t n)
{
    if (n >= 1024)
    {
#pragma omp parallel for num_threads(HOST_OMP_THREAD_NUM) schedule(static)
        for (int64_t column = 0; column < n; ++column)
        {
            std::fill_n(a + column * lda, n, Complex(0.0f, 0.0f));
        }
        return;
    }
    if (lda == n)
    {
        std::fill_n(a, n * n, Complex(0.0f, 0.0f));
        return;
    }
    for (int64_t column = 0; column < n; ++column)
    {
        std::fill_n(a + column * lda, n, Complex(0.0f, 0.0f));
    }
}

void SolveInputDiagonal(Complex* a, int64_t lda, int64_t n, bool computeVectors, float* w)
{
    bool alreadySorted = true;
    for (int64_t index = 1; index < n; ++index)
    {
        alreadySorted = alreadySorted && a[index - 1 + (index - 1) * lda].real() <= a[index + index * lda].real();
    }
    if (alreadySorted)
    {
        for (int64_t index = 0; index < n; ++index)
        {
            w[index] = a[index + index * lda].real();
        }
        if (computeVectors)
        {
            ClearInputMatrix(a, lda, n);
            for (int64_t index = 0; index < n; ++index)
            {
                a[index + index * lda] = Complex(1.0f, 0.0f);
            }
        }
        return;
    }

    std::vector<int64_t> permutation(static_cast<size_t>(n));
    for (int64_t index = 0; index < n; ++index)
    {
        permutation[static_cast<size_t>(index)] = index;
    }
    std::stable_sort(permutation.begin(), permutation.end(), [a, lda](int64_t lhs, int64_t rhs)
                     { return a[lhs + lhs * lda].real() < a[rhs + rhs * lda].real(); });
    for (int64_t column = 0; column < n; ++column)
    {
        const int64_t source = permutation[static_cast<size_t>(column)];
        w[column] = a[source + source * lda].real();
    }
    if (!computeVectors)
    {
        return;
    }
    ClearInputMatrix(a, lda, n);
    for (int64_t column = 0; column < n; ++column)
    {
        a[permutation[static_cast<size_t>(column)] + column * lda] = Complex(1.0f, 0.0f);
    }
}

template <int64_t blockSize>
bool IsCompactBlockDiagonal(const std::vector<Complex>& packed, int64_t n)
{
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            if (row / blockSize == col / blockSize)
            {
                continue;
            }
            if (packed[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))] != Complex(0.0f, 0.0f))
            {
                return false;
            }
        }
    }
    return true;
}

bool IsCompactDiagonal(const std::vector<Complex>& packed, int64_t n) { return IsCompactBlockDiagonal<1>(packed, n); }

bool IsCompactTwoByTwoBlockDiagonal(const std::vector<Complex>& packed, int64_t n)
{
    return IsCompactBlockDiagonal<2>(packed, n);
}

void SolveCompactDiagonal(std::vector<Complex>* packed, int64_t n, bool computeVectors, float* w)
{
    std::vector<int64_t> permutation(static_cast<size_t>(n));
    for (int64_t index = 0; index < n; ++index)
    {
        permutation[static_cast<size_t>(index)] = index;
    }
    std::stable_sort(permutation.begin(), permutation.end(),
                     [packed, n](int64_t lhs, int64_t rhs)
                     {
                         return (*packed)[static_cast<size_t>(CompactColumnMajorIndex(lhs, lhs, n))].real() <
                                (*packed)[static_cast<size_t>(CompactColumnMajorIndex(rhs, rhs, n))].real();
                     });
    for (int64_t column = 0; column < n; ++column)
    {
        const int64_t source = permutation[static_cast<size_t>(column)];
        w[column] = (*packed)[static_cast<size_t>(CompactColumnMajorIndex(source, source, n))].real();
    }
    if (!computeVectors)
    {
        return;
    }
    packed->assign(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    for (int64_t column = 0; column < n; ++column)
    {
        const int64_t row = permutation[static_cast<size_t>(column)];
        (*packed)[static_cast<size_t>(CompactColumnMajorIndex(row, column, n))] = Complex(1.0f, 0.0f);
    }
}

struct CompactEigenpair
{
    float value;
    int64_t firstRow;
    int64_t secondRow;
    Complex firstValue;
    Complex secondValue;
};

inline void SortCompactEigenpairs(std::vector<CompactEigenpair>* eigenpairs, int64_t n, float* w)
{
    std::stable_sort(eigenpairs->begin(), eigenpairs->end(),
                     [](const CompactEigenpair& lhs, const CompactEigenpair& rhs) { return lhs.value < rhs.value; });
    for (int64_t column = 0; column < n; ++column)
    {
        w[column] = (*eigenpairs)[static_cast<size_t>(column)].value;
    }
}

void AppendTwoByTwoEigenpairs(std::vector<CompactEigenpair>* eigenpairs, int64_t first, int64_t second, float diagonal0,
                              float diagonal1, Complex upper)
{
    const double center = 0.5 * (static_cast<double>(diagonal0) + static_cast<double>(diagonal1));
    const double halfDifference = 0.5 * (static_cast<double>(diagonal0) - static_cast<double>(diagonal1));
    const double radius = std::hypot(halfDifference, std::abs(static_cast<std::complex<double>>(upper)));
    const float low = static_cast<float>(center - radius);
    const float high = static_cast<float>(center + radius);

    if (upper == Complex(0.0f, 0.0f))
    {
        const bool firstIsLow = diagonal0 <= diagonal1;
        const Complex low0 = firstIsLow ? Complex(1.0f, 0.0f) : Complex(0.0f, 0.0f);
        const Complex low1 = firstIsLow ? Complex(0.0f, 0.0f) : Complex(1.0f, 0.0f);
        eigenpairs->push_back({low, first, second, low0, low1});
        eigenpairs->push_back({high, first, second, -std::conj(low1), std::conj(low0)});
        return;
    }

    Complex low0 = upper;
    Complex low1(low - diagonal0, 0.0f);
    const Complex alternate0(low - diagonal1, 0.0f);
    const Complex alternate1 = std::conj(upper);
    const auto squaredNorm = [](Complex value) { return std::norm(static_cast<std::complex<double>>(value)); };
    if (squaredNorm(alternate0) + squaredNorm(alternate1) > squaredNorm(low0) + squaredNorm(low1))
    {
        low0 = alternate0;
        low1 = alternate1;
    }
    const double norm = std::sqrt(squaredNorm(low0) + squaredNorm(low1));
    low0 = norm > 0.0 ? Complex(static_cast<float>(low0.real() / norm), static_cast<float>(low0.imag() / norm))
                      : Complex(0.0f, 0.0f);
    low1 = norm > 0.0 ? Complex(static_cast<float>(low1.real() / norm), static_cast<float>(low1.imag() / norm))
                      : Complex(0.0f, 0.0f);
    eigenpairs->push_back({low, first, second, low0, low1});
    // The orthogonal complement is the second eigenvector.
    eigenpairs->push_back({high, first, second, -std::conj(low1), std::conj(low0)});
}

void SolveInputTwoByTwoBlockDiagonal(Complex* a, int64_t lda, int64_t n, bool lower, bool computeVectors, float* w)
{
    std::vector<CompactEigenpair> eigenpairs;
    eigenpairs.reserve(static_cast<size_t>(n));
    for (int64_t first = 0; first < n; first += 2)
    {
        if (first + 1 == n)
        {
            eigenpairs.push_back(
                {a[first + first * lda].real(), first, first, Complex(1.0f, 0.0f), Complex(0.0f, 0.0f)});
            continue;
        }

        const int64_t second = first + 1;
        AppendTwoByTwoEigenpairs(&eigenpairs, first, second, a[first + first * lda].real(),
                                 a[second + second * lda].real(), LoadHermitianValue(a, lda, first, second, lower));
    }

    SortCompactEigenpairs(&eigenpairs, n, w);
    if (!computeVectors)
    {
        return;
    }
    ClearInputMatrix(a, lda, n);
    for (int64_t column = 0; column < n; ++column)
    {
        const CompactEigenpair& pair = eigenpairs[static_cast<size_t>(column)];
        a[pair.firstRow + column * lda] = pair.firstValue;
        if (pair.secondRow != pair.firstRow)
        {
            a[pair.secondRow + column * lda] = pair.secondValue;
        }
    }
}

void SolveCompactTwoByTwoBlockDiagonal(std::vector<Complex>* packed, int64_t n, bool computeVectors, float* w)
{
    std::vector<CompactEigenpair> eigenpairs;
    eigenpairs.reserve(static_cast<size_t>(n));
    for (int64_t first = 0; first < n; first += 2)
    {
        if (first + 1 == n)
        {
            const float value = (*packed)[static_cast<size_t>(CompactColumnMajorIndex(first, first, n))].real();
            eigenpairs.push_back({value, first, first, Complex(1.0f, 0.0f), Complex(0.0f, 0.0f)});
            continue;
        }

        const int64_t second = first + 1;
        AppendTwoByTwoEigenpairs(&eigenpairs, first, second,
                                 (*packed)[static_cast<size_t>(CompactColumnMajorIndex(first, first, n))].real(),
                                 (*packed)[static_cast<size_t>(CompactColumnMajorIndex(second, second, n))].real(),
                                 (*packed)[static_cast<size_t>(CompactColumnMajorIndex(first, second, n))]);
    }

    SortCompactEigenpairs(&eigenpairs, n, w);
    if (!computeVectors)
    {
        return;
    }
    packed->assign(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    for (int64_t column = 0; column < n; ++column)
    {
        const CompactEigenpair& pair = eigenpairs[static_cast<size_t>(column)];
        (*packed)[static_cast<size_t>(CompactColumnMajorIndex(pair.firstRow, column, n))] = pair.firstValue;
        if (pair.secondRow != pair.firstRow)
        {
            (*packed)[static_cast<size_t>(CompactColumnMajorIndex(pair.secondRow, column, n))] = pair.secondValue;
        }
    }
}

void ApplyJacobiMatrixRotation(std::vector<DoubleComplex>* work, int64_t n, int64_t p, int64_t q, double cosine,
                               DoubleComplex sine, double app, double aqq, double tangent, double magnitude)
{
    for (int64_t row = 0; row < n; ++row)
    {
        if (row == p || row == q)
        {
            continue;
        }
        const size_t rowP = static_cast<size_t>(CompactColumnMajorIndex(row, p, n));
        const size_t rowQ = static_cast<size_t>(CompactColumnMajorIndex(row, q, n));
        const DoubleComplex oldP = (*work)[rowP];
        const DoubleComplex oldQ = (*work)[rowQ];
        const DoubleComplex newP = cosine * oldP - std::conj(sine) * oldQ;
        const DoubleComplex newQ = sine * oldP + cosine * oldQ;
        (*work)[rowP] = newP;
        (*work)[static_cast<size_t>(CompactColumnMajorIndex(p, row, n))] = std::conj(newP);
        (*work)[rowQ] = newQ;
        (*work)[static_cast<size_t>(CompactColumnMajorIndex(q, row, n))] = std::conj(newQ);
    }
    const size_t pp = static_cast<size_t>(CompactColumnMajorIndex(p, p, n));
    const size_t qq = static_cast<size_t>(CompactColumnMajorIndex(q, q, n));
    const size_t pq = static_cast<size_t>(CompactColumnMajorIndex(p, q, n));
    const size_t qp = static_cast<size_t>(CompactColumnMajorIndex(q, p, n));
    const double delta = tangent * magnitude;
    (*work)[pp] = DoubleComplex(app - delta, 0.0);
    (*work)[qq] = DoubleComplex(aqq + delta, 0.0);
    (*work)[pq] = DoubleComplex(0.0, 0.0);
    (*work)[qp] = DoubleComplex(0.0, 0.0);
}

void ApplyJacobiVectorRotation(std::vector<DoubleComplex>* vectors, int64_t n, int64_t p, int64_t q, double cosine,
                               DoubleComplex sine)
{
    if (vectors == nullptr)
    {
        return;
    }
    for (int64_t row = 0; row < n; ++row)
    {
        const size_t rowP = static_cast<size_t>(CompactColumnMajorIndex(row, p, n));
        const size_t rowQ = static_cast<size_t>(CompactColumnMajorIndex(row, q, n));
        const DoubleComplex oldP = (*vectors)[rowP];
        const DoubleComplex oldQ = (*vectors)[rowQ];
        (*vectors)[rowP] = cosine * oldP - std::conj(sine) * oldQ;
        (*vectors)[rowQ] = sine * oldP + cosine * oldQ;
    }
}

double RunJacobiSweep(std::vector<DoubleComplex>* work, std::vector<DoubleComplex>* vectors, int64_t n,
                      double thresholdSquared)
{
    double maxOffDiagonalSquared = 0.0;
    for (int64_t p = 0; p + 1 < n; ++p)
    {
        for (int64_t q = p + 1; q < n; ++q)
        {
            const size_t pq = static_cast<size_t>(CompactColumnMajorIndex(p, q, n));
            const DoubleComplex offDiagonal = (*work)[pq];
            const double magnitudeSquared = std::norm(offDiagonal);
            maxOffDiagonalSquared = std::max(maxOffDiagonalSquared, magnitudeSquared);
            if (magnitudeSquared <= thresholdSquared)
            {
                continue;
            }
            const double magnitude = std::sqrt(magnitudeSquared);
            const size_t pp = static_cast<size_t>(CompactColumnMajorIndex(p, p, n));
            const size_t qq = static_cast<size_t>(CompactColumnMajorIndex(q, q, n));
            const double app = (*work)[pp].real();
            const double aqq = (*work)[qq].real();
            const double tau = (aqq - app) / (2.0 * magnitude);
            const double tangent = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
            const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
            const DoubleComplex sine =
                magnitude > 0.0 ? tangent * cosine * offDiagonal / magnitude : DoubleComplex(0.0, 0.0);
            ApplyJacobiMatrixRotation(work, n, p, q, cosine, sine, app, aqq, tangent, magnitude);
            ApplyJacobiVectorRotation(vectors, n, p, q, cosine, sine);
        }
    }
    return maxOffDiagonalSquared;
}

std::vector<int64_t> SortJacobiEigenpairs(const std::vector<DoubleComplex>& work, int64_t n, float* w)
{
    std::vector<int64_t> permutation(static_cast<size_t>(n));
    for (int64_t index = 0; index < n; ++index)
    {
        permutation[static_cast<size_t>(index)] = index;
    }
    std::stable_sort(permutation.begin(), permutation.end(),
                     [&work, n](int64_t lhs, int64_t rhs)
                     {
                         return work[static_cast<size_t>(CompactColumnMajorIndex(lhs, lhs, n))].real() <
                                work[static_cast<size_t>(CompactColumnMajorIndex(rhs, rhs, n))].real();
                     });
    for (int64_t column = 0; column < n; ++column)
    {
        const int64_t source = permutation[static_cast<size_t>(column)];
        w[column] = static_cast<float>(work[static_cast<size_t>(CompactColumnMajorIndex(source, source, n))].real());
    }
    return permutation;
}

void WriteJacobiEigenvectors(const std::vector<DoubleComplex>& vectors, const std::vector<int64_t>& permutation,
                             int64_t n, std::vector<Complex>* matrix)
{
    matrix->assign(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    for (int64_t column = 0; column < n; ++column)
    {
        const int64_t source = permutation[static_cast<size_t>(column)];
        for (int64_t row = 0; row < n; ++row)
        {
            const DoubleComplex value = vectors[static_cast<size_t>(CompactColumnMajorIndex(row, source, n))];
            (*matrix)[static_cast<size_t>(CompactColumnMajorIndex(row, column, n))] =
                Complex(static_cast<float>(value.real()), static_cast<float>(value.imag()));
        }
    }
}

bool SolveCompactHostJacobi(std::vector<Complex>* matrix, int64_t n, bool computeVectors, float* w)
{
    constexpr double relativeTolerance = 1.0e-12;
    constexpr int maxSweeps = 80;
    std::vector<DoubleComplex> work(static_cast<size_t>(n * n));
    // Keep the convergence threshold relative to A; a unit floor drops valid small spectra.
    double matrixScale = 0.0;
    for (size_t index = 0; index < work.size(); ++index)
    {
        work[index] = static_cast<DoubleComplex>((*matrix)[index]);
        matrixScale = std::max(matrixScale, std::abs(work[index]));
    }
    std::vector<DoubleComplex> vectors;
    if (computeVectors)
    {
        vectors.assign(static_cast<size_t>(n * n), DoubleComplex(0.0, 0.0));
        for (int64_t index = 0; index < n; ++index)
        {
            vectors[static_cast<size_t>(CompactColumnMajorIndex(index, index, n))] = DoubleComplex(1.0, 0.0);
        }
    }

    bool converged = false;
    for (int sweep = 0; sweep < maxSweeps; ++sweep)
    {
        const double thresholdSquared = relativeTolerance * relativeTolerance * matrixScale * matrixScale;
        const double maxOffDiagonalSquared =
            RunJacobiSweep(&work, computeVectors ? &vectors : nullptr, n, thresholdSquared);
        if (maxOffDiagonalSquared <= thresholdSquared)
        {
            converged = true;
            break;
        }
    }

    const std::vector<int64_t> permutation = SortJacobiEigenpairs(work, n, w);
    if (!computeVectors)
    {
        return converged;
    }
    WriteJacobiEigenvectors(vectors, permutation, n, matrix);
    return converged;
}

int64_t FindTridiagonalBlockEnd(const std::vector<double>& diagonal, const std::vector<double>& offDiagonal,
                                int64_t left, int64_t n, double epsilon)
{
    int64_t right = left;
    for (; right + 1 < n; ++right)
    {
        const double scale =
            std::abs(diagonal[static_cast<size_t>(right)]) + std::abs(diagonal[static_cast<size_t>(right + 1)]);
        if (std::abs(offDiagonal[static_cast<size_t>(right)]) <= epsilon * scale)
        {
            break;
        }
    }
    return right;
}

void ApplyTridiagonalRotation(std::vector<double>* vectors, int64_t n, int64_t index, double sine, double cosine)
{
    if (vectors == nullptr)
    {
        return;
    }
    for (int64_t row = 0; row < n; ++row)
    {
        const size_t first = static_cast<size_t>(row + index * n);
        const size_t second = static_cast<size_t>(row + (index + 1) * n);
        const double oldSecond = (*vectors)[second];
        (*vectors)[second] = sine * (*vectors)[first] + cosine * oldSecond;
        (*vectors)[first] = cosine * (*vectors)[first] - sine * oldSecond;
    }
}

void RunTridiagonalQlIteration(std::vector<double>* diagonal, std::vector<double>* offDiagonal,
                               std::vector<double>* vectors, int64_t n, int64_t left, int64_t right)
{
    double g = ((*diagonal)[static_cast<size_t>(left + 1)] - (*diagonal)[static_cast<size_t>(left)]) /
               (2.0 * (*offDiagonal)[static_cast<size_t>(left)]);
    double r = std::hypot(g, 1.0);
    g = (*diagonal)[static_cast<size_t>(right)] - (*diagonal)[static_cast<size_t>(left)] +
        (*offDiagonal)[static_cast<size_t>(left)] / (g + std::copysign(r, g));
    double sine = 1.0;
    double cosine = 1.0;
    double shift = 0.0;
    for (int64_t index = right; index-- > left;)
    {
        const double f = sine * (*offDiagonal)[static_cast<size_t>(index)];
        const double b = cosine * (*offDiagonal)[static_cast<size_t>(index)];
        if (std::abs(f) >= std::abs(g))
        {
            cosine = g / f;
            r = std::hypot(cosine, 1.0);
            (*offDiagonal)[static_cast<size_t>(index + 1)] = f * r;
            sine = r > 0.0 ? 1.0 / r : 1.0;
            cosine *= sine;
        }
        else
        {
            sine = f / g;
            r = std::hypot(sine, 1.0);
            (*offDiagonal)[static_cast<size_t>(index + 1)] = g * r;
            cosine = r > 0.0 ? 1.0 / r : 1.0;
            sine *= cosine;
        }
        g = (*diagonal)[static_cast<size_t>(index + 1)] - shift;
        r = ((*diagonal)[static_cast<size_t>(index)] - g) * sine + 2.0 * cosine * b;
        shift = sine * r;
        (*diagonal)[static_cast<size_t>(index + 1)] = g + shift;
        g = cosine * r - b;
        ApplyTridiagonalRotation(vectors, n, index, sine, cosine);
    }
    (*diagonal)[static_cast<size_t>(left)] -= shift;
    (*offDiagonal)[static_cast<size_t>(left)] = g;
    (*offDiagonal)[static_cast<size_t>(right)] = 0.0;
}

bool SolveRealSymmetricTridiagonal(std::vector<double>* diagonal, std::vector<double>* offDiagonal,
                                   std::vector<double>* vectors, int64_t n)
{
    constexpr int maximumIterations = 96;
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (int64_t left = 0; left < n; ++left)
    {
        for (int iterations = 0; iterations <= maximumIterations; ++iterations)
        {
            const int64_t right = FindTridiagonalBlockEnd(*diagonal, *offDiagonal, left, n, epsilon);
            if (right == left)
            {
                break;
            }
            if (iterations == maximumIterations)
            {
                return false;
            }
            RunTridiagonalQlIteration(diagonal, offDiagonal, vectors, n, left, right);
        }
    }
    return true;
}

bool BuildHouseholderReflector(const std::vector<DoubleComplex>& work, int64_t n, int64_t step,
                               std::vector<DoubleComplex>* reflector, DoubleComplex* alpha, double* beta)
{
    const int64_t begin = step + 1;
    const int64_t active = n - begin;
    reflector->resize(static_cast<size_t>(active));
    double normSquared = 0.0;
    for (int64_t index = 0; index < active; ++index)
    {
        const DoubleComplex value = work[static_cast<size_t>((begin + index) * n + step)];
        (*reflector)[static_cast<size_t>(index)] = value;
        normSquared += std::norm(value);
    }
    const double norm = std::sqrt(normSquared);
    if (norm == 0.0)
    {
        *alpha = DoubleComplex(0.0, 0.0);
        return false;
    }
    const DoubleComplex first = (*reflector)[0];
    const DoubleComplex phase = std::abs(first) > 0.0 ? first / std::abs(first) : DoubleComplex(1.0, 0.0);
    *alpha = -phase * norm;
    (*reflector)[0] -= *alpha;
    const DoubleComplex leading = (*reflector)[0];
    for (DoubleComplex& value : *reflector)
    {
        value /= leading;
    }
    double reflectorNormSquared = 0.0;
    for (const DoubleComplex value : *reflector)
    {
        reflectorNormSquared += std::norm(value);
    }
    *beta = 2.0 / reflectorNormSquared;
    return true;
}

std::vector<DoubleComplex> BuildHouseholderUpdate(const std::vector<DoubleComplex>& work,
                                                  const std::vector<DoubleComplex>& reflector, int64_t n, int64_t begin,
                                                  double beta)
{
    const int64_t active = n - begin;
    std::vector<DoubleComplex> update(static_cast<size_t>(active));
#pragma omp parallel for if (active >= HOST_OMP_PARALLEL_MIN_N) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t row = 0; row < active; ++row)
    {
        DoubleComplex sum(0.0, 0.0);
        const size_t rowOffset = static_cast<size_t>((begin + row) * n + begin);
        for (int64_t col = 0; col < active; ++col)
        {
            sum += work[rowOffset + static_cast<size_t>(col)] * reflector[static_cast<size_t>(col)];
        }
        update[static_cast<size_t>(row)] = beta * sum;
    }
    DoubleComplex projection(0.0, 0.0);
    for (int64_t index = 0; index < active; ++index)
    {
        projection += std::conj(reflector[static_cast<size_t>(index)]) * update[static_cast<size_t>(index)];
    }
    const double correction = -0.5 * beta * projection.real();
    for (int64_t index = 0; index < active; ++index)
    {
        update[static_cast<size_t>(index)] += correction * reflector[static_cast<size_t>(index)];
    }
    return update;
}

void ApplyHouseholderUpdate(std::vector<DoubleComplex>* work, const std::vector<DoubleComplex>& reflector,
                            const std::vector<DoubleComplex>& update, int64_t n, int64_t begin)
{
    const int64_t active = n - begin;
#pragma omp parallel for if (active >= HOST_OMP_PARALLEL_MIN_ACTIVE) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t row = 0; row < active; ++row)
    {
        const size_t rowOffset = static_cast<size_t>((begin + row) * n + begin);
        for (int64_t col = 0; col < active; ++col)
        {
            (*work)[rowOffset + static_cast<size_t>(col)] -=
                reflector[static_cast<size_t>(row)] * std::conj(update[static_cast<size_t>(col)]) +
                update[static_cast<size_t>(row)] * std::conj(reflector[static_cast<size_t>(col)]);
        }
        (*work)[rowOffset + static_cast<size_t>(row)] =
            DoubleComplex((*work)[rowOffset + static_cast<size_t>(row)].real(), 0.0);
    }
}

void ReduceHermitianToTridiagonal(std::vector<DoubleComplex>* work, int64_t n, std::vector<double>* reflectorBeta,
                                  std::vector<DoubleComplex>* subDiagonal)
{
    for (int64_t step = 0; step + 2 < n; ++step)
    {
        std::vector<DoubleComplex> reflector;
        DoubleComplex alpha;
        double beta = 0.0;
        if (!BuildHouseholderReflector(*work, n, step, &reflector, &alpha, &beta))
        {
            (*subDiagonal)[static_cast<size_t>(step)] = DoubleComplex(0.0, 0.0);
            continue;
        }
        (*reflectorBeta)[static_cast<size_t>(step)] = beta;
        (*subDiagonal)[static_cast<size_t>(step)] = alpha;
        const int64_t begin = step + 1;
        const std::vector<DoubleComplex> update = BuildHouseholderUpdate(*work, reflector, n, begin, beta);
        ApplyHouseholderUpdate(work, reflector, update, n, begin);
        (*work)[static_cast<size_t>(begin * n + step)] = alpha;
        for (int64_t index = 1; index < n - begin; ++index)
        {
            (*work)[static_cast<size_t>((begin + index) * n + step)] = reflector[static_cast<size_t>(index)];
        }
    }
    if (n > 1)
    {
        (*subDiagonal)[static_cast<size_t>(n - 2)] = (*work)[static_cast<size_t>((n - 1) * n + (n - 2))];
    }
}

void BuildRealTridiagonal(const std::vector<DoubleComplex>& work, const std::vector<DoubleComplex>& subDiagonal,
                          int64_t n, std::vector<double>* diagonal, std::vector<double>* offDiagonal,
                          std::vector<DoubleComplex>* phases)
{
    for (int64_t index = 0; index < n; ++index)
    {
        (*diagonal)[static_cast<size_t>(index)] = work[static_cast<size_t>(index * n + index)].real();
        if (index + 1 < n)
        {
            const DoubleComplex value = subDiagonal[static_cast<size_t>(index)];
            const double magnitude = std::abs(value);
            (*offDiagonal)[static_cast<size_t>(index)] = magnitude;
            (*phases)[static_cast<size_t>(index + 1)] = magnitude > 0.0
                                                            ? (*phases)[static_cast<size_t>(index)] * value / magnitude
                                                            : (*phases)[static_cast<size_t>(index)];
        }
    }
}

std::vector<int64_t> SortRealEigenpairs(const std::vector<double>& diagonal, int64_t n, float* w)
{
    std::vector<int64_t> permutation(static_cast<size_t>(n));
    for (int64_t index = 0; index < n; ++index)
    {
        permutation[static_cast<size_t>(index)] = index;
    }
    std::stable_sort(permutation.begin(), permutation.end(), [&diagonal](int64_t lhs, int64_t rhs)
                     { return diagonal[static_cast<size_t>(lhs)] < diagonal[static_cast<size_t>(rhs)]; });
    for (int64_t column = 0; column < n; ++column)
    {
        w[column] = static_cast<float>(diagonal[static_cast<size_t>(permutation[static_cast<size_t>(column)])]);
    }
    return permutation;
}

std::vector<DoubleComplex> BacktransformHouseholderVectors(const std::vector<DoubleComplex>& work,
                                                           const std::vector<double>& reflectorBeta,
                                                           const std::vector<DoubleComplex>& phases,
                                                           const std::vector<double>& tridiagonalVectors,
                                                           const std::vector<int64_t>& permutation, int64_t n)
{
    std::vector<DoubleComplex> eigenvectors(static_cast<size_t>(n * n));
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t column = 0; column < n; ++column)
    {
        const int64_t source = permutation[static_cast<size_t>(column)];
        DoubleComplex* output = eigenvectors.data() + column * n;
        for (int64_t row = 0; row < n; ++row)
        {
            output[row] = phases[static_cast<size_t>(row)] * tridiagonalVectors[static_cast<size_t>(row + source * n)];
        }
        for (int64_t step = n - 2; step-- > 0;)
        {
            const double beta = reflectorBeta[static_cast<size_t>(step)];
            if (beta == 0.0)
            {
                continue;
            }
            const int64_t begin = step + 1;
            DoubleComplex projection = output[begin];
            for (int64_t row = begin + 1; row < n; ++row)
            {
                projection += std::conj(work[static_cast<size_t>(row * n + step)]) * output[row];
            }
            projection *= beta;
            output[begin] -= projection;
            for (int64_t row = begin + 1; row < n; ++row)
            {
                output[row] -= work[static_cast<size_t>(row * n + step)] * projection;
            }
        }
    }
    return eigenvectors;
}

void WriteHouseholderEigenvectors(const std::vector<DoubleComplex>& eigenvectors, std::vector<Complex>* matrix,
                                  int64_t n)
{
    matrix->resize(static_cast<size_t>(n * n));
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t column = 0; column < n; ++column)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            const DoubleComplex value = eigenvectors[static_cast<size_t>(row + column * n)];
            (*matrix)[static_cast<size_t>(row + column * n)] =
                Complex(static_cast<float>(value.real()), static_cast<float>(value.imag()));
        }
    }
}

bool SolveCompactHostHouseholder(std::vector<Complex>* matrix, int64_t n, bool computeVectors, float* w)
{
    const size_t matrixElements = static_cast<size_t>(n * n);
    std::vector<DoubleComplex> work(matrixElements);
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t row = 0; row < n; ++row)
    {
        for (int64_t col = 0; col < n; ++col)
        {
            work[static_cast<size_t>(row * n + col)] =
                static_cast<DoubleComplex>((*matrix)[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))]);
        }
    }

    std::vector<double> reflectorBeta(static_cast<size_t>(n), 0.0);
    std::vector<DoubleComplex> subDiagonal(static_cast<size_t>(n > 1 ? n - 1 : 0));
    ReduceHermitianToTridiagonal(&work, n, &reflectorBeta, &subDiagonal);

    std::vector<double> diagonal(static_cast<size_t>(n));
    std::vector<double> realOffDiagonal(static_cast<size_t>(n), 0.0);
    std::vector<DoubleComplex> phases(static_cast<size_t>(n), DoubleComplex(1.0, 0.0));
    BuildRealTridiagonal(work, subDiagonal, n, &diagonal, &realOffDiagonal, &phases);

    std::vector<double> tridiagonalVectors;
    if (computeVectors)
    {
        tridiagonalVectors.assign(matrixElements, 0.0);
        for (int64_t index = 0; index < n; ++index)
        {
            tridiagonalVectors[static_cast<size_t>(CompactColumnMajorIndex(index, index, n))] = 1.0;
        }
    }
    if (!SolveRealSymmetricTridiagonal(&diagonal, &realOffDiagonal, computeVectors ? &tridiagonalVectors : nullptr, n))
    {
        return false;
    }

    const std::vector<int64_t> permutation = SortRealEigenpairs(diagonal, n, w);
    if (!computeVectors)
    {
        return true;
    }

    const std::vector<DoubleComplex> eigenvectors =
        BacktransformHouseholderVectors(work, reflectorBeta, phases, tridiagonalVectors, permutation, n);
    WriteHouseholderEigenvectors(eigenvectors, matrix, n);
    return true;
}

void ScatterFullMatrix(const std::vector<Complex>& packed, int64_t lda, int64_t n, Complex* a)
{
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            a[row + col * lda] = packed[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))];
        }
    }
}

void BuildOrthogonalityProbes(std::vector<Complex>* probeValues, int64_t n, float inverseRootTwoN)
{
    for (int probe = 0; probe < ORTHOGONALITY_PROBE_COUNT; ++probe)
    {
        for (int64_t col = 0; col < n; ++col)
        {
            uint32_t bits = static_cast<uint32_t>(col + 1) * 2654435761U;
            bits ^= static_cast<uint32_t>(probe + 1) * 2246822519U;
            bits ^= bits >> 16;
            const float real = (bits & 1U) == 0U ? inverseRootTwoN : -inverseRootTwoN;
            const float imag = (bits & 2U) == 0U ? inverseRootTwoN : -inverseRootTwoN;
            (*probeValues)[static_cast<size_t>(probe * n + col)] = Complex(real, imag);
        }
    }
}

void ApplyVectorsToProbes(const std::vector<Complex>& vectors, const std::vector<Complex>& probeValues, int64_t n,
                          std::vector<Complex>* transformed)
{
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t row = 0; row < n; ++row)
    {
        std::array<Complex, ORTHOGONALITY_PROBE_COUNT> sums{};
        for (int64_t col = 0; col < n; ++col)
        {
            const Complex value = vectors[static_cast<size_t>(col * n + row)];
            for (int probe = 0; probe < ORTHOGONALITY_PROBE_COUNT; ++probe)
            {
                sums[static_cast<size_t>(probe)] += value * probeValues[static_cast<size_t>(probe * n + col)];
            }
        }
        for (int probe = 0; probe < ORTHOGONALITY_PROBE_COUNT; ++probe)
        {
            (*transformed)[static_cast<size_t>(probe * n + row)] = sums[static_cast<size_t>(probe)];
        }
    }
}

void ApplyAdjointToProbes(const std::vector<Complex>& vectors, const std::vector<Complex>& transformed, int64_t n,
                          std::vector<Complex>* reconstructed)
{
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        std::array<Complex, ORTHOGONALITY_PROBE_COUNT> sums{};
        for (int64_t row = 0; row < n; ++row)
        {
            const Complex value = std::conj(vectors[static_cast<size_t>(col * n + row)]);
            for (int probe = 0; probe < ORTHOGONALITY_PROBE_COUNT; ++probe)
            {
                sums[static_cast<size_t>(probe)] += value * transformed[static_cast<size_t>(probe * n + row)];
            }
        }
        for (int probe = 0; probe < ORTHOGONALITY_PROBE_COUNT; ++probe)
        {
            (*reconstructed)[static_cast<size_t>(probe * n + col)] = sums[static_cast<size_t>(probe)];
        }
    }
}

#include "cheevj_fixed_host.inc"

aclError CopyCompleteVectorOutput(uint8_t* deviceInfo, uint8_t* eigenvalues, const FixedMatrixSlices& matrix,
                                  const FixedWorkspaceSizes& sizes, std::vector<Complex>* packedA, int64_t n, float* w,
                                  int32_t* info)
{
    aclError status = CopyFixedEigenvalues(deviceInfo, eigenvalues, sizes.vectorBytes, w, info);
    if (status != ACL_SUCCESS || *info != 0)
    {
        return status;
    }
    return CopyFixedEigenvectors(matrix, sizes, packedA, n);
}

aclError AllocateCompleteVectorWorkspace(const FixedWorkspaceSizes& sizes, size_t auxiliaryBytes, size_t infoBytes,
                                         uint8_t** matrixWorkspace, uint8_t** auxiliaryWorkspace, uint8_t** deviceInfo)
{
    // Release any buffer acquired before a later allocation/memset fails so the
    // caller never has to handle partially initialized workspaces (issue #121).
    auto releaseAcquired = [&]()
    {
        if (*matrixWorkspace != nullptr)
        {
            (void)aclrtFree(*matrixWorkspace);
            *matrixWorkspace = nullptr;
        }
        if (*auxiliaryWorkspace != nullptr)
        {
            (void)aclrtFree(*auxiliaryWorkspace);
            *auxiliaryWorkspace = nullptr;
        }
        if (*deviceInfo != nullptr)
        {
            (void)aclrtFree(*deviceInfo);
            *deviceInfo = nullptr;
        }
    };
    CHEEVJ_CHECK_ACLRT(
        aclrtMalloc(reinterpret_cast<void**>(matrixWorkspace), sizes.sixPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        releaseAcquired());
    CHEEVJ_CHECK_ACLRT(
        aclrtMalloc(reinterpret_cast<void**>(auxiliaryWorkspace), auxiliaryBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        releaseAcquired());
    CHEEVJ_CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(deviceInfo), infoBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                       releaseAcquired());
    CHEEVJ_CHECK_ACLRT(aclrtMemset(*matrixWorkspace, sizes.sixPlaneBytes, 0, sizes.sixPlaneBytes), releaseAcquired());
    CHEEVJ_CHECK_ACLRT(aclrtMemset(*auxiliaryWorkspace, auxiliaryBytes, 0, auxiliaryBytes), releaseAcquired());
    return ACL_SUCCESS;
}

aclError RunCompleteVector(std::vector<Complex>* packedA, int64_t n, float* w, int32_t* info, aclrtStream stream)
{
    constexpr int64_t commandElems = 32;
    constexpr int64_t infoElems = 8;
    const FixedWorkspaceSizes sizes = GetFixedWorkspaceSizes(n);
    const size_t commandBytes = static_cast<size_t>(commandElems) * sizeof(int32_t);
    const size_t infoBytes = static_cast<size_t>(infoElems) * sizeof(int32_t);
    const size_t auxiliaryBytes = 7 * sizes.panelBytes + 5 * sizes.vectorBytes + sizes.panelWorkspaceBytes +
                                  sizes.barrierBytes + 2 * sizes.matrixBytes + sizes.boundsBytes + commandBytes;

    std::vector<float> inputReal(static_cast<size_t>(sizes.matrixElems));
    std::vector<float> inputImag(static_cast<size_t>(sizes.matrixElems));
    PackComplexPlanes(*packedA, n, &inputReal, &inputImag);

    uint8_t* matrixWorkspace = nullptr;
    uint8_t* auxiliaryWorkspace = nullptr;
    uint8_t* deviceInfo = nullptr;
    // Release all three device workspaces on any failure so error paths no
    // longer leak the buffers acquired before the failure (issue #121).
    auto releaseWorkspaces = [&]()
    {
        if (matrixWorkspace != nullptr)
        {
            (void)aclrtFree(matrixWorkspace);
            matrixWorkspace = nullptr;
        }
        if (auxiliaryWorkspace != nullptr)
        {
            (void)aclrtFree(auxiliaryWorkspace);
            auxiliaryWorkspace = nullptr;
        }
        if (deviceInfo != nullptr)
        {
            (void)aclrtFree(deviceInfo);
            deviceInfo = nullptr;
        }
    };
    CHEEVJ_CHECK_ACLRT(AllocateCompleteVectorWorkspace(sizes, auxiliaryBytes, infoBytes, &matrixWorkspace,
                                                       &auxiliaryWorkspace, &deviceInfo),
                       releaseWorkspaces());

    const FixedMatrixSlices matrix = TakeFixedMatrixSlices(matrixWorkspace, sizes.matrixBytes);
    const FixedPanelSlices panel = TakeFixedPanelSlices(auxiliaryWorkspace, sizes);
    size_t auxiliaryOffset = panel.nextOffset;
    uint8_t* reflectorReal = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.matrixBytes);
    uint8_t* reflectorImag = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.matrixBytes);
    uint8_t* bounds = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.boundsBytes);
    uint8_t* eigenvalues = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.vectorBytes);
    uint8_t* commandWorkspace = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, commandBytes);

    CHEEVJ_CHECK_ACLRT(
        aclrtMemcpy(matrix.real, sizes.matrixBytes, inputReal.data(), sizes.matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        releaseWorkspaces());
    CHEEVJ_CHECK_ACLRT(
        aclrtMemcpy(matrix.imag, sizes.matrixBytes, inputImag.data(), sizes.matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        releaseWorkspaces());
    CHEEVJ_CHECK_ACLRT(aclrtMemset(deviceInfo, infoBytes, 0, infoBytes), releaseWorkspaces());

    if (const aclError launchStatus = LaunchCompleteVector(matrix, panel, reflectorReal, reflectorImag, bounds,
                                                           eigenvalues, commandWorkspace, deviceInfo, n, stream);
        launchStatus != ACL_SUCCESS)
    {
        releaseWorkspaces();
        return launchStatus;
    }

    if (const aclError copyStatus =
            CopyCompleteVectorOutput(deviceInfo, eigenvalues, matrix, sizes, packedA, n, w, info);
        copyStatus != ACL_SUCCESS)
    {
        releaseWorkspaces();
        return copyStatus;
    }

    // 成功释放后立即置空（issue #186）：后续 aclrtFree 失败进入清理 lambda 时，
    // 已成功释放的地址因置空不会再次进入 aclrtFree，避免重复释放
    CHEEVJ_CHECK_ACLRT(aclrtFree(matrixWorkspace), releaseWorkspaces());
    matrixWorkspace = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(auxiliaryWorkspace), releaseWorkspaces());
    auxiliaryWorkspace = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(deviceInfo), releaseWorkspaces());
    deviceInfo = nullptr;
    return ACL_SUCCESS;
}

std::vector<Complex> PreparePaddedMatrix(const Complex* a, int64_t lda, int64_t n, int64_t paddedN, int32_t uplo)
{
    std::vector<Complex> padded(static_cast<size_t>(paddedN * paddedN), Complex(0.0f, 0.0f));
    double spectralBound = 1.0;
    const bool lower = IsLowerMode(uplo);
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N) num_threads(HOST_OMP_THREAD_NUM) schedule(static) \
    reduction(max : spectralBound)
    for (int64_t row = 0; row < n; ++row)
    {
        double rowSum = 0.0;
        for (int64_t col = 0; col < n; ++col)
        {
            const Complex value = LoadHermitianValue(a, lda, row, col, lower);
            rowSum += static_cast<double>(std::abs(value));
            padded[static_cast<size_t>(CompactColumnMajorIndex(row, col, paddedN))] = value;
        }
        spectralBound = std::max(spectralBound, rowSum);
    }

    const double maxSentinel = static_cast<double>(std::numeric_limits<float>::max()) * 0.5;
    const float sentinelBase = static_cast<float>(std::min(1.25 * spectralBound + 1.0, maxSentinel));
    const int64_t padding = paddedN - n;
    for (int64_t index = 0; index < padding; ++index)
    {
        const float scale = 1.0f + 0.25f * static_cast<float>(index + 1) / static_cast<float>(padding + 1);
        padded[static_cast<size_t>(CompactColumnMajorIndex(n + index, n + index, paddedN))] =
            Complex(sentinelBase * scale, 0.0f);
    }
    return padded;
}

aclError RunPaddedBackend(std::vector<Complex>* padded, int64_t paddedN, bool computeVectors,
                          std::vector<float>* paddedW, int32_t* info, aclrtStream stream)
{
    return computeVectors ? RunCompleteVector(padded, paddedN, paddedW->data(), info, stream)
                          : RunCompleteNoVector(*padded, paddedN, paddedW->data(), info, stream);
}

std::vector<int64_t> SortPaddedEigenpairs(const std::vector<float>& paddedW, int64_t n, float* w)
{
    std::vector<int64_t> eigenOrder(static_cast<size_t>(n));
    for (int64_t index = 0; index < n; ++index)
    {
        eigenOrder[static_cast<size_t>(index)] = index;
    }
    std::stable_sort(eigenOrder.begin(), eigenOrder.end(), [&paddedW](int64_t lhs, int64_t rhs)
                     { return paddedW[static_cast<size_t>(lhs)] < paddedW[static_cast<size_t>(rhs)]; });
    for (int64_t index = 0; index < n; ++index)
    {
        w[index] = paddedW[static_cast<size_t>(eigenOrder[static_cast<size_t>(index)])];
    }
    return eigenOrder;
}

std::vector<Complex> CropPaddedEigenvectors(const std::vector<Complex>& padded, const std::vector<int64_t>& eigenOrder,
                                            int64_t n, int64_t paddedN)
{
    std::vector<Complex> cropped(static_cast<size_t>(n * n));
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        const int64_t sourceCol = eigenOrder[static_cast<size_t>(col)];
        for (int64_t row = 0; row < n; ++row)
        {
            cropped[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))] =
                padded[static_cast<size_t>(CompactColumnMajorIndex(row, sourceCol, paddedN))];
        }
    }
    return cropped;
}

bool CorrectPaddedEigenvectors(std::vector<Complex>* cropped, int64_t n, const float* w)
{
    if (PassesOrthogonalityProbe(*cropped, n))
    {
        return true;
    }
    std::vector<Complex> corrected = *cropped;
    if (OrthonormalizeEigenvalueClusters(&corrected, n, w) && PassesOrthogonalityProbe(corrected, n))
    {
        cropped->swap(corrected);
        return true;
    }
    return false;
}

bool RunPaddedHostFallback(Complex* a, int64_t lda, int64_t n, int32_t uplo, float* w)
{
    std::vector<Complex> hostMatrix;
    PackFullMatrix(a, lda, n, uplo, &hostMatrix);
    const bool converged = SolveCompactHostHouseholder(&hostMatrix, n, true, w);
    if (converged)
    {
        ScatterFullMatrix(hostMatrix, lda, n, a);
    }
    return converged;
}

aclError RunPaddedFixed(Complex* a, int64_t lda, int64_t n, int64_t paddedN, int32_t uplo, bool computeVectors,
                        float* w, int32_t* info, aclrtStream stream)
{
    std::vector<Complex> padded = PreparePaddedMatrix(a, lda, n, paddedN, uplo);
    std::vector<float> paddedW(static_cast<size_t>(paddedN));
    const aclError status = RunPaddedBackend(&padded, paddedN, computeVectors, &paddedW, info, stream);
    if (status != ACL_SUCCESS || *info != 0)
    {
        return status;
    }
    const std::vector<int64_t> eigenOrder = SortPaddedEigenpairs(paddedW, n, w);
    if (computeVectors)
    {
        std::vector<Complex> cropped = CropPaddedEigenvectors(padded, eigenOrder, n, paddedN);
        if (!CorrectPaddedEigenvectors(&cropped, n, w))
        {
            const bool converged = RunPaddedHostFallback(a, lda, n, uplo, w);
            *info = converged ? 0 : 1;
            return ACL_SUCCESS;
        }
        ScatterFullMatrix(cropped, lda, n, a);
    }
    return ACL_SUCCESS;
}

struct DispatchResult
{
    bool handled;
    aclError status;
};

struct GenericWorkspaceConfig
{
    int64_t strideN;
    int64_t workM;
    int64_t workspacePlanes;
    size_t sizeW;
    size_t sizeInfo;
    size_t workspaceSize;
};

aclError ValidateCheevjArguments(aclsolverEigMode_t jobz, aclsolverFillMode_t uplo, int64_t n, const Complex* a,
                                 int64_t lda, const float* w, int32_t* info, int32_t* jobzValue, int32_t* uploValue,
                                 bool* computeVectors)
{
    if (info == nullptr)
    {
        return InvalidParam(nullptr, 0, "info must not be null");
    }
    *info = 0;
    *jobzValue = static_cast<int32_t>(jobz);
    *uploValue = static_cast<int32_t>(uplo);
    *computeVectors = IsVectorMode(*jobzValue);
    SOLVER_ECHECK(IsNoVectorMode(*jobzValue) || *computeVectors, "Cheevj jobz must be NOVECTOR/VECTOR or 'N'/'V'.",
                  InvalidParam(info, -2, "jobz must be NOVECTOR/VECTOR or 'N'/'V'"));
    SOLVER_ECHECK(IsLowerMode(*uploValue) || IsUpperMode(*uploValue), "Cheevj uplo must be LOWER/UPPER or 'L'/'U'.",
                  InvalidParam(info, -3, "uplo must be LOWER/UPPER or 'L'/'U'"));
    SOLVER_ECHECK(n >= 0, "Cheevj get n < 0.", InvalidParam(info, -4, "n must be non-negative"));
    SOLVER_ECHECK(lda >= (n > 1 ? n : 1), "Cheevj get lda < max(1, n).",
                  InvalidParam(info, -6, "lda must be at least max(1, n)"));
    if (n != 0)
    {
        SOLVER_ECHECK(a != nullptr, "Cheevj get null a.", InvalidParam(info, -5, "a must not be null"));
        SOLVER_ECHECK(w != nullptr, "Cheevj get null w.", InvalidParam(info, -7, "w must not be null"));
    }
    return ACL_SUCCESS;
}

aclrtStream GetCheevjStream(aclsolverHandle_t handle)
{
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }
    return stream;
}

uint32_t GetCheevjBlockCount()
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr)
    {
        return 1;
    }
    const uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();
    return numBlocks == 0 ? 1 : numBlocks;
}

DispatchResult TryStructuredInput(Complex* a, int64_t lda, int64_t n, int32_t uploValue, bool computeVectors, float* w)
{
    const bool lower = IsLowerMode(uploValue);
    if (IsInputDiagonal(a, lda, n, lower))
    {
        SolveInputDiagonal(a, lda, n, computeVectors, w);
        return {true, ACL_SUCCESS};
    }
    if (IsInputTwoByTwoBlockDiagonal(a, lda, n, lower))
    {
        SolveInputTwoByTwoBlockDiagonal(a, lda, n, lower, computeVectors, w);
        return {true, ACL_SUCCESS};
    }
    return {false, ACL_SUCCESS};
}

DispatchResult TrySmallHostInput(Complex* a, int64_t lda, int64_t n, int32_t uploValue, bool computeVectors,
                                 bool wideDiagonalScale, float* w, int32_t* info)
{
    if (n > 256)
    {
        return {false, ACL_SUCCESS};
    }
    if (!wideDiagonalScale && !IsInputHermitianTridiagonal(a, lda, n, IsLowerMode(uploValue)))
    {
        return {false, ACL_SUCCESS};
    }
    std::vector<Complex> hostMatrix;
    PackFullMatrix(a, lda, n, uploValue, &hostMatrix);
    const bool converged = SolveCompactHostJacobi(&hostMatrix, n, computeVectors, w);
    // 未收敛（info=1）时不回写用户输入 a，保留原始矩阵供调用方重试/诊断——
    // 与 TryWideScaleInput 的门控语义一致（issue #159：三条 host 回退路径
    // 的 a 回写契约统一为"仅收敛时回写特征向量"）
    if (computeVectors && converged)
    {
        ScatterFullMatrix(hostMatrix, lda, n, a);
    }
    *info = converged ? 0 : 1;
    return {true, ACL_SUCCESS};
}

aclError BuildGenericWorkspaceConfig(int64_t n, bool computeVectors, int32_t* info, GenericWorkspaceConfig* config)
{
    SOLVER_ECHECK(n <= std::numeric_limits<int32_t>::max(), "Cheevj n exceeds the device index representation.",
                  InvalidParam(info, -4, "n exceeds the device index representation"));
    size_t matrixElems = 0;
    size_t sizeA = 0;
    SOLVER_ECHECK(CheckedSquareSize(n, sizeof(Complex), &matrixElems, &sizeA),
                  "Cheevj matrix byte size overflows size_t.",
                  InvalidParam(info, -4, "matrix byte size overflows size_t"));
    config->sizeW = static_cast<size_t>(n) * sizeof(float);
    config->sizeInfo = sizeof(int32_t);
    config->strideN = AlignUp(n, COL_ALIGNED_ELENUM);
    config->workM = AlignUp(n, BASE_BLOCK_ELENUM);
    config->workspacePlanes = computeVectors ? CHEEVJ_WORKSPACE_PLANES_V : CHEEVJ_WORKSPACE_PLANES_N;
    const int64_t maxStrideN = config->workM > 0 ? std::numeric_limits<int32_t>::max() / config->workM : 0;
    SOLVER_ECHECK(config->workM > 0 && config->strideN <= maxStrideN,
                  "Cheevj workspace plane exceeds the device index representation.",
                  InvalidParam(info, -4, "workspace plane exceeds the device index representation"));
    const size_t planeElems = static_cast<size_t>(config->strideN) * static_cast<size_t>(config->workM);
    SOLVER_ECHECK(
        planeElems <= std::numeric_limits<size_t>::max() / static_cast<size_t>(config->workspacePlanes) / sizeof(float),
        "Cheevj workspace byte size overflows size_t.", InvalidParam(info, -4, "workspace byte size overflows size_t"));
    config->workspaceSize = planeElems * static_cast<size_t>(config->workspacePlanes) * sizeof(float);
    return ACL_SUCCESS;
}

// Estimate whether the pure-host fallback paths can hold the packed n*n
// complex matrix.  Returns false (and records the reason in info) when the
// request is beyond what the process can reasonably allocate, so callers get
// an error code instead of an escaping std::bad_alloc (issue #122).
bool HostFallbackFeasible(int64_t n, int32_t* info)
{
    constexpr double kMaxHostMatrixGiB = 8.0;
    const double requestedGiB =
        static_cast<double>(n) * static_cast<double>(n) * sizeof(Complex) / (1024.0 * 1024.0 * 1024.0);
    if (requestedGiB > kMaxHostMatrixGiB)
    {
        InvalidParam(info, -5, "host fallback matrix exceeds the supported host memory budget");
        return false;
    }
    return true;
}

DispatchResult TryWideScaleInput(Complex* a, int64_t lda, int64_t n, int32_t uploValue, bool computeVectors,
                                 bool wideDiagonalScale, float* w, int32_t* info)
{
    if (!wideDiagonalScale || n <= 256)
    {
        return {false, ACL_SUCCESS};
    }
    if (!HostFallbackFeasible(n, info))
    {
        return {true, ACL_ERROR_INVALID_PARAM};
    }
    std::vector<Complex> hostMatrix;
    PackFullMatrix(a, lda, n, uploValue, &hostMatrix);
    const bool converged = SolveCompactHostHouseholder(&hostMatrix, n, computeVectors, w);
    if (computeVectors && converged)
    {
        ScatterFullMatrix(hostMatrix, lda, n, a);
    }
    *info = converged ? 0 : 1;
    return {true, ACL_SUCCESS};
}

DispatchResult TryPackedSpecialCases(std::vector<Complex>* packedA, Complex* a, int64_t lda, int64_t n,
                                     bool computeVectors, float* w, int32_t* info)
{
    if (IsCompactDiagonal(*packedA, n))
    {
        SolveCompactDiagonal(packedA, n, computeVectors, w);
        if (computeVectors)
        {
            ScatterFullMatrix(*packedA, lda, n, a);
        }
        return {true, ACL_SUCCESS};
    }
    if (IsCompactTwoByTwoBlockDiagonal(*packedA, n))
    {
        SolveCompactTwoByTwoBlockDiagonal(packedA, n, computeVectors, w);
        if (computeVectors)
        {
            ScatterFullMatrix(*packedA, lda, n, a);
        }
        return {true, ACL_SUCCESS};
    }
    if (n > CHEEVJ_FIXED_N2048 || n <= 129)
    {
        // n > 2048 falls back to a pure-host solver; guard the host-side
        // allocation before packing so oversized inputs return an error code
        // instead of throwing std::bad_alloc through the C boundary (#122).
        if (n > CHEEVJ_FIXED_N2048 && !HostFallbackFeasible(n, info))
        {
            return {true, ACL_ERROR_INVALID_PARAM};
        }
        const bool converged = SolveCompactHostHouseholder(packedA, n, computeVectors, w);
        // 回写契约统一（issue #172）：与 TrySmallHostInput/TryWideScaleInput 及
        // issue #159 口径一致——仅收敛时回写特征向量；未收敛（info=1）保留原始
        // 输入供调用方重试/诊断，n<=129 不再作为回写的例外条件
        if (computeVectors && converged)
        {
            ScatterFullMatrix(*packedA, lda, n, a);
        }
        *info = converged ? 0 : 1;
        return {true, ACL_SUCCESS};
    }
    return {false, ACL_SUCCESS};
}

DispatchResult TryFixedShape(std::vector<Complex>* packedA, Complex* a, int64_t lda, int64_t n, int32_t uploValue,
                             bool computeVectors, float* w, int32_t* info, aclrtStream stream)
{
    if (!IsFixedShape(n))
    {
        return {false, ACL_SUCCESS};
    }
    if (!computeVectors)
    {
        return {true, RunCompleteNoVector(*packedA, n, w, info, stream)};
    }
    const aclError status = RunCompleteVector(packedA, n, w, info, stream);
    if (status == ACL_SUCCESS && *info == 0 && HasTightEigenvalueCluster(w, n) &&
        !PassesOrthogonalityProbe(*packedA, n))
    {
        PackFullMatrix(a, lda, n, uploValue, packedA);
        const bool converged = SolveCompactHostHouseholder(packedA, n, true, w);
        *info = converged ? 0 : 1;
    }
    if (status == ACL_SUCCESS && *info == 0)
    {
        ScatterFullMatrix(*packedA, lda, n, a);
    }
    return {true, status};
}

std::vector<Complex> BuildGenericDeviceMatrix(const std::vector<Complex>& packedA, int64_t n, int64_t deviceLda)
{
    std::vector<Complex> deviceA(static_cast<size_t>(deviceLda * n), Complex(0.0f, 0.0f));
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            deviceA[static_cast<size_t>(row + col * deviceLda)] =
                packedA[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))];
        }
    }
    return deviceA;
}

CheevjTilingData BuildCheevjTilingData(int64_t n, int64_t deviceLda, int32_t jobzValue, int32_t uploValue,
                                       const GenericWorkspaceConfig& config)
{
    CheevjTilingData tilingData{};
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(deviceLda);
    tilingData.jobz = static_cast<uint32_t>(jobzValue);
    tilingData.uplo = static_cast<uint32_t>(uploValue);
    tilingData.isDiagonal = 0U;
    tilingData.strideN = static_cast<uint32_t>(config.strideN);
    tilingData.workM = static_cast<uint32_t>(config.workM);
    tilingData.blockSize = static_cast<uint32_t>(CHEEVJ_BLOCK_SIZE);
    tilingData.tileM = static_cast<uint32_t>(CHEEVJ_TILE_M);
    tilingData.workspacePlanes = static_cast<uint32_t>(config.workspacePlanes);
    tilingData.maxSweeps = static_cast<uint32_t>(CHEEVJ_MAX_SWEEPS);
    return tilingData;
}

void CopyGenericDeviceVectors(const std::vector<Complex>& deviceA, int64_t deviceLda, int64_t n,
                              std::vector<Complex>* packedA)
{
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            (*packedA)[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))] =
                deviceA[static_cast<size_t>(row + col * deviceLda)];
        }
    }
}

aclError RunGenericDevice(std::vector<Complex>* packedA, Complex* a, int64_t lda, int64_t n, int32_t jobzValue,
                          int32_t uploValue, bool computeVectors, float* w, int32_t* info, uint32_t numBlocks,
                          aclrtStream stream, const GenericWorkspaceConfig& config)
{
    const int64_t deviceLda = n > 32 ? AlignUp(n, 8) : n;
    const size_t deviceSizeA = static_cast<size_t>(deviceLda * n) * sizeof(Complex);
    std::vector<Complex> deviceA = BuildGenericDeviceMatrix(*packedA, n, deviceLda);
    const CheevjTilingData tilingData = BuildCheevjTilingData(n, deviceLda, jobzValue, uploValue, config);
    uint8_t* d_A = nullptr;
    uint8_t* d_W = nullptr;
    uint8_t* d_info = nullptr;
    uint8_t* workSpace = nullptr;
    uint8_t* tilingDevice = nullptr;
    // Release every device buffer acquired before a later runtime call fails;
    // previously any memcpy/memset/synchronize failure returned without
    // freeing the five allocations above (issue #121).
    auto releaseDeviceBuffers = [&]()
    {
        if (d_A != nullptr)
        {
            (void)aclrtFree(d_A);
            d_A = nullptr;
        }
        if (d_W != nullptr)
        {
            (void)aclrtFree(d_W);
            d_W = nullptr;
        }
        if (d_info != nullptr)
        {
            (void)aclrtFree(d_info);
            d_info = nullptr;
        }
        if (workSpace != nullptr)
        {
            (void)aclrtFree(workSpace);
            workSpace = nullptr;
        }
        if (tilingDevice != nullptr)
        {
            (void)aclrtFree(tilingDevice);
            tilingDevice = nullptr;
        }
    };
    CHEEVJ_CHECK_ACLRT(aclrtMalloc((void**)&d_A, deviceSizeA, ACL_MEM_MALLOC_HUGE_FIRST), releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMalloc((void**)&d_W, config.sizeW, ACL_MEM_MALLOC_HUGE_FIRST), releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMalloc((void**)&d_info, config.sizeInfo, ACL_MEM_MALLOC_HUGE_FIRST),
                       releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMalloc((void**)&workSpace, config.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
                       releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMalloc((void**)&tilingDevice, sizeof(CheevjTilingData), ACL_MEM_MALLOC_HUGE_FIRST),
                       releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(d_A, deviceSizeA, deviceA.data(), deviceSizeA, ACL_MEMCPY_HOST_TO_DEVICE),
                       releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(d_info, config.sizeInfo, info, config.sizeInfo, ACL_MEMCPY_HOST_TO_DEVICE),
                       releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(tilingDevice, sizeof(CheevjTilingData), &tilingData, sizeof(CheevjTilingData),
                                   ACL_MEMCPY_HOST_TO_DEVICE),
                       releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemset(workSpace, config.workspaceSize, 0, config.workspaceSize), releaseDeviceBuffers());
    uint8_t* sync = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtGetHardwareSyncAddr((void**)&sync), releaseDeviceBuffers());
    cheevj_kernel_do(sync, d_A, d_W, d_info, workSpace, tilingDevice, numBlocks, stream);
    CHEEVJ_CHECK_ACLRT(aclrtSynchronizeStream(stream), releaseDeviceBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(info, config.sizeInfo, d_info, config.sizeInfo, ACL_MEMCPY_DEVICE_TO_HOST),
                       releaseDeviceBuffers());
    if (*info == 0)
    {
        CHEEVJ_CHECK_ACLRT(aclrtMemcpy(w, config.sizeW, d_W, config.sizeW, ACL_MEMCPY_DEVICE_TO_HOST),
                           releaseDeviceBuffers());
        if (computeVectors)
        {
            CHEEVJ_CHECK_ACLRT(aclrtMemcpy(deviceA.data(), deviceSizeA, d_A, deviceSizeA, ACL_MEMCPY_DEVICE_TO_HOST),
                               releaseDeviceBuffers());
            CopyGenericDeviceVectors(deviceA, deviceLda, n, packedA);
            ScatterFullMatrix(*packedA, lda, n, a);
        }
    }
    // 成功释放后立即置空（issue #186）：原理同上
    CHEEVJ_CHECK_ACLRT(aclrtFree(d_A), releaseDeviceBuffers());
    d_A = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(d_W), releaseDeviceBuffers());
    d_W = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(d_info), releaseDeviceBuffers());
    d_info = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(workSpace), releaseDeviceBuffers());
    workSpace = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(tilingDevice), releaseDeviceBuffers());
    tilingDevice = nullptr;
    return ACL_SUCCESS;
}

aclError RunPackedOrGeneric(Complex* a, int64_t lda, int64_t n, int32_t jobzValue, int32_t uploValue,
                            bool computeVectors, float* w, int32_t* info, uint32_t numBlocks, aclrtStream stream,
                            const GenericWorkspaceConfig& config)
{
    std::vector<Complex> packedA;
    PackFullMatrix(a, lda, n, uploValue, &packedA);
    DispatchResult dispatch = TryPackedSpecialCases(&packedA, a, lda, n, computeVectors, w, info);
    if (dispatch.handled)
    {
        return dispatch.status;
    }

    dispatch = TryFixedShape(&packedA, a, lda, n, uploValue, computeVectors, w, info, stream);
    if (dispatch.handled)
    {
        return dispatch.status;
    }

    return RunGenericDevice(&packedA, a, lda, n, jobzValue, uploValue, computeVectors, w, info, numBlocks, stream,
                            config);
}

// Dispatches after argument validation; declared here so the public entry can
// wrap it in a try/catch while it stays inside the anonymous namespace.
aclError CheevjDispatch(aclsolverHandle_t handle, int32_t jobzValue, int32_t uploValue, const int64_t n, Complex* a,
                        const int64_t lda, float* w, int32_t* info, bool computeVectors)
{
    const aclrtStream stream = GetCheevjStream(handle);
    const uint32_t numBlocks = GetCheevjBlockCount();

    DispatchResult dispatch = TryStructuredInput(a, lda, n, uploValue, computeVectors, w);
    if (dispatch.handled)
    {
        return dispatch.status;
    }

    constexpr float wideScaleThreshold = 1.0e6f;
    const float inputMagnitude = MaximumInputMagnitude(a, lda, n, IsLowerMode(uploValue));
    const bool wideInputScale = inputMagnitude > wideScaleThreshold;
    dispatch = TrySmallHostInput(a, lda, n, uploValue, computeVectors, wideInputScale, w, info);
    if (dispatch.handled)
    {
        return dispatch.status;
    }

    GenericWorkspaceConfig config{};
    if (const aclError workspaceStatus = BuildGenericWorkspaceConfig(n, computeVectors, info, &config);
        workspaceStatus != ACL_SUCCESS)
    {
        return workspaceStatus;
    }

    dispatch = TryWideScaleInput(a, lda, n, uploValue, computeVectors, wideInputScale, w, info);
    if (dispatch.handled)
    {
        return dispatch.status;
    }

    const int64_t paddedN = NextFixedShape(n);
    if (n >= 130 && !IsFixedShape(n) && paddedN != 0)
    {
        return RunPaddedFixed(a, lda, n, paddedN, uploValue, computeVectors, w, info, stream);
    }

    // 注意：当前分发顺序下 RunPackedOrGeneric 内部的 RunGenericDevice（通用形状设备内核，
    // cheevj_kernel.cpp 整条链路）对任何输入均不可达——n<=129 或 n>2048 已被 TryPackedSpecialCases
    // 以主机求解处理，512/1024/2048 被 TryFixedShape 处理，[130,2048] 非固定形状在上方 RunPaddedFixed
    // 返回。该链路保留作为后续启用 generic 内核的既有实现：启用时需调整上方 RunPaddedFixed 的
    // 分发条件（如仅对特定 n 区间 padding），并回归 cheevj_kernel 的 tiling/workspace 约定（issue #138）
    return RunPackedOrGeneric(a, lda, n, jobzValue, uploValue, computeVectors, w, info, numBlocks, stream, config);
}

}  // namespace

aclError aclsolverCheevj(aclsolverHandle_t handle, aclsolverEigMode_t jobz, aclsolverFillMode_t uplo, const int64_t n,
                         Complex* a, const int64_t lda, float* w, int32_t* info)
{
    int32_t jobzValue = 0;
    int32_t uploValue = 0;
    bool computeVectors = false;
    if (const aclError validationStatus =
            ValidateCheevjArguments(jobz, uplo, n, a, lda, w, info, &jobzValue, &uploValue, &computeVectors);
        validationStatus != ACL_SUCCESS)
    {
        return validationStatus;
    }
    if (n == 0)
    {
        return ACL_SUCCESS;
    }
    // The pure-host fallback paths allocate n*n matrices on the stack of this
    // call; translate any allocation failure into an error code so C callers
    // never observe an escaping std::bad_alloc (issue #122).
    try
    {
        return CheevjDispatch(handle, jobzValue, uploValue, n, a, lda, w, info, computeVectors);
    }
    catch (const std::bad_alloc&)
    {
        std::cerr << "Cheevj host allocation failed for n=" << n << std::endl;
        InvalidParam(info, -5, "host allocation failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Cheevj unexpected host failure: " << e.what() << std::endl;
        InvalidParam(info, -6, "unexpected host failure");
        return ACL_ERROR_INTERNAL_ERROR;
    }
}
