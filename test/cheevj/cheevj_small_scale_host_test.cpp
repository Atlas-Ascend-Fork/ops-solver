/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "cann_ops_solver.h"

namespace
{
using Complex = std::complex<float>;
using DoubleComplex = std::complex<double>;
constexpr int64_t kOrder = 3;

bool CheckEigenvectors(const std::vector<Complex>& matrix, const std::vector<Complex>& input,
                       const std::array<float, kOrder>& eigenvalues, float scale)
{
    double residualNormSquared = 0.0;
    double orthogonalityNormSquared = 0.0;
    for (int64_t column = 0; column < kOrder; ++column)
    {
        for (int64_t row = 0; row < kOrder; ++row)
        {
            DoubleComplex product(0.0, 0.0);
            for (int64_t k = 0; k < kOrder; ++k)
            {
                product += static_cast<DoubleComplex>(input[static_cast<size_t>(row + k * kOrder)]) *
                           static_cast<DoubleComplex>(matrix[static_cast<size_t>(k + column * kOrder)]);
            }
            const DoubleComplex residual =
                product - static_cast<double>(eigenvalues[static_cast<size_t>(column)]) *
                              static_cast<DoubleComplex>(matrix[static_cast<size_t>(row + column * kOrder)]);
            residualNormSquared += std::norm(residual);
        }
        for (int64_t other = 0; other < kOrder; ++other)
        {
            DoubleComplex dot(0.0, 0.0);
            for (int64_t row = 0; row < kOrder; ++row)
            {
                dot += std::conj(static_cast<DoubleComplex>(matrix[static_cast<size_t>(row + column * kOrder)])) *
                       static_cast<DoubleComplex>(matrix[static_cast<size_t>(row + other * kOrder)]);
            }
            if (column == other)
            {
                dot -= DoubleComplex(1.0, 0.0);
            }
            orthogonalityNormSquared += std::norm(dot);
        }
    }
    return std::sqrt(residualNormSquared) <= static_cast<double>(scale) * 1.0e-4 &&
           std::sqrt(orthogonalityNormSquared) <= 1.0e-5;
}

bool CheckTridiagonalSpectrum(const char* name, float scale, aclsolverEigMode_t jobz)
{
    std::vector<Complex> matrix(static_cast<size_t>(kOrder * kOrder), Complex(0.0f, 0.0f));
    matrix[static_cast<size_t>(0 + 1 * kOrder)] = Complex(scale, 0.0f);
    matrix[static_cast<size_t>(1 + 0 * kOrder)] = Complex(scale, 0.0f);
    matrix[static_cast<size_t>(1 + 2 * kOrder)] = Complex(scale, 0.0f);
    matrix[static_cast<size_t>(2 + 1 * kOrder)] = Complex(scale, 0.0f);
    const std::vector<Complex> input = matrix;
    std::array<float, kOrder> eigenvalues{};
    eigenvalues.fill(std::numeric_limits<float>::quiet_NaN());
    int32_t info = -1;
    const aclError status = aclsolverCheevj(nullptr, jobz, ACLSOLVER_FILL_MODE_UPPER, kOrder, matrix.data(), kOrder,
                                            eigenvalues.data(), &info);
    const double rootTwo = std::sqrt(2.0);
    const std::array<double, kOrder> expected = {-rootTwo * scale, 0.0, rootTwo * scale};
    bool valid = status == ACL_SUCCESS && info == 0;
    const double tolerance = static_cast<double>(scale) * 1.0e-5;
    for (int64_t index = 0; valid && index < kOrder; ++index)
    {
        const double error = std::abs(static_cast<double>(eigenvalues[static_cast<size_t>(index)]) -
                                      expected[static_cast<size_t>(index)]);
        valid = std::isfinite(eigenvalues[static_cast<size_t>(index)]) && error <= tolerance;
    }
    if (valid && jobz == ACLSOLVER_EIG_MODE_NOVECTOR)
    {
        valid = std::memcmp(matrix.data(), input.data(), matrix.size() * sizeof(Complex)) == 0;
    }
    if (valid && jobz == ACLSOLVER_EIG_MODE_VECTOR)
    {
        valid = CheckEigenvectors(matrix, input, eigenvalues, scale);
    }
    const char* mode = jobz == ACLSOLVER_EIG_MODE_VECTOR ? "V" : "N";
    std::cout << (valid ? "[PASS] " : "[FAIL] ") << name << '/' << mode << " status=" << status << " info=" << info
              << " eigenvalues=" << eigenvalues[0] << ',' << eigenvalues[1] << ',' << eigenvalues[2] << '\n';
    return valid;
}

bool CheckZeroMatrix(aclsolverEigMode_t jobz)
{
    std::vector<Complex> matrix(static_cast<size_t>(kOrder * kOrder), Complex(0.0f, 0.0f));
    std::array<float, kOrder> eigenvalues{};
    eigenvalues.fill(std::numeric_limits<float>::quiet_NaN());
    int32_t info = -1;
    const aclError status = aclsolverCheevj(nullptr, jobz, ACLSOLVER_FILL_MODE_UPPER, kOrder, matrix.data(), kOrder,
                                            eigenvalues.data(), &info);
    bool valid = status == ACL_SUCCESS && info == 0;
    for (float value : eigenvalues)
    {
        valid = valid && value == 0.0f;
    }
    const char* mode = jobz == ACLSOLVER_EIG_MODE_VECTOR ? "V" : "N";
    std::cout << (valid ? "[PASS] " : "[FAIL] ") << "zero matrix/" << mode << " control status=" << status
              << " info=" << info << " eigenvalues=" << eigenvalues[0] << ',' << eigenvalues[1] << ',' << eigenvalues[2]
              << '\n';
    return valid;
}
}  // namespace

int main()
{
    int failed = 0;
    for (aclsolverEigMode_t jobz : {ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_EIG_MODE_VECTOR})
    {
        failed += !CheckTridiagonalSpectrum("small scale 5e-13", 5.0e-13f, jobz);
        failed += !CheckTridiagonalSpectrum("scaled control 5e-10", 5.0e-10f, jobz);
        failed += !CheckZeroMatrix(jobz);
    }
    return failed == 0 ? 0 : 1;
}
