/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_solver.h"

namespace
{

using Complex = std::complex<float>;

int gPassed = 0;
int gFailed = 0;

void Check(const std::string &name, bool condition)
{
    if (condition)
    {
        ++gPassed;
        std::cout << "[PASS] " << name << '\n';
    }
    else
    {
        ++gFailed;
        std::cout << "[FAIL] " << name << '\n';
    }
}

bool NearlyEqual(float left, float right, float tolerance) { return std::abs(left - right) <= tolerance; }

void CheckInvalidParameters()
{
    Complex a(1.0f, 0.0f);
    float w = 0.0f;
    int32_t info = 123;

    Check("null info rejected", aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, 1, &a,
                                                1, &w, nullptr) == ACL_ERROR_INVALID_PARAM);

    info = 123;
    Check("invalid jobz rejected with info=-2",
          aclsolverCheevj(nullptr, static_cast<aclsolverEigMode_t>(99), ACLSOLVER_FILL_MODE_UPPER, 1, &a, 1, &w,
                          &info) == ACL_ERROR_INVALID_PARAM &&
              info == -2);

    info = 123;
    Check("invalid uplo rejected with info=-3",
          aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, static_cast<aclsolverFillMode_t>(99), 1, &a, 1, &w,
                          &info) == ACL_ERROR_INVALID_PARAM &&
              info == -3);

    info = 123;
    Check("negative n rejected with info=-4",
          aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, -1, &a, 1, &w, &info) ==
                  ACL_ERROR_INVALID_PARAM &&
              info == -4);

    info = 123;
    Check("short lda rejected with info=-6",
          aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, 2, &a, 1, &w, &info) ==
                  ACL_ERROR_INVALID_PARAM &&
              info == -6);

    info = 123;
    Check("n=0 accepts null A/W and returns info=0",
          aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, 0, nullptr, 1, nullptr,
                          &info) == ACL_SUCCESS &&
              info == 0);

    info = 123;
    Check("positive n rejects null A with info=-5",
          aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, 1, nullptr, 1, &w, &info) ==
                  ACL_ERROR_INVALID_PARAM &&
              info == -5);

    info = 123;
    Check("positive n rejects null W with info=-7",
          aclsolverCheevj(nullptr, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, 1, &a, 1, nullptr, &info) ==
                  ACL_ERROR_INVALID_PARAM &&
              info == -7);
}

void CheckNoArtificialSizeLimit(aclsolverHandle_t handle)
{
    constexpr int64_t n = 2049;
    std::vector<Complex> a(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    std::vector<float> w(static_cast<size_t>(n), 0.0f);
    for (int64_t index = 0; index < n; ++index)
    {
        a[static_cast<size_t>(index + index * n)] = Complex(static_cast<float>(index) - 1024.0f, 0.0f);
    }
    int32_t info = -999;
    const aclError status = aclsolverCheevj(handle, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, n, a.data(),
                                            n, w.data(), &info);
    Check("n=2049 diagonal has no artificial size rejection",
          status == ACL_SUCCESS && info == 0 && w.front() == -1024.0f && w.back() == 1024.0f);
}

void CheckDenseAboveFixedBackends(aclsolverHandle_t handle)
{
    constexpr int64_t n = 2049;
    constexpr float tolerance = 5.0e-3f;
    std::vector<Complex> a(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    std::vector<float> w(static_cast<size_t>(n), 0.0f);
    for (int64_t row = 0; row < n; ++row)
    {
        const float rowEigenvalue = static_cast<float>(row) - 1024.0f;
        for (int64_t col = row; col < n; ++col)
        {
            const float colEigenvalue = static_cast<float>(col) - 1024.0f;
            float value = -2.0f * (rowEigenvalue + colEigenvalue) / static_cast<float>(n);
            if (row == col)
            {
                value += rowEigenvalue;
            }
            a[static_cast<size_t>(row + col * n)] = Complex(value, 0.0f);
        }
    }

    int32_t info = -999;
    const aclError status = aclsolverCheevj(handle, ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_FILL_MODE_UPPER, n, a.data(),
                                            n, w.data(), &info);
    bool valid = status == ACL_SUCCESS && info == 0;
    for (int64_t index = 0; valid && index < n; ++index)
    {
        valid = NearlyEqual(w[static_cast<size_t>(index)], static_cast<float>(index) - 1024.0f, tolerance);
    }
    Check("n=2049 dense known-spectrum fallback", valid);
}

std::vector<Complex> MakePoisonedMatrix(aclsolverFillMode_t uplo, int64_t lda)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Complex poison(nan, nan);
    std::vector<Complex> a(static_cast<size_t>(lda * 2), Complex(913.0f, -271.0f));

    a[0] = Complex(2.0f, nan);
    a[1 + lda] = Complex(5.0f, -nan);
    if (uplo == ACLSOLVER_FILL_MODE_UPPER)
    {
        a[lda] = Complex(1.0f, 2.0f);
        a[1] = poison;
    }
    else
    {
        a[1] = Complex(1.0f, -2.0f);
        a[lda] = poison;
    }
    return a;
}

bool PaddingUnchanged(const std::vector<Complex> &before, const std::vector<Complex> &after, int64_t n, int64_t lda)
{
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = n; row < lda; ++row)
        {
            const size_t index = static_cast<size_t>(row + col * lda);
            if (std::memcmp(&before[index], &after[index], sizeof(Complex)) != 0)
            {
                return false;
            }
        }
    }
    return true;
}

bool EigenvectorsValid(const std::vector<Complex> &vectors, int64_t lda, const std::vector<float> &w)
{
    constexpr float tolerance = 2.0e-3f;
    const Complex a00(2.0f, 0.0f);
    const Complex a01(1.0f, 2.0f);
    const Complex a10(1.0f, -2.0f);
    const Complex a11(5.0f, 0.0f);

    for (int64_t col = 0; col < 2; ++col)
    {
        const Complex v0 = vectors[static_cast<size_t>(col * lda)];
        const Complex v1 = vectors[static_cast<size_t>(1 + col * lda)];
        const Complex r0 = a00 * v0 + a01 * v1 - w[col] * v0;
        const Complex r1 = a10 * v0 + a11 * v1 - w[col] * v1;
        if (std::abs(r0) > tolerance || std::abs(r1) > tolerance)
        {
            return false;
        }
        if (!NearlyEqual(std::norm(v0) + std::norm(v1), 1.0f, tolerance))
        {
            return false;
        }
    }

    const Complex dot = std::conj(vectors[0]) * vectors[static_cast<size_t>(lda)] +
                        std::conj(vectors[1]) * vectors[static_cast<size_t>(1 + lda)];
    return std::abs(dot) <= tolerance;
}

void CheckValidContracts(aclsolverHandle_t handle)
{
    constexpr int64_t n = 2;
    constexpr int64_t lda = 5;
    const float expectedLow = 0.5f * (7.0f - std::sqrt(29.0f));
    const float expectedHigh = 0.5f * (7.0f + std::sqrt(29.0f));
    std::vector<float> referenceW;

    for (const aclsolverEigMode_t jobz : {ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_EIG_MODE_VECTOR})
    {
        for (const aclsolverFillMode_t uplo : {ACLSOLVER_FILL_MODE_UPPER, ACLSOLVER_FILL_MODE_LOWER})
        {
            std::vector<Complex> a = MakePoisonedMatrix(uplo, lda);
            const std::vector<Complex> before = a;
            std::vector<float> w(n, 0.0f);
            int32_t info = -999;

            const aclError status = aclsolverCheevj(handle, jobz, uplo, n, a.data(), lda, w.data(), &info);
            const std::string label = std::string(jobz == ACLSOLVER_EIG_MODE_VECTOR ? "V" : "N") + "/" +
                                      (uplo == ACLSOLVER_FILL_MODE_UPPER ? "U" : "L") + "/lda5/poison";

            const bool eigenvaluesValid =
                status == ACL_SUCCESS && info == 0 && std::isfinite(w[0]) && std::isfinite(w[1]) && w[0] <= w[1] &&
                NearlyEqual(w[0], expectedLow, 2.0e-3f) && NearlyEqual(w[1], expectedHigh, 2.0e-3f);
            Check(label + " status/sorted/eigenvalues", eigenvaluesValid);
            Check(label + " lda padding unchanged", PaddingUnchanged(before, a, n, lda));

            if (jobz == ACLSOLVER_EIG_MODE_NOVECTOR)
            {
                Check(label + " input A bitwise unchanged",
                      std::memcmp(before.data(), a.data(), before.size() * sizeof(Complex)) == 0);
            }
            else
            {
                Check(label + " residual/orthogonality", eigenvaluesValid && EigenvectorsValid(a, lda, w));
            }

            if (referenceW.empty())
            {
                referenceW = w;
            }
            else
            {
                Check(label + " N/V and U/L eigenvalue equivalence",
                      NearlyEqual(w[0], referenceW[0], 2.0e-3f) && NearlyEqual(w[1], referenceW[1], 2.0e-3f));
            }
        }
    }
}

bool CheckComplexTwoByTwoEigenvectors(const std::vector<Complex> &a, const std::vector<float> &w, int64_t lda,
                                      const Complex &upper, double radius)
{
    const std::complex<double> v0(static_cast<double>(a[0].real()), static_cast<double>(a[0].imag()));
    const std::complex<double> v1(static_cast<double>(a[1].real()), static_cast<double>(a[1].imag()));
    const std::complex<double> v2(static_cast<double>(a[static_cast<size_t>(lda)].real()),
                                  static_cast<double>(a[static_cast<size_t>(lda)].imag()));
    const std::complex<double> v3(static_cast<double>(a[static_cast<size_t>(1 + lda)].real()),
                                  static_cast<double>(a[static_cast<size_t>(1 + lda)].imag()));
    const std::complex<double> z(static_cast<double>(upper.real()), static_cast<double>(upper.imag()));
    const double vectorNorm = std::norm(v0) + std::norm(v1);
    const std::complex<double> residual0 = z * v1 - static_cast<double>(w[0]) * v0;
    const std::complex<double> residual1 = std::conj(z) * v0 - static_cast<double>(w[0]) * v1;
    const std::complex<double> dot = std::conj(v0) * v2 + std::conj(v1) * v3;
    if (radius == 0.0)
    {
        return std::isfinite(vectorNorm) && std::abs(vectorNorm - 1.0) <= 2.0e-6 && std::abs(residual0) <= 2.0e-6 &&
               std::abs(residual1) <= 2.0e-6 && std::abs(dot) <= 2.0e-6;
    }
    return std::isfinite(vectorNorm) && std::abs(vectorNorm - 1.0) <= 2.0e-6 &&
           std::abs(residual0) / radius <= 2.0e-6 && std::abs(residual1) / radius <= 2.0e-6 && std::abs(dot) <= 2.0e-6;
}

void CheckComplexTwoByTwoCase(aclsolverHandle_t handle, float scale, aclsolverEigMode_t jobz, aclsolverFillMode_t uplo)
{
    constexpr int64_t n = 2;
    constexpr int64_t lda = 4;
    const Complex upper(scale, 0.5f * scale);
    const double radius = std::hypot(static_cast<double>(upper.real()), static_cast<double>(upper.imag()));
    const double tolerance = std::max(2.0e-6, radius * 2.0e-6);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<Complex> a(static_cast<size_t>(lda * n), Complex(913.0f, -271.0f));
    a[0] = Complex(0.0f, 0.0f);
    a[static_cast<size_t>(1 + lda)] = Complex(0.0f, 0.0f);
    if (uplo == ACLSOLVER_FILL_MODE_UPPER)
    {
        a[static_cast<size_t>(lda)] = upper;
        a[1] = Complex(poison, poison);
    }
    else
    {
        a[1] = std::conj(upper);
        a[static_cast<size_t>(lda)] = Complex(poison, poison);
    }
    const std::vector<Complex> before = a;
    std::vector<float> w(static_cast<size_t>(n), 0.0f);
    int32_t info = -999;
    const aclError status = aclsolverCheevj(handle, jobz, uplo, n, a.data(), lda, w.data(), &info);
    bool valid = status == ACL_SUCCESS && info == 0 && std::isfinite(w[0]) && std::isfinite(w[1]) && w[0] <= w[1] &&
                 std::abs(static_cast<double>(w[0]) + radius) <= tolerance &&
                 std::abs(static_cast<double>(w[1]) - radius) <= tolerance;
    valid = valid && PaddingUnchanged(before, a, n, lda);
    if (jobz == ACLSOLVER_EIG_MODE_NOVECTOR)
    {
        valid = valid && std::memcmp(before.data(), a.data(), before.size() * sizeof(Complex)) == 0;
    }
    else
    {
        valid = valid && CheckComplexTwoByTwoEigenvectors(a, w, lda, upper, radius);
    }
    const std::string label = std::string(jobz == ACLSOLVER_EIG_MODE_VECTOR ? "V" : "N") + "/" +
                              (uplo == ACLSOLVER_FILL_MODE_UPPER ? "U" : "L") +
                              (scale > 1.0e10f ? "/large-complex-2x2" : "/ordinary-complex-2x2");
    Check(label, valid);
}

void CheckComplexTwoByTwoScale(aclsolverHandle_t handle)
{
    for (const float scale : {2.0f, 2.0e20f})
    {
        for (const aclsolverEigMode_t jobz : {ACLSOLVER_EIG_MODE_NOVECTOR, ACLSOLVER_EIG_MODE_VECTOR})
        {
            for (const aclsolverFillMode_t uplo : {ACLSOLVER_FILL_MODE_UPPER, ACLSOLVER_FILL_MODE_LOWER})
            {
                CheckComplexTwoByTwoCase(handle, scale, jobz, uplo);
            }
        }
    }
}

std::vector<Complex> RepeatedTwoByTwoInput(int64_t n, aclsolverFillMode_t uplo)
{
    std::vector<Complex> input(static_cast<size_t>(n * n), Complex(0.0f, 0.0f));
    for (int64_t first = 0; first < n; first += 2)
    {
        const float diagonal = first < 4 ? -0.5f : 2.75f;
        const Complex upper(1.0e-8f * static_cast<float>(first + 1), -1.0e-8f);
        input[static_cast<size_t>(first + first * n)] = Complex(diagonal, 0.0f);
        input[static_cast<size_t>(first + 1 + (first + 1) * n)] = Complex(diagonal, 0.0f);
        if (uplo == ACLSOLVER_FILL_MODE_UPPER)
        {
            input[static_cast<size_t>(first + (first + 1) * n)] = upper;
        }
        else
        {
            input[static_cast<size_t>(first + 1 + first * n)] = std::conj(upper);
        }
    }
    return input;
}

void CheckRepeatedTwoByTwoOrthogonality(aclsolverHandle_t handle)
{
    constexpr int64_t n = 8;
    constexpr float tolerance = 2.0e-4f;
    for (const aclsolverFillMode_t uplo : {ACLSOLVER_FILL_MODE_UPPER, ACLSOLVER_FILL_MODE_LOWER})
    {
        std::vector<Complex> input = RepeatedTwoByTwoInput(n, uplo);
        std::vector<float> w(static_cast<size_t>(n));
        int32_t info = -999;
        const aclError status =
            aclsolverCheevj(handle, ACLSOLVER_EIG_MODE_VECTOR, uplo, n, input.data(), n, w.data(), &info);
        bool valid = status == ACL_SUCCESS && info == 0;
        for (int64_t column = 0; valid && column < n; ++column)
        {
            for (int64_t other = 0; other < n; ++other)
            {
                Complex dot(0.0f, 0.0f);
                for (int64_t row = 0; row < n; ++row)
                {
                    dot += std::conj(input[static_cast<size_t>(row + column * n)]) *
                           input[static_cast<size_t>(row + other * n)];
                }
                const Complex expected = column == other ? Complex(1.0f, 0.0f) : Complex(0.0f, 0.0f);
                valid = valid && std::abs(dot - expected) <= tolerance;
            }
        }
        Check(std::string("repeated 2x2 V/") + (uplo == ACLSOLVER_FILL_MODE_UPPER ? "U" : "L") + " orthogonality",
              valid);
    }
}

}  // namespace

int main(int argc, char **argv)
{
    const int deviceId = argc > 1 ? std::atoi(argv[1]) : 0;

    CheckInvalidParameters();

    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS)
    {
        std::cerr << "[FAIL] ACL initialization on device " << deviceId << '\n';
        return 1;
    }

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS || aclsolverCreate(&handle) != ACLSOLVER_STATUS_SUCCESS ||
        aclsolverSetStream(handle, stream) != ACLSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "[FAIL] stream/solver handle initialization\n";
        return 1;
    }

    CheckValidContracts(handle);
    CheckComplexTwoByTwoScale(handle);
    CheckRepeatedTwoByTwoOrthogonality(handle);
    CheckNoArtificialSizeLimit(handle);
    CheckDenseAboveFixedBackends(handle);

    const bool cleanupOk = aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS &&
                           aclrtDestroyStream(stream) == ACL_SUCCESS && aclrtResetDevice(deviceId) == ACL_SUCCESS &&
                           aclFinalize() == ACL_SUCCESS;
    Check("ACL cleanup", cleanupOk);

    std::cout << "Cheevj Contract Gate Summary\n"
              << "  Passed checks: " << gPassed << '\n'
              << "  Failed checks: " << gFailed << '\n';
    if (gFailed == 0)
    {
        std::cout << "[SUMMARY] All configured contract and boundary checks passed.\n";
    }
    return gFailed == 0 ? 0 : 1;
}
