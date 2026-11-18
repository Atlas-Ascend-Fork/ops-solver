/**
 * Copyright (c) 2026 Xztxzt
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#define main original_sgels_main
#include "sgels_test.cpp"
#undef main

namespace
{
int fixtureMode = 0;
int failures = 0;
int cases = 0;
void Check(bool got, bool expected, const char *label)
{
    ++cases;
    if (got != expected)
    {
        ++failures;
        printf("FAILED %s\n", label);
    }
}
}  // namespace

// Controlled output producer, not an implementation or simulation of SSgels.
extern "C" aclsolverStatus_t aclsolverSSgels_bufferSize(aclsolverHandle_t, int, int, int, float *, int, float *, int,
                                                        float *, int, void *, size_t *bytes)
{
    *bytes = 1;
    return ACLSOLVER_STATUS_SUCCESS;
}
extern "C" aclsolverStatus_t aclsolverSSgels(aclsolverHandle_t, int m, int n, int nrhs, float *a, int lda, float *b,
                                             int ldb, float *x, int ldx, void *, size_t, int *niter, int *info)
{
    std::vector<float> A(a, a + lda * n), B(b, b + ldb * nrhs), X;
    GelsCpu(m, n, nrhs, lda, ldb, A, B, X, ldx);
    std::copy(X.begin(), X.end(), x);
    for (int j = 0; j < nrhs; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            if (fixtureMode == 1) x[j * ldx + i] = std::numeric_limits<float>::infinity();
            if (fixtureMode == 2) x[j * ldx + i] = -std::numeric_limits<float>::infinity();
            if (fixtureMode == 3) x[j * ldx + i] = std::numeric_limits<float>::quiet_NaN();
            if (fixtureMode == 4) x[j * ldx + i] = 0;
        }
    }
    *niter = 0;
    *info = 0;
    return ACLSOLVER_STATUS_SUCCESS;
}

int main()
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int m : {2, 3})
    {
        // Exercise the full production flow, including repeat determinism and status checks.
        for (int mode = 0; mode < 5; ++mode)
        {
            fixtureMode = mode;
            Check(RunShape(reinterpret_cast<void *>(1), nullptr, 0, m, 2, 2, m, m, 3, 2, true, true, nullptr),
                  mode == 0, "RunShape");
        }
        const int n = 2, nrhs = 2, ldx = 3;
        std::vector<float> A(m * n, 0), B(m * nrhs, 0), X(ldx * nrhs, nan);
        for (int i = 0; i < n; ++i) A[i * m + i] = 1;
        for (int j = 0; j < nrhs; ++j)
        {
            for (int i = 0; i < n; ++i) B[j * m + i] = X[j * ldx + i] = i + j + 1;
        }
        // Padding is not part of the logical solution and may be nonfinite.
        Check(GelsVerifyAcc(m, n, nrhs, m, m, ldx, A, B, X, false), true, "NaN padding");
        for (int j = 0; j < nrhs; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for (float bad : {inf, -inf, nan})
                {
                    auto broken = X;
                    broken[j * ldx + i] = bad;
                    Check(GelsVerifyAcc(m, n, nrhs, m, m, ldx, A, B, broken, false), false,
                          "single nonfinite solution component");
                }
            }
        }
        // Check reduction helpers themselves: NaN must not disappear through std::max.
        for (double bad : {static_cast<double>(inf), static_cast<double>(nan)})
        {
            std::vector<double> residual(m * nrhs, 0);
            residual.back() = bad;
            double ratio = m == n ? Sqrt16(m, n, nrhs, ldx, residual, X, 1) : Sqrt17(m, n, nrhs, m, A, residual, 1, 3);
            Check(std::isfinite(ratio), false, "nonfinite residual");
        }
    }
    printf("cases=%d failures=%d; Host verdicts with controlled outputs only\n", cases, failures);
    return failures != 0;
}
