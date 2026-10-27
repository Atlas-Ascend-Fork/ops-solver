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
#include <cstddef>

#include "cann_ops_solver.h"
#include "cholesky_config.h"

namespace Cholesky
{
constexpr int MAX_ORDER = 4096;
constexpr int MAX_RHS = 128;
constexpr int MAX_BATCH = 1000000;

inline bool IsUpper(aclsolverFillMode_t uplo) { return uplo == ACLSOLVER_FILL_MODE_UPPER; }
inline bool ValidUplo(aclsolverFillMode_t uplo)
{
    return uplo == ACLSOLVER_FILL_MODE_LOWER || uplo == ACLSOLVER_FILL_MODE_UPPER;
}
inline int FactorArg(aclsolverFillMode_t uplo, int n, const void* a, int lda, bool emptyBatch = false)
{
    if (!ValidUplo(uplo))
    {
        return 1;
    }
    if (n < 0 || n > MAX_ORDER)
    {
        return 2;
    }
    if (!a && n > 0 && !emptyBatch)
    {
        return 3;
    }
    if (lda < (n > 0 ? n : 1))
    {
        return 4;
    }
    return 0;
}
inline int SolveArg(aclsolverFillMode_t uplo, int n, int nrhs, const void* a, int lda, const void* b, int ldb,
                    bool emptyBatch = false)
{
    if (!ValidUplo(uplo))
    {
        return 1;
    }
    if (n < 0 || n > MAX_ORDER)
    {
        return 2;
    }
    if (nrhs < 0 || nrhs > MAX_RHS)
    {
        return 3;
    }
    if (!a && n > 0 && nrhs > 0 && !emptyBatch)
    {
        return 4;
    }
    if (lda < (n > 0 ? n : 1))
    {
        return 5;
    }
    if (!b && n > 0 && nrhs > 0 && !emptyBatch)
    {
        return 6;
    }
    if (ldb < (n > 0 ? n : 1))
    {
        return 7;
    }
    return 0;
}

aclsolverStatus_t ReportError(aclsolverStatus_t status, const char* message);
aclsolverStatus_t Finish(aclsolverHandle_t handle, int* info, int argument);
void Factor(aclrtStream stream, bool upper, int n, aclsolverFloatComplex* a, aclsolverFloatComplex** array, int lda,
            int* info, int batch, aclsolverFloatComplex* work = nullptr);
// Workspace helpers return complex elements for ONE matrix; callers multiply by batch.
// batch selects the algorithm only. A zero count is valid when no scratch is needed.
inline int FactorWorkspace(int n, int batch = 1)
{
    int padded = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    return n <= REGISTER_FACTOR_ORDER || (batch >= BATCH_THRESHOLD && n <= FACTOR_BLOCK)
               ? 0
               : padded * GRAM_COMPONENT_ROWS + padded * FACTOR_AUXILIARY_FLOAT_PLANES / COMPLEX_COMPONENTS;
}
void Schur(aclrtStream stream, int rows, float* real, float* imag, int capacity, int batch, bool upper, int n,
           aclsolverFloatComplex* a, aclsolverFloatComplex** array, int lda, int offset, const int* info);
void Solve(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
           const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b,
           aclsolverFloatComplex** barray, int ldb, int batch, aclsolverFloatComplex* work = nullptr);
inline bool UseNativeSolve(int n, int nrhs, int batch = 1)
{
    int registerLimit =
        batch >= BATCH_THRESHOLD && nrhs <= SMALL_RHS_LIMIT ? BATCH_REGISTER_ORDER : REGISTER_SOLVE_ORDER;
    return n > registerLimit || (n > WARP_SIZE && nrhs >= WARP_SIZE);
}
// Per-matrix complex elements used by the native matrix-product path.
inline int ProductWorkspace(int n, int nrhs)
{
    return (NATIVE_RHS_ROWS + COMPLEX_COMPONENTS * ((n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT)) *
           ((nrhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT);
}
// Host sizing and launch dispatch query the same platform and shared buffered policy.
bool UseBufferedSolve(int n, int nrhs, int batch);
// Per-matrix core solve scratch; register/buffered paths may need none.
inline int CoreSolveWorkspace(int n, int nrhs, int batch = 1)
{
    if (UseBufferedSolve(n, nrhs, batch))
    {
        return 0;
    }
    int width = SolveBlockSize(nrhs, batch);
    if (width == 0)
    {
        return 0;
    }
    return UseNativeSolve(n, nrhs, batch) ? ProductWorkspace(n, nrhs) + 2 * ((n + width - 1) / width) * width * width
                                          : 0;
}
// Per-matrix total, including RHS backups and flags; empty problems return zero.
inline int SolveWorkspace(int n, int nrhs, int batch = 1)
{
    int flags = (n + WARP_SIZE - 1) / WARP_SIZE;
    if (UseBufferedSolve(n, nrhs, batch) && flags < nrhs)
    {
        flags = nrhs;
    }
    return n && nrhs ? CoreSolveWorkspace(n, nrhs, batch) + 2 * n * nrhs + flags : 0;
}
// Single-matrix inverse workspace, including the temporary identity matrix.
inline int InverseWorkspace(int n) { return n * n + SolveWorkspace(n, n); }
inline aclsolverStatus_t QueryWorkspace(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                        aclsolverFloatComplex* a, int lda, int* lwork, bool inverse)
{
    if (!handle)
    {
        return ReportError(ACLSOLVER_STATUS_NOT_INITIALIZED, "Cholesky handle is null");
    }
    if (FactorArg(uplo, n, a, lda) || !lwork)
    {
        return ReportError(ACLSOLVER_STATUS_INVALID_VALUE, "invalid Cholesky buffer-size argument");
    }
    *lwork = inverse ? InverseWorkspace(n) : FactorWorkspace(n);
    return ACLSOLVER_STATUS_SUCCESS;
}
aclsolverStatus_t RunInPlace(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, aclsolverFloatComplex* a,
                             int lda, aclsolverFloatComplex* work, int lwork, int* info, bool inverse);
aclsolverStatus_t AcquireScratch(aclsolverHandle_t handle, aclrtStream* stream, size_t bytes,
                                 aclsolverFloatComplex** work);
void SolveProduct(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                  const aclsolverFloatComplex* const* array, int lda, float* work, int batch, int offset, int width,
                  bool backward, const int* check, bool diagonal = false, int activeRhs = -1);
void Inverse(aclrtStream stream, bool upper, int n, aclsolverFloatComplex* a, int lda, aclsolverFloatComplex* work,
             int* info);
}  // namespace Cholesky
