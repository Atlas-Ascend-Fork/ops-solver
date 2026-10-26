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
 * \file cann_ops_solver.h
 * \brief
 */

#pragma once
#include <acl/acl.h>

#include <complex>
#include <cstddef>
#include <cstdint>

#include "cann_ops_solver_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Create an aclsolver handle
     *
     * Allocates an aclsolver handle on the heap and outputs it via @p handle. The internal
     * structure is opaque to users; the returned void* is used with other APIs.
     *
     * @param handle Output parameter to receive the created handle pointer (void**). *handle
     *               must be nullptr before the call, otherwise it is treated as already existing
     *               to prevent memory leaks.
     * @return ACLSOLVER_STATUS_SUCCESS success
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle pointer is null
     *         ACLSOLVER_STATUS_INVALID_VALUE *handle is not null (prevent repeated creation causing memory leak)
     *         ACLSOLVER_STATUS_NOT_INITIALIZED CANN context not initialized
     *         ACLSOLVER_STATUS_ALLOC_FAILED memory allocation failed
     */
    aclsolverStatus_t aclsolverCreate(aclsolverHandle_t *handle);

    /**
     * @brief Destroy an aclsolver handle
     *
     * Releases all resources occupied by the handle created by aclsolverCreate.
     *
     * @param handle The handle to destroy (void*)
     * @return ACLSOLVER_STATUS_SUCCESS success
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
     */
    aclsolverStatus_t aclsolverDestroy(aclsolverHandle_t handle);

    /**
     * @brief Set the stream for a handle
     * @param handle aclsolver handle (void*)
     * @param stream The stream to set, nullptr means default stream
     * @return ACLSOLVER_STATUS_SUCCESS success
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
     */
    aclsolverStatus_t aclsolverSetStream(aclsolverHandle_t handle, aclrtStream stream);

    /**
     * @brief Get the stream of a handle
     * @param handle aclsolver handle (void*)
     * @param stream Output parameter, returns the current stream
     * @return ACLSOLVER_STATUS_SUCCESS success
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
     *         ACLSOLVER_STATUS_INVALID_VALUE stream output parameter is null
     */
    aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t handle, aclrtStream *stream);

#ifdef __cplusplus
}
#endif

/*! \brief Eigenvector computation mode for Hermitian eigensolver APIs. */
typedef enum
{
    ACLSOLVER_EIG_MODE_NOVECTOR = 0,
    ACLSOLVER_EIG_MODE_VECTOR = 1
} aclsolverEigMode_t;

/*! \brief Matrix triangle selection mode for Hermitian matrix inputs. */
typedef enum
{
    ACLSOLVER_FILL_MODE_LOWER = 0,
    ACLSOLVER_FILL_MODE_UPPER = 1
} aclsolverFillMode_t;

/**
 * @brief Batched complex matrix inversion for small matrices (n < 32)
 *
 * Computes the inverse of a batch of complex matrices. When n >= 32, the call is
 * automatically forwarded to aclsolverCgetriBatched semantics.
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param n order of each square matrix, valid range (0, 256]; this API targets n < 32
 * @param A input matrices, COMPLEX64, contiguous layout [batchSize, n, n]
 * @param lda leading dimension; current version requires lda == n
 * @param Ainv output inverse matrices, COMPLEX64, [batchSize, n, n]
 * @param lda_inv leading dimension of Ainv; current version requires lda_inv == n
 * @param info result info per batch (may be nullptr; issue #244); the current version zero-initializes only
 *             info[0] on entry and does not write per-batch status. Argument
 *             errors are reported via the return value, not via info.
 * @param batchSize number of matrices, valid range (0, 3000]
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverCmatinvBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                                 std::complex<float> *Ainv, const int64_t lda_inv, int32_t *info, int64_t batchSize);

/**
 * @brief Batched complex matrix inversion (n >= 32)
 *
 * Computes the inverse of a batch of complex matrices using LU-based flow.
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param n order of each square matrix, valid range [32, 256]
 * @param A input matrices, COMPLEX64, contiguous layout [batchSize, n, n]
 * @param lda leading dimension; current version requires lda == n
 * @param Ainv output inverse matrices, COMPLEX64, [batchSize, n, n]
 * @param lda_inv leading dimension of Ainv; current version requires lda_inv == n
 * @param info result info per batch (may be nullptr; issue #244); the current version zero-initializes only
 *             info[0] on entry and does not write per-batch status. Argument
 *             errors are reported via the return value, not via info.
 * @param batchSize number of matrices, valid range (0, 3000]
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverCgetriBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                                std::complex<float> *Ainv, const int64_t lda_inv, int32_t *info, int64_t batchSize);

/**
 * @brief Inversion of a single complex matrix
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param n order of the square matrix, valid range [1, 8192]
 * @param A input matrix: the original matrix A to invert. The operator performs
 *             LU decomposition internally (it does NOT consume LU factors from
 *             aclsolverCgetrf; passing packed LU output would silently produce
 *             a pivot-permuted result). COMPLEX64, contiguous n * n layout
 * @param lda leading dimension; current version requires lda == n
 * @param info result info (may be nullptr, issue #244); zero-initialized on entry.
 *             Argument errors are reported via the return value, not via info.
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverCgetri(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                         int32_t *info);

/**
 * @brief Inversion of a single real float matrix
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param n order of the square matrix, valid range [1, 8192]
 * @param A input matrix: the original matrix A to invert. The operator performs
 *             LU decomposition internally (it does NOT consume LU factors from
 *             aclsolverSgetrf; passing packed LU output would silently produce
 *             a pivot-permuted result). FLOAT32, contiguous n * n layout
 * @param lda leading dimension; current version requires lda == n
 * @param info result info (may be nullptr, issue #244); zero-initialized on entry.
 *             Argument errors are reported via the return value, not via info.
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverSgetri(aclsolverHandle_t handle, const int64_t n, float *A, const int64_t lda, int32_t *info);

/**
 * @brief LU decomposition of a single complex matrix (A = P * L * U)
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param m rows of matrix A, valid range [1, 8192]
 * @param n columns of matrix A, valid range [1, 8192]
 * @param A input/output matrix, COMPLEX64, contiguous m * n layout; holds L and U on return
 * @param lda leading dimension; current version requires lda == n
 * @param ipiv output pivot indices (1-based), length min(m, n). If the matrix is
 *             singular, the entry at the first singular column stays 0 (sentinel).
 * @param info result info (may be nullptr, issue #244); zero-initialized on entry.
 *             On success reports the 1-based index of the first singular column
 *             (LAPACK positive-info convention) if the matrix is singular, else
 *             remains 0. Argument errors (including n > 8192) are reported via
 *             the return value, not via info.
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverCgetrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, std::complex<float> *A,
                         const int64_t lda, int32_t *ipiv, int32_t *info);

/**
 * @brief LU decomposition of a single real float matrix (A = P * L * U)
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param m rows of matrix A, valid range [1, 8192]
 * @param n columns of matrix A, valid range [1, 8192]
 * @param A input/output matrix, FLOAT32, contiguous m * n layout; holds L and U on return
 * @param lda leading dimension; current version requires lda == n
 * @param ipiv output pivot indices (1-based), length min(m, n). If the matrix is
 *             singular, the entry at the first singular column stays 0 (sentinel).
 * @param info result info (may be nullptr, issue #244); zero-initialized on entry.
 *             On success reports the 1-based index of the first singular column
 *             (LAPACK positive-info convention) if the matrix is singular, else
 *             remains 0. Argument errors (including n > 8192) are reported via
 *             the return value, not via info.
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverSgetrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, float *A, const int64_t lda,
                         int32_t *ipiv, int32_t *info);


/**
 * @brief Compute eigenvalues and optionally eigenvectors of a complex Hermitian matrix.
 *
 * A uses column-major storage. W returns eigenvalues in ascending order. When jobz is
 * ACLSOLVER_EIG_MODE_VECTOR, A is overwritten by column-major eigenvectors.
 *
 * @param handle Solver handle created by aclsolverCreate.
 * @param jobz Selects eigenvalues only or eigenvalues and eigenvectors.
 * @param uplo Selects the lower or upper triangle of A as input.
 * @param n Non-negative order of the matrix. The practical size is constrained by
 *          available host and device memory.
 * @param A Input Hermitian matrix in column-major storage. In vector mode, overwritten
 *          with the eigenvectors stored by column.
 * @param lda Leading dimension of A; must be at least max(1, n).
 * @param W Output array of n real eigenvalues in ascending order.
 * @param info Output status: 0 on success, negative for an invalid argument, and
 *             positive for a numerical failure.
 * @return ACL_SUCCESS on a completed call, otherwise an ACL error code.
 */
aclError aclsolverCheevj(aclsolverHandle_t handle, aclsolverEigMode_t jobz, aclsolverFillMode_t uplo, const int64_t n,
                         std::complex<float> *A, const int64_t lda, float *W, int32_t *info);

/**
 * @brief QR decomposition of a single real float matrix (A = Q * R) using Householder reflections
 *
 * A uses column-major storage. On return, the upper triangle contains R and the elements
 * below the diagonal, together with tau, represent the orthogonal matrix Q. Q is not formed
 * explicitly.
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param m rows of matrix A, must be non-negative
 * @param n columns of matrix A, must be non-negative
 * @param A input/output matrix, FLOAT32, column-major m * n layout; holds R and Householder
 *          vectors on return
 * @param lda leading dimension of A; must be at least max(1, m)
 * @param tau output scalar factors of the Householder reflectors, length min(m, n)
 * @param info result info; not written in current version
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverSgeqrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, float *A, const int64_t lda,
                         float *tau, int32_t *info);

aclError aclsolverSgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs, float *A, const int64_t lda, int32_t *ipiv, float *B, const int64_t ldb, int32_t *info);

aclError aclsolverCgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs, std::complex<float> *A, const int64_t lda, int32_t *ipiv, std::complex<float> *B, const int64_t ldb, int32_t *info);