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
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#include <complex>
#endif

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
     * Calls using the same handle, including SetStream and Destroy, must be serialized
     * by the caller. Distinct handles may be used concurrently in valid CANN contexts.
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
    aclsolverStatus_t aclsolverCreate(aclsolverHandle_t* handle);

    /**
     * @brief Destroy an aclsolver handle
     *
     * Waits for streams associated with internal Cholesky scratch and releases the handle.
     * Those streams must remain valid until this call completes. Do not use the handle
     * concurrently or after destruction. A null stream denotes the CANN default stream.
     * On failure, other safely releasable blocks are still reclaimed. A valid handle
     * remains owned by the caller only for retrying Destroy; releases are not repeated.
     * Creation, compute and destruction must use the handle's owning CANN context.
     *
     * @param handle The handle to destroy (void*)
     * @return ACLSOLVER_STATUS_SUCCESS success
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR handle is null
     *         ACLSOLVER_STATUS_EXECUTION_FAILED stream/device synchronization failed
     *         ACLSOLVER_STATUS_INTERNAL_ERROR a scratch release failed
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
    aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t handle, aclrtStream* stream);

    /**
     * @brief Query the workspace for a complex Hermitian Cholesky factorization.
     *
     * No device work is launched. Workspace counts are complex elements, not bytes.
     * All Cholesky matrices are column major; unselected triangles are not read.
     * Compute calls and info writes are asynchronous on the handle stream, including
     * the default stream. Check the host return status before interpreting info:
     * SUCCESS and completed stream: info=0 or a documented positive numerical result;
     * INVALID_VALUE with valid handle/stream/info: a negative argument index excluding
     * handle. For allocation, initialization or execution failures, info is undefined.
     * Valid dimension bounds do not guarantee that inputs and scratch fit available HBM.
     * Cholesky handles are not safe for concurrent host-thread use. The caller must
     * serialize queries, SetStream, compute and Destroy on the same handle; Destroy
     * must not overlap any other thread using that handle. Different handles are independent.
     *
     * @param handle Initialized solver handle; calls on one handle must be serialized.
     * @param uplo LOWER or UPPER triangle.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param A Device matrix; may be null only for n=0. Not accessed by the query.
     * @param lda Column stride in complex elements, at least max(1,n).
     * @param Lwork Host output count; zero is valid and permits a null workspace.
     * @return SUCCESS, INVALID_VALUE for invalid arguments, or NOT_INITIALIZED for a null handle.
     */
    aclsolverStatus_t aclsolverCpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                                 aclsolverFloatComplex* A, int lda, int* Lwork);

    /**
     * @brief Factor a complex Hermitian positive definite matrix into L or U in place.
     * @param handle Initialized, caller-serialized handle; see aclsolverCpotrf_bufferSize for info validity.
     * @param uplo LOWER computes A=L*L^H; UPPER computes A=U^H*U.
     * @param n Matrix order, 0 <= n <= 4096; n=0 succeeds without matrix access.
     * @param A Device matrix; selected triangle is overwritten, lda padding is preserved.
     * @param lda Column stride in complex elements, at least max(1,n).
     * @param Workspace Device workspace; may be null when the queried count is zero.
     * @param Lwork Available workspace count in complex elements, at least the queried count.
     * @param devInfo Device scalar: zero on success; positive first non-positive pivot (one based);
     *                negative invalid argument index (excluding handle). Earlier columns are completed.
     * @return SUCCESS (including numerical failure reported in devInfo), INVALID_VALUE,
     *         NOT_INITIALIZED, or a handle/stream error. See the shared info validity contract.
     */
    aclsolverStatus_t aclsolverCpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                      aclsolverFloatComplex* A, int lda, aclsolverFloatComplex* Workspace, int Lwork,
                                      int* devInfo);

    /**
     * @brief Solve A*X=B using an existing complex Cholesky factor; overwrite B with X.
     * @param handle Initialized, caller-serialized handle; see aclsolverCpotrf_bufferSize for info validity.
     * @param uplo Select the stored LOWER factor L or UPPER factor U.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param nrhs Right-hand side count, 0 <= nrhs <= 128. n=0 or nrhs=0 is empty.
     * @param A Read-only device factor; may be null for an empty problem. Positive definiteness
     *          must be checked by the preceding factorization, not by this solve.
     * @param lda Factor column stride in complex elements, at least max(1,n).
     * @param B Device n-by-nrhs matrix, overwritten by X; may be null for an empty problem.
     * @param ldb B column stride in complex elements, at least max(1,n); padding is preserved.
     * @param devInfo Device scalar, zero or negative invalid argument index excluding handle.
     * @return SUCCESS, INVALID_VALUE, NOT_INITIALIZED, ALLOC_FAILED, or a handle/stream error.
     *         Internal scratch is retained until handle destruction; its streams must outlive the handle.
     */
    aclsolverStatus_t aclsolverCpotrs(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                      const aclsolverFloatComplex* A, int lda, aclsolverFloatComplex* B, int ldb,
                                      int* devInfo);

    /**
     * @brief Query workspace for the inverse from a complex Cholesky factor, without device work.
     * @param handle Initialized, caller-serialized solver handle.
     * @param uplo Select the stored LOWER or UPPER factor.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param A Device factor; may be null only for n=0. Not accessed by the query.
     * @param lda Column stride in complex elements, at least max(1,n).
     * @param Lwork Host output count in complex elements; zero permits a null workspace.
     * @return SUCCESS, INVALID_VALUE, or NOT_INITIALIZED. See aclsolverCpotrf_bufferSize for shared contracts.
     */
    aclsolverStatus_t aclsolverCpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                                 aclsolverFloatComplex* A, int lda, int* Lwork);

    /**
     * @brief Form the selected triangle of the inverse from a complex Cholesky factor in place.
     * @param handle Initialized, caller-serialized handle; see aclsolverCpotrf_bufferSize for info validity.
     * @param uplo Select the stored LOWER or UPPER factor and the inverse triangle to produce.
     * @param n Matrix order, 0 <= n <= 4096; n=0 succeeds without matrix access.
     * @param A Device factor; selected triangle is overwritten, lda padding is preserved.
     * @param lda Column stride in complex elements, at least max(1,n).
     * @param Workspace Device workspace; may be null when the queried count is zero.
     * @param Lwork Available workspace count in complex elements, at least the queried count.
     * @param devInfo Device scalar: zero, positive first zero real diagonal (one based), or
     *                negative invalid argument index excluding handle.
     * @return SUCCESS (including numerical failure in devInfo), INVALID_VALUE, NOT_INITIALIZED,
     *         or a handle/stream error. See the shared info validity contract.
     */
    aclsolverStatus_t aclsolverCpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                      aclsolverFloatComplex* A, int lda, aclsolverFloatComplex* Workspace, int Lwork,
                                      int* devInfo);

    /**
     * @brief Factor a batch of independent complex Hermitian matrices in place.
     * @param handle Initialized, caller-serialized handle; see aclsolverCpotrf_bufferSize for info validity.
     * @param uplo LOWER or UPPER triangle for every matrix.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param Aarray Device array of device matrix pointers; matrices may be disjoint. May be null
     *               when n=0 or batchSize=0; selected triangles are overwritten.
     * @param lda Column stride in complex elements, at least max(1,n); padding is preserved.
     * @param infoArray Device array of max(1,batchSize) integers: one result per matrix;
     *                  positive first non-positive pivot (one based). Argument errors use infoArray[0].
     *                  Numerical results require SUCCESS and stream completion; allocation,
     *                  initialization or execution failure leaves info undefined, even if zero.
     * @param batchSize Batch count, 0 <= batchSize <= 1000000. Empty batches still require writable info.
     * @return SUCCESS, INVALID_VALUE, NOT_INITIALIZED, ALLOC_FAILED, or a handle/stream error.
     *         Inputs and internal workspace must fit available HBM; dimension limits are independent.
     *         Large n and batchSize maxima cannot generally be combined; see docs/zh/cholesky.md.
     */
    aclsolverStatus_t aclsolverCpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                             aclsolverFloatComplex* Aarray[], int lda, int* infoArray, int batchSize);

    /**
     * @brief Solve a batch using read-only complex Cholesky factors; overwrite each B with X.
     * @param handle Initialized, caller-serialized handle; see aclsolverCpotrf_bufferSize for info validity.
     * @param uplo Select LOWER or UPPER factors for the whole batch.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param nrhs Must be 1 for n>0 and batchSize>0; empty problems accept 0 <= nrhs <= 128.
     * @param Aarray Read-only device pointer table of read-only factors, including disjoint matrices.
     *               May be null for n=0 or batchSize=0; factors must already be valid.
     * @param lda Factor column stride in complex elements, at least max(1,n).
     * @param Barray Device pointer table of writable n-by-1 matrices; the table is not modified.
     *               May be null for n=0 or batchSize=0.
     * @param ldb B column stride in complex elements, at least max(1,n); padding is preserved.
     * @param info Device scalar, zero or negative invalid argument index excluding handle.
     * @param batchSize Batch count, 0 <= batchSize <= 1000000; zero performs no matrix access.
     * @return SUCCESS, INVALID_VALUE, NOT_INITIALIZED, ALLOC_FAILED, or a handle/stream error.
     *         Inputs and scratch must fit available HBM. Scratch streams must outlive the handle.
     */
    aclsolverStatus_t aclsolverCpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                             const aclsolverFloatComplex* const Aarray[], int lda,
                                             aclsolverFloatComplex* Barray[], int ldb, int* info, int batchSize);

#ifdef __cplusplus
}
#endif

/*! \brief Eigenvector computation mode for Hermitian eigensolver APIs. */
typedef enum
{
    ACLSOLVER_EIG_MODE_NOVECTOR = 0,
    ACLSOLVER_EIG_MODE_VECTOR = 1
} aclsolverEigMode_t;

#ifdef __cplusplus
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
aclError aclsolverCmatinvBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float>* A, const int64_t lda,
                                 std::complex<float>* Ainv, const int64_t lda_inv, int32_t* info, int64_t batchSize);

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
aclError aclsolverCgetriBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float>* A, const int64_t lda,
                                std::complex<float>* Ainv, const int64_t lda_inv, int32_t* info, int64_t batchSize);

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
aclError aclsolverCgetri(aclsolverHandle_t handle, const int64_t n, std::complex<float>* A, const int64_t lda,
                         int32_t* info);

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
aclError aclsolverSgetri(aclsolverHandle_t handle, const int64_t n, float* A, const int64_t lda, int32_t* info);

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
aclError aclsolverCgetrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, std::complex<float>* A,
                         const int64_t lda, int32_t* ipiv, int32_t* info);

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

aclError aclsolverSpotrf(aclsolverHandle_t handle, const char uplo, const int64_t m, const int64_t n, float* A,
                         const int64_t lda, int32_t* info);

aclError aclsolverSgetrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, float* A, const int64_t lda,
                         int32_t* ipiv, int32_t* info);

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
                         std::complex<float>* A, const int64_t lda, float* W, int32_t* info);

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
aclError aclsolverSgeqrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, float* A, const int64_t lda,
                         float* tau, int32_t* info);

aclError aclsolverSgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs, float* A,
                         const int64_t lda, int32_t* ipiv, float* B, const int64_t ldb, int32_t* info);

aclError aclsolverCgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs,
                         std::complex<float>* A, const int64_t lda, int32_t* ipiv, std::complex<float>* B,
                         const int64_t ldb, int32_t* info);

/**
 * @brief Cholesky factorization of a single complex Hermitian positive-definite matrix (A = L * L^H or U^H * U)
 *
 * This C++ overload uses host row-major matrices. The Ascend950 C API above uses device
 * column-major matrices and an explicit workspace. The two contracts are independent.
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param uplo selects which triangular part of A is stored and where the factor is written:
 *             'U' for upper triangular (factorization A = U^H * U),
 *             'L' for lower triangular (factorization A = L * L^H).
 *             Both 'U' and 'L' are supported; only the uppercase forms are accepted, other
 *             values are rejected with ACL_ERROR_INVALID_PARAM. The factor is written into
 *             the triangular part selected by uplo (if uplo='U' the upper triangle contains U;
 *             if uplo='L' the lower triangle contains L).
 * @param n order of matrix A, must be positive
 * @param A input/output matrix, complex float (std::complex<float>), row-major with leading
 *          dimension lda; on return, contains the factor U (if uplo='U') or L (if uplo='L') in the
 *          specified triangular part; the other triangular part is unchanged.
 * @param lda Leading dimension of A; must be at least max(1, n).
 * @param info result info; not written in current version
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverCpotrf(aclsolverHandle_t handle, const char uplo, const int64_t n, std::complex<float>* A,
                         const int64_t lda, int32_t* info);

/**
 * @brief QR decomposition of a single complex matrix (A = Q * R)
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param m rows of matrix A, must be positive
 * @param n columns of matrix A, must be positive
 * @param A input/output matrix, complex float (std::complex<float>), contiguous m * n layout;
 *          on return, contains the upper triangular matrix R and the elementary reflectors
 *          used to construct Q (see tau)
 * @param lda leading dimension; must be at least max(1, m).
 * @param tau output scalar factors of the elementary reflectors (length min(m, n))
 * @param info result info; not written in current version
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverCgeqrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, std::complex<float>* A,
                         const int64_t lda, std::complex<float>* tau, int32_t* info);
#endif
