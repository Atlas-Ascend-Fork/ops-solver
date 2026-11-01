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
     * FLOAT32 dense real solver C APIs. Matrix, workspace, pivot and info buffers
     * are Device memory. Matrices use column-major storage with caller-provided
     * leading dimensions. Lwork is a Host output, measured in float elements;
     * query A may be null. Null pivots select factorization/solve without pivoting.
     * getrf returns packed LU and 1-based row pivots. getri consumes those factors
     * and pivots, then overwrites A with its inverse. Batched APIs use Device arrays
     * of Device pointers and independent output matrices. matinv accepts n<=32.
     * info is 0 on success, -i for an illegal argument excluding handle, or the
     * first positive zero U diagonal. Errors in an unusable info buffer cannot
     * be written to that buffer. Device metadata/value checks require caller-stream
     * synchronization to return INVALID_VALUE; matrix arithmetic stays on AI Core.
     */
    aclsolverStatus_t aclsolverSgetrf_bufferSize(aclsolverHandle_t handle, int m, int n, float* A, int lda, int* Lwork);
    aclsolverStatus_t aclsolverSgetrf(aclsolverHandle_t handle, int m, int n, float* A, int lda, float* Workspace,
                                      int* devIpiv, int* devInfo);
    aclsolverStatus_t aclsolverSgetri_bufferSize(aclsolverHandle_t handle, int n, float* A, int lda, int* Lwork);
    aclsolverStatus_t aclsolverSgetri(aclsolverHandle_t handle, int n, float* A, int lda, int* devIpiv,
                                      float* Workspace, int Lwork, int* devInfo);
    aclsolverStatus_t aclsolverSgetriBatched(aclsolverHandle_t handle, int n, const float* const Aarray[], int lda,
                                             const int* PivotArray, float* const Carray[], int ldc, int* infoArray,
                                             int batchSize);
    aclsolverStatus_t aclsolverSmatinvBatched(aclsolverHandle_t handle, int n, const float* const A[], int lda,
                                              float* const Ainv[], int lda_inv, int* info, int batchSize);

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


    /**
     * @brief Query the workspace for a real symmetric Cholesky factorization on Ascend 950PR/950DT.
     *
     * No device work is launched. Workspace counts are float elements, not bytes.
     * Real Cholesky matrices are column major; unselected triangles and padding are preserved.
     * Compute calls and info writes are asynchronous on the handle stream. Check the host
     * status and synchronize the stream before reading outputs. A successful enqueue can
     * still report a numerical failure in info. Info is undefined after allocation or
     * execution failures. Invalid argument indices exclude handle; a null handle writes no info.
     * Calls sharing a handle must be serialized, and scratch streams must outlive the handle.
     * Non-950 builds return ARCH_MISMATCH without modifying outputs; a null handle is checked first.
     *
     * @param handle Solver handle created by aclsolverCreate; calls on one handle must be serialized.
     * @param uplo Select the LOWER or UPPER triangle.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param A Device FLOAT32 matrix; may be null only for n=0. Not accessed by the query.
     * @param lda Column stride in float elements, at least max(1,n).
     * @param Lwork Non-null host output count in float elements; zero means no workspace is required.
     * @return ACLSOLVER_STATUS_SUCCESS, ACLSOLVER_STATUS_INVALID_VALUE,
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, or ACLSOLVER_STATUS_ARCH_MISMATCH.
     */
    aclsolverStatus_t aclsolverSpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float *A,
                                                 int lda, int *Lwork);

    /**
     * @brief Factor a real symmetric positive definite matrix into L or U in place.
     * @param handle Initialized, caller-serialized solver handle; see aclsolverSpotrf_bufferSize for shared contracts.
     * @param uplo LOWER computes A=L*L^T; UPPER computes A=U^T*U.
     * @param n Matrix order, 0 <= n <= 4096; n=0 performs no matrix access.
     * @param A Device FLOAT32 matrix; selected triangle is overwritten. May be null for n=0.
     * @param lda Column stride in float elements, at least max(1,n); padding is preserved.
     * @param Workspace Device workspace; may be null when Lwork=0 and the queried count is zero.
     * @param Lwork Available workspace count in float elements, at least the queried count.
     * @param devInfo Device scalar: zero on success, positive first non-positive pivot (one based),
     *                or negative first invalid argument index excluding handle. Earlier factor columns are retained.
     * @return ACLSOLVER_STATUS_SUCCESS (including numerical failure reported in devInfo),
     *         ACLSOLVER_STATUS_INVALID_VALUE, ACLSOLVER_STATUS_EXECUTION_FAILED,
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, or ACLSOLVER_STATUS_ARCH_MISMATCH.
     */
    aclsolverStatus_t aclsolverSpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float *A, int lda,
                                      float *Workspace, int Lwork, int *devInfo);

    /**
     * @brief Solve A*X=B using an existing real Cholesky factor; overwrite B with X.
     * @param handle Initialized, caller-serialized solver handle; see aclsolverSpotrf_bufferSize for shared contracts.
     * @param uplo Select the stored LOWER factor L or UPPER factor U.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param nrhs Right-hand side count, 0 <= nrhs <= 128; n=0 or nrhs=0 is empty.
     * @param A Read-only device FLOAT32 factor; may be null for an empty problem. The preceding
     *          factorization must have succeeded; this solve does not check positive definiteness or zero diagonals.
     * @param lda Factor column stride in float elements, at least max(1,n); all factor storage is preserved.
     * @param B Device n-by-nrhs matrix, overwritten by X; may be null for an empty problem.
     * @param ldb B column stride in float elements, at least max(1,n); padding is preserved.
     * @param devInfo Device scalar, zero or negative first invalid argument index excluding handle.
     * @return ACLSOLVER_STATUS_SUCCESS, ACLSOLVER_STATUS_INVALID_VALUE, ACLSOLVER_STATUS_ALLOC_FAILED,
     *         ACLSOLVER_STATUS_EXECUTION_FAILED, ACLSOLVER_STATUS_HANDLE_IS_NULLPTR,
     *         or ACLSOLVER_STATUS_ARCH_MISMATCH. Internal scratch is retained until handle destruction.
     */
    aclsolverStatus_t aclsolverSpotrs(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                      const float *A, int lda, float *B, int ldb, int *devInfo);

    /**
     * @brief Query workspace for the inverse from a real Cholesky factor, without device work.
     * @param handle Initialized, caller-serialized solver handle; see aclsolverSpotrf_bufferSize for shared contracts.
     * @param uplo Select the stored LOWER or UPPER factor.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param A Device FLOAT32 factor; may be null only for n=0. Not accessed by the query.
     * @param lda Column stride in float elements, at least max(1,n).
     * @param Lwork Non-null host output count in float elements; zero is returned for n=0.
     * @return ACLSOLVER_STATUS_SUCCESS, ACLSOLVER_STATUS_INVALID_VALUE,
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, or ACLSOLVER_STATUS_ARCH_MISMATCH.
     */
    aclsolverStatus_t aclsolverSpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float *A,
                                                 int lda, int *Lwork);

    /**
     * @brief Form the selected triangle of the inverse from a real Cholesky factor in place.
     * @param handle Initialized, caller-serialized solver handle; see aclsolverSpotrf_bufferSize for shared contracts.
     * @param uplo Select the stored LOWER or UPPER factor and the inverse triangle to produce.
     * @param n Matrix order, 0 <= n <= 4096; n=0 performs no matrix access.
     * @param A Device FLOAT32 factor; selected triangle is overwritten. May be null for n=0.
     * @param lda Column stride in float elements, at least max(1,n); padding is preserved.
     * @param Workspace Device workspace; query aclsolverSpotri_bufferSize independently of factorization.
     *                  May be null when n=0 and Lwork=0.
     * @param Lwork Available workspace count in float elements, at least the queried count.
     * @param devInfo Device scalar: zero on success, positive first non-finite or non-positive factor
     *                diagonal (one based), or negative first invalid argument index excluding handle.
     *                A is unchanged when the input diagonal check fails.
     * @return ACLSOLVER_STATUS_SUCCESS (including numerical failure reported in devInfo),
     *         ACLSOLVER_STATUS_INVALID_VALUE, ACLSOLVER_STATUS_EXECUTION_FAILED,
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, or ACLSOLVER_STATUS_ARCH_MISMATCH.
     */
    aclsolverStatus_t aclsolverSpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float *A, int lda,
                                      float *Workspace, int Lwork, int *devInfo);

    /**
     * @brief Factor a batch of real symmetric positive definite matrices in place using device pointer arrays.
     * @param handle Initialized, caller-serialized solver handle; see aclsolverSpotrf_bufferSize for shared contracts.
     * @param uplo Select the LOWER or UPPER triangle of every matrix.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param Aarray Device array of device FLOAT32 matrix pointers, including separately allocated matrices.
     *               Selected triangles are overwritten; pointer slots, other triangles and padding are preserved.
     *               May be null for n=0 or batchSize=0. A null slot sets only its infoArray entry to -3.
     * @param lda Matrix column stride in float elements, at least max(1,n).
     * @param infoArray Device array of batchSize entries: zero, positive first non-positive pivot (one based),
     *                  or negative invalid argument index excluding handle. Host argument errors fill all entries
     *                  when batchSize>0; batchSize=0 writes none. A successful enqueue can include failed slots.
     * @param batchSize Non-negative batch count; matrices and internal scratch must fit available device memory.
     * @return ACLSOLVER_STATUS_SUCCESS, ACLSOLVER_STATUS_INVALID_VALUE, ACLSOLVER_STATUS_ALLOC_FAILED,
     *         ACLSOLVER_STATUS_EXECUTION_FAILED, ACLSOLVER_STATUS_HANDLE_IS_NULLPTR,
     *         or ACLSOLVER_STATUS_ARCH_MISMATCH. Internal scratch is retained until handle destruction.
     */
    aclsolverStatus_t aclsolverSpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float *Aarray[],
                                             int lda, int *infoArray, int batchSize);

    /**
     * @brief Solve a batch of real systems from Cholesky factors; overwrite each right-hand side with its solution.
     * @param handle Initialized, caller-serialized solver handle; see aclsolverSpotrf_bufferSize for shared contracts.
     * @param uplo Select the stored LOWER or UPPER factor of every matrix.
     * @param n Matrix order, 0 <= n <= 4096.
     * @param nrhs Must be 1 when n>0 and batchSize>0; empty problems accept any non-negative value.
     * @param Aarray Device array of read-only device factors. All factor storage and pointer slots are preserved.
     *               May be null for n=0 or batchSize=0; every factor must have been successfully computed.
     * @param lda Factor column stride in float elements, at least max(1,n).
     * @param Barray Device array of device right-hand sides, overwritten by their solutions; pointer slots are
     *               preserved. May be null for n=0 or batchSize=0. Output buffers must be disjoint.
     * @param ldb Right-hand side column stride in float elements, at least max(1,n); padding is preserved.
     * @param info Device scalar, zero or negative invalid argument index excluding handle. Any null Aarray slot
     *             sets -4; otherwise any null Barray slot sets -6. These slot errors leave all right-hand sides intact.
     * @param batchSize Non-negative batch count, constrained by available device memory.
     * @return ACLSOLVER_STATUS_SUCCESS, ACLSOLVER_STATUS_INVALID_VALUE, ACLSOLVER_STATUS_EXECUTION_FAILED,
     *         ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, or ACLSOLVER_STATUS_ARCH_MISMATCH.
     */
    aclsolverStatus_t aclsolverSpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                             float *Aarray[], int lda, float *Barray[], int ldb, int *info,
                                             int batchSize);

    /**
     * @brief Query the workspace size, in elements, required by aclsolverSgeqrf.
     *
     * A is not read, so passing nullptr is accepted: callers commonly query the size
     * before allocating A.
     *
     * @param handle Solver handle created by aclsolverCreate. Unused by the query.
     * @param m Number of rows of A; must be non-negative. The kernel further limits m to at
     *          most 8192 (see MaxRowsSupported in the implementation); a larger m is rejected
     *          with ACLSOLVER_STATUS_NOT_SUPPORTED.
     * @param n Number of columns of A; must be non-negative.
     * @param A Input matrix in column-major storage. Not read by this query.
     * @param lda Leading dimension of A; must be at least max(1, m).
     * @param Lwork Output workspace size counted in float elements. 0 when m or n is 0.
     * @return ACLSOLVER_STATUS_SUCCESS on success, ACLSOLVER_STATUS_INVALID_VALUE on an
     *         invalid argument.
     */
    aclsolverStatus_t aclsolverSgeqrf_bufferSize(aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork);

    /**
     * @brief QR factorization of a general single-precision real matrix, A = Q * R.
     *
     * A uses column-major storage and any lda >= max(1, m) is accepted. On return the upper
     * triangle of A holds R and the elements below the diagonal hold the Householder vectors,
     * with TAU[i] the scalar of the i-th reflector. The sign convention follows LAPACK
     * sgeqrf, so repeated runs on the same input are bit-wise identical.
     *
     * @param handle Solver handle created by aclsolverCreate. The computation runs on the
     *               stream bound to the handle.
     * @param m Number of rows of A; must be non-negative. The kernel further limits m to at
     *          most 8192 (see MaxRowsSupported in the implementation); a larger m is rejected
     *          with ACLSOLVER_STATUS_NOT_SUPPORTED.
     * @param n Number of columns of A; must be non-negative.
     * @param A Input/output matrix in column-major storage, overwritten by R and V.
     *          Element magnitudes must satisfy 5e-20 < max|A| < 1.8e19. The panel
     *          factorization applies each reflector before normalizing it, which bounds the
     *          usable range on both sides: the in-panel dot products reach max|A|^2 (overflow
     *          above the upper bound) and the stored reflector scale is 1/max|A|^2 (overflow
     *          below the lower bound). Outside the window the output contains non-finite
     *          values, which callers can detect; no error code is returned. Both bounds were
     *          measured on Ascend950PR, and are roughly sqrt(FLT_MAX) and 1/sqrt(FLT_MAX).
     * @param lda Leading dimension of A; must be at least max(1, m).
     * @param TAU Output array of min(m, n) reflector scalars.
     * @param Workspace Device workspace of at least Lwork float elements.
     * @param Lwork Workspace size in elements, as returned by aclsolverSgeqrf_bufferSize.
     * @param devInfo Device int32 output: 0 on success, -i when the i-th argument is
     *                invalid (the handle is not counted).
     * @return ACLSOLVER_STATUS_SUCCESS when the kernel is submitted, otherwise a status code.
     */
    aclsolverStatus_t aclsolverSgeqrf(aclsolverHandle_t handle, int m, int n, float *A, int lda, float *TAU,
                                      float *Workspace, int Lwork, int *devInfo);

    /**
     * @brief Query the workspace size, in elements, required by aclsolverSorgqr.
     *
     * A and tau are not read, so passing nullptr is accepted: callers commonly query the
     * size before allocating them.
     *
     * @param handle Solver handle created by aclsolverCreate. Unused by the query.
     * @param m Number of rows of Q; must be non-negative and at most 4096; a larger m is
     *          rejected with ACLSOLVER_STATUS_NOT_SUPPORTED.
     * @param n Number of columns of Q; must satisfy 0 <= n <= m.
     * @param k Number of elementary reflectors; must satisfy 0 <= k <= n.
     * @param A Householder representation from aclsolverSgeqrf. Not read by this query.
     * @param lda Leading dimension of A; must be at least max(1, m).
     * @param tau Reflector scalars of length k. Not read by this query.
     * @param lwork Output workspace size counted in float elements. 0 when m or n is 0.
     * @return ACLSOLVER_STATUS_SUCCESS on success, ACLSOLVER_STATUS_INVALID_VALUE on an
     *         invalid argument.
     */
    aclsolverStatus_t aclsolverSorgqr_bufferSize(aclsolverHandle_t handle, int m, int n, int k, const float *A, int lda,
                                                 const float *tau, int *lwork);

    /**
     * @brief Generate the first n columns of the orthogonal factor Q = H(1) H(2) ... H(k).
     *
     * The input is the Householder representation produced by aclsolverSgeqrf, not an
     * unfactored matrix. A uses column-major storage and any lda >= max(1, m) is accepted;
     * Q overwrites A. With k = 0 the result is the first n columns of the identity, which is
     * written out explicitly (matching LAPACK sorgqr).
     *
     * @param handle Solver handle created by aclsolverCreate. The computation runs on the
     *               stream bound to the handle.
     * @param m Number of rows of Q; must be non-negative and at most 4096; a larger m is
     *          rejected with ACLSOLVER_STATUS_NOT_SUPPORTED.
     * @param n Number of columns of Q; must satisfy 0 <= n <= m.
     * @param k Number of elementary reflectors; must satisfy 0 <= k <= n.
     * @param A Input/output matrix in column-major storage, overwritten by Q.
     * @param lda Leading dimension of A; must be at least max(1, m).
     * @param tau Input array of k reflector scalars from aclsolverSgeqrf.
     * @param work Device workspace of at least lwork float elements.
     * @param lwork Workspace size in elements, as returned by aclsolverSorgqr_bufferSize.
     * @param devInfo Device int32 output: 0 on success, -i when the i-th argument is
     *                invalid (the handle is not counted).
     * @return ACLSOLVER_STATUS_SUCCESS when the kernel is submitted, otherwise a status code.
     */
    aclsolverStatus_t aclsolverSorgqr(aclsolverHandle_t handle, int m, int n, int k, float *A, int lda, const float *tau,
                                      float *work, int lwork, int *devInfo);

    /**
     * @brief Query the workspace size, in elements, required by aclsolverSormqr.
     *
     * A, tau and C are not read, so passing nullptr is accepted: callers commonly query the
     * size before allocating them.
     *
     * @param handle Solver handle created by aclsolverCreate. Unused by the query.
     * @param side ACLSOLVER_SIDE_LEFT applies op(Q) from the left, ACLSOLVER_SIDE_RIGHT from
     *             the right.
     * @param trans ACLSOLVER_OP_N applies Q; ACLSOLVER_OP_T and ACLSOLVER_OP_C both apply Q
     *              transposed (identical for real data).
     * @param m Number of rows of C; must be non-negative.
     * @param n Number of columns of C; must be non-negative.
     * @param k Number of elementary reflectors; must satisfy 0 <= k <= nq, where nq is m for
     *          side LEFT and n for side RIGHT; nq must be at most 4096, otherwise the call is
     *          rejected with ACLSOLVER_STATUS_NOT_SUPPORTED. nq is m for
     *          side LEFT and n for side RIGHT.
     * @param A Householder representation from aclsolverSgeqrf. Not read by this query.
     * @param lda Leading dimension of A; must be at least max(1, nq).
     * @param tau Reflector scalars of length k. Not read by this query.
     * @param C Matrix acted on by op(Q). Not read by this query.
     * @param ldc Leading dimension of C; must be at least max(1, m).
     * @param lwork Output workspace size counted in float elements. 0 when m, n or k is 0.
     * @return ACLSOLVER_STATUS_SUCCESS on success, ACLSOLVER_STATUS_INVALID_VALUE on an
     *         invalid argument.
     */
    aclsolverStatus_t aclsolverSormqr_bufferSize(aclsolverHandle_t handle, aclsolverSideMode_t side,
                                                 aclsolverOperation_t trans, int m, int n, int k, const float *A, int lda,
                                                 const float *tau, const float *C, int ldc, int *lwork);

    /**
     * @brief Multiply C by the orthogonal factor Q = H(1) H(2) ... H(k) of a QR factorization.
     *
     * Computes op(Q) * C for side LEFT and C * op(Q) for side RIGHT, overwriting C. The input
     * is the Householder representation produced by aclsolverSgeqrf, not an explicit Q. A and
     * C use column-major storage and any lda >= max(1, nq), ldc >= max(1, m) is accepted. With
     * k = 0 the operator is the identity and C is left unchanged.
     *
     * @param handle Solver handle created by aclsolverCreate. The computation runs on the
     *               stream bound to the handle.
     * @param side Side from which op(Q) is applied.
     * @param trans ACLSOLVER_OP_N applies Q; ACLSOLVER_OP_T and ACLSOLVER_OP_C both apply Q
     *              transposed (identical for real data).
     * @param m Number of rows of C; must be non-negative.
     * @param n Number of columns of C; must be non-negative.
     * @param k Number of elementary reflectors; must satisfy 0 <= k <= nq (nq is m for side
     *          LEFT and n for side RIGHT); nq must be at most 4096, otherwise the call is
     *          rejected with ACLSOLVER_STATUS_NOT_SUPPORTED.
     * @param A Input matrix in column-major storage holding the reflector vectors.
     * @param lda Leading dimension of A; must be at least max(1, nq).
     * @param tau Input array of k reflector scalars from aclsolverSgeqrf.
     * @param C Input/output matrix in column-major storage, overwritten in place.
     * @param ldc Leading dimension of C; must be at least max(1, m).
     * @param work Device workspace of at least lwork float elements.
     * @param lwork Workspace size in elements, as returned by aclsolverSormqr_bufferSize.
     * @param devInfo Device int32 output: 0 on success, -i when the i-th argument is
     *                invalid (the handle is not counted).
     * @return ACLSOLVER_STATUS_SUCCESS when the kernel is submitted, otherwise a status code.
     */
    aclsolverStatus_t aclsolverSormqr(aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                      int m, int n, int k, const float *A, int lda, const float *tau, float *C, int ldc,
                                      float *work, int lwork, int *devInfo);

    /**
     * @brief Query the workspace sizes, in bytes, required by aclsolverXlarft.
     *
     * V, tau and T are not read, so passing nullptr is accepted: callers commonly query the
     * sizes before allocating them. The host workspace is always 0 bytes for this
     * implementation.
     *
     * @param handle Solver handle created by aclsolverCreate. Unused by the query.
     * @param params Must be nullptr: this implementation ships no parameter object.
     * @param direct ACLSOLVER_DIRECT_FORWARD for H(1) H(2) ... H(k), ACLSOLVER_DIRECT_BACKWARD
     *               for H(k) ... H(2) H(1).
     * @param storev Must be ACLSOLVER_STOREV_COLUMNWISE.
     * @param n Order of the block reflector; must satisfy n >= k >= 0.
     * @param k Number of elementary reflectors, also the order of T; must be non-negative.
     * @param dataTypeV Element type of V; must be ACLSOLVER_R_32F.
     * @param V Reflector vectors in column-major storage. Not read by this query.
     * @param ldv Leading dimension of V; must be at least max(1, n).
     * @param dataTypeTau Element type of tau; must be ACLSOLVER_R_32F.
     * @param tau Reflector scalars of length k. Not read by this query.
     * @param dataTypeT Element type of T; must be ACLSOLVER_R_32F.
     * @param T Output triangular factor. Not read by this query.
     * @param ldt Leading dimension of T; must be at least max(1, k).
     * @param computeType Compute precision; must be ACLSOLVER_R_32F.
     * @param workspaceInBytesOnDevice Output device workspace size in bytes. 0 when n or k is 0.
     * @param workspaceInBytesOnHost Output host workspace size in bytes. Always 0.
     * @return ACLSOLVER_STATUS_SUCCESS on success, ACLSOLVER_STATUS_INVALID_VALUE on an invalid
     *         argument, ACLSOLVER_STATUS_NOT_SUPPORTED when the data types are not all FLOAT32.
     */
    aclsolverStatus_t aclsolverXlarft_bufferSize(aclsolverHandle_t handle, aclsolverParams_t params,
                                                 aclsolverDirectMode_t direct, aclsolverStorevMode_t storev, int64_t n,
                                                 int64_t k, aclsolverDataType_t dataTypeV, const void *V, int64_t ldv,
                                                 aclsolverDataType_t dataTypeTau, const void *tau,
                                                 aclsolverDataType_t dataTypeT, void *T, int64_t ldt,
                                                 aclsolverDataType_t computeType, size_t *workspaceInBytesOnDevice,
                                                 size_t *workspaceInBytesOnHost);

    /**
     * @brief Form the triangular factor T of a block reflector built from k elementary reflectors.
     *
     * With direct = FORWARD the block reflector is H(1) H(2) ... H(k) = I - V T V^T and T is
     * upper triangular; with BACKWARD it is H(k) ... H(2) H(1) and T is lower triangular. Only
     * the referenced triangle of T is written, matching LAPACK slarft; the other triangle is
     * left untouched. V uses column-major storage with the unit diagonal implicit: column i
     * has a 1 at row i (FORWARD) or at row n-k+i (BACKWARD), and the entries on the other side
     * of that 1 are not read. There is no devInfo output: argument errors are reported through
     * the return code only.
     *
     * @param handle Solver handle created by aclsolverCreate. The computation runs on the
     *               stream bound to the handle.
     * @param params Must be nullptr: this implementation ships no parameter object.
     * @param direct Order in which the reflectors are applied.
     * @param storev Must be ACLSOLVER_STOREV_COLUMNWISE.
     * @param n Order of the block reflector; must satisfy n >= k >= 0 and n <= 4096.
     * @param k Number of elementary reflectors; must satisfy 0 <= k <= min(n, 256).
     * @param dataTypeV Element type of V; must be ACLSOLVER_R_32F.
     * @param V Input reflector vectors in column-major storage.
     * @param ldv Leading dimension of V; must be at least max(1, n).
     * @param dataTypeTau Element type of tau; must be ACLSOLVER_R_32F.
     * @param tau Input array of k reflector scalars.
     * @param dataTypeT Element type of T; must be ACLSOLVER_R_32F.
     * @param T Output k x k triangular factor in column-major storage.
     * @param ldt Leading dimension of T; must be at least max(1, k).
     * @param computeType Compute precision; must be ACLSOLVER_R_32F.
     * @param bufferOnDevice Device workspace of at least workspaceInBytesOnDevice bytes.
     * @param workspaceInBytesOnDevice Device workspace size in bytes, as returned by the query.
     * @param bufferOnHost Host workspace; unused, may be nullptr.
     * @param workspaceInBytesOnHost Host workspace size in bytes; always 0 for this
     *                               implementation.
     * @return ACLSOLVER_STATUS_SUCCESS when the kernel is submitted, otherwise a status code.
     */
    aclsolverStatus_t aclsolverXlarft(aclsolverHandle_t handle, aclsolverParams_t params, aclsolverDirectMode_t direct,
                                      aclsolverStorevMode_t storev, int64_t n, int64_t k, aclsolverDataType_t dataTypeV,
                                      const void *V, int64_t ldv, aclsolverDataType_t dataTypeTau, const void *tau,
                                      aclsolverDataType_t dataTypeT, void *T, int64_t ldt,
                                      aclsolverDataType_t computeType, void *bufferOnDevice,
                                      size_t workspaceInBytesOnDevice, void *bufferOnHost,
                                      size_t workspaceInBytesOnHost);

    /**
     * @brief Query the device workspace size, in bytes, required by aclsolverSSgels.
     *
     * dA, dB, dX and dwork are not read, so passing nullptr is accepted: callers commonly
     * query the size before allocating them.
     *
     * @param handle Solver handle created by aclsolverCreate; must not be nullptr.
     * @param m Number of rows of A; must satisfy m >= n >= 0.
     * @param n Number of columns of A; must be non-negative.
     * @param nrhs Number of right-hand sides; must be non-negative.
     * @param dA Coefficient matrix. Not read by this query.
     * @param ldda Leading dimension of dA; must be at least max(1, m).
     * @param dB Right-hand side matrix. Not read by this query.
     * @param lddb Leading dimension of dB; must be at least max(1, m).
     * @param dX Solution matrix. Not read by this query.
     * @param lddx Leading dimension of dX; must be at least max(1, n).
     * @param dwork Ignored, kept for signature compatibility.
     * @param lwork_bytes Output workspace size in bytes. 0 when m, n or nrhs is 0.
     * @return ACLSOLVER_STATUS_SUCCESS on success, ACLSOLVER_STATUS_INVALID_VALUE on an
     *         invalid argument.
     */
    aclsolverStatus_t aclsolverSSgels_bufferSize(aclsolverHandle_t handle, int m, int n, int nrhs, float *dA, int ldda,
                                                 float *dB, int lddb, float *dX, int lddx, void *dwork,
                                                 size_t *lwork_bytes);

    /**
     * @brief Solve the least squares problem min ||A X - B|| for an m-by-n matrix with m >= n.
     *
     * The solution is computed by a QR factorization: A = Q R, then X = R^-1 (Q^T B)(1:n, :).
     *
     * A is assumed to have full column rank. If a diagonal element of R is exactly zero the
     * triangular solve follows IEEE semantics and propagates Inf/NaN into the affected
     * columns of dX, matching a BLAS strsm on a singular triangle; rank deficiency is NOT
     * reported through dinfo (unlike LAPACK sgels, whose info > 0 names the zero diagonal).
     * Callers that must detect rank deficiency should inspect dX for non-finite values.
     * All matrices use column-major storage and any ldda >= max(1, m), lddb >= max(1, m),
     * lddx >= max(1, n) is accepted. dA is overwritten by the factorization, dB is read only,
     * and dX must not alias dB. niter is a host scalar; this implementation performs a direct
     * QR solve with no iterative refinement, so it is always set to 0.
     *
     * @param handle Solver handle created by aclsolverCreate. The computation runs on the
     *               stream bound to the handle.
     * @param m Number of rows of A; must satisfy m >= n >= 0 and m <= 4096.
     * @param n Number of columns of A; must be non-negative and at most 4096.
     * @param nrhs Number of right-hand sides; must be non-negative.
     * @param dA Input/output matrix in column-major storage; its contents are destroyed.
     *           The internal QR step carries the same magnitude window as aclsolverSgeqrf:
     *           5e-20 < max|dA| < 1.8e19.
     * @param ldda Leading dimension of dA; must be at least max(1, m).
     * @param dB Input right-hand side matrix in column-major storage; not modified.
     * @param lddb Leading dimension of dB; must be at least max(1, m).
     * @param dX Output solution matrix (n by nrhs) in column-major storage. It must alias
     *           neither dB (dB is a read-only input) nor dA: the solve writes dX before it
     *           reads the triangular factor left in dA, so sharing either buffer silently
     *           corrupts the result. Both cases are rejected with dinfo = -8.
     * @param lddx Leading dimension of dX; must be at least max(1, n).
     * @param dWorkspace Device workspace of at least lwork_bytes bytes.
     * @param lwork_bytes Workspace size in bytes, as returned by aclsolverSSgels_bufferSize.
     * @param niter Host output: number of refinement iterations performed. Always 0 here.
     * @param dinfo Device int32 output: 0 on success, -i when the i-th argument is invalid
     *              (the handle is not counted).
     * @return ACLSOLVER_STATUS_SUCCESS when the work is submitted, otherwise a status code.
     */
    aclsolverStatus_t aclsolverSSgels(aclsolverHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB,
                                      int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter,
                                      int *dinfo);

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
 * @brief Cholesky factorization of a single real float symmetric positive-definite matrix
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param uplo 'L' computes the lower triangular factor (A = L * L^T), 'U' the upper factor
 * @param m rows of matrix A; must equal n (Cholesky requires a square n-by-n matrix)
 * @param n matrix order, valid range [1, 8192]
 * @param A input/output matrix, FLOAT32, n * n layout with column stride lda; holds the
 *          factor in the selected triangle on return
 * @param lda leading dimension, lda >= n and lda <= 16384 (the kernel stores the row
 *            gap of DataCopyExtParams in a uint16_t byte counter; with the 'U' factor
 *            the member column count shrinks per iteration, so the worst-case gap is
 *            bounded by lda itself, not by lda - n)
 * @param info result info pointer; must be a valid pointer (not dereferenced in the
 *             current version, reserved for future use)
 * @return aclError ACL_SUCCESS on success, ACL_ERROR_INVALID_PARAM on invalid input
 */
aclError aclsolverSpotrf(aclsolverHandle_t handle, const char uplo, const int64_t m, const int64_t n, float* A,
                         const int64_t lda, int32_t* info);

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
