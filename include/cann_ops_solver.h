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
#ifdef __cplusplus
#include <acl/acl.h>
#else
#include <acl/acl_rt.h>
#endif
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
    aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t handle, aclrtStream* stream);

    /**
     * FLOAT32 dense real solver C APIs. Matrix, workspace, pivot and info buffers
     * are Device memory. Matrices use column-major storage with caller-provided
     * leading dimensions. Lwork is a Host output, measured in float elements;
     * query A may be null. Null pivots select factorization/solve without pivoting.
     * getrf returns packed LU and 1-based row pivots. getri consumes those factors
     * and pivots, then overwrites A with its inverse. Batched APIs use Device arrays
     * of Device pointers and independent output matrices. getrf enforces m<=8192,
     * getri and getriBatched enforce n<=4096, and matinv accepts n<=32; shapes
     * beyond these limits return INVALID_VALUE in both entries (the queries
     * write no status buffer; the execute entries additionally write devInfo=-1).
     * info is 0 on success, -i for an illegal argument excluding handle, or the
     * first positive zero U diagonal. Errors in an unusable info buffer cannot
     * be written to that buffer. Device metadata/value checks require caller-stream
     * synchronization to return INVALID_VALUE; matrix arithmetic stays on AI Core.
     * The solve stage below is the exception: aclsolverSgetrs/Cgetrs take Host
     * buffers and stage them internally, see their own contract comments.
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
     * @param devInfo Device scalar: zero, positive one-based index of the first invalid diagonal
     *                (non-positive real part, non-zero imaginary part, or non-finite value), or
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
     * @brief Query complex QR workspace without reading device data or launching kernels.
     * @param handle Non-null solver handle. QR APIs use column-major device matrices.
     * @param m Non-negative row count.
     * @param n Non-negative column count.
     * @param A Device matrix; may be null for an empty problem.
     * @param lda Column stride, at least max(1,m).
     * @param Lwork Host output workspace count in complex elements, not bytes.
     * @return SUCCESS, INVALID_VALUE, or HANDLE_IS_NULLPTR. On non-950 builds a non-null
     *         handle returns ARCH_MISMATCH without accessing inputs or outputs.
     */
    aclsolverStatus_t aclsolverCgeqrf_bufferSize(aclsolverHandle_t handle, int m, int n, aclsolverFloatComplex* A,
                                                 int lda, int* Lwork);
    /**
     * @brief Factor A=Q*R in place, retaining Householder vectors below the diagonal.
     * @param handle Solver handle; input validation synchronizes its stream and reads back
     *               a four-byte status. Main computation and info writes use the same stream.
     * @param m Non-negative row count.
     * @param n Non-negative column count.
     * @param A Device input/output matrix; padding is preserved. Null is allowed if empty.
     * @param lda Column stride, at least max(1,m).
     * @param TAU Device output coefficients, min(m,n) complex elements; may be null if empty.
     * @param Workspace Device scratch allocated according to the matching bufferSize query.
     * @param Lwork Available scratch size in complex elements.
     * @param devInfo Writable device integer: zero on success or negative argument index
     *                excluding handle, when valid storage is available for error reporting.
     * @return SUCCESS, INVALID_VALUE, HANDLE_IS_NULLPTR, or a handle/stream error.
     *         Non-950 calls with a non-null handle return ARCH_MISMATCH without memory access.
     */
    aclsolverStatus_t aclsolverCgeqrf(aclsolverHandle_t handle, int m, int n, aclsolverFloatComplex* A, int lda,
                                      aclsolverFloatComplex* TAU, aclsolverFloatComplex* Workspace, int Lwork,
                                      int* devInfo);
    /**
     * @brief Query workspace for explicitly forming Q; no device data is read.
     * @param handle Non-null solver handle.
     * @param m Row count, m >= n.
     * @param n Column count, n >= k.
     * @param k Non-negative number of reflectors.
     * @param A Device column-major packed reflectors; may be null for an empty problem.
     * @param lda Column stride, at least max(1,m).
     * @param tau Device reflector coefficients; may be null when k=0.
     * @param lwork Host output count in complex elements.
     * @return SUCCESS, INVALID_VALUE, HANDLE_IS_NULLPTR, or ARCH_MISMATCH on non-950 builds.
     */
    aclsolverStatus_t aclsolverCungqr_bufferSize(aclsolverHandle_t handle, int m, int n, int k,
                                                 const aclsolverFloatComplex* A, int lda,
                                                 const aclsolverFloatComplex* tau, int* lwork);
    /**
     * @brief Form m-by-n Q in place from k Householder reflectors.
     * @param handle Solver handle; synchronization follows aclsolverCgeqrf.
     * @param m Row count, m >= n.
     * @param n Column count, n >= k.
     * @param k Non-negative reflector count; k=0 produces identity columns.
     * @param A Device packed reflectors, overwritten with Q; padding is preserved.
     * @param lda Column stride, at least max(1,m).
     * @param tau Read-only device coefficients; may be null when k=0.
     * @param work Device scratch from the matching query; must not overlap input/output.
     * @param lwork Scratch size in complex elements.
     * @param devInfo Writable device status, zero or negative argument index excluding handle.
     * @return SUCCESS, INVALID_VALUE, HANDLE_IS_NULLPTR, a handle/stream error, or
     *         ARCH_MISMATCH for a non-null handle on non-950 builds.
     */
    aclsolverStatus_t aclsolverCungqr(aclsolverHandle_t handle, int m, int n, int k, aclsolverFloatComplex* A, int lda,
                                      const aclsolverFloatComplex* tau, aclsolverFloatComplex* work, int lwork,
                                      int* devInfo);
    /**
     * @brief Query workspace for applying Q or Q^H; no device data is read.
     * @param handle Non-null solver handle.
     * @param side LEFT or RIGHT application; reflector order is m or n respectively.
     * @param trans N or C; non-conjugate transpose T is unsupported.
     * @param m Non-negative row count of C.
     * @param n Non-negative column count of C.
     * @param k Reflector count between zero and the reflector order.
     * @param A Read-only device column-major packed reflectors.
     * @param lda Reflector stride, at least max(1,order).
     * @param tau Read-only device reflector coefficients.
     * @param C Device column-major target matrix.
     * @param ldc Target stride, at least max(1,m).
     * @param lwork Host output workspace count in complex elements.
     * @return SUCCESS, INVALID_VALUE, HANDLE_IS_NULLPTR, or ARCH_MISMATCH on non-950 builds.
     */
    aclsolverStatus_t aclsolverCunmqr_bufferSize(aclsolverHandle_t handle, aclsolverSideMode_t side,
                                                 aclsolverOperation_t trans, int m, int n, int k,
                                                 const aclsolverFloatComplex* A, int lda,
                                                 const aclsolverFloatComplex* tau, const aclsolverFloatComplex* C,
                                                 int ldc, int* lwork);
    /**
     * @brief Apply Q or Q^H from the left or right, overwriting C.
     * @param handle Solver handle; synchronization follows aclsolverCgeqrf.
     * @param side LEFT or RIGHT application.
     * @param trans N or C; T is unsupported.
     * @param m Non-negative target row count.
     * @param n Non-negative target column count.
     * @param k Reflector count, at most m for LEFT or n for RIGHT; zero is a no-op.
     * @param A Read-only device packed reflectors; inactive storage is not read.
     * @param lda Column stride, at least max(1,reflector order).
     * @param tau Read-only device coefficients.
     * @param C Device input/output target; leading-dimension padding is preserved.
     * @param ldc Target stride, at least max(1,m).
     * @param work Device scratch from the matching workspace query.
     * @param lwork Scratch size in complex elements.
     * @param devInfo Writable device status, zero or negative argument index excluding handle.
     * @return SUCCESS, INVALID_VALUE, HANDLE_IS_NULLPTR, a handle/stream error, or
     *         ARCH_MISMATCH for a non-null handle on non-950 builds.
     */
    aclsolverStatus_t aclsolverCunmqr(aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                      int m, int n, int k, const aclsolverFloatComplex* A, int lda,
                                      const aclsolverFloatComplex* tau, aclsolverFloatComplex* C, int ldc,
                                      aclsolverFloatComplex* work, int lwork, int* devInfo);
    /**
     * @brief Query complex least-squares workspace in bytes; no device data is read.
     * @param handle Non-null solver handle.
     * @param m Row count, m >= n; both entries enforce m,n <= 4096.
     * @param n Non-negative column count, n <= 4096.
     * @param nrhs Non-negative right-hand side count, nrhs <= 128; exceeding the
     *             range (when m, n, nrhs are all non-zero) makes this query return
     *             INVALID_VALUE (the query writes no status buffer; the execute
     *             entry writes dinfo=-1 for m/n, -3 for nrhs).
     * @param dA Device column-major coefficient matrix.
     * @param ldda Coefficient stride, at least max(1,m).
     * @param dB Device column-major right-hand sides.
     * @param lddb Right-hand-side stride, at least max(1,m).
     * @param dX Device column-major solution matrix.
     * @param lddx Solution stride, at least max(1,n).
     * @param dwork Reserved device pointer; ignored by this query.
     * @param lwork_bytes Host output scratch size in bytes, unlike the QR element counts.
     * @return SUCCESS, INVALID_VALUE, HANDLE_IS_NULLPTR, or ARCH_MISMATCH on non-950 builds.
     */
    aclsolverStatus_t aclsolverCCgels_bufferSize(aclsolverHandle_t handle, int m, int n, int nrhs,
                                                 aclsolverFloatComplex* dA, int ldda, aclsolverFloatComplex* dB,
                                                 int lddb, aclsolverFloatComplex* dX, int lddx, void* dwork,
                                                 size_t* lwork_bytes);
    /**
     * @brief Solve a square or overdetermined complex least-squares problem using QR.
     * @param handle Solver handle; synchronization follows aclsolverCgeqrf.
     * @param m Row count, m >= n with m, n <= 4096; underdetermined problems are
     *          rejected even if m=0, and exceeding the range (with n, nrhs non-zero)
     *          returns INVALID_VALUE with dinfo=-1.
     * @param n Non-negative column count, n <= 4096; exceeding (with m, nrhs
     *          non-zero) returns INVALID_VALUE with dinfo=-1.
     * @param nrhs Non-negative right-hand-side count, nrhs <= 128; n=0 or nrhs=0
     *             is empty, and nrhs > 128 (with m, n non-zero) returns
     *             INVALID_VALUE with dinfo=-3.
     * @param dA Device coefficient matrix, overwritten with packed QR factors.
     * @param ldda Coefficient stride, at least max(1,m).
     * @param dB Read-only device right-hand sides.
     * @param lddb Right-hand-side stride, at least max(1,m).
     * @param dX Device output solution, n-by-nrhs; padding is preserved.
     * @param lddx Solution stride, at least max(1,n).
     * @param dWorkspace Device scratch from the matching query; no input/output overlap.
     * @param lwork_bytes Available scratch in bytes.
     * @param niter Host output iteration count, zero for this direct solver.
     * @param dinfo Writable device status: zero, negative argument index excluding handle,
     *              or positive one-based zero-pivot position during back substitution.
     * @return SUCCESS (including numerical failure reported in dinfo), INVALID_VALUE,
     *         HANDLE_IS_NULLPTR, a handle/stream error, or ARCH_MISMATCH on non-950 builds.
     */
    aclsolverStatus_t aclsolverCCgels(aclsolverHandle_t handle, int m, int n, int nrhs, aclsolverFloatComplex* dA,
                                      int ldda, aclsolverFloatComplex* dB, int lddb, aclsolverFloatComplex* dX,
                                      int lddx, void* dWorkspace, size_t lwork_bytes, int* niter, int* dinfo);

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
    aclsolverStatus_t aclsolverSpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float* A,
                                                 int lda, int* Lwork);

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
    aclsolverStatus_t aclsolverSpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float* A, int lda,
                                      float* Workspace, int Lwork, int* devInfo);

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
                                      const float* A, int lda, float* B, int ldb, int* devInfo);

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
    aclsolverStatus_t aclsolverSpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float* A,
                                                 int lda, int* Lwork);

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
    aclsolverStatus_t aclsolverSpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float* A, int lda,
                                      float* Workspace, int Lwork, int* devInfo);

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
    aclsolverStatus_t aclsolverSpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, float* Aarray[],
                                             int lda, int* infoArray, int batchSize);

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
                                             float* Aarray[], int lda, float* Barray[], int ldb, int* info,
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
    aclsolverStatus_t aclsolverSgeqrf_bufferSize(aclsolverHandle_t handle, int m, int n, float* A, int lda, int* Lwork);

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
    aclsolverStatus_t aclsolverSgeqrf(aclsolverHandle_t handle, int m, int n, float* A, int lda, float* TAU,
                                      float* Workspace, int Lwork, int* devInfo);

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
    aclsolverStatus_t aclsolverSorgqr_bufferSize(aclsolverHandle_t handle, int m, int n, int k, const float* A, int lda,
                                                 const float* tau, int* lwork);

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
    aclsolverStatus_t aclsolverSorgqr(aclsolverHandle_t handle, int m, int n, int k, float* A, int lda,
                                      const float* tau, float* work, int lwork, int* devInfo);

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
                                                 aclsolverOperation_t trans, int m, int n, int k, const float* A,
                                                 int lda, const float* tau, const float* C, int ldc, int* lwork);

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
                                      int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc,
                                      float* work, int lwork, int* devInfo);

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
     *         argument, ACLSOLVER_STATUS_NOT_SUPPORTED when the data types are not all FLOAT32
     *         or when n > 4096 / k > 256 (same kernel limits as aclsolverXlarft, enforced here
     *         before the workspace size derivation to keep both entries consistent).
     */
    aclsolverStatus_t aclsolverXlarft_bufferSize(aclsolverHandle_t handle, aclsolverParams_t params,
                                                 aclsolverDirectMode_t direct, aclsolverStorevMode_t storev, int64_t n,
                                                 int64_t k, aclsolverDataType_t dataTypeV, const void* V, int64_t ldv,
                                                 aclsolverDataType_t dataTypeTau, const void* tau,
                                                 aclsolverDataType_t dataTypeT, void* T, int64_t ldt,
                                                 aclsolverDataType_t computeType, size_t* workspaceInBytesOnDevice,
                                                 size_t* workspaceInBytesOnHost);

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
                                      const void* V, int64_t ldv, aclsolverDataType_t dataTypeTau, const void* tau,
                                      aclsolverDataType_t dataTypeT, void* T, int64_t ldt,
                                      aclsolverDataType_t computeType, void* bufferOnDevice,
                                      size_t workspaceInBytesOnDevice, void* bufferOnHost,
                                      size_t workspaceInBytesOnHost);

    /**
     * @brief Query the device workspace size, in bytes, required by aclsolverSSgels.
     *
     * dA, dB, dX and dwork are not read, so passing nullptr is accepted: callers commonly
     * query the size before allocating them.
     *
     * @param handle Solver handle created by aclsolverCreate; must not be nullptr.
     * @param m Number of rows of A; must satisfy m >= n >= 0, with m, n <= 4096
     *          (GELS_MAX_DIM; both entries enforce it, exceeding returns INVALID_VALUE).
     * @param n Number of columns of A; must be non-negative, n <= 4096.
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
    aclsolverStatus_t aclsolverSSgels_bufferSize(aclsolverHandle_t handle, int m, int n, int nrhs, float* dA, int ldda,
                                                 float* dB, int lddb, float* dX, int lddx, void* dwork,
                                                 size_t* lwork_bytes);

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
    aclsolverStatus_t aclsolverSSgels(aclsolverHandle_t handle, int m, int n, int nrhs, float* dA, int ldda, float* dB,
                                      int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* niter,
                                      int* dinfo);

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
extern "C"
{
#endif

    /** Query workspace in aclFloatComplex elements. A is a column-major Device matrix.
     *  m<=4096 and n<=8192; exceeding makes this query return INVALID_VALUE (the
     *  query writes no status buffer; the execute entry writes devInfo=-1/-2). */
    aclsolverStatus_t aclsolverCgetrf_bufferSize(aclsolverHandle_t handle, int m, int n, aclFloatComplex* A, int lda,
                                                 int* Lwork);

    /** In-place LU with deterministic partial pivoting. Null devIpiv selects no pivoting.
     * A, Workspace, devIpiv and devInfo are Device pointers; lda >= max(1, m).
     * devInfo is 0, the first zero diagonal (1-based), or a negative argument index excluding handle.
      m<=4096 and n<=8192;
     * exceeding returns INVALID_VALUE with devInfo=-1 (m) / -2 (n). */
    aclsolverStatus_t aclsolverCgetrf(aclsolverHandle_t handle, int m, int n, aclFloatComplex* A, int lda,
                                      aclFloatComplex* Workspace, int* devIpiv, int* devInfo);

    /** Query inverse workspace in aclFloatComplex elements. n<=4096; exceeding
     *  makes this query return INVALID_VALUE (the query writes no status buffer;
     *  the execute entry writes devInfo=-1). */
    aclsolverStatus_t aclsolverCgetri_bufferSize(aclsolverHandle_t handle, int n, aclFloatComplex* A, int lda,
                                                 int* Lwork);

    /** In-place inverse from column-major LU and 1-based pivots; null pivots denote no pivoting.
     * All buffers are Device pointers. Workspace must not overlap A.
      n<=4096, exceeding returns INVALID_VALUE
     * with devInfo=-1. */
    aclsolverStatus_t aclsolverCgetri(aclsolverHandle_t handle, int n, aclFloatComplex* A, int lda, int* devIpiv,
                                      aclFloatComplex* Workspace, int Lwork, int* devInfo);

    /** Out-of-place inverse from LU. Aarray and Carray are Device arrays of Device pointers.
     * PivotArray contains n * batchSize pivots, or is null for no pivoting.
     * Input and output matrices must not overlap; output matrices must be mutually disjoint.
      n<=4096, exceeding returns INVALID_VALUE
     * with devInfo=-1 per batch. */
    aclsolverStatus_t aclsolverCgetriBatched(aclsolverHandle_t handle, int n, const aclFloatComplex* const Aarray[],
                                             int lda, const int* PivotArray, aclFloatComplex* const Carray[], int ldc,
                                             int* infoArray, int batchSize);

    /** Direct out-of-place inverse of original column-major matrices, 0 <= n <= 32.
     * A and Ainv are Device arrays of Device pointers. Each batch writes its own Device info.
     * Invalid dimensions return INVALID_VALUE. n > 32 never forwards to CgetriBatched.
     */
    aclsolverStatus_t aclsolverCmatinvBatched(aclsolverHandle_t handle, int n, const aclFloatComplex* const A[],
                                              int lda, aclFloatComplex* const Ainv[], int lda_inv, int* info,
                                              int batchSize);

#ifdef __cplusplus
}

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
 * A, W and info are HOST buffers: the call stages A to Device internally, runs the
 * selected path (device kernels or the documented host fallback), synchronizes, and
 * writes the eigenvalues to W, the eigenvectors back into A (vector mode) and the
 * status to info before returning. This differs from the Ascend 950 C APIs above,
 * which take Device buffers directly.
 *
 * @param handle Solver handle created by aclsolverCreate.
 * @param jobz ACLSOLVER_EIG_MODE_NOVECTOR or ACLSOLVER_EIG_MODE_VECTOR; the character
 *             forms 'N' and 'V' cast to the enum are accepted as well.
 * @param uplo ACLSOLVER_FILL_MODE_LOWER or ACLSOLVER_FILL_MODE_UPPER; the character
 *             forms 'L' and 'U' cast to the enum are accepted as well.
 * @param n Non-negative order of the matrix. n = 0 returns ACL_SUCCESS with info = 0
 *          without accessing A or W. The practical size is constrained by available
 *          host and device memory.
 * @param A Input Hermitian matrix in column-major storage. In vector mode, overwritten
 *          with the eigenvectors stored by column.
 * @param lda Leading dimension of A; must be at least max(1, n).
 * @param W Output array of n real eigenvalues in ascending order.
 * @param info Output status: 0 on success; negative for an invalid argument (-2 jobz,
 *             -3 uplo, -4 n, -5 A, -6 lda, -7 W); -9 host fallback exceeds the host
 *             memory budget, -10 host allocation failed, -11 unexpected host failure
 *             (-8 is reserved for the kernel-side "workspace not allocated" marker);
 *             and positive for a numerical failure.
 * @return ACL_SUCCESS on a completed call, otherwise an ACL error code.
 */
aclError aclsolverCheevj(aclsolverHandle_t handle, aclsolverEigMode_t jobz, aclsolverFillMode_t uplo, const int64_t n,
                         std::complex<float>* A, const int64_t lda, float* W, int32_t* info);

/* aclsolverSgetrs / aclsolverCgetrs 参数均为 C 兼容类型，按 C ABI 导出（issue #301：
 * 此前声明位于 extern "C" 块之外，C 消费者链接时得到 C++ mangled 符号导致 undefined reference） */
#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Solve a real system of linear equations using LU factors from aclsolverSgetrf.
     *
     * Unlike the other dense real solver C APIs above, A, ipiv, B and info are HOST
     * buffers: the call stages them to Device internally, runs the triangular solves
     * on AI Core, synchronizes, and overwrites B with the solution X before returning.
     *
     * @param handle Solver handle used to select the stream for the staged work.
     * @param trans 0 solves A * X = B, 1 solves A^T * X = B; other values are rejected.
     * @param n Order of the factored matrix, 1 <= n <= 8192.
     * @param nrhs Number of right-hand-side columns, 1 <= nrhs <= 8192.
     * @param A Host buffer with the packed LU factors; at least lda * n floats.
     * @param lda Leading dimension of A; lda >= n and lda - n <= 16383.
     * @param ipiv Host buffer with the 1-based row-interchange sequence as returned by
     *             aclsolverSgetrf; every entry must lie in [1, n], otherwise the call
     *             is rejected before any Device work with info set to -4.
     * @param B Host buffer with the right-hand sides; overwritten with the solution X;
     *          at least ldb * nrhs floats.
     * @param ldb Leading dimension of B; ldb >= nrhs and ldb - nrhs <= 16383.
     * @param info Host output; set to 0 once the call completes successfully, set to -4
     *             when an ipiv entry is outside [1, n], and left unmodified when the
     *             call is rejected for other invalid arguments.
     * @return ACL_SUCCESS, or ACL_ERROR_INVALID_PARAM for invalid arguments.
     */
    aclError aclsolverSgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs,
                             float* A, const int64_t lda, int32_t* ipiv, float* B, const int64_t ldb, int32_t* info);

    /**
     * @brief Solve a complex system of linear equations using LU factors from aclsolverCgetrf.
     *
     * A, ipiv, B and info are HOST buffers, staged to Device internally and written
     * back before returning, exactly like aclsolverSgetrs; B is overwritten with X.
     * A and B use the C-compatible aclFloatComplex type (two consecutive IEEE float32,
     * layout-identical to std::complex<float>), so C consumers can call this entry
     * without reinterpret_cast.
     *
     * @param handle Solver handle used to select the stream for the staged work.
     * @param trans 0 solves A * X = B, 1 solves A^T * X = B, 2 solves A^H * X = B;
     *              other values are rejected.
     * @param n Order of the factored matrix, 1 <= n <= 8192.
     * @param nrhs Number of right-hand-side columns, 1 <= nrhs <= 8192.
     * @param A Host buffer with the packed LU factors; at least lda * n complex elements.
     * @param lda Leading dimension of A; lda >= n and lda - n <= 8191.
     * @param ipiv Host buffer with the 1-based row-interchange sequence as returned by
     *             aclsolverCgetrf; every entry must lie in [1, n], otherwise the call
     *             is rejected before any Device work with info set to -4.
     * @param B Host buffer with the right-hand sides; overwritten with the solution X;
     *          at least ldb * nrhs complex elements.
     * @param ldb Leading dimension of B; ldb >= nrhs, ldb - nrhs <= 8191, and
     *            ldb - nrhs <= 4095 when nrhs > 4096.
     * @param info Host output; set to 0 once the call completes successfully, set to -4
     *             when an ipiv entry is outside [1, n], and left unmodified when the
     *             call is rejected for other invalid arguments.
     * @return ACL_SUCCESS, or ACL_ERROR_INVALID_PARAM for invalid arguments.
     */
    aclError aclsolverCgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs,
                             aclFloatComplex* A, const int64_t lda, int32_t* ipiv, aclFloatComplex* B,
                             const int64_t ldb, int32_t* info);

#ifdef __cplusplus
}
#endif

/**
 * @brief Cholesky factorization of a single complex Hermitian positive-definite matrix (A = L * L^H or U^H * U)
 *
 * @param handle aclsolver handle created by aclsolverCreate
 * @param uplo selects which triangular part of A is stored and where the factor is written:
 *             'U' for upper triangular (factorization A = U^H * U),
 *             'L' for lower triangular (factorization A = L * L^H).
 *             Both 'U' and 'L' are supported; only the uppercase forms are accepted, other
 *             values are rejected with ACL_ERROR_INVALID_PARAM. The factor is written into
 *             the triangular part selected by uplo (if uplo='U' the upper triangle contains U;
 *             if uplo='L' the lower triangle contains L).
 * @param n order of matrix A, valid range [1, 8192]
 * @param A input/output matrix, complex float (std::complex<float>), row-major with leading
 *          dimension lda; on return, contains the factor U (if uplo='U') or L (if uplo='L') in the
 *          specified triangular part; the other triangular part is unchanged.
 * @param lda Leading dimension of A; must be at least max(1, n) and at most 16384.
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
