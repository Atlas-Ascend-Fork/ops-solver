/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SOLVER_ADAPTER_H
#define SOLVER_ADAPTER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
#define SOLVER_ADAPTER_VERSION 1u
#define SOLVER_ADAPTER_VERSION_V2 2u
#if defined(__GNUC__)
#define SOLVER_OPTIONAL __attribute__((weak))
#else
#define SOLVER_OPTIONAL
#endif
    /* Transport: tightly packed row-major Host buffers. complex64 is interleaved
     * float32(real,imag). Actual DUT memory/layout is described separately. */
    typedef struct
    {
        uint32_t version, struct_size;
        const char *op, *dtype, *dut_abi, *dut_layout, *info_kind;
        const void *target_symbol; /* Address of the actual public DUT function. */
    } SolverAdapterDesc;
    typedef struct
    {
        uint32_t version, struct_size;
        int64_t n, nrhs, batch, lda, ldb;
        char uplo;
        const char *info_probe, *input_kind; /* matrix or cholesky_factor */
        size_t a_bytes, b_bytes, out_bytes, info_count;
    } SolverAdapterCase;
    const SolverAdapterDesc *solver_adapter_descriptor(void);
    /* Handle and stream are owned by harness. create must publish partially
     * allocated context through state so destroy can clean it after failure. */
    int solver_adapter_create(const SolverAdapterCase *c, void *handle, void *stream, void **state) SOLVER_OPTIONAL;
    /* Restore all modified inputs, outputs, info; never call the target operator. */
    int solver_adapter_reset(void *state, const void *a, size_t a_bytes, const void *b, size_t b_bytes) SOLVER_OPTIONAL;
    /* Return adapter status; separately return unmodified DUT API return code. */
    int solver_adapter_execute(void *state, int *dut_status);
    /* Harness synchronizes the supplied stream before readback. */
    int solver_adapter_readback(void *state, void *out, size_t out_bytes, int32_t *info,
                                size_t info_count) SOLVER_OPTIONAL;
    /* v2 keeps v1 structures/callbacks intact. All buffers remain compact Host
     * row-major; adapters marshal to the declared DUT layout/memory domain.
     * ipiv is int32, 1-based. tau has the same dtype as the matrix. */
    typedef struct
    {
        SolverAdapterCase base;
        int64_t m, k, ldc, ldx;
        char side, trans;
        size_t in_ipiv_count, in_tau_count, out_ipiv_count, out_tau_count;
    } SolverAdapterCaseV2;
    int solver_adapter_create_v2(const SolverAdapterCaseV2 *c, void *handle, void *stream,
                                 void **state) SOLVER_OPTIONAL;
    int solver_adapter_reset_v2(void *state, const void *a, size_t a_bytes, const void *b, size_t b_bytes,
                                const int32_t *ipiv, size_t ipiv_count, const void *tau,
                                size_t tau_bytes) SOLVER_OPTIONAL;
    int solver_adapter_readback_v2(void *state, void *out, size_t out_bytes, int32_t *info, size_t info_count,
                                   int32_t *ipiv, size_t ipiv_count, void *tau, size_t tau_bytes) SOLVER_OPTIONAL;
    int solver_adapter_destroy(void *state); /* null/partially initialized is valid */
#ifdef __cplusplus
}
#endif
#endif
