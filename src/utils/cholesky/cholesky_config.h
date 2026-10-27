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

#include <cstdint>

#include "cann_ops_solver_common.h"

namespace Cholesky
{
// Arch35 execution widths and DMA alignment are distinct units, even when values coincide.
constexpr int WARP_SIZE = 32;
constexpr int MIN_SUBGROUP_LANES = WARP_SIZE / 8;
constexpr int MEDIUM_SUBGROUP_LANES = WARP_SIZE / 4;
constexpr int HALF_WARP_LANES = WARP_SIZE / 2;
constexpr int SIMD_FLOAT_LANES = 64;
constexpr int DMA_ALIGNMENT_BYTES = 32;
constexpr int COMPLEX_COMPONENTS = 2;
constexpr int COMPLEX_BYTES = sizeof(aclsolverFloatComplex);
constexpr int DMA_COMPLEX_ALIGNMENT = DMA_ALIGNMENT_BYTES / COMPLEX_BYTES;
constexpr int CUBE_ALIGNMENT = 16;
constexpr int FACTOR_BLOCK = 64;
constexpr int REGISTER_FACTOR_ORDER = WARP_SIZE;
constexpr int FACTOR_STRIP = 8;
constexpr int DIAGONAL_PREFETCH_COLUMNS = FACTOR_STRIP / COMPLEX_COMPONENTS;
constexpr int WHOLE_FACTOR_STRIP = 4;
constexpr int PANEL_STRIP = 4;
constexpr int ORDINARY_PANEL_STRIP = 3;
constexpr int INTERLEAVED_PANEL_ROWS = SIMD_FLOAT_LANES / COMPLEX_COMPONENTS;
constexpr int BATCH_THRESHOLD = 16;
constexpr int SMALL_RHS_LIMIT = 4;
constexpr int SMALL_SOLVE_BLOCK = 32;
constexpr int SINGLE_RHS_BLOCK = 64;
constexpr int MAX_SOLVE_BLOCK = 128;
constexpr int REGISTER_SOLVE_ORDER = 128;
constexpr int BATCH_REGISTER_ORDER = 256;
constexpr int FUSED_REGISTER_ORDER = 512;
constexpr int BUFFERED_WORKSPACE_ORDER = 1024;
constexpr int BUFFERED_SOLVE_ORDER = 2048;
constexpr int MIN_BATCHES_PER_CORE = 4;
constexpr int WHOLE_MATRICES_PER_AIV = 1;
constexpr int RHS_ACCUMULATORS = 4;

// These bounds select compensated arithmetic; retaining their values preserves FP32 behavior.
constexpr float COLUMN_ENERGY_BOUND = 2.0f;
constexpr float CONSERVATIVE_ENERGY_BOUND = 1.0f;
constexpr float FINITE_ROOT_BOUND = 3.4e38f;

// Each packed row contains both real and imaginary components of a factor block.
constexpr int GRAM_COMPONENT_ROWS = COMPLEX_COMPONENTS * FACTOR_BLOCK;
// First-column product remainders and a compensated diagonal state per matrix row.
constexpr int FACTOR_DIAGONAL_COMPONENTS = 2;
constexpr int FACTOR_AUXILIARY_FLOAT_PLANES = COMPLEX_COMPONENTS + FACTOR_DIAGONAL_COMPONENTS;
constexpr int NATIVE_RHS_ROWS = COMPLEX_COMPONENTS * MAX_SOLVE_BLOCK;
constexpr int NATIVE_SHARED_ELEMENTS = MAX_SOLVE_BLOCK;
constexpr int NATIVE_PREFETCH_PADDING = DMA_COMPLEX_ALIGNMENT;
constexpr int SCHUR_TILE = 128;
constexpr int BATCH_SCHUR_TILE = 64;
constexpr int SCHUR_REDUCTION_TILE = 64;
constexpr int SOLVE_REDUCTION_TILE = 64;
constexpr int SOLVE_L0_REDUCTION_TILE = 32;
constexpr int SCHUR_BUFFER_COUNT = 2;
constexpr int CROSS_CORE_MODE = 4;
constexpr int SECOND_AIV_FLAG_OFFSET = 16;
constexpr int GEMM_SWIZZLE_LOG = 3;
constexpr int GEMM_INPUT_STAGES = 2;
constexpr int GEMM_ACCUMULATOR_STAGES = 1;
constexpr int FLAGS_PER_SCHUR_BUFFER = 2;

// UB strides are in complex elements unless their name explicitly says FLOATS or BYTES.
constexpr int SIMD_PANEL_ROWS = SIMD_FLOAT_LANES;
constexpr int SIMD_PANEL_STRIDE = FACTOR_BLOCK + DMA_COMPLEX_ALIGNMENT;
constexpr int WHOLE_FLOAT_STRIDE = COMPLEX_COMPONENTS * SIMD_PANEL_STRIDE;
constexpr int WHOLE_MATRIX_BYTES = FACTOR_BLOCK * WHOLE_FLOAT_STRIDE * sizeof(float);
constexpr int WHOLE_UB_BYTES = 2 * WHOLE_MATRIX_BYTES + FACTOR_BLOCK * COMPLEX_BYTES + DMA_ALIGNMENT_BYTES;
// Reserve the original diagonal plane; raw DMA uses the padded diagonal plane below.
constexpr int PANEL_OFFSET = FACTOR_BLOCK * (FACTOR_BLOCK + 1);
constexpr int PANEL_ELEMENTS = FACTOR_BLOCK * SIMD_PANEL_STRIDE;
constexpr int PANEL_LOW_OFFSET = PANEL_OFFSET + PANEL_ELEMENTS;
constexpr int PANEL_INFO_OFFSET_BYTES = (PANEL_LOW_OFFSET + SIMD_PANEL_ROWS) * COMPLEX_BYTES;
constexpr int PANEL_RAW_DIAGONAL_BYTES = PANEL_INFO_OFFSET_BYTES + DMA_ALIGNMENT_BYTES;
constexpr int PANEL_RAW_PANEL_BYTES = PANEL_RAW_DIAGONAL_BYTES + PANEL_ELEMENTS * COMPLEX_BYTES;
constexpr int SIMD_PANEL_BYTES = PANEL_RAW_PANEL_BYTES + PANEL_ELEMENTS * COMPLEX_BYTES;

constexpr int PACKED_MATRICES_PER_AIV = 4;
// Inactive lanes address the first loaded diagonal; only live matrix slots are needed.
constexpr int PACKED_DIAGONAL_SLOTS = PACKED_MATRICES_PER_AIV;
constexpr int PACKED_DIAGONAL_STRIDE = PANEL_ELEMENTS + DMA_COMPLEX_ALIGNMENT;
constexpr int PACKED_PANEL_OFFSET = PACKED_DIAGONAL_SLOTS * PACKED_DIAGONAL_STRIDE;
constexpr int PACKED_LOW_OFFSET = PACKED_PANEL_OFFSET + PANEL_ELEMENTS;
constexpr int METADATA_FIELDS = 4;
constexpr int METADATA_COMPLETED_OFFSET = SIMD_FLOAT_LANES;
constexpr int METADATA_BASE_OFFSET = 2 * SIMD_FLOAT_LANES;
constexpr int METADATA_ROW_OFFSET = 3 * SIMD_FLOAT_LANES;
constexpr int METADATA_ELEMENTS = METADATA_FIELDS * SIMD_FLOAT_LANES;
constexpr int PACKED_METADATA_BYTES = (PACKED_LOW_OFFSET + SIMD_FLOAT_LANES) * COMPLEX_BYTES;
constexpr int PACKED_TAIL_BYTES = PACKED_METADATA_BYTES + METADATA_ELEMENTS * sizeof(uint32_t);
constexpr int PACKED_TAIL_ROWS = CUBE_ALIGNMENT;
constexpr int PACKED_PANEL_BYTES = PACKED_TAIL_BYTES + PACKED_TAIL_ROWS * SIMD_PANEL_STRIDE * COMPLEX_BYTES;

// Schur correction: two row planes, followed by two half-tile column planes, in FP32 elements.
constexpr int SCHUR_ROW_LOW = COMPLEX_COMPONENTS * SCHUR_TILE;
constexpr int SCHUR_COLUMN_HIGH = 2 * SCHUR_ROW_LOW;
constexpr int SCHUR_COLUMN_LOW = SCHUR_COLUMN_HIGH + COMPLEX_COMPONENTS * (SCHUR_TILE / 2);
constexpr int SCHUR_COLUMN_LOW_RELATIVE = SCHUR_COLUMN_LOW - SCHUR_COLUMN_HIGH;
constexpr int SCHUR_CORRECTION_FLOATS = SCHUR_COLUMN_LOW + COMPLEX_COMPONENTS * (SCHUR_TILE / 2);

// CANN 9.0 permits 224 KiB per vector launch, including 8 KiB declared by the SIMT kernel.
constexpr int VECTOR_KERNEL_UB_BYTES = 224 * 1024;
constexpr int SIMT_KERNEL_SHARED_BYTES = 8 * 1024;
constexpr int BUFFERED_UB_BYTES = VECTOR_KERNEL_UB_BYTES - SIMT_KERNEL_SHARED_BYTES;
constexpr int MAX_BUFFERED_WARPS = 32;
constexpr int MEDIUM_BUFFERED_WARPS = 24;
constexpr int INITIAL_BUFFERED_WARPS = 16;
constexpr int MIN_BUFFERED_WARPS = 4;

static_assert((MAX_SOLVE_BLOCK * (MAX_SOLVE_BLOCK + 1) + NATIVE_SHARED_ELEMENTS) * COMPLEX_BYTES <= BUFFERED_UB_BYTES,
              "native solve UB budget");
static_assert((3 * SCHUR_TILE * SCHUR_TILE + SCHUR_CORRECTION_FLOATS) * sizeof(float) <= BUFFERED_UB_BYTES,
              "Schur UB budget");

enum class NativeStage
{
    PackRhs,
    ApplyTrailing,
    CopyDiagonal,
    SolveDiagonal
};

// The host workspace query and AICore dispatch must select the same tile and UB allocation.
#if defined(__CCE__)
#define CHOLESKY_POLICY_FUNC __forceinline__[host, aicore]
#else
#define CHOLESKY_POLICY_FUNC inline
#endif
CHOLESKY_POLICY_FUNC constexpr int SolveBlockSize(int nrhs, int batch = 1)
{
    return nrhs <= SMALL_RHS_LIMIT && batch >= BATCH_THRESHOLD ? SMALL_SOLVE_BLOCK
           : nrhs == 1                                         ? SINGLE_RHS_BLOCK
                                                               : MAX_SOLVE_BLOCK;
}
CHOLESKY_POLICY_FUNC constexpr bool BufferedSolvePolicy(int n, int nrhs, int batch, int vectorCores)
{
    return vectorCores > 0 && n > 0 && nrhs > 0 && batch >= BATCH_THRESHOLD && nrhs <= SMALL_RHS_LIMIT &&
           n <= BUFFERED_SOLVE_ORDER &&
           (n <= BUFFERED_WORKSPACE_ORDER || batch > static_cast<int64_t>(MIN_BATCHES_PER_CORE) * vectorCores);
}
CHOLESKY_POLICY_FUNC int BufferedWarps(int n)
{
    int padded = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    int warps = padded <= BUFFERED_UB_BYTES / (MAX_BUFFERED_WARPS * COMPLEX_BYTES)      ? MAX_BUFFERED_WARPS
                : padded <= BUFFERED_UB_BYTES / (MEDIUM_BUFFERED_WARPS * COMPLEX_BYTES) ? MEDIUM_BUFFERED_WARPS
                                                                                        : INITIAL_BUFFERED_WARPS;
    while (warps > MIN_BUFFERED_WARPS && padded * warps * COMPLEX_BYTES > BUFFERED_UB_BYTES)
    {
        warps /= 2;
    }
    return warps;
}
CHOLESKY_POLICY_FUNC constexpr int NativeSolveBytes(int block)
{
    int stride = block + (block == SINGLE_RHS_BLOCK ? NATIVE_PREFETCH_PADDING : 1);
    return (block * stride + NATIVE_SHARED_ELEMENTS) * COMPLEX_BYTES;
}
CHOLESKY_POLICY_FUNC constexpr int PackedRowShift(int rows)
{
    int shift = 0;
    while ((1 << shift) < rows)
    {
        ++shift;
    }
    return shift;
}
CHOLESKY_POLICY_FUNC constexpr int UbPaddingBlocks(int stride, int elements)
{
    int aligned = (elements + DMA_COMPLEX_ALIGNMENT - 1) / DMA_COMPLEX_ALIGNMENT * DMA_COMPLEX_ALIGNMENT;
    return (stride - aligned) / DMA_COMPLEX_ALIGNMENT;
}
#undef CHOLESKY_POLICY_FUNC

static_assert(COMPLEX_BYTES == COMPLEX_COMPONENTS * sizeof(float), "complex component layout");
static_assert(SIMD_PANEL_ROWS == FACTOR_BLOCK, "panel rows match vector lanes");
static_assert(PACKED_MATRICES_PER_AIV * PACKED_TAIL_ROWS <= SIMD_FLOAT_LANES, "packed panel lane capacity");
static_assert(WHOLE_UB_BYTES <= BUFFERED_UB_BYTES, "whole factor UB budget");
static_assert(SIMD_PANEL_BYTES <= BUFFERED_UB_BYTES, "panel UB budget");
static_assert(PACKED_PANEL_BYTES <= BUFFERED_UB_BYTES, "packed panel UB budget");
static_assert(PACKED_TAIL_BYTES % DMA_ALIGNMENT_BYTES == 0, "packed tail alignment");
}  // namespace Cholesky
