/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

// Device API declarations precede the implementation fragments below.
#include "../utils/complex_solver_kernel.hpp"
#include "../utils/complex_solver_launch.h"
#include "../utils/complex_solver_small.hpp"
#include "../utils/complex_solver_update.hpp"

namespace
{
constexpr uint64_t RESIDENT_ELEMENTS = 14 * 1024;
}

#if __NPU_ARCH__ == 3510
using namespace ComplexSolverKernel;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void FactorPanelVf(ComplexSolverShape shape,
                                                                       __gm__ aclFloatComplex* a, __gm__ int* pivots,
                                                                       __gm__ int* info, int begin, int end)
{
    __ubuf__ float scores[THREADS / 32];
    __ubuf__ int rows[THREADS / 32];
    FactorPanel(a, shape.m, shape.n, shape.lda, begin, end, pivots, info, scores, rows);
}

__simt_callee__ __aicore__ inline aclFloatComplex NormalizeCachedValue(aclFloatComplex value,
                                                                       const PreparedDivisor& divisor,
                                                                       float inverseScale)
{
    if (inverseScale == 0.0f) return PreparedDivide(value, divisor);
    float real = value.real * inverseScale;
    float imag = value.imag * inverseScale;
    return {PreparedQuotient(real, divisor.real, imag, divisor.imag, divisor),
            PreparedQuotient(imag, divisor.real, -real, divisor.imag, divisor)};
}

__simt_callee__ __aicore__ inline void ReduceCachedPivot(float& score, int& row, int height)
{
    float maximum = asc_reduce_max(score);
    row = asc_reduce_min(score == maximum ? row : height);
    score = maximum;
}

__simt_callee__ __aicore__ inline int FindCachedPivot(__ubuf__ aclFloatComplex* panel, int height, int column,
                                                      __ubuf__ float* scores, __ubuf__ int* rows)
{
    float best = -1.0f;
    int row = height;
    for (int r = column + threadIdx.x; r < height; r += blockDim.x)
    {
        auto value = Load(panel, column * height + r);
        float score = Abs(value.real) + Abs(value.imag);
        if (score > best || (score == best && r < row))
        {
            best = score;
            row = r;
        }
    }
    ReduceCachedPivot(best, row, height);
    if (threadIdx.x % 32 == 0)
    {
        scores[threadIdx.x / 32] = best;
        rows[threadIdx.x / 32] = row;
    }
    asc_syncthreads();
    if (threadIdx.x < 32)
    {
        best = threadIdx.x < blockDim.x / 32 ? scores[threadIdx.x] : -1.0f;
        row = threadIdx.x < blockDim.x / 32 ? rows[threadIdx.x] : height;
        ReduceCachedPivot(best, row, height);
        if (threadIdx.x == 0) rows[0] = row;
    }
    asc_syncthreads();
    return rows[0];
}

__simt_callee__ __aicore__ inline void FactorCachedColumn(__ubuf__ aclFloatComplex* panel, int rowsCount, int columns,
                                                          int column, int begin, __gm__ int* pivots, __gm__ int* info,
                                                          __ubuf__ float* scores, __ubuf__ int* rows)
{
    int pivot = pivots == nullptr ? column : FindCachedPivot(panel, rowsCount, column, scores, rows);
    if (threadIdx.x == 0)
    {
        if (pivots != nullptr)
        {
            pivots[begin + column] = begin + pivot + 1;
        }
        if (*info == 0 && Zero(Load(panel, column * rowsCount + pivot))) *info = begin + column + 1;
    }
    SwapRows(panel, columns, rowsCount, column, pivot);
    auto diagonal = Load(panel, column * rowsCount + column);
    bool nonzero = !Zero(diagonal);
    auto divisor = PrepareDivisor(nonzero ? diagonal : aclFloatComplex{1, 0});
    float inverseScale = divisor.scale >= 0x1p-126f ? 1.0f / divisor.scale : 0.0f;
    // One thread owns a complete panel row; normalized L never crosses threads.
    for (int row = column + 1 + threadIdx.x; row < rowsCount; row += blockDim.x)
    {
        auto lower = Load(panel, column * rowsCount + row);
        if (nonzero)
        {
            lower = NormalizeCachedValue(lower, divisor, inverseScale);
            Store(panel, column * rowsCount + row, lower);
        }
        for (int target = column + 1; target < columns; ++target)
        {
            int address = target * rowsCount + row;
            Store(panel, address, SubMul(Load(panel, address), lower, Load(panel, target * rowsCount + column)));
        }
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void FactorCachedPanelVf(ComplexSolverShape shape,
                                                                          __gm__ aclFloatComplex* a, __gm__ int* pivots,
                                                                          __gm__ int* info, int begin, int end)
{
    __ubuf__ float storage[2 * COMPLEX_SOLVER_PANEL_CAPACITY];
    __ubuf__ float scores[1024 / 32];
    __ubuf__ int rows[1024 / 32];
    auto* panel = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    int height = shape.m - begin;
    int width = end - begin;
    for (int index = threadIdx.x; index < height * width; index += blockDim.x)
        Store(panel, index,
              Load(a, static_cast<uint64_t>(begin + index / height) * shape.lda + begin + index % height));
    asc_syncthreads();
    for (int column = 0; column < width; ++column)
        FactorCachedColumn(panel, height, width, column, begin, pivots, info, scores, rows);
    for (int index = threadIdx.x; index < height * width; index += blockDim.x)
        Store(a, static_cast<uint64_t>(begin + index / height) * shape.lda + begin + index % height,
              Load(panel, index));
    asc_syncthreads();
}

// Reverse the panel swaps for each destination row independently. Load every
// source before writing, so repeated pivots preserve their sequential semantics.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void PermutePanelRowsVf(ComplexSolverShape shape,
                                                                        __gm__ aclFloatComplex* matrix,
                                                                        __gm__ const int* pivots, int begin, int end)
{
    __ubuf__ int pivotRows[PANEL];
    int width = end - begin;
    if (threadIdx.x < width) pivotRows[threadIdx.x] = pivots[begin + threadIdx.x] - 1;
    asc_syncthreads();
    int slot = threadIdx.x % (2 * PANEL);
    int target = slot < width ? begin + slot : slot >= PANEL && slot - PANEL < width ? pivotRows[slot - PANEL] : -1;
    int source = target;
    bool active = target >= 0 && (slot < PANEL || target >= end);
    for (int step = width - 1; step >= 0; --step)
    {
        int pivot = pivotRows[step];
        if (slot >= PANEL && step < slot - PANEL && pivot == target) active = false;
        source = source == begin + step ? pivot : source == pivot ? begin + step : source;
    }
    active = active && source != target;
    for (int first = blockIdx.x * 8; first < shape.n; first += gridDim.x * 8)
    {
        int column = first + threadIdx.x / (2 * PANEL);
        bool firstValid = active && column < shape.n && (column < begin || column >= end);
        bool secondValid = active && column + 4 < shape.n && (column + 4 < begin || column + 4 >= end);
        aclFloatComplex firstValue{0, 0}, secondValue{0, 0};
        if (firstValid) firstValue = Load(matrix, static_cast<uint64_t>(column) * shape.lda + source);
        if (secondValid) secondValue = Load(matrix, static_cast<uint64_t>(column + 4) * shape.lda + source);
        asc_syncthreads();
        if (firstValid) Store(matrix, static_cast<uint64_t>(column) * shape.lda + target, firstValue);
        if (secondValid) Store(matrix, static_cast<uint64_t>(column + 4) * shape.lda + target, secondValue);
        asc_syncthreads();
    }
}

template <int Capacity>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void FactorResidentVf(ComplexSolverShape shape,
                                                                          __gm__ aclFloatComplex* a, __gm__ int* pivots,
                                                                          __gm__ int* info)
{
    __ubuf__ float storage[2 * Capacity];
    __ubuf__ float scores[THREADS / 32];
    __ubuf__ int rows[THREADS / 32];
    auto* local = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    for (int index = threadIdx.x; index < shape.m * shape.n; index += blockDim.x)
        Store(local, index, Load(a, static_cast<uint64_t>(index / shape.m) * shape.lda + index % shape.m));
    asc_syncthreads();
    int limit = shape.m < shape.n ? shape.m : shape.n;
    for (int column = 0; column < limit; ++column)
        FactorColumn(local, shape.m, shape.n, shape.m, column, shape.n, pivots, info, scores, rows);
    for (int index = threadIdx.x; index < shape.m * shape.n; index += blockDim.x)
        Store(a, static_cast<uint64_t>(index / shape.m) * shape.lda + index % shape.m, Load(local, index));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void FactorShortVf(ComplexSolverShape shape,
                                                                       __gm__ aclFloatComplex* a, __gm__ int* pivots,
                                                                       __gm__ int* info)
{
    __ubuf__ float storage[2 * PANEL * PANEL];
    __ubuf__ float scores[THREADS / 32];
    __ubuf__ int rows[THREADS / 32];
    auto* local = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    for (int index = threadIdx.x; index < shape.m * shape.m; index += blockDim.x)
        Store(local, index, Load(a, static_cast<uint64_t>(index / shape.m) * shape.lda + index % shape.m));
    asc_syncthreads();
    FactorPanel(local, shape.m, shape.m, shape.m, 0, shape.m, pivots, info, scores, rows);
    for (int index = threadIdx.x; index < shape.m * shape.m; index += blockDim.x)
        Store(a, static_cast<uint64_t>(index / shape.m) * shape.lda + index % shape.m, Load(local, index));
}

template <bool ApplyPivots>
__simt_callee__ __aicore__ inline void SolveUpperColumn(__gm__ aclFloatComplex* a,
                                                        __ubuf__ const aclFloatComplex* lower, __gm__ const int* pivots,
                                                        int lda, int column, int begin, int end)
{
    int lane = threadIdx.x % 32;
    int row = begin + lane;
    aclFloatComplex value = row < end ? Load(a, static_cast<uint64_t>(column) * lda + row) : aclFloatComplex{0, 0};
    if constexpr (ApplyPivots)
        if (pivots != nullptr)
            for (int step = begin; step < end; ++step)
            {
                int target = pivots[step] - 1;
                int source = row == step ? target - begin : row == target ? step - begin : lane;
                value = {asc_shfl(value.real, source), asc_shfl(value.imag, source)};
            }
    for (int pivot = 0; pivot < end - begin; ++pivot)
    {
        aclFloatComplex solved{asc_shfl(value.real, pivot), asc_shfl(value.imag, pivot)};
        if (lane > pivot && row < end) value = SubMul(value, Load(lower, pivot * PANEL + lane), solved);
    }
    if (row < end) Store(a, static_cast<uint64_t>(column) * lda + row, value);
}

template <bool ApplyPivots>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void SolveUpperVf(ComplexSolverShape shape,
                                                                      __gm__ aclFloatComplex* a,
                                                                      __gm__ const int* pivots, int begin, int end)
{
    __ubuf__ float storage[2 * PANEL * PANEL];
    auto* lower = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    for (int index = threadIdx.x; index < PANEL * PANEL; index += blockDim.x)
    {
        int row = begin + index % PANEL;
        int column = begin + index / PANEL;
        Store(lower, index,
              row < end && column < end ? Load(a, static_cast<uint64_t>(column) * shape.lda + row)
                                        : aclFloatComplex{0, 0});
    }
    asc_syncthreads();
    for (int column = end + blockIdx.x * (THREADS / 32) + threadIdx.x / 32; column < shape.n;
         column += (THREADS / 32) * gridDim.x)
        SolveUpperColumn<ApplyPivots>(a, lower, pivots, shape.lda, column, begin, end);
}

#endif

__global__ __aicore__ void complex_factor_panel(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivots,
                                                __gm__ uint8_t* info, int begin, int end)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    if (static_cast<uint64_t>(shape.m - begin) * (end - begin) <= COMPLEX_SOLVER_PANEL_CAPACITY)
        asc_vf_call<FactorCachedPanelVf>(dim3{1024, 1, 1}, shape, reinterpret_cast<__gm__ aclFloatComplex*>(a),
                                         reinterpret_cast<__gm__ int*>(pivots), reinterpret_cast<__gm__ int*>(info),
                                         begin, end);
    else
        asc_vf_call<FactorPanelVf>(dim3{THREADS, 1, 1}, shape, reinterpret_cast<__gm__ aclFloatComplex*>(a),
                                   reinterpret_cast<__gm__ int*>(pivots), reinterpret_cast<__gm__ int*>(info), begin,
                                   end);
#endif
}

__global__ __aicore__ void complex_permute_panel_rows(ComplexSolverShape shape, __gm__ uint8_t* matrix,
                                                      __gm__ uint8_t* pivots, int begin, int end)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<PermutePanelRowsVf>(dim3{256, 1, 1}, shape, reinterpret_cast<__gm__ aclFloatComplex*>(matrix),
                                    reinterpret_cast<__gm__ const int*>(pivots), begin, end);
#endif
}

void LaunchPanelPermutation(ComplexSolverShape shape, __gm__ uint8_t* matrix, __gm__ uint8_t* pivots, int begin,
                            int end, uint32_t blocks, void* stream)
{
    if (pivots == nullptr) return;
    uint32_t used = ComplexSolverLinearBlocks<8>(shape.n, blocks);
    complex_permute_panel_rows<<<used, nullptr, stream>>>(shape, matrix, pivots, begin, end);
}

__global__ __aicore__ void complex_factor_resident(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivots,
                                                   __gm__ uint8_t* info, bool shortPanel)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    auto* matrix = reinterpret_cast<__gm__ aclFloatComplex*>(a);
    auto* pivot = reinterpret_cast<__gm__ int*>(pivots);
    auto* status = reinterpret_cast<__gm__ int*>(info);
    if (shortPanel)
        asc_vf_call<FactorShortVf>(dim3{256, 1, 1}, shape, matrix, pivot, status);
    else if (static_cast<uint64_t>(shape.m) * shape.n <= 1024)
        asc_vf_call<FactorResidentVf<1024>>(dim3{256, 1, 1}, shape, matrix, pivot, status);
    else
        asc_vf_call<FactorResidentVf<RESIDENT_ELEMENTS>>(dim3{256, 1, 1}, shape, matrix, pivot, status);
#endif
}

__global__ __aicore__ void complex_solve_upper(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivots,
                                               int begin, int end, bool shortPanel)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    auto* matrix = reinterpret_cast<__gm__ aclFloatComplex*>(a);
    auto* pivot = reinterpret_cast<__gm__ const int*>(pivots);
    if (shortPanel)
        asc_vf_call<SolveUpperVf<true>>(dim3{256, 1, 1}, shape, matrix, pivot, begin, end);
    else
        asc_vf_call<SolveUpperVf<false>>(dim3{256, 1, 1}, shape, matrix, pivot, begin, end);
#endif
}

__global__ __aicore__ void complex_update_trailing(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry,
                                                   __gm__ uint8_t* a, int begin, int end)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<BlockUpdateVf<false, false>>(dim3{256, 1, 1}, shape, geometry, a, a, nullptr, begin, end);
#endif
}

void LaunchUpper(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivots, int begin, int end,
                 bool shortPanel, uint32_t blocks, void* stream)
{
    uint32_t used = ComplexSolverLinearBlocks<256 / 32>(shape.n - end, blocks);
    complex_solve_upper<<<used, nullptr, stream>>>(shape, a, pivots, begin, end, shortPanel);
}

void LaunchTrailingPanel(ComplexSolverShape shape, __gm__ uint8_t* matrix, int begin, int end, uint32_t blocks,
                         void* stream)
{
    auto geometry = ComplexSolverUpdateRegion(shape, false, begin, end, true);
    if (geometry.tasks != 0 && ComplexSolverUseCube(shape, geometry, begin, end))
        ComplexSolverCubeUpdate(shape, geometry, matrix, matrix, nullptr, false, false, begin, end, blocks, stream);
    else if (geometry.tasks != 0)
        complex_update_trailing<<<ComplexSolverUsedBlocks(geometry.tasks, blocks), nullptr, stream>>>(
            shape, geometry, matrix, begin, end);
}

void FactorWidePanel(ComplexSolverShape shape, __gm__ uint8_t* matrix, __gm__ uint8_t* pivot, __gm__ uint8_t* status,
                     int begin, int end, uint32_t blocks, void* stream)
{
    auto panelShape = shape;
    panelShape.n = end;
    for (int column = begin; column < end;)
    {
        int width = ComplexSolverPanelWidth(shape.m - column);
        int next = column + width < end ? column + width : end;
        complex_factor_panel<<<1, nullptr, stream>>>(shape, matrix, pivot, status, column, next);
        LaunchPanelPermutation(shape, matrix, pivot, column, next, blocks, stream);
        if (next < end) LaunchUpper(panelShape, matrix, pivot, column, next, false, blocks, stream);
        LaunchTrailingPanel(panelShape, matrix, column, next, blocks, stream);
        column = next;
    }
}

void ComplexSolverGetrf(ComplexSolverShape shape, void* a, void* workspace, int* pivots, int* info, uint32_t blocks,
                        void* stream)
{
    auto matrix = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a));
    auto pivot = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(pivots));
    auto status = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info));
    shape.ldc = shape.lda;
    int limit = shape.m < shape.n ? shape.m : shape.n;
    if (limit <= 32 && shape.n <= 128 && static_cast<uint64_t>(shape.m) * shape.n <= RESIDENT_ELEMENTS)
    {
        complex_factor_resident<<<1, nullptr, stream>>>(shape, matrix, pivot, status, false);
        return;
    }
    if (shape.m <= 32)
    {
        complex_factor_resident<<<1, nullptr, stream>>>(shape, matrix, pivot, status, true);
        LaunchUpper(shape, matrix, pivot, 0, shape.m, true, blocks, stream);
        return;
    }
    for (int begin = 0; begin < limit;)
    {
        int width = COMPLEX_SOLVER_PANEL;
        int end = begin + width < limit ? begin + width : limit;
        if (ComplexSolverUseDistributedPanel(shape.m - begin, shape.n - begin) &&
            ComplexSolverDistributedBlocks(shape.m - begin, blocks) != 0)
        {
            ComplexSolverDistributedPanel(shape, a, workspace, pivots, info, begin, end, blocks, stream);
            LaunchPanelPermutation(shape, matrix, pivot, begin, end, blocks, stream);
        }
        else
            FactorWidePanel(shape, matrix, pivot, status, begin, end, blocks, stream);
        if (end < shape.n) LaunchUpper(shape, matrix, pivot, begin, end, false, blocks, stream);
        LaunchTrailingPanel(shape, matrix, begin, end, blocks, stream);
        begin = end;
    }
}
