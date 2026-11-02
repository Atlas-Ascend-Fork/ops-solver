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
#include "complex_solver_kernel.hpp"
#include "complex_solver_launch.h"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
constexpr int UPDATE_ROWS = COMPLEX_SOLVER_UPDATE_ROWS;
constexpr int UPDATE_COLUMNS = COMPLEX_SOLVER_UPDATE_COLUMNS;
constexpr int UPDATE_VALUES = UPDATE_ROWS * UPDATE_COLUMNS / THREADS;

struct UpdateAccumulator
{
    float real[UPDATE_VALUES];
    float imag[UPDATE_VALUES];
};

__simt_callee__ __aicore__ inline void ReadUpdate(__gm__ const aclFloatComplex* c, int ldc, int row, int col,
                                                  int lastRow, int lastColumn, UpdateAccumulator& sum)
{
#pragma unroll
    for (int i = 0; i < UPDATE_VALUES; ++i)
    {
        int column = col + i * (THREADS / UPDATE_ROWS);
        auto value = row < lastRow && column < lastColumn ? Load(c, static_cast<uint64_t>(column) * ldc + row)
                                                          : aclFloatComplex{0, 0};
        sum.real[i] = value.real;
        sum.imag[i] = value.imag;
    }
}

__simt_callee__ __aicore__ inline void StageUpdate(__gm__ const aclFloatComplex* a, __gm__ const aclFloatComplex* b,
                                                   __ubuf__ aclFloatComplex* left, __ubuf__ aclFloatComplex* right,
                                                   int lda, int ldb, int firstRow, int firstCol, int lastRow,
                                                   int lastCol, int begin, int end)
{
    for (int index = threadIdx.x; index < PANEL * UPDATE_ROWS; index += blockDim.x)
    {
        int row = firstRow + index % UPDATE_ROWS;
        int k = begin + index / UPDATE_ROWS;
        Store(left, index,
              row < lastRow && k < end ? Load(a, static_cast<uint64_t>(k) * lda + row) : aclFloatComplex{0, 0});
    }
    for (int index = threadIdx.x; index < PANEL * UPDATE_COLUMNS; index += blockDim.x)
    {
        int k = begin + index % PANEL;
        int col = firstCol + index / PANEL;
        Store(right, index,
              col < lastCol && k < end ? Load(b, static_cast<uint64_t>(col) * ldb + k) : aclFloatComplex{0, 0});
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void AccumulateUpdate(__ubuf__ const aclFloatComplex* left,
                                                        __ubuf__ const aclFloatComplex* right, int count,
                                                        UpdateAccumulator& sum)
{
    int row = threadIdx.x % UPDATE_ROWS;
    int col = threadIdx.x / UPDATE_ROWS;
    for (int k = 0; k < count; ++k)
    {
        auto value = Load(left, k * UPDATE_ROWS + row);
#pragma unroll
        for (int i = 0; i < UPDATE_VALUES; ++i)
        {
            auto updated =
                SubMul({sum.real[i], sum.imag[i]}, value, Load(right, (col + i * (THREADS / UPDATE_ROWS)) * PANEL + k));
            sum.real[i] = updated.real;
            sum.imag[i] = updated.imag;
        }
    }
    // All UB readers finish before the next task reuses these tiles.
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void WriteUpdate(__gm__ aclFloatComplex* c, int ldc, int row, int col, int lastRow,
                                                   int lastColumn, const UpdateAccumulator& sum)
{
#pragma unroll
    for (int i = 0; i < UPDATE_VALUES; ++i)
    {
        int column = col + i * (THREADS / UPDATE_ROWS);
        if (row < lastRow && column < lastColumn)
            Store(c, static_cast<uint64_t>(column) * ldc + row, {sum.real[i], sum.imag[i]});
    }
}

template <bool Batched, bool Inverse>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void BlockUpdateVf(ComplexSolverShape shape,
                                                                       ComplexSolverUpdateGeometry geometry,
                                                                       __gm__ uint8_t* a, __gm__ uint8_t* c,
                                                                       __gm__ const int* info, int begin, int end)
{
    __ubuf__ float leftStorage[2 * PANEL * UPDATE_ROWS];
    __ubuf__ float rightStorage[2 * PANEL * UPDATE_COLUMNS];
    auto* left = reinterpret_cast<__ubuf__ aclFloatComplex*>(leftStorage);
    auto* right = reinterpret_cast<__ubuf__ aclFloatComplex*>(rightStorage);
    uint64_t perMatrix = static_cast<uint64_t>(geometry.rowTiles) * geometry.columnTiles;
    for (uint64_t task = blockIdx.x; task < geometry.tasks; task += gridDim.x)
    {
        int batch = task / perMatrix;
        if constexpr (Inverse)
        {
            if (info[batch] != 0) continue;
        }
        uint64_t tile = task % perMatrix;
        int firstRow = geometry.firstRow + (tile % geometry.rowTiles) * UPDATE_ROWS;
        int firstColumn = geometry.firstColumn + (tile / geometry.rowTiles) * UPDATE_COLUMNS;
        auto* input = Batched ? MatrixPointer(a, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(a);
        auto* output = Batched ? MatrixPointer(c, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(c);
        int row = firstRow + threadIdx.x % UPDATE_ROWS;
        int column = firstColumn + threadIdx.x / UPDATE_ROWS;
        UpdateAccumulator sum;
        ReadUpdate(output, shape.ldc, row, column, geometry.lastRow, shape.n, sum);
        StageUpdate(input, output, left, right, shape.lda, shape.ldc, firstRow, firstColumn, geometry.lastRow, shape.n,
                    begin, end);
        AccumulateUpdate(left, right, end - begin, sum);
        WriteUpdate(output, shape.ldc, row, column, geometry.lastRow, shape.n, sum);
    }
}
}  // namespace ComplexSolverKernel
#endif
