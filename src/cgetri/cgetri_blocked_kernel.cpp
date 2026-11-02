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
#include "../utils/complex_solver_diagonal_vector.hpp"
#include "../utils/complex_solver_small.hpp"
#include "../utils/complex_solver_update.hpp"

#if __NPU_ARCH__ == 3510
using namespace ComplexSolverKernel;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void InitializeInverseVf(ComplexSolverShape shape,
                                                                             __gm__ uint8_t* a,
                                                                             __gm__ const int* pivots,
                                                                             __gm__ uint8_t* c, __gm__ int* info,
                                                                             bool batched)
{
    __ubuf__ int status[1];
    __ubuf__ int checkGroups[THREADS / 32];
    int columnTiles = (shape.n + 31) / 32;
    for (uint64_t task = blockIdx.x; task < static_cast<uint64_t>(shape.batch) * columnTiles; task += gridDim.x)
    {
        int batch = task / columnTiles;
        int first = (task % columnTiles) * 32;
        int columns = shape.n - first < 32 ? shape.n - first : 32;
        auto* input = batched ? MatrixPointer(a, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(a);
        auto* output = batched ? MatrixPointer(c, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(c);
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        int code = ParallelCheckLu(input, shape.n, shape.lda, pivot, checkGroups);
        if (threadIdx.x == 0)
        {
            status[0] = code;
            if (first == 0) info[batch] = code;
        }
        asc_syncthreads();
        if (status[0] == 0)
        {
            for (uint64_t index = threadIdx.x; index < static_cast<uint64_t>(shape.n) * columns; index += blockDim.x)
            {
                int row = index % shape.n;
                int column = first + index / shape.n;
                Store(output, static_cast<uint64_t>(column) * shape.ldc + row, {row == column ? 1.0f : 0.0f, 0.0f});
            }
        }
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PermuteInverseColumnsVf(ComplexSolverShape shape,
                                                                                 __gm__ const int* pivots,
                                                                                 __gm__ uint8_t* c,
                                                                                 __gm__ const int* info, bool batched)
{
    int rowTiles = (shape.n + THREADS - 1) / THREADS;
    for (uint64_t task = blockIdx.x; task < static_cast<uint64_t>(shape.batch) * rowTiles; task += gridDim.x)
    {
        int batch = task / rowTiles;
        int row = (task % rowTiles) * THREADS + threadIdx.x;
        if (info[batch] != 0 || row >= shape.n) continue;
        auto* output = batched ? MatrixPointer(c, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(c);
        auto* pivot = pivots + static_cast<uint64_t>(batch) * shape.n;
        for (int column = shape.n - 1; column >= 0; --column)
        {
            int swapped = pivot[column] - 1;
            if (swapped == column) continue;
            uint64_t left = static_cast<uint64_t>(column) * shape.ldc + row;
            uint64_t right = static_cast<uint64_t>(swapped) * shape.ldc + row;
            auto value = Load(output, left);
            Store(output, left, Load(output, right));
            Store(output, right, value);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void CachedPermuteVf(ComplexSolverShape shape,
                                                                     __ubuf__ const float* stage, __ubuf__ int* sources,
                                                                     __gm__ const int* pivot,
                                                                     __gm__ aclFloatComplex* output, int pitch)
{
    for (int column = threadIdx.x; column < shape.n; column += blockDim.x) sources[column] = column;
    asc_syncthreads();
    if (threadIdx.x == 0)
        for (int column = shape.n - 1; column >= 0; --column)
        {
            int swapped = pivot[column] - 1;
            int source = sources[column];
            sources[column] = sources[swapped];
            sources[swapped] = source;
        }
    asc_syncthreads();
    for (int index = threadIdx.x; index < shape.n * shape.n; index += blockDim.x)
    {
        int column = index / shape.n, row = index % shape.n;
        int source = (sources[column] * pitch + row) * 2;
        Store(output, static_cast<uint64_t>(column) * shape.ldc + row, {stage[source], stage[source + 1]});
    }
}
__aicore__ inline void CachedPermuteMatrix(ComplexSolverShape shape, __gm__ const int* pivots, __gm__ uint8_t* c,
                                           __gm__ const int* info, bool batched, int batch, __ubuf__ float* memory)
{
    if (info[batch] != 0) return;
    int pitch = (shape.n + 3) / 4 * 4;
    AscendC::LocalTensor<float> stage(AscendC::TPosition::VECCALC,
                                      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)),
                                      COMPLEX_SOLVER_LU_VECTOR_BYTES / sizeof(float));
    AscendC::GlobalTensor<uint64_t> pointers;
    pointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(c));
    auto* output = batched ? reinterpret_cast<__gm__ aclFloatComplex*>(pointers.GetValue(batch))
                           : reinterpret_cast<__gm__ aclFloatComplex*>(c);
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    AscendC::DataCopyExtParams cp{static_cast<uint16_t>(shape.n), static_cast<uint32_t>(shape.n * 8),
                                  static_cast<uint32_t>((shape.ldc - shape.n) * 8), 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::DataCopyPad(stage, global, cp, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    asc_vf_call<CachedPermuteVf>(dim3{256, 1, 1}, shape, memory,
                                 reinterpret_cast<__ubuf__ int*>(memory + shape.n * pitch * 2),
                                 pivots + static_cast<uint64_t>(batch) * shape.n, output, pitch);
}

#endif

__global__ __aicore__ void complex_inverse_initialize(ComplexSolverShape shape, __gm__ uint8_t* a,
                                                      __gm__ uint8_t* pivots, __gm__ uint8_t* c, __gm__ uint8_t* info,
                                                      bool batched)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<InitializeInverseVf>(dim3{256, 1, 1}, shape, a, reinterpret_cast<__gm__ const int*>(pivots), c,
                                     reinterpret_cast<__gm__ int*>(info), batched);
#endif
}

__global__ __aicore__ void complex_inverse_diagonal(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* c,
                                                    __gm__ uint8_t* info, bool batched, int begin, int end,
                                                    bool forward)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    alignas(32) __ubuf__ float memory[ComplexDiagonalVector::MEMORY_FLOATS];
    ComplexDiagonalVector::Inverse inverse;
    inverse.Run(shape, a, c, reinterpret_cast<__gm__ const int*>(info), begin, end, forward, batched, memory);
#endif
}

__global__ __aicore__ void complex_inverse_permute(ComplexSolverShape shape, __gm__ uint8_t* pivots, __gm__ uint8_t* c,
                                                   __gm__ uint8_t* info, bool batched)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<PermuteInverseColumnsVf>(dim3{256, 1, 1}, shape, reinterpret_cast<__gm__ const int*>(pivots), c,
                                         reinterpret_cast<__gm__ const int*>(info), batched);
#endif
}

__global__ __aicore__ void complex_inverse_cached_permute(ComplexSolverShape shape, __gm__ uint8_t* pivots,
                                                          __gm__ uint8_t* c, __gm__ uint8_t* info, bool batched)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    __ubuf__ float memory[COMPLEX_SOLVER_LU_VECTOR_BYTES / sizeof(float)];
    for (int batch = AscendC::GetBlockIdx(); batch < shape.batch; batch += AscendC::GetBlockNum())
        CachedPermuteMatrix(shape, reinterpret_cast<__gm__ const int*>(pivots), c,
                            reinterpret_cast<__gm__ const int*>(info), batched, batch, memory);
#endif
}

void LaunchInversePhase(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* c, __gm__ uint8_t* status,
                        bool batched, int begin, int end, bool forward, uint32_t blocks, void* stream)
{
    ComplexSolverUpdateGeometry geometry{0, forward ? end : shape.n, forward ? end : 0, 0, 0, 0};
    uint64_t tasks = static_cast<uint64_t>(shape.batch) *
                     ((geometry.lastRow + COMPLEX_SOLVER_INVERSE_COLUMNS - 1) / COMPLEX_SOLVER_INVERSE_COLUMNS);
    complex_inverse_diagonal<<<ComplexSolverUsedBlocks(tasks, blocks), nullptr, stream>>>(shape, a, c, status, batched,
                                                                                          begin, end, forward);
    if (!forward) shape.n = begin;
    geometry.rowTiles = (geometry.lastRow + COMPLEX_SOLVER_UPDATE_ROWS - 1) / COMPLEX_SOLVER_UPDATE_ROWS;
    geometry.columnTiles =
        (shape.n - geometry.firstColumn + COMPLEX_SOLVER_UPDATE_COLUMNS - 1) / COMPLEX_SOLVER_UPDATE_COLUMNS;
    geometry.tasks = static_cast<uint64_t>(geometry.rowTiles) * geometry.columnTiles * shape.batch;
    if (geometry.tasks != 0)
        ComplexSolverCubeUpdate(shape, geometry, a, c,
                                reinterpret_cast<const int*>(reinterpret_cast<uintptr_t>(status)), batched, true, begin,
                                end, blocks, stream);
}

void ComplexSolverBlockedGetri(ComplexSolverShape shape, const void* a, const int* pivots, const void* c, int* info,
                               bool batched, uint32_t blocks, void* stream)
{
    auto input = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a));
    auto pivot = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(pivots));
    auto output = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c));
    auto status = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info));
    complex_inverse_initialize<<<
        ComplexSolverUsedBlocks(static_cast<uint64_t>(shape.batch) * ((shape.n + 31) / 32), blocks), nullptr, stream>>>(
        shape, input, pivot, output, status, batched);
    for (int begin = 0; begin < shape.n; begin += COMPLEX_SOLVER_INVERSE_PANEL)
    {
        int end = begin + COMPLEX_SOLVER_INVERSE_PANEL < shape.n ? begin + COMPLEX_SOLVER_INVERSE_PANEL : shape.n;
        LaunchInversePhase(shape, input, output, status, batched, begin, end, true, blocks, stream);
    }
    for (int end = shape.n; end > 0;)
    {
        int begin = ((end - 1) / COMPLEX_SOLVER_INVERSE_PANEL) * COMPLEX_SOLVER_INVERSE_PANEL;
        LaunchInversePhase(shape, input, output, status, batched, begin, end, false, blocks, stream);
        end = begin;
    }
    if (pivots != nullptr)
    {
        uint64_t cachedBytes =
            static_cast<uint64_t>(shape.n) * ((shape.n + 3) / 4 * 4) * 8 + static_cast<uint64_t>(shape.n) * sizeof(int);
        if (batched && cachedBytes <= COMPLEX_SOLVER_LU_VECTOR_BYTES)
        {
            complex_inverse_cached_permute<<<ComplexSolverUsedBlocks(shape.batch, blocks), nullptr, stream>>>(
                shape, pivot, output, status, batched);
            return;
        }
        uint64_t tasks = static_cast<uint64_t>(shape.batch) * ((shape.n + 255) / 256);
        complex_inverse_permute<<<ComplexSolverUsedBlocks(tasks, blocks), nullptr, stream>>>(shape, pivot, output,
                                                                                             status, batched);
    }
}
