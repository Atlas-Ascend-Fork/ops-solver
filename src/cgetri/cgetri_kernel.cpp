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
#include "../utils/complex_solver_grouped.hpp"
#include "../utils/complex_solver_kernel.hpp"
#include "../utils/complex_solver_launch.h"
#include "../utils/complex_solver_lu_vector.hpp"
#include "../utils/complex_solver_medium.hpp"
#include "../utils/complex_solver_resident.hpp"
#include "../utils/complex_solver_small.hpp"

#if __NPU_ARCH__ == 3510
using namespace ComplexSolverKernel;
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void CopyInverseVf(int n, int lda,
                                                                       __gm__ const aclFloatComplex* workspace,
                                                                       __gm__ aclFloatComplex* a,
                                                                       __gm__ const int* info)
{
    if (info[0] != 0) return;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < static_cast<uint64_t>(n) * n; index += static_cast<uint64_t>(gridDim.x) * blockDim.x)
        Store(a, (index / n) * lda + index % n, Load(workspace, index));
}
#endif

__global__ __aicore__ void complex_inverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivots,
                                           __gm__ uint8_t* c, __gm__ uint8_t* info, bool batched)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    if (batched && shape.n <= 32)
        CallRegisterInverse<false>(shape, a, pivots, c, info);
    else if (batched && shape.n <= COMPLEX_SOLVER_GROUPED_DENSE_COLUMNS)
        asc_vf_call<GroupedLuInverseVf<64, COMPLEX_SOLVER_GROUPED_BATCH_GROUPS, COMPLEX_SOLVER_GROUPED_DENSE_COLUMNS,
                                       COMPLEX_SOLVER_GROUPED_PANEL_COLUMNS>>(
            dim3{32 * COMPLEX_SOLVER_GROUPED_BATCH_GROUPS, 1, 1}, shape, a, reinterpret_cast<__gm__ const int*>(pivots),
            c, reinterpret_cast<__gm__ int*>(info));
    else if (batched && shape.n <= 64)
        asc_vf_call<
            GroupedLuInverseVf<64, COMPLEX_SOLVER_RESIDENT_BATCH_GROUPS, 64, COMPLEX_SOLVER_GROUPED_PANEL_COLUMNS>>(
            dim3{32 * COMPLEX_SOLVER_RESIDENT_BATCH_GROUPS, 1, 1}, shape, a,
            reinterpret_cast<__gm__ const int*>(pivots), c, reinterpret_cast<__gm__ int*>(info));
    else
        CallResidentInverse<false>(shape, a, pivots, c, info);
#endif
}

__global__ __aicore__ void complex_tiny_inverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivots,
                                                __gm__ uint8_t* c, __gm__ uint8_t* info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<TinyLuInverseVf<2, true>>(dim3{COMPLEX_SOLVER_TINY_THREADS, 1, 1}, shape, a,
                                          reinterpret_cast<__gm__ const int*>(pivots), c,
                                          reinterpret_cast<__gm__ int*>(info));
#endif
}

__global__ __aicore__ void complex_copy_inverse(int n, int lda, __gm__ uint8_t* workspace, __gm__ uint8_t* a,
                                                __gm__ uint8_t* info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<CopyInverseVf>(dim3{THREADS, 1, 1}, n, lda, reinterpret_cast<__gm__ const aclFloatComplex*>(workspace),
                               reinterpret_cast<__gm__ aclFloatComplex*>(a), reinterpret_cast<__gm__ const int*>(info));
#endif
}

__global__ __aicore__ void complex_lu_vector_inverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivot,
                                                     __gm__ uint8_t* c, __gm__ uint8_t* info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    __ubuf__ float memory[COMPLEX_SOLVER_LU_VECTOR_BYTES / sizeof(float)];
    ComplexLuVector::Inverse<96> inverse;
    inverse.Run(shape, a, reinterpret_cast<__gm__ const int*>(pivot), c, reinterpret_cast<__gm__ int*>(info), memory);
#endif
}

void ComplexSolverGetri(ComplexSolverShape shape, const void* a, const int* pivots, const void* c, int* info,
                        bool batched, uint32_t blocks, void* stream)
{
    if (batched && shape.n == 2)
    {
        complex_tiny_inverse<<<ComplexSolverUsedBlocks(ComplexSolverInverseBatchTasks(shape.n, shape.batch), blocks),
                               nullptr, stream>>>(
            shape, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(pivots)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)));
        return;
    }
    if (shape.n > 32 && !batched)
    {
        ComplexSolverBlockedGetri(shape, a, pivots, c, info, false, blocks, stream);
        return;
    }
    int rows = ComplexSolverLuVectorRows(shape.n);
    if (batched && shape.n > 64 && rows != 0)
    {
        uint64_t tasks = static_cast<uint64_t>(shape.batch) * ((shape.n + rows - 1) / rows);
        complex_lu_vector_inverse<<<ComplexSolverUsedBlocks(tasks, blocks), nullptr, stream>>>(
            shape, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(pivots)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)));
        return;
    }
    if (shape.n > 160 || (batched && shape.n > 64))
    {
        ComplexSolverBlockedGetri(shape, a, pivots, c, info, batched, blocks, stream);
        return;
    }
    uint64_t tasks = static_cast<uint64_t>(shape.batch) * ((shape.n + 7) / 8);
    if (shape.n <= 32)
        tasks = static_cast<uint64_t>(shape.batch) *
                ((shape.n + COMPLEX_SOLVER_SMALL_RESIDENT_COLUMNS - 1) / COMPLEX_SOLVER_SMALL_RESIDENT_COLUMNS);
    if (batched && shape.n <= 64) tasks = ComplexSolverInverseBatchTasks(shape.n, shape.batch);
    complex_inverse<<<ComplexSolverUsedBlocks(tasks, blocks), nullptr, stream>>>(
        shape, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(pivots)),
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)), batched);
}

void ComplexSolverCopyInverse(int n, int lda, const void* workspace, void* a, const int* info, uint32_t blocks,
                              void* stream)
{
    auto used = ComplexSolverLinearBlocks<256>(static_cast<uint64_t>(n) * n, blocks);
    complex_copy_inverse<<<used, nullptr, stream>>>(
        n, lda, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(workspace)),
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)));
}
