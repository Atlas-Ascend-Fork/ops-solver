/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// clang-format off
#include "kernel_operator.h"

// Device API declarations precede the implementation fragments below.
#include "../utils/complex_solver_small.hpp"
#include "../utils/complex_solver_batch_vector.hpp"
// clang-format on

__global__ __aicore__ void complex_matinv(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* c,
                                          __gm__ uint8_t* info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    if (shape.n > 2 && shape.n <= 4)
        asc_vf_call<ComplexSolverKernel::DirectRowInverseVf<4, 512>>(dim3{512, 1, 1}, shape, a, c,
                                                                     reinterpret_cast<__gm__ int*>(info));
    else
        ComplexSolverKernel::CallRegisterInverse<true>(shape, a, nullptr, c, info);
#endif
}

template <int Group>
__global__ __aicore__ void complex_matinv_batch_vector(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* c,
                                                       __gm__ uint8_t* info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    __ubuf__ float storage[61 * 1024];
    ComplexBatchVector::Inverse<Group> inverse;
    inverse.Run(shape, a, c, reinterpret_cast<__gm__ int*>(info), storage);
#endif
}
template <int Group>
void LaunchBatchVector(ComplexSolverShape shape, const void* a, const void* c, int* info, uint32_t blocks, void* stream)
{
    auto used = ComplexSolverUsedBlocks((static_cast<uint64_t>(shape.batch) + Group - 1) / Group, blocks);
    complex_matinv_batch_vector<Group>
        <<<used, nullptr, stream>>>(shape, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
                                    reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
                                    reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)));
}

void ComplexSolverMatinv(ComplexSolverShape shape, const void* a, const void* c, int* info, uint32_t blocks,
                         void* stream)
{
    if (shape.n > 8 && shape.lda == shape.n && shape.ldc == shape.n && shape.batch >= static_cast<int>(blocks) * 64)
    {
        int group = 64;
        while (group > 16 && ((2 * shape.n * shape.n + 8 * shape.n + 7) * group + 256) * sizeof(float) > 244 * 1024)
            group /= 2;
        if (group == 64)
            LaunchBatchVector<64>(shape, a, c, info, blocks, stream);
        else if (group == 32)
            LaunchBatchVector<32>(shape, a, c, info, blocks, stream);
        else
            LaunchBatchVector<16>(shape, a, c, info, blocks, stream);
        return;
    }
    int groups = ComplexSolverDirectThreads(ComplexSolverRegisterWidth(shape.n)) / ComplexSolverRegisterWidth(shape.n);
    auto used = ComplexSolverUsedBlocks((static_cast<uint64_t>(shape.batch) + groups - 1) / groups, blocks);
    complex_matinv<<<used, nullptr, stream>>>(shape, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
                                              reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
                                              reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)));
}
