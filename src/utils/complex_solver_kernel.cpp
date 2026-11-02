/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "complex_solver_kernel.hpp"

#include "complex_solver_launch.h"

#if __NPU_ARCH__ == 3510
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void FillInfoVf(__gm__ int* info, int count, int value)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
        info[index] = value;
}
#endif

__global__ __aicore__ void complex_solver_info(__gm__ uint8_t* info, int count, int value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<FillInfoVf>(dim3{256, 1, 1}, reinterpret_cast<__gm__ int*>(info), count, value);
#endif
}

void ComplexSolverFillInfo(int* info, int count, int value, uint32_t blocks, void* stream)
{
    complex_solver_info<<<blocks, nullptr, stream>>>(
        reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(info)), count, value);
}
