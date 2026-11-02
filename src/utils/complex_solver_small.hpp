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
#include "complex_solver_row_register.hpp"
#include "complex_solver_tiny.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int Width, bool Direct>
__aicore__ inline void CallSizedRegisterInverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivot,
                                                __gm__ uint8_t* c, __gm__ int* info)
{
    constexpr int threads = COMPLEX_SOLVER_BATCH_THREADS<Width, Direct>;
    if (!Direct && Width <= 8 && shape.n == Width)
        asc_vf_call<RegisterInverseVf<Width, Direct, true>>(dim3{threads, 1, 1}, shape, a, pivot, c, info);
    else
        asc_vf_call<RegisterInverseVf<Width, Direct, false>>(dim3{threads, 1, 1}, shape, a, pivot, c, info);
}

template <bool Direct>
__aicore__ inline void CallRegisterInverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivot,
                                           __gm__ uint8_t* c, __gm__ uint8_t* info)
{
    auto* pivots = reinterpret_cast<__gm__ const int*>(pivot);
    auto* status = reinterpret_cast<__gm__ int*>(info);
    if (!Direct && shape.n == 2)
        asc_vf_call<TinyLuInverseVf<2, true>>(dim3{COMPLEX_SOLVER_TINY_THREADS, 1, 1}, shape, a, pivots, c, status);
    else if (!Direct && shape.n == 1)
        asc_vf_call<TinyLuInverseVf<2, false>>(dim3{COMPLEX_SOLVER_TINY_THREADS, 1, 1}, shape, a, pivots, c, status);
    else if (!Direct && shape.n <= 4)
    {
        if (shape.n == 4)
            CallRowRegisterInverse<4, 512, true>(shape, a, pivots, c, status);
        else
            CallRowRegisterInverse<4, 512, false>(shape, a, pivots, c, status);
    }
    else if (shape.n <= 2)
        CallSizedRegisterInverse<2, Direct>(shape, a, pivots, c, status);
    else if (shape.n <= 4)
        CallSizedRegisterInverse<4, Direct>(shape, a, pivots, c, status);
    else if (!Direct && shape.n <= 8)
    {
        if (shape.n == 8)
            CallRowRegisterInverse<8, 256, true>(shape, a, pivots, c, status);
        else
            CallRowRegisterInverse<8, 256, false>(shape, a, pivots, c, status);
    }
    else if (shape.n <= 8)
        CallSizedRegisterInverse<8, Direct>(shape, a, pivots, c, status);
    else if (shape.n <= 16)
        CallSizedRegisterInverse<16, Direct>(shape, a, pivots, c, status);
    else if (!Direct && shape.n <= 24)
        asc_vf_call<RegisterInverseVf<32, false, false, 1024, 24>>(dim3{1024, 1, 1}, shape, a, pivots, c, status);
    else
        CallSizedRegisterInverse<32, Direct>(shape, a, pivots, c, status);
}
}  // namespace ComplexSolverKernel
#endif
