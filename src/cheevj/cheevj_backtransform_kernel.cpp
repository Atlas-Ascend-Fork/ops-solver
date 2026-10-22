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
 * \file cheevj_backtransform_kernel.cpp
 * \brief Fixed-size Householder eigenvector backtransformation.
 */

#include "kernel/cheevj_jobzv_assembly.hpp"
#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace
{
constexpr int kN512 = 512;
constexpr int kN1024 = 1024;
constexpr int kN2048 = 2048;
constexpr uint32_t kBacktransformBlocks = 32;
}  // namespace

__global__ __aicore__ void cheevj_fixed_backtransform_kernel(GM_ADDR tridiagonalEigenvectors, GM_ADDR reflectorReal,
                                                             GM_ADDR reflectorImag, GM_ADDR tauReal, GM_ADDR tauImag,
                                                             GM_ADDR eigenvectorReal, GM_ADDR eigenvectorImag,
                                                             GM_ADDR info, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    AscendC::TPipe pipe;
    // Non-fixed orders are padded to one of these backend sizes by RunPaddedFixed;
    // small and greater-than-2048 orders use the host fallback instead.
    if (n == kN512)
    {
        Cheevj::CheevjE2EN512JobzVBacktransform backtransform;
        backtransform.Init(&pipe, tridiagonalEigenvectors, reflectorReal, reflectorImag, tauReal, tauImag,
                           eigenvectorReal, eigenvectorImag, info);
        backtransform.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjE2EN1024JobzVBacktransform backtransform;
        backtransform.Init(&pipe, tridiagonalEigenvectors, reflectorReal, reflectorImag, tauReal, tauImag,
                           eigenvectorReal, eigenvectorImag, info);
        backtransform.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjE2EN2048JobzVBacktransform backtransform;
        backtransform.Init(&pipe, tridiagonalEigenvectors, reflectorReal, reflectorImag, tauReal, tauImag,
                           eigenvectorReal, eigenvectorImag, info);
        backtransform.Process();
    }
    pipe.Destroy();
#endif
}

void cheevj_fixed_backtransform_do(GM_ADDR tridiagonalEigenvectors, GM_ADDR reflectorReal, GM_ADDR reflectorImag,
                                   GM_ADDR tauReal, GM_ADDR tauImag, GM_ADDR eigenvectorReal, GM_ADDR eigenvectorImag,
                                   GM_ADDR info, int n, void *stream)
{
    cheevj_fixed_backtransform_kernel<<<kBacktransformBlocks, nullptr, stream>>>(
        tridiagonalEigenvectors, reflectorReal, reflectorImag, tauReal, tauImag, eigenvectorReal, eigenvectorImag, info,
        n);
}
