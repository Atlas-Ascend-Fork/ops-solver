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
 * \file cheevj_finalize_kernel.cpp
 * \brief Fixed-size tridiagonal reduction finalization.
 */

#include "kernel/cheevj_e2e_fixed.hpp"
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
constexpr uint32_t kFinalizeBlocks = 32;
}  // namespace

__global__ __aicore__ void cheevj_fixed_finalize_kernel(GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal,
                                                        GM_ADDR tauImag, GM_ADDR matrixReal, GM_ADDR matrixImag,
                                                        GM_ADDR bounds, GM_ADDR info, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        Cheevj::CheevjE2EFinalize<kN512> finalize;
        finalize.Init(&pipe, diagonal, offDiagonal, tauReal, tauImag, matrixReal, matrixImag, bounds, info);
        finalize.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjE2EFinalize<kN1024> finalize;
        finalize.Init(&pipe, diagonal, offDiagonal, tauReal, tauImag, matrixReal, matrixImag, bounds, info);
        finalize.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjE2EFinalize<kN2048> finalize;
        finalize.Init(&pipe, diagonal, offDiagonal, tauReal, tauImag, matrixReal, matrixImag, bounds, info);
        finalize.Process();
    }
    pipe.Destroy();
#endif
}

void cheevj_fixed_finalize_do(GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal, GM_ADDR tauImag,
                              GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR bounds, GM_ADDR info, int n, void *stream)
{
    cheevj_fixed_finalize_kernel<<<kFinalizeBlocks, nullptr, stream>>>(diagonal, offDiagonal, tauReal, tauImag,
                                                                       matrixReal, matrixImag, bounds, info, n);
}
