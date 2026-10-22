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
 * \file cheevj_vector_kernel.cpp
 * \brief Optimized fixed-size tridiagonal eigenvector reconstruction.
 */

#include "cheevj_launchers.hpp"
#include "kernel/cheevj_tridiag_inverse.hpp"
#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

__global__ __aicore__ void cheevj_fixed_tridiag_inverse_kernel(GM_ADDR diagonal, GM_ADDR offDiagonal,
                                                               GM_ADDR eigenvalues, GM_ADDR eigenvectors, int waveStart,
                                                               int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    AscendC::TPipe pipe;
    Cheevj::CheevjTridiagInverseIteration solver;
    solver.Init(&pipe, diagonal, offDiagonal, eigenvalues, eigenvectors, n, waveStart);
    solver.Process();
    pipe.Destroy();
#endif
}

void cheevj_fixed_vectors_do(GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR eigenvalues,
                             GM_ADDR tridiagonalEigenvectors, GM_ADDR reflectorReal, GM_ADDR reflectorImag,
                             GM_ADDR tauReal, GM_ADDR tauImag, GM_ADDR eigenvectorReal, GM_ADDR eigenvectorImag,
                             GM_ADDR info, int n, void *stream)
{
    for (int waveStart = 0; waveStart < n; waveStart += Cheevj::CHEEVJ_TRIDIAG_INVERSE_WAVE_COLUMNS)
    {
        cheevj_fixed_tridiag_inverse_kernel<<<Cheevj::CHEEVJ_TRIDIAG_INVERSE_WORKERS, nullptr, stream>>>(
            diagonal, offDiagonal, eigenvalues, tridiagonalEigenvectors, waveStart, n);
    }
    cheevj_fixed_backtransform_do(tridiagonalEigenvectors, reflectorReal, reflectorImag, tauReal, tauImag,
                                  eigenvectorReal, eigenvectorImag, info, n, stream);
}
