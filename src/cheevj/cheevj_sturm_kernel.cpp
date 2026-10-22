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
 * \file cheevj_sturm_kernel.cpp
 * \brief Optimized fixed-size Sturm eigenvalue solve.
 */

#include "kernel/cheevj_tridiag_sturm.hpp"
#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

__global__ __aicore__ void cheevj_fixed_sturm_kernel(GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR bounds,
                                                     GM_ADDR eigenvalues, GM_ADDR lowWorkspace, GM_ADDR highWorkspace,
                                                     int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> boundsGlobal;
    AscendC::TBuf<AscendC::TPosition::VECCALC> boundsBuf;
    boundsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bounds));
    pipe.InitBuffer(boundsBuf, 8 * sizeof(float));
    auto boundsLocal = boundsBuf.Get<float>();
    AscendC::DataCopyExtParams copy{1, 2 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{true, 0, 6, 0.0f};
    AscendC::DataCopyPad(boundsLocal, boundsGlobal, copy, pad);
    AscendC::PipeBarrier<PIPE_ALL>();
    const float lower = boundsLocal.GetValue(0);
    const float upper = boundsLocal.GetValue(1);

    Cheevj::CheevjTridiagSturm sturm;
    sturm.Init(&pipe, diagonal, offDiagonal, eigenvalues, lowWorkspace, highWorkspace, n, lower, upper, 32);
    sturm.Process();
    pipe.Destroy();
#endif
}

void cheevj_fixed_sturm_do(GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR bounds, GM_ADDR eigenvalues,
                           GM_ADDR lowWorkspace, GM_ADDR highWorkspace, int n, void *stream)
{
    const uint32_t blocks = static_cast<uint32_t>(n / Cheevj::CHEEVJ_STURM_LANES);
    cheevj_fixed_sturm_kernel<<<blocks, nullptr, stream>>>(diagonal, offDiagonal, bounds, eigenvalues, lowWorkspace,
                                                           highWorkspace, n);
}
