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
 * \file cheevj_panel_kernel.cpp
 * \brief Fixed-size Householder panel reduction.
 */

#include "kernel/cheevj_householder_fixed.hpp"
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
constexpr uint32_t kPanelBlocks = 33;
constexpr uint32_t kN1024PanelBlocks = 17;
}  // namespace

__global__ __aicore__ void cheevj_fixed_panel_kernel(GM_ADDR packedReal, GM_ADDR packedImag, GM_ADDR panelVReal,
                                                     GM_ADDR panelVImag, GM_ADDR panelWReal, GM_ADDR panelWImag,
                                                     GM_ADDR wHReal, GM_ADDR wHImag, GM_ADDR wHImagNeg,
                                                     GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal,
                                                     GM_ADDR tauImag, GM_ADDR panelWorkspace, GM_ADDR barrierWorkspace,
                                                     int n, int activeN)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        Cheevj::CheevjHouseholderNb4Persistent panel;
        panel.Init(&pipe, packedReal, packedImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal, wHImag,
                   wHImagNeg, diagonal, offDiagonal, tauReal, tauImag, panelWorkspace, barrierWorkspace, activeN);
        panel.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjHouseholderN1024Nb4Streamed panel;
        panel.Init(&pipe, packedReal, packedImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal, wHImag,
                   wHImagNeg, diagonal, offDiagonal, tauReal, tauImag, panelWorkspace, barrierWorkspace, activeN);
        panel.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjHouseholderN2048Nb4Streamed panel;
        panel.Init(&pipe, packedReal, packedImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal, wHImag,
                   wHImagNeg, diagonal, offDiagonal, tauReal, tauImag, panelWorkspace, barrierWorkspace, activeN);
        panel.Process();
    }
    pipe.Destroy();
#endif
}

void cheevj_fixed_panel_do(GM_ADDR packedReal, GM_ADDR packedImag, GM_ADDR panelVReal, GM_ADDR panelVImag,
                           GM_ADDR panelWReal, GM_ADDR panelWImag, GM_ADDR wHReal, GM_ADDR wHImag, GM_ADDR wHImagNeg,
                           GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal, GM_ADDR tauImag,
                           GM_ADDR panelWorkspace, GM_ADDR barrierWorkspace, int n, int activeN, void *stream)
{
    const uint32_t panelBlocks = n == kN1024 ? kN1024PanelBlocks : kPanelBlocks;
    cheevj_fixed_panel_kernel<<<panelBlocks, nullptr, stream>>>(
        packedReal, packedImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal, wHImag, wHImagNeg, diagonal,
        offDiagonal, tauReal, tauImag, panelWorkspace, barrierWorkspace, n, activeN);
}
