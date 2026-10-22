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
 * \file sgetri_kernel.cpp
 * \brief
 */

#ifndef SGETRI_MIX_H
#define SGETRI_MIX_H

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "../utils/kernel/c32/lu_custom.hpp"
#include "../utils/kernel/c32/pad.hpp"
#include "../utils/kernel/c32/trsm_custom.hpp"
#include "../utils/kernel/c32/trsm_upper_custom.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

__global__ __aicore__ void sgetri_kernel(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM,
                                         GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm,
                                         GM_ADDR gather2_gm, GM_ADDR eye_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    int N = (orgN + 15) / 16 * 16;

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> aGlobal1;
    GlobalTensor<float> eyeGlobal;
    GlobalTensor<uint32_t> wGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    eyeGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eye_gm));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(W));
    aGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    MatmulCustom<float> opmm;
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    pipe.InitBufPool(tbufPool, 192 * 1024);
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    opmm.Init(&pipe);
#endif
    pad(tbufPool, orgM, orgN, A_org, A_work);
    custom_lu(tbufPool, opmm, orgM, orgN, blockM, blockN, tileM, A_work, W, work_gm, gather1_gm, gather2_gm, eye_gm);
    custom_trsm(tbufPool, opmm, M, N, strideN, blockM, eye_gm, A_work);
    custom_trsm_upper(tbufPool, opmm, orgM, M, N, strideN, blockM, eye_gm, A_work);
    restore(tbufPool, orgM, orgN, A_org, eye_gm);
}

void sgetri_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                      GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                      GM_ADDR eye_gm, uint32_t numBlocks, void *stream)
{
    sgetri_kernel<<<numBlocks, nullptr, stream>>>(sync, orgM, orgN, blockM, blockN, tileM, A_org, A_work, W, work_gm,
                                                  gather1_gm, gather2_gm, eye_gm);
}
#endif  // SGETRI_MIX_H
