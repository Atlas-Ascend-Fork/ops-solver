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
 * \file cgetrs_kernel.cpp
 * \brief
 */

#ifndef CGETRS_MIX_H
#define CGETRS_MIX_H

#include <cstdint>

#include "cgetrs_launchers.hpp"
#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>

#include "./kernel/trsm_custom.hpp"
#include "./kernel/trsm_upper_custom.hpp"
#include "./kernel/pad.hpp"
#include "./kernel/laswp.hpp"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void custom_getrs(int orgN, int orgNRHS, int trans, int lda, int ldb, int blockM, int blockN, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W, GM_ADDR P, GM_ADDR B_org, GM_ADDR B_work, GM_ADDR gather) {
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> bGlobal;
    GlobalTensor<uint32_t> wGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B_work));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(W));

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    CMatmulCustom<float> opmm;
    #ifdef __DAV_C220_VEC__
        pipe.InitBufPool(tbufPool, 192 * 1024);
    #elif  __DAV_C220_CUBE__
        opmm.Init(&pipe);
    #endif
    pad(tbufPool, orgN, orgN, lda, trans, A_org, A_work);
    if (trans == 0) {
        laswp(tbufPool, orgN, orgNRHS, ldb, 0, B_org, B_work, W, P, gather);
        custom_trsm(tbufPool, opmm, orgN, orgNRHS, blockM, 0, 1, B_work, A_work);
        custom_trsm_upper(tbufPool, opmm, orgN, orgNRHS, blockM, 0, 0, B_work, A_work);
        restore(tbufPool, orgN, orgNRHS, ldb, B_org, B_work, gather);
    } else {
        pad(tbufPool, orgN, orgNRHS, ldb, 0, B_org, B_work);
        custom_trsm(tbufPool, opmm, orgN, orgNRHS, blockM, 1, 0, B_work, A_work);
        custom_trsm_upper(tbufPool, opmm, orgN, orgNRHS, blockM, 1, 1, B_work, A_work);
        laswp(tbufPool, orgN, orgNRHS, ldb, 1, B_org, B_work, W, P, gather);
    }
}

__global__ __aicore__ void cgetrs_kernel(GM_ADDR sync, int orgN, int orgNRHS, int trans, int lda, int ldb, int blockM, int blockN,
                                         GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W, GM_ADDR P, GM_ADDR B_org, GM_ADDR B_work, GM_ADDR gather) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  custom_getrs(orgN, orgNRHS, trans, lda, ldb, blockM, blockN, A_org, A_work, W, P, B_org, B_work, gather);
}

void cgetrs_kernel_do(GM_ADDR sync, int orgN, int orgNRHS, int trans, int lda, int ldb, int blockM, int blockN,
                      GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W, GM_ADDR P, GM_ADDR B_org, GM_ADDR B_work, GM_ADDR gather, 
                      uint32_t numBlocks, void *stream) {
  cgetrs_kernel<<<numBlocks, nullptr, stream>>>(
      sync, orgN, orgNRHS, trans, lda, ldb, blockM, blockN, A_org, A_work, W, P, B_org, B_work, gather);
}

#endif  // CGETRS_MIX_H
