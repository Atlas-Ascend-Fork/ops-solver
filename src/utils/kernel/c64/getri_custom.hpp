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
 * \file getri_custom.hpp
 * \brief
 */

#ifndef _GETRI_CUSTOM_HPP_
#define _GETRI_CUSTOM_HPP_

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "cast.hpp"
#include "kernel_operator.h"
#include "lu_custom.hpp"
#include "pad.hpp"
#include "trsm_custom.hpp"
#include "trsm_upper_custom.hpp"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void custom_getri(int dtype, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                                    GM_ADDR A_inv, GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm,
                                    GM_ADDR gather2_gm, GM_ADDR gather3_gm, GM_ADDR eye_gm, GM_ADDR workspace)
{
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    int N = (orgN + 15) / 16 * 16;

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    CMatmulCustom<float> opmm;
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    pipe.InitBufPool(tbufPool, 192 * 1024);
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    opmm.Init(&pipe);
#endif

    if (dtype > 0)
    {
        pad(tbufPool, orgM, orgN, A_org, A_work);
    }
    else
    {
        cast_up(tbufPool, orgM, orgN, workspace, A_org);
        pad(tbufPool, orgM, orgN, workspace, A_work);
    }

    custom_lu(tbufPool, opmm, orgM, orgN, blockM, blockN, tileM, A_work, W, work_gm, gather1_gm, gather2_gm, eye_gm);
    custom_trsm(tbufPool, opmm, M, N, strideN, blockM, eye_gm, A_work);
    custom_trsm_upper(tbufPool, opmm, orgM, M, N, strideN, blockM, eye_gm, A_work);

    if (dtype > 0)
    {
        restore(tbufPool, orgM, orgN, A_inv, eye_gm, gather3_gm);
    }
    else
    {
        restore(tbufPool, orgM, orgN, workspace, eye_gm, gather3_gm);
        cast_down(tbufPool, orgM, orgN, A_inv, workspace);
    }
    pipe.Destroy();
}

#endif  // _GETRI_CUSTOM_HPP_
