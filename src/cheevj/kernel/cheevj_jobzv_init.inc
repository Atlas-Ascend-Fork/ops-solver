/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR tridiagonalEigenvectorsRowMajor, GM_ADDR reflectorReal,
                                GM_ADDR reflectorImag, GM_ADDR tauReal, GM_ADDR tauImag, GM_ADDR eigenvectorReal,
                                GM_ADDR eigenvectorImag, GM_ADDR info)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        BindCommonGlobals(tridiagonalEigenvectorsRowMajor, reflectorReal, reflectorImag, tauReal, tauImag,
                          eigenvectorReal, eigenvectorImag, info);
        InitBuffers(pipe);
#else
        IgnoreInitArgs(pipe, tridiagonalEigenvectorsRowMajor, reflectorReal, reflectorImag, tauReal, tauImag,
                       eigenvectorReal, eigenvectorImag, info);
#endif
    }
