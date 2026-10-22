/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR panelVReal,
                                GM_ADDR panelVImag, GM_ADDR panelWReal, GM_ADDR panelWImag, GM_ADDR wHReal,
                                GM_ADDR wHImag, GM_ADDR wHImagNeg, GM_ADDR diagonal, GM_ADDR offDiagonal,
                                GM_ADDR tauReal, GM_ADDR tauImag, GM_ADDR panelWorkspace, GM_ADDR barrierWorkspace,
                                int activeSize)
    {
        activeN = activeSize;
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        BindGlobals(matrixReal, matrixImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal, wHImag, wHImagNeg,
                    diagonal, offDiagonal, tauReal, tauImag, panelWorkspace, barrierWorkspace);
        InitBuffers(pipe);
#else
        CheevjIgnoreHouseholderInitArgs(pipe, matrixReal, matrixImag, panelVReal, panelVImag, panelWReal, panelWImag,
                                        wHReal, wHImag, wHImagNeg, diagonal, offDiagonal, tauReal, tauImag,
                                        panelWorkspace, barrierWorkspace);
#endif
    }
