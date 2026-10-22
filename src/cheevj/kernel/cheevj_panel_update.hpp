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
 * \file cheevj_panel_update.hpp
 * \brief Mixed AIV/CUBE panel update helpers for Cheevj planar C64 workspaces.
 */

#ifndef CHEEVJ_C64_PANEL_UPDATE_HPP
#define CHEEVJ_C64_PANEL_UPDATE_HPP

#include <cstdint>

#include "../../utils/kernel/c64/gemm.hpp"
#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

using namespace AscendC;

namespace Cheevj
{

constexpr int CHEEVJ_PANEL_PACK_READY_FLAG = 0x0;
constexpr int CHEEVJ_PANEL_GEMM_DONE_FLAG = 0x1;

__aicore__ inline void CheevjBindFloatGlobal(GlobalTensor<float> *tensor, GM_ADDR addr)
{
    tensor->SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addr));
}

__aicore__ inline void CheevjBindFloatGlobal(GlobalTensor<float> *tensor, GlobalTensor<float> source)
{
    *tensor = source;
}

__aicore__ inline void CheevjPanelCleanEntireDataCache()
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    GlobalTensor<uint64_t> cacheGlobal;
    cacheGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(0));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(cacheGlobal);
#endif
}

__aicore__ inline void CheevjRunComplexGemmCube(GM_ADDR aReal, GM_ADDR aImag, GM_ADDR bReal, GM_ADDR bImag,
                                                GM_ADDR bImagNeg, GM_ADDR cReal, GM_ADDR cImag, int m, int n, int k)
{
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    TPipe pipe;
    CMatmulCustom<float> opmm;
    opmm.Init(&pipe);

    GlobalTensor<float> aRealGlobal;
    GlobalTensor<float> aImagGlobal;
    GlobalTensor<float> bRealGlobal;
    GlobalTensor<float> bImagGlobal;
    GlobalTensor<float> bImagNegGlobal;
    GlobalTensor<float> cRealGlobal;
    GlobalTensor<float> cImagGlobal;
    CheevjBindFloatGlobal(&aRealGlobal, aReal);
    CheevjBindFloatGlobal(&aImagGlobal, aImag);
    CheevjBindFloatGlobal(&bRealGlobal, bReal);
    CheevjBindFloatGlobal(&bImagGlobal, bImag);
    CheevjBindFloatGlobal(&bImagNegGlobal, bImagNeg);
    CheevjBindFloatGlobal(&cRealGlobal, cReal);
    CheevjBindFloatGlobal(&cImagGlobal, cImag);

    opmm.SetMatrix(aRealGlobal, aImagGlobal, bRealGlobal, bImagGlobal, bImagNegGlobal, cRealGlobal, cImagGlobal, n, k);
    opmm.Process(0, 0, 0, m, n, k);
#else
    (void)aReal;
    (void)aImag;
    (void)bReal;
    (void)bImag;
    (void)bImagNeg;
    (void)cReal;
    (void)cImag;
    (void)m;
    (void)n;
    (void)k;
#endif
}

class CheevjPanelUpdate
{
   public:
    __aicore__ inline CheevjPanelUpdate() : hasOutputImagNeg(false) {}

    __aicore__ inline void SetShape(int matrixN, int strideN, int panelBegin, int panelDim)
    {
        this->matrixN = matrixN;
        this->strideN = strideN;
        this->panelBegin = panelBegin;
        this->panelDim = panelDim;
        this->usePairShape = false;
        this->leftBegin = panelBegin;
        this->leftDim = panelDim;
        this->rightBegin = panelBegin + panelDim;
        this->rightDim = 0;
        this->packReadyFlag = CHEEVJ_PANEL_PACK_READY_FLAG;
        this->gemmDoneFlag = CHEEVJ_PANEL_GEMM_DONE_FLAG;
    }

    __aicore__ inline void SetPairShape(int matrixN, int strideN, int leftBegin, int leftDim, int rightBegin,
                                        int rightDim)
    {
        this->matrixN = matrixN;
        this->strideN = strideN;
        this->panelBegin = leftBegin;
        this->panelDim = leftDim + rightDim;
        this->usePairShape = true;
        this->leftBegin = leftBegin;
        this->leftDim = leftDim;
        this->rightBegin = rightBegin;
        this->rightDim = rightDim;
        this->packReadyFlag = CHEEVJ_PANEL_PACK_READY_FLAG;
        this->gemmDoneFlag = CHEEVJ_PANEL_GEMM_DONE_FLAG;
    }

    __aicore__ inline void SetFlagIds(int packReadyFlag, int gemmDoneFlag)
    {
        this->packReadyFlag = packReadyFlag;
        this->gemmDoneFlag = gemmDoneFlag;
    }

    __aicore__ inline void SetMatrix(GM_ADDR matrixReal, GM_ADDR matrixImag)
    {
        CheevjBindFloatGlobal(&matrixRealGlobal, matrixReal);
        CheevjBindFloatGlobal(&matrixImagGlobal, matrixImag);
    }

    __aicore__ inline void SetMatrix(GlobalTensor<float> matrixReal, GlobalTensor<float> matrixImag)
    {
        CheevjBindFloatGlobal(&matrixRealGlobal, matrixReal);
        CheevjBindFloatGlobal(&matrixImagGlobal, matrixImag);
    }

    __aicore__ inline void SetUnitary(GM_ADDR unitaryReal, GM_ADDR unitaryImag, GM_ADDR unitaryImagNeg)
    {
        CheevjBindFloatGlobal(&unitaryRealGlobal, unitaryReal);
        CheevjBindFloatGlobal(&unitaryImagGlobal, unitaryImag);
        CheevjBindFloatGlobal(&unitaryImagNegGlobal, unitaryImagNeg);
    }

    __aicore__ inline void SetUnitary(GlobalTensor<float> unitaryReal, GlobalTensor<float> unitaryImag,
                                      GlobalTensor<float> unitaryImagNeg)
    {
        CheevjBindFloatGlobal(&unitaryRealGlobal, unitaryReal);
        CheevjBindFloatGlobal(&unitaryImagGlobal, unitaryImag);
        CheevjBindFloatGlobal(&unitaryImagNegGlobal, unitaryImagNeg);
    }

    __aicore__ inline void SetColumnMajorUnitary(GM_ADDR unitaryRealColumnMajor, GM_ADDR unitaryImagColumnMajor,
                                                 int unitaryStride)
    {
        CheevjBindFloatGlobal(&unitaryColumnMajorRealGlobal, unitaryRealColumnMajor);
        CheevjBindFloatGlobal(&unitaryColumnMajorImagGlobal, unitaryImagColumnMajor);
        this->unitaryStride = unitaryStride;
    }

    __aicore__ inline void SetColumnMajorUnitary(GlobalTensor<float> unitaryRealColumnMajor,
                                                 GlobalTensor<float> unitaryImagColumnMajor, int unitaryStride)
    {
        CheevjBindFloatGlobal(&unitaryColumnMajorRealGlobal, unitaryRealColumnMajor);
        CheevjBindFloatGlobal(&unitaryColumnMajorImagGlobal, unitaryImagColumnMajor);
        this->unitaryStride = unitaryStride;
    }

    __aicore__ inline void SetPacked(GM_ADDR packedAReal, GM_ADDR packedAImag, GM_ADDR packedCReal, GM_ADDR packedCImag)
    {
        SetPackedInput(packedAReal, packedAImag);
        SetPackedOutput(packedCReal, packedCImag);
    }

    __aicore__ inline void SetPacked(GlobalTensor<float> packedAReal, GlobalTensor<float> packedAImag,
                                     GlobalTensor<float> packedCReal, GlobalTensor<float> packedCImag)
    {
        SetPackedInput(packedAReal, packedAImag);
        SetPackedOutput(packedCReal, packedCImag);
    }

    __aicore__ inline void SetPackedInput(GM_ADDR packedAReal, GM_ADDR packedAImag)
    {
        CheevjBindFloatGlobal(&packedARealGlobal, packedAReal);
        CheevjBindFloatGlobal(&packedAImagGlobal, packedAImag);
    }

    __aicore__ inline void SetPackedInput(GlobalTensor<float> packedAReal, GlobalTensor<float> packedAImag)
    {
        CheevjBindFloatGlobal(&packedARealGlobal, packedAReal);
        CheevjBindFloatGlobal(&packedAImagGlobal, packedAImag);
    }

    __aicore__ inline void SetPackedOutput(GM_ADDR packedCReal, GM_ADDR packedCImag)
    {
        CheevjBindFloatGlobal(&packedCRealGlobal, packedCReal);
        CheevjBindFloatGlobal(&packedCImagGlobal, packedCImag);
    }

    __aicore__ inline void SetPackedOutput(GlobalTensor<float> packedCReal, GlobalTensor<float> packedCImag)
    {
        CheevjBindFloatGlobal(&packedCRealGlobal, packedCReal);
        CheevjBindFloatGlobal(&packedCImagGlobal, packedCImag);
    }

    __aicore__ inline void SetOutput(GM_ADDR outputReal, GM_ADDR outputImag)
    {
        CheevjBindFloatGlobal(&outputRealGlobal, outputReal);
        CheevjBindFloatGlobal(&outputImagGlobal, outputImag);
        hasOutputImagNeg = false;
    }

    __aicore__ inline void SetOutput(GlobalTensor<float> outputReal, GlobalTensor<float> outputImag)
    {
        CheevjBindFloatGlobal(&outputRealGlobal, outputReal);
        CheevjBindFloatGlobal(&outputImagGlobal, outputImag);
        hasOutputImagNeg = false;
    }

    __aicore__ inline void SetOutput(GM_ADDR outputReal, GM_ADDR outputImag, GM_ADDR outputImagNeg)
    {
        CheevjBindFloatGlobal(&outputRealGlobal, outputReal);
        CheevjBindFloatGlobal(&outputImagGlobal, outputImag);
        CheevjBindFloatGlobal(&outputImagNegGlobal, outputImagNeg);
        hasOutputImagNeg = true;
    }

    __aicore__ inline void SetOutput(GlobalTensor<float> outputReal, GlobalTensor<float> outputImag,
                                     GlobalTensor<float> outputImagNeg)
    {
        CheevjBindFloatGlobal(&outputRealGlobal, outputReal);
        CheevjBindFloatGlobal(&outputImagGlobal, outputImag);
        CheevjBindFloatGlobal(&outputImagNegGlobal, outputImagNeg);
        hasOutputImagNeg = true;
    }

    __aicore__ inline void PackInputPanel()
    {
        for (int row = 0; row < matrixN; ++row)
        {
            for (int localCol = 0; localCol < panelDim; ++localCol)
            {
                const int matrixOffset = MatrixOffset(row, localCol);
                const int packedOffset = PackedOffset(row, localCol);
                packedARealGlobal.SetValue(packedOffset, matrixRealGlobal.GetValue(matrixOffset));
                packedAImagGlobal.SetValue(packedOffset, matrixImagGlobal.GetValue(matrixOffset));
            }
        }
    }

    __aicore__ inline void ScatterOutputPanel()
    {
        for (int row = 0; row < matrixN; ++row)
        {
            for (int localCol = 0; localCol < panelDim; ++localCol)
            {
                const int matrixOffset = MatrixOffset(row, localCol);
                const int packedOffset = PackedOffset(row, localCol);
                outputRealGlobal.SetValue(matrixOffset, packedCRealGlobal.GetValue(packedOffset));
                const float outputImag = packedCImagGlobal.GetValue(packedOffset);
                outputImagGlobal.SetValue(matrixOffset, outputImag);
                if (hasOutputImagNeg)
                {
                    outputImagNegGlobal.SetValue(matrixOffset, -outputImag);
                }
            }
        }
    }

    __aicore__ inline void PackColumnMajorUnitaryToRowMajor()
    {
        for (int row = 0; row < panelDim; ++row)
        {
            for (int col = 0; col < panelDim; ++col)
            {
                const int srcOffset = row + col * unitaryStride;
                const int dstOffset = row * panelDim + col;
                const float imagValue = unitaryColumnMajorImagGlobal.GetValue(srcOffset);
                unitaryRealGlobal.SetValue(dstOffset, unitaryColumnMajorRealGlobal.GetValue(srcOffset));
                unitaryImagGlobal.SetValue(dstOffset, imagValue);
                unitaryImagNegGlobal.SetValue(dstOffset, -imagValue);
            }
        }
    }

    __aicore__ inline void ProcessMixedAiv()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        PackInputPanel();
        PublishPackedPanelToCube();
        WaitCubeGemmDone();
        CheevjPanelCleanEntireDataCache();
        ScatterOutputPanel();
#endif
    }

    __aicore__ inline void ProcessMixedAivWithUnitaryPack()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        PackInputPanel();
        PackColumnMajorUnitaryToRowMajor();
        PublishPackedPanelToCube();
        WaitCubeGemmDone();
        CheevjPanelCleanEntireDataCache();
        ScatterOutputPanel();
#endif
    }

    __aicore__ inline void ProcessMixedCube()
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        WaitPackedPanel();
        TPipe pipe;
        CMatmulCustom<float> opmm;
        opmm.Init(&pipe);
        opmm.SetMatrix(packedARealGlobal, packedAImagGlobal, unitaryRealGlobal, unitaryImagGlobal, unitaryImagNegGlobal,
                       packedCRealGlobal, packedCImagGlobal, panelDim, panelDim);
        opmm.Process(0, 0, 0, matrixN, panelDim, panelDim);
        PublishCubeGemmDone();
#endif
    }

    __aicore__ inline void ProcessMixedCube(CMatmulCustom<float> *opmm)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        if (opmm == nullptr)
        {
            return;
        }
        WaitPackedPanel();
        opmm->SetMatrix(packedARealGlobal, packedAImagGlobal, unitaryRealGlobal, unitaryImagGlobal,
                        unitaryImagNegGlobal, packedCRealGlobal, packedCImagGlobal, panelDim, panelDim);
        opmm->Process(0, 0, 0, matrixN, panelDim, panelDim);
        PublishCubeGemmDone();
#else
        (void)opmm;
#endif
    }

   private:
    GlobalTensor<float> matrixRealGlobal;
    GlobalTensor<float> matrixImagGlobal;
    GlobalTensor<float> unitaryRealGlobal;
    GlobalTensor<float> unitaryImagGlobal;
    GlobalTensor<float> unitaryImagNegGlobal;
    GlobalTensor<float> unitaryColumnMajorRealGlobal;
    GlobalTensor<float> unitaryColumnMajorImagGlobal;
    GlobalTensor<float> packedARealGlobal;
    GlobalTensor<float> packedAImagGlobal;
    GlobalTensor<float> packedCRealGlobal;
    GlobalTensor<float> packedCImagGlobal;
    GlobalTensor<float> outputRealGlobal;
    GlobalTensor<float> outputImagGlobal;
    GlobalTensor<float> outputImagNegGlobal;
    int matrixN;
    int strideN;
    int panelBegin;
    int panelDim;
    bool usePairShape;
    int leftBegin;
    int leftDim;
    int rightBegin;
    int rightDim;
    int unitaryStride;
    int packReadyFlag;
    int gemmDoneFlag;
    bool hasOutputImagNeg;

    __aicore__ inline int GlobalColForLocal(int localCol) const
    {
        if (!usePairShape)
        {
            return panelBegin + localCol;
        }
        if (localCol < leftDim)
        {
            return leftBegin + localCol;
        }
        return rightBegin + localCol - leftDim;
    }

    __aicore__ inline int MatrixOffset(int row, int localCol) const
    {
        const int globalCol = GlobalColForLocal(localCol);
        return globalCol * strideN + row;
    }

    __aicore__ inline int PackedOffset(int row, int localCol) const { return row * panelDim + localCol; }

    __aicore__ inline void PublishPackedPanelToCube()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        PipeBarrier<PIPE_ALL>();
        CheevjPanelCleanEntireDataCache();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(packReadyFlag);
#endif
    }

    __aicore__ inline void WaitPackedPanel()
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        CrossCoreWaitFlag(packReadyFlag);
#endif
    }

    __aicore__ inline void PublishCubeGemmDone()
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        CrossCoreSetFlag<0x2, PIPE_FIX>(gemmDoneFlag);
#endif
    }

    __aicore__ inline void WaitCubeGemmDone()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        CrossCoreWaitFlag(gemmDoneFlag);
#endif
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_PANEL_UPDATE_HPP
