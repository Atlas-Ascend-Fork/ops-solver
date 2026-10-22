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
 * \file gemm.hpp
 * \brief
 */

#ifndef _GEMM_HPP_
#define _GEMM_HPP_

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using AscendC::DataCopy;
using AscendC::Fixpipe;
using AscendC::FixpipeParamsV220;
using AscendC::GlobalTensor;
using AscendC::LoadData;
using AscendC::LoadData3DParamsV2Pro;
using AscendC::LocalTensor;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Nd2NzParams;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

template <typename T>
class CMatmulCustom
{
    GlobalTensor<T> aRealGlobal, bRealGlobal, cRealGlobal;
    GlobalTensor<T> aImagGlobal, bImagGlobal, cImagGlobal;
    GlobalTensor<T> aImagNegGlobal;
    GlobalTensor<T> bImagNegGlobal;
    TQue<QuePosition::A1, 2> inQueueA1Real, inQueueA1Imag;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1Real, inQueueB1Imag;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1Real;
    TBuf<TPosition::CO1> outBufCO1Imag;
    int N, K;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int baseM, baseN, baseK;
    int tailM, tailN, tailK;
    int stepM, stepN;
    int aligned_n;
    int alignToN;
    int stepK;
    bool useAImagNeg;
    bool segmentedMode;
    int segmentedMatrixLeadingDim;
    int segmentedLeftBegin;
    int segmentedLeftDim;
    int segmentedRightBegin;
    int segmentedRightDim;
    int32_t segmentedEventMToFix;
    int32_t segmentedEventFixToM;

   public:
    __aicore__ inline CMatmulCustom() : useAImagNeg(false), segmentedMode(false) {}
    __aicore__ inline void Init(TPipe *pipe)
    {
        this->stepK = 2;
        pipe->InitBuffer(inQueueA1Real, 2, 65536);
        pipe->InitBuffer(inQueueA1Imag, 2, 65536);
        pipe->InitBuffer(inQueueB1Real, 2, 65536);
        pipe->InitBuffer(inQueueB1Imag, 2, 65536);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1Imag, 65536);
        pipe->InitBuffer(outBufCO1Real, 65536);
    }
    __aicore__ inline void InitSegmentedEvents()
    {
        // Only the segmented multi-Fixpipe path needs these event classes.
        // Keep legacy LU/TRSM/Getri users of CMatmulCustom on their original
        // event footprint.
        segmentedEventMToFix = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::M_FIX));
        segmentedEventFixToM = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_M));
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aRealGlobal, GlobalTensor<T> aImagGlobal,
                                     GlobalTensor<T> bRealGlobal, GlobalTensor<T> bImagGlobal,
                                     GlobalTensor<T> bImagNegGlobal, GlobalTensor<T> cRealGlobal,
                                     GlobalTensor<T> cImagGlobal, int N, int K)
    {
        this->N = N;
        this->K = K;
        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->bImagNegGlobal = bImagNegGlobal;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
        this->useAImagNeg = false;
        this->segmentedMode = false;
    }
    __aicore__ inline void SetMatrixWithAImagNeg(GlobalTensor<T> aRealGlobal, GlobalTensor<T> aImagGlobal,
                                                 GlobalTensor<T> aImagNegGlobal, GlobalTensor<T> bRealGlobal,
                                                 GlobalTensor<T> bImagGlobal, GlobalTensor<T> cRealGlobal,
                                                 GlobalTensor<T> cImagGlobal, int N, int K)
    {
        this->N = N;
        this->K = K;
        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->aImagNegGlobal = aImagNegGlobal;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
        this->useAImagNeg = true;
        this->segmentedMode = false;
    }
    __aicore__ inline void SetMatrixWithAImagNegSegmented(GlobalTensor<T> aRealGlobal, GlobalTensor<T> aImagGlobal,
                                                          GlobalTensor<T> aImagNegGlobal, GlobalTensor<T> bRealGlobal,
                                                          GlobalTensor<T> bImagGlobal, GlobalTensor<T> cRealGlobal,
                                                          GlobalTensor<T> cImagGlobal, int logicalN, int k,
                                                          int matrixLeadingDim, int leftBegin, int leftDim,
                                                          int rightBegin, int rightDim)
    {
        // In segmented mode logicalN bounds the full matrix, while
        // matrixLeadingDim is its physical GM row stride. Process() consumes
        // one column tile: offsetB/offsetC are its column offsets and blockN
        // is its width. The three A planes must already point at one 32x32 U^T.
        this->N = logicalN;
        this->K = k;
        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->aImagNegGlobal = aImagNegGlobal;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
        this->useAImagNeg = true;
        this->segmentedMode = true;
        this->segmentedMatrixLeadingDim = matrixLeadingDim;
        this->segmentedLeftBegin = leftBegin;
        this->segmentedLeftDim = leftDim;
        this->segmentedRightBegin = rightBegin;
        this->segmentedRightDim = rightDim;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK)
    {
        if (segmentedMode)
        {
            ProcessSegmented(offsetA, offsetB, offsetC, blockM, blockN, blockK);
            return;
        }

        // K-granularity contract: this class packs operands with Nd2Nz and feeds raw
        // Mmad, whose fractal K unit is 8 for FP32. blockK must be a multiple of 8
        // (callers pad K with zeros when needed); otherwise the packing and the Mmad
        // K granularity disagree and the result is silently wrong.
        PrepareStandardProcess(offsetA, offsetB, offsetC, blockM, blockN, blockK);
        const int mBlocks = (singleCoreM + baseM - 1) / baseM;
        const int nBlocks = (singleCoreN + baseN - 1) / baseN;
        const int kBlocks = (singleCoreK + baseK - 1) / baseK;

        int32_t eventIDFIXToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
        ProcessStandardTiles(mBlocks, nBlocks, kBlocks);
    }

   private:
    __aicore__ inline void PrepareStandardProcess(int offsetA, int offsetB, int offsetC, int blockM, int blockN,
                                                  int blockK)
    {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;

        // default: (128, 128, 64)
        baseN = min(singleCoreN, 128);
        int nn = 64;
        while (nn < baseN)
        {
            nn <<= 1;
        }
        baseM = min(singleCoreM, 16384 / nn);
        int mm = 16;
        while (mm < baseM)
        {
            mm <<= 1;
        }
        baseK = min(singleCoreK, min(8192 / nn, 8192 / mm));
        // Cap baseK at the FP32 fractal K unit (C0=8, see gemm_segmented.inc):
        // Nd2Nz packs K in units of 8, so a larger baseK lets a non-multiple-of-8
        // tailK cross a fractal K boundary and silently corrupt the Mmad result.
        baseK = min(baseK, 8);
    }

    __aicore__ inline void ProcessStandardTiles(int mBlocks, int nBlocks, int kBlocks)
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        for (int i = 0; i < nBlocks; ++i)
        {
            tailN = min(singleCoreN - i * baseN, baseN);
            for (int j = 0; j < mBlocks; ++j)
            {
                tailM = min(singleCoreM - j * baseM, baseM);
                AscendC::Load3DSetFMatrixCal(1, tailM, padList);
                int offsetC_1 = offsetC + j * baseM * N + i * baseN;
                LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
                LocalTensor<T> c1ImagLocal = outBufCO1Imag.Get<T>();
                for (int k = 0; k < kBlocks; k += stepK)
                {
                    ProcessStandardKGroup(i, j, k, kBlocks, c1RealLocal, c1ImagLocal, padList);
                }
                CopyOut(c1RealLocal, offsetC_1, 0);
                CopyOut(c1ImagLocal, offsetC_1, 1);
            }
        }
    }

    __aicore__ inline void ProcessStandardKGroup(int i, int j, int k, int kBlocks, const LocalTensor<T> &c1RealLocal,
                                                 const LocalTensor<T> &c1ImagLocal, const uint8_t (&padList)[4])
    {
        tailK = baseK;
        const int cnt = min(kBlocks - k, stepK);
        AscendC::Load3DSetFMatrixBCal(1, baseK * cnt, padList);
        const int offsetA_1 = offsetA + j * baseM * K + k * baseK;
        const int offsetB_1 = offsetB + k * baseK * N + i * baseN;
        CopyInAReal(offsetA_1, cnt);
        CopyInBReal(offsetB_1, cnt);
        CopyInBImag(offsetB_1, cnt, 0);
        CopyInAPlane(aImagGlobal, offsetA_1, cnt);
        if (useAImagNeg)
        {
            CopyInAPlane(aImagNegGlobal, offsetA_1, cnt);
        }
        else
        {
            CopyInBImag(offsetB_1, cnt, 1);
        }

        LocalTensor<T> a1RealLocal = inQueueA1Real.DeQue<T>();
        LocalTensor<T> b1RealLocal = inQueueB1Real.DeQue<T>();
        LocalTensor<T> b1ImagLocal = inQueueB1Imag.DeQue<T>();
        AccumulateRealProducts(a1RealLocal, b1RealLocal, b1ImagLocal, c1RealLocal, c1ImagLocal, k, cnt);
        inQueueA1Real.FreeTensor(a1RealLocal);

        LocalTensor<T> a1ImagLocal = inQueueA1Imag.DeQue<T>();
        if (useAImagNeg)
        {
            AccumulateImagProductsWithNeg(a1ImagLocal, b1RealLocal, b1ImagLocal, c1RealLocal, c1ImagLocal, k, kBlocks,
                                          cnt);
        }
        else
        {
            AccumulateImagProducts(a1ImagLocal, b1RealLocal, b1ImagLocal, c1RealLocal, c1ImagLocal, k, kBlocks, cnt);
        }
        inQueueB1Imag.FreeTensor(b1ImagLocal);
        inQueueA1Imag.FreeTensor(a1ImagLocal);
        inQueueB1Real.FreeTensor(b1RealLocal);
    }

    __aicore__ inline void AccumulateRealProducts(const LocalTensor<T> &a1RealLocal, const LocalTensor<T> &b1RealLocal,
                                                  const LocalTensor<T> &b1ImagLocal, const LocalTensor<T> &c1RealLocal,
                                                  const LocalTensor<T> &c1ImagLocal, int k, int cnt)
    {
        for (int x = 0; x < cnt; ++x)
        {
            tailK = min(singleCoreK - (k + x) * baseK, baseK);
            SplitA(a1RealLocal, x);
            SplitB(b1RealLocal, x);
            LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
            Compute(a2Local, c1RealLocal, !k && !x, 0);
            SplitB(b1ImagLocal, x);
            Compute(a2Local, c1ImagLocal, !k && !x, 0);
            inQueueA2.FreeTensor(a2Local);
        }
    }

    __aicore__ inline void AccumulateImagProductsWithNeg(const LocalTensor<T> &a1ImagLocal,
                                                         const LocalTensor<T> &b1RealLocal,
                                                         const LocalTensor<T> &b1ImagLocal,
                                                         const LocalTensor<T> &c1RealLocal,
                                                         const LocalTensor<T> &c1ImagLocal, int k, int kBlocks, int cnt)
    {
        LocalTensor<T> a1ImagNegLocal = inQueueA1Imag.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            tailK = min(singleCoreK - (k + x) * baseK, baseK);
            SplitA(a1ImagLocal, x);
            SplitB(b1RealLocal, x);
            LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
            Compute(a2Local, c1ImagLocal, 0, k + x == kBlocks - 1);
            inQueueA2.FreeTensor(a2Local);

            SplitA(a1ImagNegLocal, x);
            SplitB(b1ImagLocal, x);
            a2Local = inQueueA2.DeQue<T>();
            Compute(a2Local, c1RealLocal, 0, k + x == kBlocks - 1);
            inQueueA2.FreeTensor(a2Local);
        }
        inQueueA1Imag.FreeTensor(a1ImagNegLocal);
    }

    __aicore__ inline void AccumulateImagProducts(const LocalTensor<T> &a1ImagLocal, const LocalTensor<T> &b1RealLocal,
                                                  LocalTensor<T> &b1ImagLocal, const LocalTensor<T> &c1RealLocal,
                                                  const LocalTensor<T> &c1ImagLocal, int k, int kBlocks, int cnt)
    {
        inQueueB1Imag.FreeTensor(b1ImagLocal);
        b1ImagLocal = inQueueB1Imag.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            tailK = min(singleCoreK - (k + x) * baseK, baseK);
            SplitA(a1ImagLocal, x);
            SplitB(b1RealLocal, x);
            LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
            Compute(a2Local, c1ImagLocal, 0, k + x == kBlocks - 1);
            SplitB(b1ImagLocal, x);
            Compute(a2Local, c1RealLocal, 0, k + x == kBlocks - 1);
            inQueueA2.FreeTensor(a2Local);
        }
    }

#include "gemm_segmented.inc"

    __aicore__ inline void CopyInAReal(int offsetA_1, int cnt)
    {
        LocalTensor<T> a1Local = inQueueA1Real.AllocTensor<T>();
        Nd2NzParams intriParamsMatA{1,
                                    static_cast<uint16_t>(tailM),
                                    static_cast<uint16_t>(tailK * cnt),
                                    0,
                                    static_cast<uint16_t>(K),
                                    static_cast<uint16_t>(tailM),
                                    1,
                                    1};
        DataCopy(a1Local, aRealGlobal[offsetA_1], intriParamsMatA);
        inQueueA1Real.EnQue(a1Local);
    }
    __aicore__ inline void CopyInAPlane(GlobalTensor<T> source, int offsetA_1, int cnt)
    {
        LocalTensor<T> a1Local = inQueueA1Imag.AllocTensor<T>();
        Nd2NzParams intriParamsMatA{1,
                                    static_cast<uint16_t>(tailM),
                                    static_cast<uint16_t>(tailK * cnt),
                                    0,
                                    static_cast<uint16_t>(K),
                                    static_cast<uint16_t>(tailM),
                                    1,
                                    1};
        DataCopy(a1Local, source[offsetA_1], intriParamsMatA);
        inQueueA1Imag.EnQue(a1Local);
    }
    __aicore__ inline void CopyInBReal(int offsetB_1, int cnt)
    {
        LocalTensor<T> b1Local = inQueueB1Real.AllocTensor<T>();
        Nd2NzParams intriParamsMatB{
            1,
            static_cast<uint16_t>(tailK * cnt),
            static_cast<uint16_t>(tailN),
            0,
            static_cast<uint16_t>(N),
            static_cast<uint16_t>(tailK * cnt),
            1,
            1,
        };
        DataCopy(b1Local, bRealGlobal[offsetB_1], intriParamsMatB);
        inQueueB1Real.EnQue(b1Local);
    }
    __aicore__ inline void CopyInBImag(int offsetB_1, int cnt, int neg)
    {
        LocalTensor<T> b1Local = inQueueB1Imag.AllocTensor<T>();
        Nd2NzParams intriParamsMatB{1,
                                    static_cast<uint16_t>(tailK * cnt),
                                    static_cast<uint16_t>(tailN),
                                    0,
                                    static_cast<uint16_t>(N),
                                    static_cast<uint16_t>(tailK * cnt),
                                    1,
                                    1};
        if (!neg)
            DataCopy(b1Local, bImagGlobal[offsetB_1], intriParamsMatB);
        else
            DataCopy(b1Local, bImagNegGlobal[offsetB_1], intriParamsMatB);
        inQueueB1Imag.EnQue(b1Local);
    }
    __aicore__ inline void SplitA(const LocalTensor<T> &a1Local, int splitIdx)
    {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        uint16_t mPos = 0;
        uint16_t sAL1NOffset_ = 0;
        loadData3DV2.channelSize = tailK;
        loadData3DV2.extConfig =
            ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) | ((uint64_t)tailM << 16) | (uint64_t)tailK;
        LoadData<T>(a2Local, a1Local[splitIdx * baseK * tailM], loadData3DV2);
        inQueueA2.EnQue(a2Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<T> &b1Local, int splitIdx)
    {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        uint16_t mPos = 0;
        uint16_t sBL1NOffset_ = 0;
        loadData3DV2.channelSize = tailN;
        loadData3DV2.extConfig =
            ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) | ((uint64_t)tailK << 16) | (uint64_t)tailN;
        loadData3DV2.fMatrixCtrl = true;
        LoadData<T>(b2Local, b1Local[splitIdx * baseK * 8], loadData3DV2);
        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<T> &a2Local, const LocalTensor<T> &c1Local, int first,
                                   const int unitFlag)
    {
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
        MmadParams mmadParams;
        mmadParams.m = tailM;
        mmadParams.n = tailN;
        mmadParams.k = tailK;
        mmadParams.unitFlag = unitFlag ? 3 : 2;
        if (first)
        {
            Mmad(c1Local, a2Local, b2Local, mmadParams);
        }
        else
        {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        }
        if (((tailM * tailN) >> 8) < 10)
        {
            PipeBarrier<PIPE_M>();
        }
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut(const LocalTensor<T> &c1Local, int offsetC_1, int imag)
    {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = tailN;
        fixpipeParams.mSize = tailM;
        fixpipeParams.srcStride = tailM;
        fixpipeParams.dstStride = N;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if (!imag)
        {
            Fixpipe(cRealGlobal[offsetC_1], c1Local, fixpipeParams);
        }
        else
        {
            Fixpipe(cImagGlobal[offsetC_1], c1Local, fixpipeParams);
        }
    }
};

#endif  // _GEMM_HPP_
