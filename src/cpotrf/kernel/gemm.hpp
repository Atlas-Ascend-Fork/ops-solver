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

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class CMatmulCustom
{
    GlobalTensor<T> aRealGlobal, bRealGlobal, cRealGlobal;
    GlobalTensor<T> aImagGlobal, bImagGlobal, cImagGlobal;
    GlobalTensor<T> extGlobal;
    TQue<QuePosition::A1, 2> inQueueA1Real, inQueueA1Imag;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1Real, inQueueB1Imag;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1Real;
    TBuf<TPosition::CO1> outBufCO1Imag;
    int strideA, strideB, strideC;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int baseM, baseN, baseK;
    int tailM, tailN, tailK;
    int stepM, stepN;
    int aligned_n;
    int alignToN;
    int stepK;

   public:
    __aicore__ inline CMatmulCustom() {}
    __aicore__ inline void Init(TPipe* pipe)
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
    __aicore__ inline void SetMatrix(GlobalTensor<T> aRealGlobal, GlobalTensor<T> aImagGlobal,
                                     GlobalTensor<T> bRealGlobal, GlobalTensor<T> bImagGlobal,
                                     GlobalTensor<T> extGlobal, GlobalTensor<T> cRealGlobal,
                                     GlobalTensor<T> cImagGlobal, int strideA, int strideB, int strideC)
    {
        this->strideA = strideA;
        this->strideB = strideB;
        this->strideC = strideC;
        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->extGlobal = extGlobal;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
    }
    template <bool trans_A = false, bool trans_B = false>
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK)
    {
        // extGlobal == bImagNegGlobal
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;

        // default: (128, 128, 64)

        baseN = min(singleCoreN, 128);
        int nn = 64;
        while (nn < baseN) nn <<= 1;
        baseM = min(singleCoreM, 16384 / nn);
        int mm = 16;
        while (mm < baseM) mm <<= 1;
        baseK = min(singleCoreK, min(8192 / nn, 8192 / mm));

        int mBlocks = (singleCoreM + baseM - 1) / baseM;
        int nBlocks = (singleCoreN + baseN - 1) / baseN;
        int kBlocks = (singleCoreK + baseK - 1) / baseK;

        static constexpr uint8_t padList[]{0, 0, 0, 0};
        int cnt;
        int32_t eventIDFIXToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);

        for (int i = 0; i < nBlocks; ++i)
        {
            tailN = min(singleCoreN - i * baseN, baseN);
            for (int j = 0; j < mBlocks; ++j)
            {
                tailM = min(singleCoreM - j * baseM, baseM);
                if constexpr (!trans_A) Load3DSetFMatrixCal(1, tailM, padList);
                int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
                LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
                LocalTensor<T> c1ImagLocal = outBufCO1Imag.Get<T>();

                for (int k = 0; k < kBlocks; k += stepK)
                {
                    tailK = baseK;
                    cnt = min(kBlocks - k, stepK);
                    if constexpr (trans_A) Load3DSetFMatrixCal(1, baseK * cnt, padList);
                    Load3DSetFMatrixBCal(1, baseK * cnt, padList);
                    int offsetA_1, offsetB_1;
                    if constexpr (trans_A)
                        offsetA_1 = offsetA + k * baseK * strideA + j * baseM;
                    else
                        offsetA_1 = offsetA + j * baseM * strideA + k * baseK;
                    if constexpr (trans_B)
                        offsetB_1 = offsetB + i * baseN * strideB + k * baseK;
                    else
                        offsetB_1 = offsetB + k * baseK * strideB + i * baseN;
                    CopyInAReal<trans_A>(offsetA_1, cnt);
                    CopyInBReal<trans_B>(offsetB_1, cnt);
                    CopyInBImag<trans_B, 0>(offsetB_1, cnt);
                    CopyInAImag<trans_A>(offsetA_1, cnt);
                    CopyInBImag<trans_B, 1>(offsetB_1, cnt);

                    LocalTensor<T> a1RealLocal = inQueueA1Real.DeQue<T>();
                    LocalTensor<T> b1RealLocal = inQueueB1Real.DeQue<T>();
                    LocalTensor<T> b1ImagLocal = inQueueB1Imag.DeQue<T>();
                    // C1 += A1B1
                    // C2 += A1B2
                    for (int x = 0; x < cnt; ++x)
                    {
                        tailK = min(singleCoreK - (k + x) * baseK, baseK);
                        SplitA<trans_A>(a1RealLocal, x);
                        SplitB<trans_B>(b1RealLocal, x);
                        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
                        Compute(a2Local, c1RealLocal, !k && !x, 0);
                        SplitB<trans_B>(b1ImagLocal, x);
                        Compute(a2Local, c1ImagLocal, !k && !x, 0);
                        inQueueA2.FreeTensor(a2Local);
                    }
                    inQueueA1Real.FreeTensor(a1RealLocal);
                    inQueueB1Imag.FreeTensor(b1ImagLocal);

                    LocalTensor<T> a1ImagLocal = inQueueA1Imag.DeQue<T>();
                    b1ImagLocal = inQueueB1Imag.DeQue<T>();
                    // C2 += A2B1
                    // C1 -= A2B2
                    for (int x = 0; x < cnt; ++x)
                    {
                        tailK = min(singleCoreK - (k + x) * baseK, baseK);
                        SplitA<trans_A>(a1ImagLocal, x);
                        SplitB<trans_B>(b1RealLocal, x);
                        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
                        Compute(a2Local, c1ImagLocal, 0, k + x == kBlocks - 1);
                        SplitB<trans_B>(b1ImagLocal, x);
                        Compute(a2Local, c1RealLocal, 0, k + x == kBlocks - 1);
                        inQueueA2.FreeTensor(a2Local);
                    }
                    inQueueB1Imag.FreeTensor(b1ImagLocal);
                    inQueueA1Imag.FreeTensor(a1ImagLocal);
                    inQueueB1Real.FreeTensor(b1RealLocal);
                }
                CopyOut<false>(c1RealLocal, offsetC_1);
                CopyOut<true>(c1ImagLocal, offsetC_1);
            }
        }
    }

   private:
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInAReal(int offsetA_1, int cnt)
    {
        LocalTensor<T> a1Local = inQueueA1Real.AllocTensor<T>();
        if constexpr (trans)
        {
            Nd2NzParams intriParamsMatA{1,
                                        static_cast<uint16_t>(tailK * cnt),
                                        static_cast<uint16_t>(tailM),
                                        0,
                                        static_cast<uint16_t>(strideA),
                                        static_cast<uint16_t>(tailK * cnt),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(a1Local, aRealGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, extGlobal[offsetA_1], intriParamsMatA);
        }
        else
        {
            Nd2NzParams intriParamsMatA{1,
                                        static_cast<uint16_t>(tailM),
                                        static_cast<uint16_t>(tailK * cnt),
                                        0,
                                        static_cast<uint16_t>(strideA),
                                        static_cast<uint16_t>(tailM),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(a1Local, aRealGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, extGlobal[offsetA_1], intriParamsMatA);
        }
        inQueueA1Real.EnQue(a1Local);
    }
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInAImag(int offsetA_1, int cnt)
    {
        LocalTensor<T> a1Local = inQueueA1Imag.AllocTensor<T>();
        if constexpr (trans)
        {
            Nd2NzParams intriParamsMatA{1,
                                        static_cast<uint16_t>(tailK * cnt),
                                        static_cast<uint16_t>(tailM),
                                        0,
                                        static_cast<uint16_t>(strideA),
                                        static_cast<uint16_t>(tailK * cnt),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(a1Local, aImagGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, extGlobal[offsetA_1], intriParamsMatA);
        }
        else
        {
            Nd2NzParams intriParamsMatA{1,
                                        static_cast<uint16_t>(tailM),
                                        static_cast<uint16_t>(tailK * cnt),
                                        0,
                                        static_cast<uint16_t>(strideA),
                                        static_cast<uint16_t>(tailM),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(a1Local, aImagGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, extGlobal[offsetA_1], intriParamsMatA);
        }
        inQueueA1Imag.EnQue(a1Local);
    }
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInBReal(int offsetB_1, int cnt)
    {
        LocalTensor<T> b1Local = inQueueB1Real.AllocTensor<T>();
        if constexpr (trans)
        {
            Nd2NzParams intriParamsMatB{1,
                                        static_cast<uint16_t>(tailN),
                                        static_cast<uint16_t>(tailK * cnt),
                                        0,
                                        static_cast<uint16_t>(strideB),
                                        static_cast<uint16_t>(tailN),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(b1Local, bRealGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, extGlobal[offsetB_1], intriParamsMatB);
        }
        else
        {
            Nd2NzParams intriParamsMatB{1,
                                        static_cast<uint16_t>(tailK * cnt),
                                        static_cast<uint16_t>(tailN),
                                        0,
                                        static_cast<uint16_t>(strideB),
                                        static_cast<uint16_t>(tailK * cnt),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(b1Local, bRealGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, extGlobal[offsetB_1], intriParamsMatB);
        }
        inQueueB1Real.EnQue(b1Local);
    }
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInBImag(int offsetB_1, int cnt)
    {
        LocalTensor<T> b1Local = inQueueB1Imag.AllocTensor<T>();
        if constexpr (trans)
        {
            Nd2NzParams intriParamsMatB{1,
                                        static_cast<uint16_t>(tailN),
                                        static_cast<uint16_t>(tailK * cnt),
                                        0,
                                        static_cast<uint16_t>(strideB),
                                        static_cast<uint16_t>(tailN),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(b1Local, bImagGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, extGlobal[offsetB_1], intriParamsMatB);
        }
        else
        {
            Nd2NzParams intriParamsMatB{1,
                                        static_cast<uint16_t>(tailK * cnt),
                                        static_cast<uint16_t>(tailN),
                                        0,
                                        static_cast<uint16_t>(strideB),
                                        static_cast<uint16_t>(tailK * cnt),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(b1Local, bImagGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, extGlobal[offsetB_1], intriParamsMatB);
        }
        inQueueB1Imag.EnQue(b1Local);
    }
    template <bool trans = false>
    __aicore__ inline void SplitA(const LocalTensor<T>& a1Local, int splitIdx)
    {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        if constexpr (trans)
        {
            uint16_t mPos = splitIdx * baseK, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = tailM;
            loadData3DV2.extConfig =
                ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) | ((uint64_t)tailK << 16) | (uint64_t)tailM;
            loadData3DV2.enTranspose = true;
            LoadData<T>(a2Local, a1Local, loadData3DV2);
        }
        else
        {
            uint16_t mPos = 0, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = tailK;
            loadData3DV2.extConfig =
                ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) | ((uint64_t)tailM << 16) | (uint64_t)tailK;
            LoadData<T>(a2Local, a1Local[splitIdx * baseK * tailM], loadData3DV2);
        }
        inQueueA2.EnQue(a2Local);
    }
    template <bool trans = false>
    __aicore__ inline void SplitB(const LocalTensor<T>& b1Local, int splitIdx)
    {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        if constexpr (trans)
        {
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = tailK * tailN / 128;
            loadDataParams.srcStride = 1;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = 0;
            LoadData<T>(b2Local, b1Local[splitIdx * baseK * tailN], loadDataParams);
        }
        else
        {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = splitIdx * baseK, sBL1NOffset_ = 0;
            loadData3DV2.channelSize = tailN;
            loadData3DV2.extConfig =
                ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) | ((uint64_t)tailK << 16) | (uint64_t)tailN;
            loadData3DV2.fMatrixCtrl = true;
            LoadData<T>(b2Local, b1Local, loadData3DV2);
        }
        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<T>& a2Local, const LocalTensor<T>& c1Local, int first,
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
        if (tailM * tailN >> 8 < 10) PipeBarrier<PIPE_M>();
        inQueueB2.FreeTensor(b2Local);
    }
    template <bool imag = false>
    __aicore__ inline void CopyOut(const LocalTensor<T>& c1Local, int offsetC_1)
    {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = tailN;
        fixpipeParams.mSize = tailM;
        fixpipeParams.srcStride = tailM;
        fixpipeParams.dstStride = strideC;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr (!imag)
            Fixpipe(cRealGlobal[offsetC_1], c1Local, fixpipeParams);
        else
            Fixpipe(cImagGlobal[offsetC_1], c1Local, fixpipeParams);
    }
};
