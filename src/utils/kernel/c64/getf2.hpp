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
 * \file getf2.hpp
 * \brief
 */

#ifndef C64_GETF2_HPP_
#define C64_GETF2_HPP_

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

// Runtime check for critical kernel constraints: assert() is compiled out in
// release builds (NDEBUG), so buffer-size/shape invariants must be guarded by
// an explicit branch (see issue #12).
#define SOLVER_KERNEL_CHECK(cond, action) \
    do                                    \
    {                                     \
        if (!(cond))                      \
        {                                 \
            action;                       \
        }                                 \
    } while (0)

template <typename T>
class LUCustom1
{  // for M * N <= 8192
    GlobalTensor<T> aGlobalReal;
    GlobalTensor<T> aGlobalImag;
    GlobalTensor<uint32_t> wGlobal;
    int M, N;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    int lineCount;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 1> inQueueSrcReal;
    TQue<QuePosition::VECIN, 1> inQueueSrcImag;
    TBuf<TPosition::VECCALC> workBuf;

   public:
    __aicore__ inline LUCustom1() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc1,
                                TQue<QuePosition::VECIN, 1> inQueueSrc2, GlobalTensor<T> &aGlobalReal,
                                GlobalTensor<T> &aGlobalImag, GlobalTensor<uint32_t> &wGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->wGlobal = wGlobal;

        elementsPerBlock = 32 / sizeof(T);

        this->inQueueSrcReal = inQueueSrc1;
        this->inQueueSrcImag = inQueueSrc2;
        pipe->InitBuffer(workBuf, TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Process(int MatAOffset, int offsetM, int offsetN, int blockM, int realM, int realN)
    {
        SOLVER_KERNEL_CHECK(blockM * blockN <= 8192, return);
        SOLVER_KERNEL_CHECK(blockM >= blockN, return);
        this->blockM = blockM;
        this->realM = realM;
        LocalTensor<T> srcLocalReal;
        LocalTensor<T> srcLocalImag;
        {  // CopyIn
            srcLocalReal = inQueueSrcReal.AllocTensor<T>();
            srcLocalImag = inQueueSrcImag.AllocTensor<T>();
            DataCopyParams copyInParams{static_cast<uint16_t>(realN), static_cast<uint16_t>(blockM / elementsPerBlock),
                                        static_cast<uint16_t>((M - blockM) / elementsPerBlock), 0};
            DataCopy(srcLocalReal, aGlobalReal[MatAOffset], copyInParams);
            DataCopy(srcLocalImag, aGlobalImag[MatAOffset], copyInParams);
            inQueueSrcReal.EnQue(srcLocalReal);
            inQueueSrcImag.EnQue(srcLocalImag);
        }
        {  // Compute
            srcLocalReal = inQueueSrcReal.DeQue<T>();
            srcLocalImag = inQueueSrcImag.DeQue<T>();
            int t = min(realM, realN);
            for (int k = 0; k < t; ++k)
            {
                int p = choosePivot(k, srcLocalReal, srcLocalImag);
                for (int i = 0; i < realN * blockM; i += blockM)
                {
                    swap((blockM - 1 - k) + i, p + i, srcLocalReal);
                    swap((blockM - 1 - k) + i, p + i, srcLocalImag);
                }
                Divide(k * blockM, k, srcLocalReal, srcLocalImag);
                wGlobal.SetValue(offsetM + k, offsetM + (blockM - 1 - p));
                PipeBarrier<PIPE_ALL>();
                if (k == t - 1)
                {
                    break;
                }
                for (int i = k + 1; i < realN; ++i)
                {
                    eliminate(k * blockM, i * blockM, k, srcLocalReal, srcLocalImag);
                }
            }
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wGlobal[offsetM]);
            Barrier();
        }
        {  // CopyOut
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopyParams coptOutParams{static_cast<uint16_t>(blockN),
                                         static_cast<uint16_t>(blockM / elementsPerBlock), 0,
                                         static_cast<uint16_t>((M - blockM) / elementsPerBlock)};
            DataCopy(aGlobalReal[MatAOffset], srcLocalReal, coptOutParams);
            DataCopy(aGlobalImag[MatAOffset], srcLocalImag, coptOutParams);
            inQueueSrcReal.FreeTensor(srcLocalReal);
            inQueueSrcImag.FreeTensor(srcLocalImag);
        }
    }

   private:
    __aicore__ inline int choosePivot(int k, LocalTensor<T> &colLocalReal, LocalTensor<T> &colLocalImag)
    {
        LocalTensor<T> workLocal = workBuf.Get<T>();
        Mul(workLocal, colLocalReal[blockM * k], colLocalReal[blockM * k], blockM - k);
        PipeBarrier<PIPE_V>();
        MulAddDst(workLocal, colLocalImag[blockM * k], colLocalImag[blockM * k], blockM - k);
        PipeBarrier<PIPE_V>();
        if (blockM != realM)
        {
            Duplicate(workLocal, T(-1), blockM - realM);
            PipeBarrier<PIPE_V>();
        }
        ReduceMax(workLocal, workLocal, workLocal, blockM - k, true);
        PipeBarrier<PIPE_ALL>();
        T maxIndex = workLocal.GetValue(1);
        return *reinterpret_cast<uint32_t *>(&maxIndex);
    }
    __aicore__ inline void swap(int a, int b, LocalTensor<T> &colLocal)
    {
        T tmp = colLocal.GetValue(a);
        PipeBarrier<PIPE_ALL>();
        colLocal.SetValue(a, colLocal.GetValue(b));
        colLocal.SetValue(b, tmp);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Divide(int offset, int k, LocalTensor<T> &srcLocalReal, LocalTensor<T> &srcLocalImag)
    {
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIDVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIDVToS);
        T realScalar = srcLocalReal.GetValue(offset + (blockM - 1 - k));
        T imagScalar = srcLocalImag.GetValue(offset + (blockM - 1 - k));
        T denominator = realScalar * realScalar + imagScalar * imagScalar;
        // Guard against zero pivot (singular matrix): division by zero would
        // produce NaN/Inf and silently corrupt the LU factors (issue #29/#92).
        SOLVER_KERNEL_CHECK(denominator != T(0), denominator = T(1); realScalar = T(0); imagScalar = T(0));
        T realInvScalar = realScalar / denominator;
        T imagInvScalar = imagScalar / denominator;
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T> workLocal = workBuf.Get<T>();

        Muls(workLocal, srcLocalReal[offset], T(1), blockM - k - 1);
        Muls(srcLocalReal[offset], srcLocalReal[offset], realInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(srcLocalReal[offset], srcLocalImag[offset], imagInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Muls(srcLocalImag[offset], srcLocalImag[offset], realInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(srcLocalImag[offset], workLocal, -imagInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void eliminate(int srcOffset, int dstOffset, int k, LocalTensor<T> &srcLocalReal,
                                     LocalTensor<T> &srcLocalImag)
    {
        T realScalar = -srcLocalReal.GetValue(dstOffset + (blockM - 1 - k));
        T imagScalar = srcLocalImag.GetValue(dstOffset + (blockM - 1 - k));
        PipeBarrier<PIPE_ALL>();

        Axpy(srcLocalReal[dstOffset], srcLocalReal[srcOffset], realScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(srcLocalReal[dstOffset], srcLocalImag[srcOffset], imagScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(srcLocalImag[dstOffset], srcLocalReal[srcOffset], -imagScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(srcLocalImag[dstOffset], srcLocalImag[srcOffset], realScalar, blockM - k - 1);
        PipeBarrier<PIPE_ALL>();
    }
};

template <typename T>
class LUCustom3
{
    GlobalTensor<T> aGlobalReal;
    GlobalTensor<T> aGlobalImag;
    GlobalTensor<uint32_t> wGlobal;
    int M, N;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    int blockIdx, blockNum;
    int lineCount;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 1> inQueueSrcReal[2];
    TQue<QuePosition::VECIN, 1> inQueueSrcImag[2];
    TQue<QuePosition::VECOUT, 1> inQueueAugReal;
    TQue<QuePosition::VECOUT, 1> inQueueAugImag;
    LocalTensor<T> colLocalReal[2];
    LocalTensor<T> colLocalImag[2];
    LocalTensor<T> augLocalReal;
    LocalTensor<T> augLocalImag;

   public:
    __aicore__ inline LUCustom3() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc1,
                                TQue<QuePosition::VECIN, 1> inQueueSrc2, TQue<QuePosition::VECOUT, 1> outQueueDst,
                                GlobalTensor<T> &aGlobalReal, GlobalTensor<T> &aGlobalImag,
                                GlobalTensor<uint32_t> &wGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->wGlobal = wGlobal;
        this->blockIdx = GetBlockIdx();
        this->blockNum = 8;
        this->lineCount = blockN / blockNum;

        elementsPerBlock = 32 / sizeof(T);
        this->inQueueSrcReal[0] = inQueueSrc1;
        this->inQueueSrcImag[0] = inQueueSrc2;
        for (int i = 1; i < lineCount; ++i)
        {
            pipe->InitBuffer(this->inQueueSrcReal[i], 1, TILE_LENGTH * sizeof(T));
            pipe->InitBuffer(this->inQueueSrcImag[i], 1, TILE_LENGTH * sizeof(T));
        }
        this->inQueueAugReal = outQueueDst;
        pipe->InitBuffer(this->inQueueAugImag, 1, TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Process(int MatAOffset, int offsetM, int offsetN, int blockM, int realM, int realN)
    {
        this->blockM = blockM;
        this->realM = realM;
        int startLine = blockIdx * lineCount;
        int endLine = startLine + lineCount;
        for (int i = startLine; i < endLine; ++i)
        {
            CopyInColumn(MatAOffset + i * M, i - startLine);
        }
        int t = min(realM, realN);
        int curIdx = 0;
        for (int k = 0; k < t; ++k)
        {
            int p;
            if (k >= startLine && k < endLine)
            {
                p = choosePivot(k, curIdx);
            }
            else
            {
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    wGlobal[offsetM + k]);
                Barrier();
                while (!(~wGlobal.GetValue(offsetM + k)))
                {
                    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        wGlobal[offsetM + k]);
                    Barrier();
                }
                p = blockM - 1 - (wGlobal.GetValue(offsetM + k) - offsetM);
            }
            for (int i = 0; i < lineCount; ++i)
            {
                swap(blockM - 1 - k, p, colLocalReal[i]);
                swap(blockM - 1 - k, p, colLocalImag[i]);
            }
            if (k >= startLine && k < endLine)
            {
                CopyOutColumn(MatAOffset + k * M, curIdx, k);
                int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
                wGlobal.SetValue(offsetM + k, offsetM + (blockM - 1 - p));
                PipeBarrier<PIPE_ALL>();
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    wGlobal[offsetM + k]);
                ++curIdx;
                if (k + 1 == t)
                {
                    break;
                }
            }
            else
            {
                if (k + 1 == t)
                {
                    break;
                }
                CopyInAug(MatAOffset + k * M);
                int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
            }
            for (int i = curIdx; i < lineCount; ++i)
            {
                eliminate(i, k);
            }
            if (k < startLine || k >= endLine)
            {
                inQueueAugReal.FreeTensor(augLocalReal);
                inQueueAugImag.FreeTensor(augLocalImag);
            }
        }
        for (int i = 0; i < lineCount; ++i)
        {
            CopyOutColumnFinal(MatAOffset + (i + startLine) * M, i);
        }
    }

   private:
    __aicore__ inline void CopyInColumn(int offset, int idx)
    {
        colLocalReal[idx] = inQueueSrcReal[idx].AllocTensor<T>();
        colLocalImag[idx] = inQueueSrcImag[idx].AllocTensor<T>();
        DataCopy(colLocalReal[idx], aGlobalReal[offset], blockM);
        DataCopy(colLocalImag[idx], aGlobalImag[offset], blockM);
        inQueueSrcReal[idx].EnQue(colLocalReal[idx]);
        inQueueSrcImag[idx].EnQue(colLocalImag[idx]);
        inQueueSrcReal[idx].DeQue<T>();
        inQueueSrcImag[idx].DeQue<T>();
    }
    __aicore__ inline int choosePivot(int k, int idx)
    {
        LocalTensor<T> workLocal = inQueueAugReal.AllocTensor<T>();
        Mul(workLocal, colLocalReal[idx], colLocalReal[idx], blockM - k);
        PipeBarrier<PIPE_V>();
        MulAddDst(workLocal, colLocalImag[idx], colLocalImag[idx], blockM - k);
        PipeBarrier<PIPE_V>();
        if (blockM != realM)
        {
            Duplicate(workLocal, T(-1), blockM - realM);
            PipeBarrier<PIPE_V>();
        }
        ReduceMax(workLocal, workLocal, workLocal, blockM - k, true);
        PipeBarrier<PIPE_ALL>();
        T maxIndex = workLocal.GetValue(1);
        inQueueAugReal.FreeTensor(workLocal);
        return *reinterpret_cast<uint32_t *>(&maxIndex);
    }
    __aicore__ inline void swap(int a, int b, LocalTensor<T> &colLocal)
    {
        T tmp = colLocal.GetValue(a);
        PipeBarrier<PIPE_ALL>();
        colLocal.SetValue(a, colLocal.GetValue(b));
        colLocal.SetValue(b, tmp);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CopyOutColumn(int offset, int idx, int k)
    {
        T realScalar = colLocalReal[idx].GetValue(blockM - 1 - k);
        T imagScalar = colLocalImag[idx].GetValue(blockM - 1 - k);
        T denominator = realScalar * realScalar + imagScalar * imagScalar;
        // Guard against zero pivot (singular matrix): division by zero would
        // produce NaN/Inf and silently corrupt the LU factors (issue #29/#92).
        SOLVER_KERNEL_CHECK(denominator != T(0), denominator = T(1); realScalar = T(0); imagScalar = T(0));
        T realInvScalar = realScalar / denominator;
        T imagInvScalar = imagScalar / denominator;
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T> workLocal = inQueueAugReal.AllocTensor<T>();

        Muls(workLocal, colLocalReal[idx], T(1), blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Muls(colLocalReal[idx], colLocalReal[idx], realInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(colLocalReal[idx], colLocalImag[idx], imagInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Muls(colLocalImag[idx], colLocalImag[idx], realInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(colLocalImag[idx], workLocal, -imagInvScalar, blockM - k - 1);
        PipeBarrier<PIPE_ALL>();

        inQueueAugReal.FreeTensor(workLocal);

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(aGlobalReal[offset], colLocalReal[idx], blockM);
        DataCopy(aGlobalImag[offset], colLocalImag[idx], blockM);
        augLocalReal = colLocalReal[idx];
        augLocalImag = colLocalImag[idx];
    }
    __aicore__ inline void CopyInAug(int offset)
    {
        augLocalReal = inQueueAugReal.AllocTensor<T>();
        augLocalImag = inQueueAugImag.AllocTensor<T>();
        DataCopy(augLocalReal, aGlobalReal[offset], blockM);
        DataCopy(augLocalImag, aGlobalImag[offset], blockM);
        inQueueAugReal.EnQue(augLocalReal);
        inQueueAugImag.EnQue(augLocalImag);
        inQueueAugReal.DeQue<T>();
        inQueueAugImag.DeQue<T>();
    }
    __aicore__ inline void eliminate(int idx, int k)
    {
        T realScalar = -colLocalReal[idx].GetValue(blockM - 1 - k);
        T imagScalar = colLocalImag[idx].GetValue(blockM - 1 - k);

        PipeBarrier<PIPE_ALL>();

        Axpy(colLocalReal[idx], augLocalReal, realScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(colLocalReal[idx], augLocalImag, imagScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(colLocalImag[idx], augLocalReal, -imagScalar, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Axpy(colLocalImag[idx], augLocalImag, realScalar, blockM - k - 1);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CopyOutColumnFinal(int offset, int idx)
    {
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(aGlobalReal[offset], colLocalReal[idx], blockM);
        inQueueSrcReal[idx].FreeTensor(colLocalReal[idx]);

        DataCopy(aGlobalImag[offset], colLocalImag[idx], blockM);
        inQueueSrcImag[idx].FreeTensor(colLocalImag[idx]);
    }
};

template <typename T>
class TransPose
{
    GlobalTensor<T> aGlobalReal, aGlobalImag;
    GlobalTensor<T> workGlobalReal, workGlobalImag;
    GlobalTensor<uint32_t> gather1, gather2;
    TQue<QuePosition::VECIN, 1> inQueueSrc;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    int M, N;
    int tileM, tileN;
    int elementsPerBlock;

   public:
    __aicore__ inline TransPose() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc1,
                                TQue<QuePosition::VECIN, 1> inQueueSrc2, TQue<QuePosition::VECOUT, 1> outQueueDst,
                                GlobalTensor<T> aGlobalReal, GlobalTensor<T> aGlobalImag,
                                GlobalTensor<T> workGlobalReal, GlobalTensor<T> workGlobalImag,
                                GlobalTensor<uint32_t> gather1, GlobalTensor<uint32_t> gather2, int M, int N, int tileM,
                                int tileN)
    {
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->workGlobalReal = workGlobalReal;
        this->workGlobalImag = workGlobalImag;
        this->gather1 = gather1;
        this->gather2 = gather2;
        this->M = M;
        this->N = N;
        this->tileM = tileM;
        this->tileN = tileN;
        this->elementsPerBlock = 32 / sizeof(T);

        this->inQueueSrc = inQueueSrc1;
        this->gatherQueue = inQueueSrc2;
        this->outQueueDst = outQueueDst;
    }
    __aicore__ inline void Process1(int offsetSrc, int offsetDst, int blockM)
    {
        Process1Kernel(offsetSrc, offsetDst, blockM, aGlobalReal, workGlobalReal);
        Process1Kernel(offsetSrc, offsetDst, blockM, aGlobalImag, workGlobalImag);
    }
    __aicore__ void Process1Kernel(int offsetSrc, int offsetDst, int blockM, GlobalTensor<T> aGlobal,
                                   GlobalTensor<T> workGlobal)
    {
        LocalTensor<uint32_t> gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, gather1, tileM * tileN);
        gatherQueue.EnQue(gatherLocal);
        gatherQueue.DeQue<T>();

        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopyParams copyInParams{static_cast<uint16_t>(blockM), static_cast<uint16_t>(tileN / elementsPerBlock),
                                    static_cast<uint16_t>((N - tileN) / elementsPerBlock), 0};
        DataCopy(srcLocal, aGlobal[offsetSrc], copyInParams);
        inQueueSrc.EnQue(srcLocal);

        srcLocal = inQueueSrc.DeQue<T>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, tileM * tileN);
        inQueueSrc.FreeTensor(srcLocal);
        outQueueDst.EnQue(dstLocal);
        gatherQueue.FreeTensor(gatherLocal);

        dstLocal = outQueueDst.DeQue<T>();
        DataCopyParams copyOutParams{static_cast<uint16_t>(tileN), static_cast<uint16_t>(tileM / elementsPerBlock), 0,
                                     static_cast<uint16_t>((M - tileM) / elementsPerBlock)};
        DataCopy(workGlobal[offsetDst], dstLocal, copyOutParams);
        outQueueDst.FreeTensor(dstLocal);
    }
    __aicore__ inline void Process2(int offsetSrc, int offsetDst, int blockM)
    {
        Process2Kernel(offsetSrc, offsetDst, blockM, aGlobalReal, workGlobalReal);
        Process2Kernel(offsetSrc, offsetDst, blockM, aGlobalImag, workGlobalImag);
    }
    __aicore__ void Process2Kernel(int offsetSrc, int offsetDst, int blockM, GlobalTensor<T> aGlobal,
                                   GlobalTensor<T> workGlobal)
    {
        LocalTensor<uint32_t> gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, gather2, tileM * tileN);
        gatherQueue.EnQue(gatherLocal);
        gatherQueue.DeQue<T>();

        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopyParams copyInParams{static_cast<uint16_t>(tileN), static_cast<uint16_t>(tileM / elementsPerBlock),
                                    static_cast<uint16_t>((M - tileM) / elementsPerBlock), 0};
        DataCopy(srcLocal, workGlobal[offsetDst], copyInParams);
        inQueueSrc.EnQue(srcLocal);

        srcLocal = inQueueSrc.DeQue<T>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, tileM * tileN);
        inQueueSrc.FreeTensor(srcLocal);
        outQueueDst.EnQue(dstLocal);
        gatherQueue.FreeTensor(gatherLocal);

        dstLocal = outQueueDst.DeQue<T>();
        DataCopyParams copyOutParams{static_cast<uint16_t>(blockM), static_cast<uint16_t>(tileN / elementsPerBlock), 0,
                                     static_cast<uint16_t>((N - tileN) / elementsPerBlock)};
        DataCopy(aGlobal[offsetSrc], dstLocal, copyOutParams);
        outQueueDst.FreeTensor(dstLocal);
    }
};

template <typename T>
class SwapRows
{
    static constexpr int TILE_LENGTH = 8192;
    GlobalTensor<T> aGlobalReal, aGlobalImag, eyeGlobal;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind2;
    int elementsPerBlock;

   public:
    __aicore__ inline SwapRows() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> aGlobalReal,
                                GlobalTensor<T> aGlobalImag, GlobalTensor<T> eyeGlobal)
    {
        pipe->InitBuffer(queBind1, 1, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(queBind2, 1, TILE_LENGTH * sizeof(T));
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->eyeGlobal = eyeGlobal;
        this->elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int N)
    {
        if (offsetA == offsetB || N <= 0)
        {
            return;
        }
        ProcessKernel(offsetA, offsetB, N, aGlobalReal);
        ProcessKernel(offsetA, offsetB, N, aGlobalImag);
    }
    __aicore__ inline void ProcessKernel(int offsetA, int offsetB, int N, GlobalTensor<T> aGlobal)
    {
        LocalTensor<T> bindLocal1 = queBind1.AllocTensor<T>();
        LocalTensor<T> bindLocal2 = queBind2.AllocTensor<T>();
        DataCopy(bindLocal1, aGlobal[offsetA], N);
        DataCopy(bindLocal2, aGlobal[offsetB], N);
        queBind1.EnQue(bindLocal1);
        queBind2.EnQue(bindLocal2);
        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        queBind1.DeQue<T>();
        queBind2.DeQue<T>();
        DataCopy(aGlobal[offsetB], bindLocal1, N);
        DataCopy(aGlobal[offsetA], bindLocal2, N);
        queBind1.FreeTensor(bindLocal1);
        queBind2.FreeTensor(bindLocal2);

        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
    __aicore__ inline void ProcessEye(int offsetA, int offsetB, int N)
    {
        if (offsetA == offsetB || N <= 0)
        {
            return;
        }
        LocalTensor<T> bindLocal1 = queBind1.AllocTensor<T>();
        LocalTensor<T> bindLocal2 = queBind2.AllocTensor<T>();
        DataCopy(bindLocal1, eyeGlobal[offsetA], N);
        DataCopy(bindLocal2, eyeGlobal[offsetB], N);
        queBind1.EnQue(bindLocal1);
        queBind2.EnQue(bindLocal2);
        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        bindLocal1 = queBind1.DeQue<T>();
        bindLocal2 = queBind2.DeQue<T>();
        DataCopy(eyeGlobal[offsetB], bindLocal1, N);
        DataCopy(eyeGlobal[offsetA], bindLocal2, N);
        queBind1.FreeTensor(bindLocal1);
        queBind2.FreeTensor(bindLocal2);

        int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
};

#endif  // C64_GETF2_HPP_
