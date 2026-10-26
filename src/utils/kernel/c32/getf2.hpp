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

#ifndef C32_GETF2_HPP_
#define C32_GETF2_HPP_

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

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

using namespace AscendC;
using namespace matmul;

template <typename T>
class LUCustom1
{  // for M * N <= 8192
    GlobalTensor<T> aGlobal;
    GlobalTensor<uint32_t> wGlobal;
    int M, N;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    int lineCount;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 1> inQueueSrc;
    TBuf<TPosition::VECCALC> workBuf;

   public:
    __aicore__ inline LUCustom1() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc,
                                GlobalTensor<T> &aGlobal, GlobalTensor<uint32_t> &wGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aGlobal = aGlobal;
        this->wGlobal = wGlobal;

        elementsPerBlock = 32 / sizeof(T);

        this->inQueueSrc = inQueueSrc;
        pipe->InitBuffer(workBuf, TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Process(int MatAOffset, int offsetM, int offsetN, int blockM, int realM, int realN)
    {
        SOLVER_KERNEL_CHECK(blockM * blockN <= 8192, return);
        SOLVER_KERNEL_CHECK(blockM >= blockN, return);
        this->blockM = blockM;
        this->realM = realM;
        LocalTensor<T> srcLocal;
        {  // CopyIn
            srcLocal = inQueueSrc.AllocTensor<T>();
            DataCopyParams copyInParams{static_cast<uint16_t>(realN), static_cast<uint16_t>(blockM / elementsPerBlock),
                                        static_cast<uint16_t>((M - blockM) / elementsPerBlock), 0};
            DataCopy(srcLocal, aGlobal[MatAOffset], copyInParams);
            inQueueSrc.EnQue(srcLocal);
        }
        {  // Compute
            srcLocal = inQueueSrc.DeQue<T>();
            int t = min(realM, realN);
            for (int k = 0; k < t; ++k)
            {
                int p = choosePivot(k, srcLocal);

                for (int i = 0; i < realN * blockM; i += blockM) swap((blockM - 1 - k) + i, p + i, srcLocal);
                Divide(k * blockM, k, srcLocal);
                wGlobal.SetValue(offsetM + k, offsetM + (blockM - 1 - p));
                if (k == t - 1) break;
                for (int i = k + 1; i < realN; ++i) eliminate(k * blockM, i * blockM, k, srcLocal);
            }
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wGlobal[offsetM]);
            Barrier();
        }
        {  // CopyOut
            int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopyParams coptOutParams{static_cast<uint16_t>(blockN),
                                         static_cast<uint16_t>(blockM / elementsPerBlock), 0,
                                         static_cast<uint16_t>((M - blockM) / elementsPerBlock)};
            DataCopy(aGlobal[MatAOffset], srcLocal, coptOutParams);
            inQueueSrc.FreeTensor(srcLocal);
        }
    }

   private:
    __aicore__ inline int choosePivot(int k, LocalTensor<T> &colLocal)
    {
        LocalTensor<T> workLocal = workBuf.Get<T>();
        Abs(workLocal, colLocal[blockM * k], blockM - k);
        if (blockM != realM) Duplicate(workLocal, T(-1), blockM - realM);
        ReduceMax(workLocal, workLocal, workLocal, blockM - k, true);
        // ReduceMax(V) -> GetValue(S) 补同步屏障（issue #258，对齐本文件
        // LUCustom3 与 c64 两处实现按 #93 修复的写法）：V/S 管线时序紧张时
        // 无屏障读结果可能取到旧值，选主元错误导致分解结果静默错误
        PipeBarrier<PIPE_ALL>();
        T maxIndex = workLocal.GetValue(1);
        return __builtin_bit_cast(uint32_t, maxIndex);
    }
    __aicore__ inline void swap(int a, int b, LocalTensor<T> &colLocal)
    {
        T tmp = colLocal.GetValue(a);
        colLocal.SetValue(a, colLocal.GetValue(b));
        colLocal.SetValue(b, tmp);
    }
    __aicore__ inline void Divide(int offset, int k, LocalTensor<T> &srcLocal)
    {
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIDVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIDVToS);
        T tmpScalar = T(1) / srcLocal.GetValue(offset + (blockM - 1 - k));
        Muls(srcLocal[offset], srcLocal[offset], tmpScalar, blockM - k - 1);
    }
    __aicore__ inline void eliminate(int srcOffset, int dstOffset, int k, LocalTensor<T> &srcLocal)
    {
        T tmpScalar = -srcLocal.GetValue(dstOffset + (blockM - 1 - k));
        Axpy(srcLocal[dstOffset], srcLocal[srcOffset], tmpScalar, blockM - k - 1);
    }
};

template <typename T>
class LUCustom3
{
    GlobalTensor<T> aGlobal;
    GlobalTensor<uint32_t> wGlobal;
    int M, N;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    int blockIdx, blockNum;
    int lineCount;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 1> inQueueSrc[4];
    TQue<QuePosition::VECIN, 1> inQueueAug;
    TQue<QuePosition::VECOUT, 1> workQueue;
    LocalTensor<T> colLocal[4];
    LocalTensor<T> augLocal;

   public:
    __aicore__ inline LUCustom3() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc1,
                                TQue<QuePosition::VECIN, 1> inQueueSrc2, TQue<QuePosition::VECOUT, 1> outQueueDst,
                                GlobalTensor<T> &aGlobal, GlobalTensor<uint32_t> &wGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aGlobal = aGlobal;
        this->wGlobal = wGlobal;
        this->blockIdx = GetBlockIdx();

        this->blockNum = 4;
        this->lineCount = blockN / blockNum;

        elementsPerBlock = 32 / sizeof(T);
        this->inQueueSrc[0] = inQueueSrc1;
        for (int i = 1; i < lineCount; ++i)
        {
            pipe->InitBuffer(this->inQueueSrc[i], 1, TILE_LENGTH * sizeof(T));
        }
        this->inQueueAug = inQueueSrc2;
        this->workQueue = outQueueDst;
    }
    __aicore__ inline void Process(int MatAOffset, int offsetM, int offsetN, int blockM, int realM, int realN)
    {
        this->blockM = blockM;
        this->realM = realM;
        int startLine = blockIdx * lineCount;
        int endLine = startLine + lineCount;
        for (int i = startLine; i < endLine; ++i) CopyInColumn(MatAOffset + i * M, i - startLine);

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
                while (wGlobal.GetValue(offsetM + k) == -1)
                {
                    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        wGlobal[offsetM + k]);
                    Barrier();
                }
                p = blockM - 1 - (wGlobal.GetValue(offsetM + k) - offsetM);
            }
            int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIDSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIDSToV);
            for (int i = 0; i < lineCount; ++i) swap(blockM - 1 - k, p, colLocal[i]);
            if (k >= startLine && k < endLine)
            {
                CopyOutColumn(MatAOffset + k * M, curIdx, k);
                int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
                wGlobal.SetValue(offsetM + k, offsetM + (blockM - 1 - p));
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    wGlobal[offsetM + k]);
                ++curIdx;
                if (k + 1 == t) break;
            }
            else
            {
                if (k + 1 == t) break;
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
                inQueueAug.FreeTensor(augLocal);
            }
        }

        for (int i = 0; i < lineCount; ++i) CopyOutColumnFinal(MatAOffset + (i + startLine) * M, i);
    }

   private:
    __aicore__ inline void CopyInColumn(int offset, int idx)
    {
        colLocal[idx] = inQueueSrc[idx].AllocTensor<T>();
        DataCopy(colLocal[idx], aGlobal[offset], blockM);
        inQueueSrc[idx].EnQue(colLocal[idx]);
        inQueueSrc[idx].DeQue<T>();
    }
    __aicore__ inline int choosePivot(int k, int idx)
    {
        LocalTensor<T> workLocal = workQueue.AllocTensor<T>();
        Abs(workLocal, colLocal[idx], blockM - k);
        if (blockM != realM) Duplicate(workLocal, T(-1), blockM - realM);
        ReduceMax(workLocal, workLocal, workLocal, blockM - k, true);
        // 先读结果再归还 tensor：FreeTensor 后句柄失效，读取已释放内存构成
        // use-after-free；并按 c64 版实现补 ReduceMax 后的同步屏障（issue #93）
        PipeBarrier<PIPE_ALL>();
        T maxIndex = workLocal.GetValue(1);
        workQueue.FreeTensor(workLocal);
        return __builtin_bit_cast(uint32_t, maxIndex);
    }
    __aicore__ inline void swap(int a, int b, LocalTensor<T> &colLocal)
    {
        T tmp = colLocal.GetValue(a);
        colLocal.SetValue(a, colLocal.GetValue(b));
        colLocal.SetValue(b, tmp);
    }
    __aicore__ inline void CopyOutColumn(int offset, int idx, int k)
    {
        T tmpScalar = T(1) / colLocal[idx].GetValue(blockM - 1 - k);
        Muls(colLocal[idx], colLocal[idx], tmpScalar, blockM - k - 1);
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(aGlobal[offset], colLocal[idx], blockM);
        augLocal = colLocal[idx];
    }
    __aicore__ inline void CopyInAug(int offset)
    {
        augLocal = inQueueAug.AllocTensor<T>();
        DataCopy(augLocal, aGlobal[offset], blockM);
        inQueueAug.EnQue(augLocal);
        inQueueAug.DeQue<T>();
    }
    __aicore__ inline void eliminate(int idx, int k)
    {
        T tmpScalar = -colLocal[idx].GetValue(blockM - 1 - k);
        Axpy(colLocal[idx], augLocal, tmpScalar, blockM - k - 1);
    }
    __aicore__ inline void CopyOutColumnFinal(int offset, int idx)
    {
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(aGlobal[offset], colLocal[idx], blockM);
        inQueueSrc[idx].FreeTensor(colLocal[idx]);
    }
};

template <typename T>
class TransPose
{
    GlobalTensor<T> aGlobal;
    GlobalTensor<T> workGlobal;
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
                                GlobalTensor<T> aGlobal, GlobalTensor<T> workGlobal, GlobalTensor<uint32_t> gather1,
                                GlobalTensor<uint32_t> gather2, int M, int N, int tileM, int tileN)
    {
        this->aGlobal = aGlobal;
        this->workGlobal = workGlobal;
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
    GlobalTensor<T> aGlobal, eyeGlobal;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind2;
    int elementsPerBlock;

   public:
    __aicore__ inline SwapRows() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> aGlobal,
                                GlobalTensor<T> eyeGlobal)
    {
        pipe->InitBuffer(queBind1, 1, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(queBind2, 1, TILE_LENGTH * sizeof(T));
        this->aGlobal = aGlobal;
        this->eyeGlobal = eyeGlobal;
        this->elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int N)
    {
        if (offsetA == offsetB || N <= 0) return;
        LocalTensor<T> bindLocal1 = queBind1.AllocTensor<T>();
        LocalTensor<T> bindLocal2 = queBind2.AllocTensor<T>();
        DataCopy(bindLocal1, aGlobal[offsetA], N);
        DataCopy(bindLocal2, aGlobal[offsetB], N);
        queBind1.EnQue(bindLocal1);
        queBind2.EnQue(bindLocal2);
        int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        bindLocal1 = queBind1.DeQue<T>();
        bindLocal2 = queBind2.DeQue<T>();
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
        if (offsetA == offsetB || N <= 0) return;
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

#endif  // C32_GETF2_HPP_
