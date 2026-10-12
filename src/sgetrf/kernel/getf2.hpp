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

#ifndef SGETRF_GETF2_HPP_
#define SGETRF_GETF2_HPP_

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
                for (int i = k + 1; i < realN; ++i) elimimate(k * blockM, i * blockM, k, srcLocal);
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
        const T pivot = srcLocal.GetValue(offset + (blockM - 1 - k));
        LocalTensor<T> divisor = workBuf.Get<T>();
        Duplicate(divisor, pivot, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Div(srcLocal[offset], srcLocal[offset], divisor, blockM - k - 1);
    }
    __aicore__ inline void elimimate(int srcOffset, int dstOffset, int k, LocalTensor<T> &srcLocal)
    {
        T tmpScalar = -srcLocal.GetValue(dstOffset + (blockM - 1 - k));
        Axpy(srcLocal[dstOffset], srcLocal[srcOffset], tmpScalar, blockM - k - 1);
    }
};

template <typename T>
class LUCustom2
{  // for M * N <= 8192
    GlobalTensor<T> aGlobal;
    GlobalTensor<uint32_t> wGlobal;
    int M, N;
    int blockM, blockN;
    int elementsPerBlock;
    int lineCount;
    int singleCoreN;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 1> inQueueSrc[2];
    TQue<QuePosition::VECOUT, 1> outQueueDst[2];
    TBuf<TPosition::VECCALC> workBuf;

   public:
    __aicore__ inline LUCustom2() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> &aGlobal,
                                GlobalTensor<uint32_t> &wGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;
        this->singleCoreN = blockN / 2;

        this->aGlobal = aGlobal;
        this->wGlobal = wGlobal;

        elementsPerBlock = 32 / sizeof(T);

        for (int i = 0; i < 2; ++i)
        {
            pipe->InitBuffer(inQueueSrc[i], 1, TILE_LENGTH * sizeof(T));
            pipe->InitBuffer(outQueueDst[i], 1, TILE_LENGTH * sizeof(T));
        }
        pipe->InitBuffer(workBuf, TILE_LENGTH * sizeof(T));
    }
    __aicore__ inline void Process(int offsetM, int offsetN, int blockM)
    {
        SOLVER_KERNEL_CHECK(blockM * blockN <= 8192 * 2, return);
        this->blockM = blockM;
        int offset = offsetM + offsetN * M;
        {  // CopyIn
            LocalTensor<T> srcLocal[2]{inQueueSrc[0].AllocTensor<T>(), inQueueSrc[1].AllocTensor<T>()};
            DataCopyParams copyInParams{static_cast<uint16_t>(singleCoreN),
                                        static_cast<uint16_t>(blockM / elementsPerBlock),
                                        static_cast<uint16_t>((M - blockM) / elementsPerBlock), 0};
            DataCopy(srcLocal[0], aGlobal[offset], copyInParams);
            DataCopy(srcLocal[1], aGlobal[offset + M * singleCoreN], copyInParams);
            inQueueSrc[0].EnQue(srcLocal[0]);
            inQueueSrc[1].EnQue(srcLocal[1]);
        }
        {  // Compute
            LocalTensor<T> srcLocal[2]{inQueueSrc[0].DeQue<T>(), inQueueSrc[1].DeQue<T>()};
            LocalTensor<T> dstLocal[2]{outQueueDst[0].AllocTensor<T>(), outQueueDst[1].AllocTensor<T>()};
            int t = min(blockM, blockN);
            for (int k = 0; k < t; ++k)
            {
                int p = choosePivot(k, srcLocal[k >= singleCoreN]);
                for (int i = k; i < blockN; ++i)
                    swap(k + (i % singleCoreN) * blockM, p + (i % singleCoreN) * blockM, srcLocal[i >= singleCoreN]);
                Divide((k % singleCoreN) * blockM, k, srcLocal[k >= singleCoreN], dstLocal[k >= singleCoreN]);
                wGlobal.SetValue(offsetM + k, offsetM + p);
                if (k == t - 1) break;
                for (int i = k + 1; i < blockN; ++i)
                    elimimate((k % singleCoreN) * blockM, (i % singleCoreN) * blockM, k, srcLocal[i >= singleCoreN],
                              dstLocal[k >= singleCoreN]);
            }
            for (int i = 0; i < 2; ++i)
            {
                inQueueSrc[i].FreeTensor(srcLocal[i]);
                outQueueDst[i].EnQue(dstLocal[i]);
            }
        }
        {  // CopyOut
            LocalTensor<T> dstLocal[2]{outQueueDst[0].DeQue<T>(), outQueueDst[1].DeQue<T>()};
            DataCopyParams coptOutParams{static_cast<uint16_t>(singleCoreN),
                                         static_cast<uint16_t>(blockM / elementsPerBlock), 0,
                                         static_cast<uint16_t>((M - blockM) / elementsPerBlock)};
            DataCopy(aGlobal[offset], dstLocal[0], coptOutParams);
            DataCopy(aGlobal[offset + M * singleCoreN], dstLocal[1], coptOutParams);
            outQueueDst[0].FreeTensor(dstLocal[0]);
            outQueueDst[1].FreeTensor(dstLocal[1]);
        }
    }

   private:
    __aicore__ inline int choosePivot(int k, LocalTensor<T> &colLocal)
    {
        LocalTensor<T> workLocal = workBuf.Get<T>();
        Abs(workLocal, colLocal[blockM * k], blockM);
        Duplicate(workLocal, T(-1), k);
        ReduceMax(workLocal, workLocal, workLocal, blockM, true);
        T maxIndex = workLocal.GetValue(1);
        return __builtin_bit_cast(uint32_t, maxIndex);
    }
    __aicore__ inline void swap(int a, int b, LocalTensor<T> &colLocal)
    {
        T tmp = colLocal.GetValue(a);
        colLocal.SetValue(a, colLocal.GetValue(b));
        colLocal.SetValue(b, tmp);
    }
    __aicore__ inline void Divide(int offset, int k, LocalTensor<T> &srcLocal, LocalTensor<T> &dstLocal)
    {
        T tmpScalar = T(1) / srcLocal.GetValue(offset + k);
        Muls(dstLocal[offset], srcLocal[offset], tmpScalar, blockM);
        Copy(dstLocal[offset], srcLocal[offset], k + 1, 1, {1, 1, 8, 8});
    }
    __aicore__ inline void elimimate(int srcOffset, int dstOffset, int k, LocalTensor<T> &srcLocal,
                                     LocalTensor<T> &dstLocal)
    {
        T tmpScalar = -srcLocal.GetValue(dstOffset + k);
        int length = blockM - k - 1;
        int x = (k + elementsPerBlock) / elementsPerBlock * elementsPerBlock;
        Axpy(srcLocal[dstOffset + x], dstLocal[srcOffset + x], tmpScalar, max(0, length - x + k + 1));
        if (x != k + 1)
        {
            uint64_t mask[2] = {0xffull ^ (1 << (elementsPerBlock - x + k + 1)) - 1, 0};
            Axpy(srcLocal[dstOffset + x - elementsPerBlock], dstLocal[srcOffset + x - elementsPerBlock], tmpScalar,
                 mask, 1, {1, 1, 8, 8});
        }
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
    TQue<QuePosition::VECIN, 1> inQueueSrc[2];
    TQue<QuePosition::VECIN, 1> inQueueAug;
    TBuf<TPosition::VECCALC> workBuf;
    LocalTensor<T> colLocal[2];
    LocalTensor<T> augLocal;

   public:
    __aicore__ inline LUCustom3() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc,
                                GlobalTensor<T> &aGlobal, GlobalTensor<uint32_t> &wGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aGlobal = aGlobal;
        this->wGlobal = wGlobal;
        this->blockIdx = GetBlockIdx();

        this->blockNum = 8;
        this->lineCount = blockN / blockNum;
        SOLVER_KERNEL_CHECK(blockNum > 0 && blockN % blockNum == 0, lineCount = 0; return);
        SOLVER_KERNEL_CHECK(lineCount <= 2, lineCount = 2);

        elementsPerBlock = 32 / sizeof(T);
        for (int i = 0; i < lineCount; ++i)
        {
            pipe->InitBuffer(this->inQueueSrc[i], 1, TILE_LENGTH * sizeof(T));
        }
        this->inQueueAug = inQueueSrc;
        pipe->InitBuffer(workBuf, TILE_LENGTH * sizeof(T));
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
                while (!(~wGlobal.GetValue(offsetM + k)))
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
                elimimate(i, k);
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
        LocalTensor<T> workLocal = workBuf.Get<T>();
        Abs(workLocal, colLocal[idx], blockM - k);
        if (blockM != realM) Duplicate(workLocal, T(-1), blockM - realM);
        ReduceMax(workLocal, workLocal, workLocal, blockM - k, true);
        T maxIndex = workLocal.GetValue(1);
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
        const T pivot = colLocal[idx].GetValue(blockM - 1 - k);
        LocalTensor<T> divisor = workBuf.Get<T>();
        Duplicate(divisor, pivot, blockM - k - 1);
        PipeBarrier<PIPE_V>();
        Div(colLocal[idx], colLocal[idx], divisor, blockM - k - 1);
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(aGlobal[offset], colLocal[idx], blockM);
        // 跨核竞态修复（issue #94）：本函数返回后 Process 会经 S-pipe 发布 pivot
        // 信号（wGlobal.SetValue + dcci），reader 核看到信号后即 MTE2 读取 pivot
        // 列。需保证两件事：① S-pipe 的 SetValue 排序在本列 MTE3 store 之后
        // （MTE3_S fence）；② 本列 store 对 reader 核 MTE2 全局可见（逐 cacheline
        // flush）。缺失时 reader 可能读到旧的 col-k，LU 结果间歇性静默错误。
        int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
        constexpr int32_t CACHE_LINE_BYTES = 64;  // AI Core cacheline 64B（issue #94）
        for (int32_t line = 0; line < blockM * static_cast<int32_t>(sizeof(T)); line += CACHE_LINE_BYTES)
        {
            DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                aGlobal[offset + line / static_cast<int32_t>(sizeof(T))]);
        }
        augLocal = colLocal[idx];
    }
    __aicore__ inline void CopyInAug(int offset)
    {
        augLocal = inQueueAug.AllocTensor<T>();
        DataCopy(augLocal, aGlobal[offset], blockM);
        inQueueAug.EnQue(augLocal);
        inQueueAug.DeQue<T>();
    }
    __aicore__ inline void elimimate(int idx, int k)
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
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 1> inQueueSrc,
                                TQue<QuePosition::VECOUT, 1> outQueueDst, GlobalTensor<T> aGlobal,
                                GlobalTensor<T> workGlobal, GlobalTensor<uint32_t> gather1,
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

        this->inQueueSrc = inQueueSrc;
        this->outQueueDst = outQueueDst;

        pipe->InitBuffer(gatherQueue, 1, tileM * tileN * sizeof(uint32_t));
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
    GlobalTensor<T> aGlobal;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind2;
    int elementsPerBlock;

   public:
    __aicore__ inline SwapRows() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> aGlobal)
    {
        pipe->InitBuffer(queBind1, 1, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(queBind2, 1, TILE_LENGTH * sizeof(T));
        this->aGlobal = aGlobal;
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
};

#endif
