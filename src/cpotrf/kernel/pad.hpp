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
 * \file pad.hpp
 * \brief
 */

#ifndef CPOTRF_KERNEL_PAD_HPP
#define CPOTRF_KERNEL_PAD_HPP

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

enum class NegateImag
{
    NOTNEED,
    NEED
};

template <typename T>
class SplitRealImagWithTrans
{
    static const int TILE_LENGTH = 4096;
    static const int m = 64;
    static const int n = 64;
    GlobalTensor<T> aOrgGlobal;
    GlobalTensor<T> aGlobalReal, aGlobalImag;
    GlobalTensor<uint32_t> aGather;
    int orgM, orgN, lda, padStride;
    int strideBatchSrc, strideBatchDst;
    int elementsPerBlock;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueueReal, outQueueImag;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    LocalTensor<uint32_t> gatherLocal;

   public:
    __aicore__ inline SplitRealImagWithTrans() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<uint32_t> aGather)
    {
        pipe->InitBuffer(inQueue, 2, 2 * TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueReal, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueImag, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(gatherQueue, 1, 2 * TILE_LENGTH * sizeof(uint32_t));
        elementsPerBlock = 32 / sizeof(T);
        this->aGather = aGather;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                     GlobalTensor<T> aGlobalImag, int orgM, int orgN, int lda, int padStride,
                                     int strideBatchSrc = 0, int strideBatchDst = 0)
    {
        this->aOrgGlobal = aOrgGlobal;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->orgM = orgM;
        this->orgN = orgN;
        this->lda = lda;
        this->padStride = padStride;
        if (strideBatchSrc)
            this->strideBatchSrc = strideBatchSrc;
        else
            this->strideBatchSrc = orgM * orgN * 2;
        if (strideBatchDst)
            this->strideBatchDst = strideBatchDst;
        else
            this->strideBatchDst = (orgN + 127) / 128 * 128 * padStride;
    }
    __aicore__ inline void SplitRealImagWork(int p, int id, int batch)
    {
        // p 是参与本阶段的核数（blockNum << 1 >= 2），退化入参直接返回，避免除零
        if (p == 0)
        {
            return;
        }
        int cntN = (orgN + n - 1) / n, cntM = (orgM + m - 1) / m;
        int totalNum = batch * cntN * cntM;
        int k = totalNum / p, r = totalNum % p;
        int startIdx = k * id + min(id, r);
        int calcNum = (k + (id < r));
        if (calcNum)
        {
            LoadGather();
            for (int i = startIdx; i < startIdx + calcNum; ++i)
            {
                int batchIdx = i / cntN / cntM;
                int idM = (i - batchIdx * cntN * cntM) / cntN, idN = i % cntN;
                Process(idM, idN, batchIdx);
            }
            gatherQueue.FreeTensor(gatherLocal);
        }
    }
    __aicore__ inline void Process(int idM, int idN, int batch_idx)
    {
        int realM = min(m, orgM - idM * m);
        int realN = min(n, orgN - idN * n);
        CopyIn(batch_idx * strideBatchSrc + 2 * (idM * m * lda + idN * n), realM, realN);
        Compute(realM, realN);
        CopyOut(batch_idx * strideBatchDst + idN * n * padStride + idM * m, realN, realM);
    }
    __aicore__ inline void LoadGather()
    {
        gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, 2 * m * n);
        gatherQueue.EnQue(gatherLocal);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }
    __aicore__ inline void CopyIn(int offset, int rows, int cols)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        int copyDB = (cols + elementsPerBlock - 1) / elementsPerBlock;
        DataCopyExtParams intriParams{static_cast<uint16_t>(rows), static_cast<uint16_t>(cols * 2 * sizeof(T)),
                                      static_cast<uint16_t>((lda - cols) * 2 * sizeof(T)),
                                      static_cast<uint16_t>((n - cols) * 2 / elementsPerBlock), 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, aOrgGlobal[offset], intriParams, padParams);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int realM, int realN)
    {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        Duplicate<T>(srcLocal[realM * n * 2], 0, (m - realM) * n * 2);

        LocalTensor<T> dstLocalReal = outQueueReal.AllocTensor<T>();
        Gather(dstLocalReal, srcLocal, gatherLocal, 0, realN * m);
        outQueueReal.EnQue(dstLocalReal);

        LocalTensor<T> dstLocalImag = outQueueImag.AllocTensor<T>();
        Gather(dstLocalImag, srcLocal, gatherLocal, sizeof(T), realN * m);
        outQueueImag.EnQue(dstLocalImag);

        inQueue.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut(int offset, int rows, int cols)
    {
        DataCopyParams copyOutParams{static_cast<uint16_t>(rows), static_cast<uint16_t>(m / elementsPerBlock),
                                     static_cast<uint16_t>(0),
                                     static_cast<uint16_t>((padStride - m) / elementsPerBlock)};
        LocalTensor<T> dstLocalReal = outQueueReal.DeQue<T>();
        DataCopy(aGlobalReal[offset], dstLocalReal, copyOutParams);
        outQueueReal.FreeTensor(dstLocalReal);

        LocalTensor<T> dstLocalImag = outQueueImag.DeQue<T>();
        DataCopy(aGlobalImag[offset], dstLocalImag, copyOutParams);
        outQueueImag.FreeTensor(dstLocalImag);
    }
};

template <typename T, NegateImag neg = NegateImag::NOTNEED>
class SplitRealImag
{
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> srcGlobal, dstGlobalReal, dstGlobalImag, dstGlobalImagNeg;
    int N;
    int srcN;
    int dstN;
    int padN;
    int alignedN;
    int linesPerIter;
    int elementsPerBlock;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueueReal, outQueueImag;
    LocalTensor<T> realLocal, imagLocal, negimagLocal;

   public:
    __aicore__ inline SplitRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe)
    {
        pipe->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueReal, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueImag, 2, TILE_LENGTH * sizeof(T));
        elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobalReal,
                                     GlobalTensor<T> dstGlobalImag, int N, int srcN, int dstN)
    {
        this->srcGlobal = srcGlobal;
        this->dstGlobalReal = dstGlobalReal;
        this->dstGlobalImag = dstGlobalImag;
        this->N = N;
        this->srcN = srcN;
        this->dstN = dstN;
        alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        linesPerIter = TILE_LENGTH / alignedN;
        linesPerIter = max(1, linesPerIter & ~1);
        padN = (elementsPerBlock - (N * 2 % elementsPerBlock)) % elementsPerBlock;
    }
    __aicore__ inline void SetNegMatrix(GlobalTensor<T> dstGlobalImagNeg) { this->dstGlobalImagNeg = dstGlobalImagNeg; }
    __aicore__ inline void SplitRealImagWork(int M, int p, int id)
    {
        // p 是参与本阶段的核数（blockNum << 1 >= 2），退化入参直接返回，避免除零
        if (p == 0)
        {
            return;
        }
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN * 2, (k * id + min(id, r)) * dstN, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        int loopCount = lines / linesPerIter;
        srcOffset += loopCount * linesPerIter * srcN * 2;
        dstOffset += loopCount * linesPerIter * dstN;
        int tailLines = lines - linesPerIter * loopCount;
        if (tailLines)
        {
            if (N << 1 <= TILE_LENGTH)
            {
                CopyIn(srcOffset, tailLines / 2);
                Compute(0, 0, tailLines / 2 * alignedN * 2);
                CopyIn(srcOffset + tailLines / 2 * srcN * 2, (tailLines + 1) / 2);
                Compute(1, tailLines / 2 * alignedN, (tailLines + 1) / 2 * alignedN * 2);
            }
            else
            {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, tailLines);
        }
        srcOffset -= linesPerIter * srcN * 2;
        dstOffset -= linesPerIter * dstN;
        for (int i = 0; i < loopCount; ++i)
        {
            if (N << 1 <= TILE_LENGTH)
            {
                CopyIn(srcOffset, linesPerIter / 2);
                Compute(0, 0, linesPerIter / 2 * alignedN * 2);
                CopyIn(srcOffset + linesPerIter / 2 * srcN * 2, (linesPerIter + 1) / 2);
                Compute(1, linesPerIter / 2 * alignedN, (linesPerIter + 1) / 2 * alignedN * 2);
            }
            else
            {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, linesPerIter);
            srcOffset -= linesPerIter * srcN * 2;
            dstOffset -= linesPerIter * dstN;
        }
    }
    __aicore__ inline void CopyIn(int offset, int lines)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        if (N << 1 <= TILE_LENGTH)
        {
            DataCopyExtParams intriParams{
                static_cast<uint16_t>(lines), static_cast<uint16_t>(N * 2 * sizeof(T)),
                static_cast<uint16_t>((srcN * 2 - N * 2) * sizeof(T)),
                static_cast<uint16_t>(N % elementsPerBlock && N % elementsPerBlock <= elementsPerBlock / 2), 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padN), 0};
            DataCopyPad(srcLocal, srcGlobal[offset], intriParams, padParams);
        }
        else
        {
            DataCopy(srcLocal, srcGlobal[offset], lines);
        }
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int progress, int calcedLength, int length)
    {
        if (!progress)
        {
            realLocal = outQueueReal.AllocTensor<T>();
            imagLocal = outQueueImag.AllocTensor<T>();
            if constexpr (neg == NegateImag::NEED) negimagLocal = outQueueImag.AllocTensor<T>();
        }
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint64_t rsvdCnt;
        GatherMask(realLocal[calcedLength], srcLocal, 1, false, 0,
                   {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        GatherMask(imagLocal[calcedLength], srcLocal, 2, false, 0,
                   {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        inQueue.FreeTensor(srcLocal);
        if constexpr (neg == NegateImag::NEED)
        {
            Muls(negimagLocal[calcedLength], imagLocal[calcedLength], static_cast<T>(-1), length / 2);
            if (progress)
            {
                outQueueReal.EnQue(realLocal);
                outQueueImag.EnQue(imagLocal);
                outQueueImag.EnQue(negimagLocal);
            }
        }
        else
        {
            if (progress)
            {
                outQueueReal.EnQue(realLocal);
                outQueueImag.EnQue(imagLocal);
            }
        }
    }
    __aicore__ inline void CopyOut(int offset, int lines)
    {
        LocalTensor<T> realLocal = outQueueReal.DeQue<T>();
        LocalTensor<T> imagLocal = outQueueImag.DeQue<T>();
        DataCopyExtParams intriParams{static_cast<uint16_t>(lines), static_cast<uint16_t>(N * sizeof(T)), 0,
                                      static_cast<uint16_t>((dstN - N) * sizeof(T)), 0};
        DataCopyPad(dstGlobalImag[offset], imagLocal, intriParams);
        DataCopyPad(dstGlobalReal[offset], realLocal, intriParams);
        outQueueImag.FreeTensor(imagLocal);
        outQueueReal.FreeTensor(realLocal);
        if constexpr (neg == NegateImag::NEED)
        {
            LocalTensor<T> negimagLocal = outQueueImag.DeQue<T>();
            DataCopyPad(dstGlobalImagNeg[offset], negimagLocal, intriParams);
            outQueueImag.FreeTensor(negimagLocal);
        }
    }
};

#endif  // CPOTRF_KERNEL_PAD_HPP
