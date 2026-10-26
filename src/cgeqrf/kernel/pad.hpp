/*
 * PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
 *
 * Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
 * All rights reserved.
 *
 * This software is dual-licensed:
 *   1) Community license: use, modify, and redistribute freely under the
 *      GNU General Public License v3 or later (GPL-3.0-or-later);
 *   2) Commercial license: for closed-source or commercial use / redistribution,
 *      contact the copyright holder for a separate license.
 * You may choose either. Default terms are GPL-3.0-or-later.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you may redistribute it and/or modify it under
 * the terms of the GPL. It is distributed "as is", without warranty.
 * See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
 * Commercial licensing contact: yuantang@fudan.edu.cn
 *
 * -------------------------------------------------------------------------
 * 中文版 (Chinese):
 *
 * PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
 *
 * 本软件采用双重许可（dual licensing）：
 *   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
 *      （GPL-3.0-or-later）自由使用、修改与再分发；
 *   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
 * 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
 *
 * 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
 * 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */

/*!
 * \file pad.hpp
 * \brief
 */
#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

#define ceil(x, y) (((x) + (y) - 1) / (y) * (y))
#define floor(x, y) ((x) / (y) * (y))

enum class NegateImag
{
    NOTNEED,
    NEED
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
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN * 2, (k * id + min(id, r)) * dstN, k + (id < r));
    }
    __aicore__ inline void CopyOneTile(int srcOffset, int dstOffset, int lines)
    {
        if (N << 1 <= TILE_LENGTH)
        {
            CopyIn(srcOffset, lines / 2);
            Compute(0, 0, lines / 2 * alignedN * 2);
            CopyIn(srcOffset + lines / 2 * srcN * 2, (lines + 1) / 2);
            Compute(1, lines / 2 * alignedN, (lines + 1) / 2 * alignedN * 2);
        }
        else
        {
            CopyIn(srcOffset, TILE_LENGTH);
            Compute(0, 0, TILE_LENGTH);
            CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
            Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
        }
        CopyOut(dstOffset, lines);
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        int loopCount = lines / linesPerIter;
        srcOffset += loopCount * linesPerIter * srcN * 2;
        dstOffset += loopCount * linesPerIter * dstN;
        int tailLines = lines - linesPerIter * loopCount;
        if (tailLines)
        {
            CopyOneTile(srcOffset, dstOffset, tailLines);
        }
        srcOffset -= linesPerIter * srcN * 2;
        dstOffset -= linesPerIter * dstN;
        for (int i = 0; i < loopCount; ++i)
        {
            CopyOneTile(srcOffset, dstOffset, linesPerIter);
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
    __aicore__ inline void AllocOutBuffers()
    {
        realLocal = outQueueReal.AllocTensor<T>();
        imagLocal = outQueueImag.AllocTensor<T>();
        if constexpr (neg == NegateImag::NEED) negimagLocal = outQueueImag.AllocTensor<T>();
    }
    __aicore__ inline void EnQueOut()
    {
        outQueueReal.EnQue(realLocal);
        outQueueImag.EnQue(imagLocal);
        if constexpr (neg == NegateImag::NEED) outQueueImag.EnQue(negimagLocal);
    }
    __aicore__ inline void Compute(int progress, int calcedLength, int length)
    {
        if (!progress) AllocOutBuffers();
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
        }
        if (progress) EnQueOut();
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

template <typename T>
class MergeRealImag
{
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> aOrgGlobal;
    GlobalTensor<T> aGlobalReal, aGlobalImag;
    GlobalTensor<uint32_t> aGather;
    int N;
    int alignedN;
    int srcN;
    int dstN;
    int linesPerIter;
    int elementsPerBlock;
    int baseN, aligned_n_cache, nFullBlocks, nTailBlocks;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    LocalTensor<T> realLocal, imagLocal;
    LocalTensor<uint32_t> gatherLocal;

   public:
    __aicore__ inline MergeRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* tbufPool, GlobalTensor<uint32_t> aGather)
    {
        tbufPool->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        tbufPool->InitBuffer(outQueue, 2, TILE_LENGTH * sizeof(T));
        tbufPool->InitBuffer(gatherQueue, 1, TILE_LENGTH * sizeof(uint32_t));
        elementsPerBlock = 32 / sizeof(T);
        this->aGather = aGather;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                     GlobalTensor<T> aGlobalImag, int N, int srcN, int dstN)
    {
        this->aOrgGlobal = aOrgGlobal;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->N = N;
        this->alignedN = ceil(N, elementsPerBlock);
        this->srcN = srcN;
        this->dstN = dstN;
        linesPerIter = max(1, TILE_LENGTH / ceil(dstN, 16) / 2);
    }
    __aicore__ inline void MergeRealImagWork(int M, int p, int id)
    {
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN, (k * id + min(id, r)) * dstN * 2, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        LoadGather();
        int loopCount = lines / linesPerIter;
        for (int i = 0; i < loopCount; ++i)
        {
            CopyIn(srcOffset, linesPerIter, alignedN);
            Compute(linesPerIter * alignedN * 2);
            CopyOut(dstOffset, linesPerIter, N);
            srcOffset += linesPerIter * srcN;
            dstOffset += linesPerIter * dstN * 2;
        }
        lines %= linesPerIter;
        if (!lines)
        {
            gatherQueue.FreeTensor(gatherLocal);
            return;
        }
        CopyIn(srcOffset, lines, alignedN);
        Compute(lines * alignedN * 2);
        CopyOut(dstOffset, lines, N);
        gatherQueue.FreeTensor(gatherLocal);
    }
    __aicore__ inline void LoadGather()
    {
        gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, linesPerIter * alignedN * 2);
        gatherQueue.EnQue(gatherLocal);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }
    __aicore__ inline void CopyIn(int offset, int blockM, int blockN)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();

        DataCopyParams copyInParams{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                    static_cast<uint16_t>((srcN - blockN) / elementsPerBlock), 0};
        DataCopy(srcLocal, aGlobalReal[offset], copyInParams);
        DataCopy(srcLocal[TILE_LENGTH / 2], aGlobalImag[offset], copyInParams);

        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int length)
    {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut(int offset, int blockM, int blockN)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();

        DataCopyExtParams copyOutParams{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN * 2 * sizeof(T)),
                                        static_cast<uint16_t>((alignedN * 2 - blockN * 2) / elementsPerBlock),
                                        static_cast<uint16_t>((dstN * 2 - blockN * 2) * sizeof(T)), 0};
        DataCopyPad(aOrgGlobal[offset], dstLocal, copyOutParams);

        outQueue.FreeTensor(dstLocal);
    }
};

template <typename T>
class MergeRealImagWithTrans
{
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> aOrgGlobal;
    GlobalTensor<T> aGlobalReal, aGlobalImag;
    GlobalTensor<uint32_t> aGather;
    int orgM, orgN, M, strideN, tileN;
    int strideBatchSrc, strideBatchDst;
    int elementsPerBlock;
    int baseN, aligned_n_cache, nFullBlocks, nTailBlocks;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    LocalTensor<T> realLocal, imagLocal;
    LocalTensor<uint32_t> gatherLocal;

   public:
    __aicore__ inline MergeRealImagWithTrans() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<uint32_t> aGather)
    {
        pipe->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(gatherQueue, 1, TILE_LENGTH * sizeof(uint32_t));
        elementsPerBlock = 32 / sizeof(T);
        this->aGather = aGather;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                     GlobalTensor<T> aGlobalImag, int orgM, int orgN, int strideN,
                                     int strideBatchSrc = 0, int strideBatchDst = 0)
    {
        this->aOrgGlobal = aOrgGlobal;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->orgM = orgM;
        this->orgN = orgN;
        this->M = (orgM + 15) / 16 * 16;
        this->strideN = strideN;
        this->tileN = min(strideN, TILE_LENGTH / 2 / M / 8 * 8);
        this->strideBatchSrc = strideBatchSrc;
        this->strideBatchDst = strideBatchDst;
    }
    __aicore__ inline void ProcessTiles(int startIdx, int tileNum, int batch)
    {
        for (int i = 0; i < batch; ++i) Process(startIdx * tileN, tileNum, i);
    }
    __aicore__ inline void ProcessLastTiles(int startIdx, int tileNum, int totalNum, int batch)
    {
        for (int i = 0; i < batch; ++i)
        {
            Process(startIdx * tileN, tileNum - 1, i);
            PipeBarrier<PIPE_MTE3>();
            ProcessTail((totalNum - 1) * tileN, orgN - (totalNum - 1) * tileN, i);
        }
    }
    __aicore__ inline void MergeRealImagWork(int p, int id, int batch)
    {
        LoadGather();
        int totalNum = (orgN + tileN - 1) / tileN;
        int k = totalNum / p, r = totalNum % p;
        int startIdx = k * id + min(id, r);
        int tileNum = (k + (id < r));
        if (tileNum && startIdx + tileNum == totalNum)
        {
            ProcessLastTiles(startIdx, tileNum, totalNum, batch);
        }
        else
        {
            ProcessTiles(startIdx, tileNum, batch);
        }
        gatherQueue.FreeTensor(gatherLocal);
    }
    __aicore__ inline void Process(int startN, int tileNum, int batch_idx)
    {
        for (int i = 0; i < tileNum; ++i)
        {
            CopyIn(batch_idx * strideBatchSrc + startN, tileN);
            Compute(2 * tileN * M);
            CopyOut(batch_idx * strideBatchDst + startN * orgM * 2, tileN);
            startN += tileN;
        }
    }
    __aicore__ inline void ProcessTail(int startN, int length, int batch_idx)
    {
        CopyIn(batch_idx * strideBatchSrc + startN,
               (length + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock);
        Compute(2 * tileN * M);
        CopyOut(batch_idx * strideBatchDst + startN * orgM * 2, length);
    }
    __aicore__ inline void LoadGather()
    {
        gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, 2 * tileN * M);
        gatherQueue.EnQue(gatherLocal);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }
    __aicore__ inline void CopyIn(int offset, int length)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        int copyDB = (length + elementsPerBlock - 1) / elementsPerBlock;
        DataCopyParams copyInParams{static_cast<uint16_t>(orgM), static_cast<uint16_t>(copyDB),
                                    static_cast<uint16_t>(strideN / elementsPerBlock - copyDB),
                                    static_cast<uint16_t>(tileN / elementsPerBlock - copyDB)};
        DataCopy(srcLocal, aGlobalReal[offset], copyInParams);
        DataCopy(srcLocal[TILE_LENGTH / 2], aGlobalImag[offset], copyInParams);

        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int length)
    {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut(int offset, int length)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();

        DataCopyExtParams copyOutParams{static_cast<uint16_t>(length), static_cast<uint16_t>(orgM * 2 * sizeof(T)),
                                        static_cast<uint16_t>((M * 2 - orgM * 2) / elementsPerBlock), 0, 0};
        DataCopyPad(aOrgGlobal[offset], dstLocal, copyOutParams);
        outQueue.FreeTensor(dstLocal);
    }
};

template <NegateImag neg = NegateImag::NOTNEED>
__aicore__ inline void pad(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, GM_ADDR A_org,
                           GM_ADDR A_work, int batch)
{
#ifdef __DAV_C220_VEC__
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int strideM = (orgM + 127) / 128 * 128;
    int strideN = (orgN + 127) / 128 * 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobalReal;
    GlobalTensor<float> aGlobalImag;
    GlobalTensor<float> aGlobalImagNeg;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_org));
    aGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
    aGlobalImag = aGlobalReal[batch * strideN * strideM];

    SplitRealImag<float, neg> opsp;
    opsp.Init(&tbufPool);
    opsp.SetMatrix(aGlobalOrg, aGlobalReal, aGlobalImag, orgN, orgN, strideN);
    if constexpr (neg == NegateImag::NEED)
    {
        aGlobalImagNeg = aGlobalImag[batch * strideN * strideM];
        opsp.SetNegMatrix(aGlobalImagNeg);
    }
    opsp.SplitRealImagWork(orgM, blockNum << 1, blockIdx);

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);
#endif
}

__aicore__ inline void restore(TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgM, int orgN, int strideN,
                               GlobalTensor<float> aGlobalOrg, GlobalTensor<float> aGlobalReal,
                               GlobalTensor<float> aGlobalImag, GlobalTensor<uint32_t> gather3)
{
#ifdef __DAV_C220_VEC__
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();

    MergeRealImag<float> opmg;
    static constexpr int TILE_LENGTH = 4096;
    opmg.Init(&tbufPool, gather3);
    opmg.SetMatrix(aGlobalOrg, aGlobalReal, aGlobalImag, min(orgN, TILE_LENGTH), strideN, orgN);
    opmg.MergeRealImagWork(orgM, blockNum << 1, blockIdx);
    if (orgN > TILE_LENGTH)
    {
        opmg.SetMatrix(aGlobalOrg[TILE_LENGTH * 2], aGlobalReal[TILE_LENGTH], aGlobalImag[TILE_LENGTH],
                       orgN - TILE_LENGTH, strideN, orgN);
        opmg.MergeRealImagWork(orgM, blockNum << 1, blockIdx);
    }
    tbufPool.Reset();
#endif
}
