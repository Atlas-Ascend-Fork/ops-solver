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
 * \file cgetrf_kernel.cpp
 * \brief
 */

#ifndef CGETRF_MIX_H
#define CGETRF_MIX_H

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "../utils/kernel/c64/trsm.hpp"
#include "./kernel/getf2.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

#ifndef LU_CEIL_ALIGN
#define LU_CEIL_ALIGN(x, y) (((x) + (y) - 1) / (y) * (y))
#endif

template <typename T>
class SplitRealImag
{
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> srcGlobal, dstGlobalReal, dstGlobalImag;
    int N;
    int srcN;
    int dstN;
    int padN;
    int alignedN;
    int linesPerIter;
    int elementsPerBlock;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueueReal, outQueueImag;
    LocalTensor<T> realLocal, imagLocal;

   public:
    __aicore__ inline SplitRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
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
    __aicore__ inline void SplitRealImagWork(int M, int p, int id)
    {
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
        }
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint64_t rsvdCnt;
        GatherMask(realLocal[calcedLength], srcLocal, 1, false, 0,
                   {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        GatherMask(imagLocal[calcedLength], srcLocal, 2, false, 0,
                   {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        inQueue.FreeTensor(srcLocal);
        if (progress)
        {
            outQueueReal.EnQue(realLocal);
            outQueueImag.EnQue(imagLocal);
        }
    }
    __aicore__ inline void CopyOut(int offset, int lines)
    {
        LocalTensor<T> realLocal = outQueueReal.DeQue<T>();
        LocalTensor<T> imagLocal = outQueueImag.DeQue<T>();
        DataCopyExtParams intriParams{static_cast<uint16_t>(lines), static_cast<uint16_t>(N * sizeof(T)), 0,
                                      static_cast<uint16_t>((dstN - N) * sizeof(T)), 0};
        DataCopyPad(dstGlobalReal[offset], realLocal, intriParams);
        DataCopyPad(dstGlobalImag[offset], imagLocal, intriParams);
        outQueueReal.FreeTensor(realLocal);
        outQueueImag.FreeTensor(imagLocal);
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
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<uint32_t> aGather)
    {
        pipe->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueue, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(gatherQueue, 1, TILE_LENGTH * sizeof(uint32_t));
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
        this->alignedN = LU_CEIL_ALIGN(N, elementsPerBlock);
        this->srcN = srcN;
        this->dstN = dstN;
        linesPerIter = max(1, TILE_LENGTH / LU_CEIL_ALIGN(dstN, 16) / 2);
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
class GTRF2Solver
{
    TransPose<T> optranspose;
    LUCustom1<T> opLU1;
    LUCustom3<T> opLU3;
    GlobalTensor<T> aGlobalReal;
    GlobalTensor<T> aGlobalImag;
    GlobalTensor<T> workGlobalReal;
    GlobalTensor<T> workGlobalImag;
    GlobalTensor<uint32_t> wGlobal;
    GlobalTensor<uint32_t> gather1, gather2;
    TQue<QuePosition::VECIN, 1> inQueueSrc1, inQueueSrc2;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    int M, N;
    int orgM;
    int blockN;
    int workM;
    int tileM;
    int blockIdx;
    TBufPool<TPosition::VECCALC, 16> *pipe;
    static constexpr int TILE_LENGTH = 8192;

   public:
    __aicore__ inline GTRF2Solver() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> aGlobalReal,
                                GlobalTensor<T> aGlobalImag, GlobalTensor<T> workGlobal, GlobalTensor<uint32_t> wGlobal,
                                GlobalTensor<uint32_t> gather1, GlobalTensor<uint32_t> gather2, int M, int orgM, int N,
                                int blockN, int workM, int tileM)
    {
        this->pipe = pipe;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->workGlobalReal = workGlobal;
        // work 虚平面基址按 int64 展开（issue #166）
        this->workGlobalImag = workGlobal[static_cast<int64_t>(LU_CEIL_ALIGN(M, 512)) * blockN];
        this->wGlobal = wGlobal;
        this->gather1 = gather1;
        this->gather2 = gather2;
        this->M = M;
        this->N = N;
        this->orgM = orgM;
        this->blockN = blockN;
        this->workM = workM;
        this->tileM = tileM;
        this->blockIdx = GetBlockIdx();
    }
    __aicore__ inline void Process(int offsetM, int offsetN, int blockM, int realN)
    {
        if (blockM > 512)
        {
            if (blockIdx >= 8)
            {
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x6);
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x6);
                return;
            }
            if (!offsetM)
            {
                pipe->InitBuffer(inQueueSrc1, 1, 8192 * sizeof(T));
                pipe->InitBuffer(inQueueSrc2, 1, 8192 * sizeof(T));
                pipe->InitBuffer(outQueueDst, 1, 8192 * sizeof(T));
                optranspose.Init(pipe, inQueueSrc1, inQueueSrc2, outQueueDst, aGlobalReal, aGlobalImag, workGlobalReal,
                                 workGlobalImag, gather1, gather2, workM, N, tileM, blockN);
                opLU3.Init(pipe, inQueueSrc1, inQueueSrc2, outQueueDst, workGlobalReal, workGlobalImag, wGlobal, workM,
                           blockN, blockN);
            }
            int totalBlocks = (blockM + tileM - 1) / tileM;
            int transposeM1 = 0, transposeM2 = 0;
            if ((blockIdx + 1) * tileM <= blockM)
                transposeM1 = tileM;
            else if (blockIdx * tileM <= blockM)
                transposeM1 = blockM % tileM;
            if ((blockIdx + 9) * tileM <= blockM)
                transposeM2 = tileM;
            else if ((blockIdx + 8) * tileM <= blockM)
                transposeM2 = blockM % tileM;

            int srcOffset = (offsetM + blockIdx * tileM) * N + offsetN;
            int dstOffset = (totalBlocks - 1 - blockIdx) * tileM;

            // > 4096 need to do twice
            if (transposeM1)
            {
                optranspose.Process1(srcOffset, dstOffset, transposeM1);
                if (transposeM2)
                {
                    optranspose.Process1(srcOffset + 8 * tileM * N, dstOffset - 8 * tileM, transposeM2);
                }
            }
            // need all core sync
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x6);
            CrossCoreWaitFlag(0x6);

            opLU3.Process((tileM - blockM % tileM) % tileM, offsetM, offsetN, blockM, blockM + orgM - M, realN);

            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x6);
            CrossCoreWaitFlag(0x6);

            if (transposeM1)
            {
                optranspose.Process2(srcOffset, dstOffset, transposeM1);
                if (transposeM2)
                {
                    optranspose.Process2(srcOffset + 8 * tileM * N, dstOffset - 8 * tileM, transposeM2);
                }
            }
        }
        else
        {
            if (blockIdx) return;

            int srcOffset = offsetM * N + offsetN;
            int dstOffset = 0;
            if (blockM == 512 || !offsetM)
            {
                pipe->Reset();
                pipe->InitBuffer(inQueueSrc1, 1, 8192 * sizeof(T));
                pipe->InitBuffer(inQueueSrc2, 1, 8192 * sizeof(T));
                pipe->InitBuffer(outQueueDst, 1, 8192 * sizeof(T));
                optranspose.Init(pipe, inQueueSrc1, inQueueSrc2, outQueueDst, aGlobalReal, aGlobalImag, workGlobalReal,
                                 workGlobalImag, gather1, gather2, workM, N, tileM, blockN);
                opLU1.Init(pipe, inQueueSrc1, inQueueSrc2, workGlobalReal, workGlobalImag, wGlobal, workM, blockN,
                           blockN);
            }
            optranspose.Process1(srcOffset, dstOffset, blockM);

            int32_t eventIDMTE3ToMTE2 =
                static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

            opLU1.Process(tileM - blockM, offsetM, offsetN, blockM, blockM + orgM - M, realN);

            eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

            optranspose.Process2(srcOffset, dstOffset, blockM);
        }
    }
};

// for N * M <= 192MB
__aicore__ inline void custom_lu(int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org, GM_ADDR A_work,
                                 GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm, GM_ADDR gather3_gm)
{
    // 平面尺寸用 int64：host 允许 m*n 远超 int32（详见 cgetrf_host 的规模校验），
    // strideN*M 类平面基址偏移若按 int 计算会回绕为负偏移（issue #166）。
    // M/N/strideN 本身受 host 校验上界约束仍在 int32 范围，保持 int 以匹配
    // kernel 内其余索引算术；仅平面基址按 int64 展开。
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    int N = (orgN + 15) / 16 * 16;
    const int64_t planeStride = static_cast<int64_t>(strideN) * M;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int coreIdx = blockIdx;
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    coreIdx >>= 1;
#endif

    int workM = (M + tileM - 1) / tileM * tileM;
    int limK = 128;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> aGlobalReal;
    GlobalTensor<float> aGlobalImag;
    GlobalTensor<float> aGlobalImagNeg;
    GlobalTensor<uint32_t> wGlobal;
    GlobalTensor<float> workGlobal;
    GlobalTensor<uint32_t> gather1;
    GlobalTensor<uint32_t> gather2;
    GlobalTensor<uint32_t> gather3;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    aGlobalReal = aGlobal;
    // 平面基址偏移按 int64 计算（issue #166）：strideN*M 与 strideN*M*2 在
    // m*n 较大时超 int32 回绕为负偏移
    aGlobalImag = aGlobalReal[planeStride];
    aGlobalImagNeg = aGlobalReal[planeStride * 2];
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(W));
    workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work_gm));
    gather1.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather1_gm));
    gather2.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather2_gm));
    gather3.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather3_gm));

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    CMatmulCustom<float> opmm;
    SolveTrsm<float> optrsm;

    int t = min(M, N);
    int cnt = t / blockN;

#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    pipe.InitBufPool(tbufPool, 192 * 1024);

    SplitRealImag<float> opsp;
    opsp.Init(&tbufPool);
    opsp.SetMatrix(aGlobalOrg, aGlobalReal, aGlobalImag, orgN, orgN, strideN);
    opsp.SplitRealImagWork(orgM, blockNum << 1, blockIdx);

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    GTRF2Solver<float> getf2solver;
    getf2solver.Init(&tbufPool, aGlobalReal, aGlobalImag, workGlobal, wGlobal, gather1, gather2, M, orgM, strideN,
                     blockN, workM, tileM);
    getf2solver.Process(0, 0, M, min(orgN, blockN));

    if (blockIdx >= 8 && M > 512)
    {
        CrossCoreWaitFlag(0x6);
        CrossCoreWaitFlag(0x6);
    }

    int swapIdx = -1;
    SwapRows<float> opswap;
    if (coreIdx >= 4 && ~blockIdx & 1)
    {
        swapIdx = coreIdx - 4;
        opswap.Init(&tbufPool, aGlobalReal, aGlobalImag);
    }

    NegateCustom<float> opng;
    if (coreIdx >= 4)
    {
        optrsm.Init(&tbufPool, blockM);
        if (blockIdx & 1) opng.Init(&tbufPool);
    }
    for (int i = 1, offset = blockN; i < cnt; ++i, offset += blockN)
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
        CrossCoreWaitFlag(0x2);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            wGlobal[offset - blockN]);
        if (offset % limK)
        {
            if (~swapIdx && swapIdx < 8)
            {
                int swapTotalN = offset / limK * limK + limK - offset;
                int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
                int swapOffsetN = swapBlockN * swapIdx;
                int realBlockN = max(0, min(swapBlockN, swapTotalN - swapOffsetN));
                for (int j = offset - blockN; j < offset; ++j)
                {
                    int p = wGlobal.GetValue(j);
                    opswap.Process(offset + swapOffsetN + j * strideN, offset + swapOffsetN + p * strideN, realBlockN);
                }
            }
        }
        if (offset % limK != blockN)
        {
            if (~swapIdx && swapIdx >= 8)
            {
                int swapTotalN = offset % limK - blockN;
                int swapOffset = offset / limK * limK;
                if (offset % limK == 0)
                {
                    swapTotalN += limK;
                    swapOffset -= limK;
                }
                int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
                int swapOffsetN = swapBlockN * (swapIdx - 8);
                int realBlockN = min(swapBlockN, swapTotalN - swapOffsetN);
                if (realBlockN > 0)
                {
                    for (int j = offset - blockN; j < offset; ++j)
                    {
                        int p = wGlobal.GetValue(j);
                        opswap.Process(swapOffset + swapOffsetN + j * strideN, swapOffset + swapOffsetN + p * strideN,
                                       realBlockN);
                    }
                }
            }
        }
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
        CrossCoreWaitFlag(0x2);
        if (offset % limK)
        {
            int trsmTotalN = min(N - offset, limK - offset % limK);
            int trsmM = blockM;
            int trsmBlocks = blockNum - 4;
            int trsmBlockN = LU_CEIL_ALIGN((trsmTotalN + trsmBlocks - 1) / trsmBlocks, 128);
            int trsmOffsetN = trsmBlockN * (coreIdx - 4);
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            if (coreIdx >= 4 && realBlockN > 0)
            {
                solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                                 aGlobalReal[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalImag[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalImagNeg[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalReal[offset - trsmM + (offset - trsmM) * strideN],
                                 aGlobalImag[offset - trsmM + (offset - trsmM) * strideN]);
            }
        }
        else
        {
            int trsmTotalN = N - offset;
            int trsmM = limK;
            int trsmBlocks = blockNum - 4;
            int trsmBlockN = LU_CEIL_ALIGN((trsmTotalN + trsmBlocks - 1) / trsmBlocks, 128);
            int trsmOffsetN = trsmBlockN * (coreIdx - 4);
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            if (coreIdx >= 4 && realBlockN > 0)
            {
                solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                                 aGlobalReal[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalImag[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalImagNeg[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalReal[offset - trsmM + (offset - trsmM) * strideN],
                                 aGlobalImag[offset - trsmM + (offset - trsmM) * strideN]);
            }
        }
        // trsm finish, all core sync
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
        CrossCoreWaitFlag(0x2);

        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
        if (offset % limK)
            CrossCoreWaitFlag(0x8);
        else
            CrossCoreWaitFlag(0x9);
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
        CrossCoreWaitFlag(0x2);  // wait for all gemm1 to finish

        getf2solver.Process(offset, offset, M - offset, min(orgN - offset, blockN));
        if (offset % limK)
        {
            CrossCoreWaitFlag(0x8);  // wait for small gemm2 to finish
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
            CrossCoreWaitFlag(0x2);
            if (coreIdx >= 4 && blockIdx & 1)
            {
                int negBlockM = blockM / (blockNum - 4);
                opng.SetMatrix(aGlobalReal[offset + (offset - blockM + (coreIdx - 4) * negBlockM) * strideN], strideN);
                opng.Process(negBlockM, limK - (offset % limK));
            }
        }
        if (offset % limK == limK - blockN)
        {
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wGlobal[offset]);
            int startIndex = offset / limK * limK;
            int endIndex = min(offset / limK * limK + limK, min(orgM, orgN));
            if (offset < limK)
            {
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
                CrossCoreWaitFlag(0x2);
                if (~swapIdx && swapIdx < 8)
                {
                    int swapOffset = offset / limK * limK + limK;
                    int swapTotalN = N - swapOffset;
                    int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
                    int swapOffsetN = swapBlockN * swapIdx;
                    int realBlockN = max(0, min(swapBlockN, swapTotalN - swapOffsetN));
                    for (int j = startIndex; j < endIndex; ++j)
                    {
                        int p = wGlobal.GetValue(j);
                        opswap.Process(swapOffset + swapOffsetN + j * strideN, swapOffset + swapOffsetN + p * strideN,
                                       realBlockN);
                    }
                }
            }
            else
            {
                CrossCoreWaitFlag(0x9);  // wait for big gemm2 to finish
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
                CrossCoreWaitFlag(0x2);
                if (coreIdx >= 4 && blockIdx & 1)
                {
                    int negOffset = offset / limK * limK;
                    int negBlockM = limK / (blockNum - 4);
                    opng.SetMatrix(aGlobalReal[negOffset + (negOffset - limK + (coreIdx - 4) * negBlockM) * strideN],
                                   strideN);
                    opng.Process(negBlockM, N - negOffset);
                }
                if (~swapIdx && swapIdx < 8)
                {
                    int swapOffset = offset / limK * limK + limK;
                    int swapTotalN = N - swapOffset;
                    int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
                    int swapOffsetN = swapBlockN * swapIdx;
                    int realBlockN = max(0, min(swapBlockN, swapTotalN - swapOffsetN));
                    for (int j = startIndex; j < endIndex; ++j)
                    {
                        int p = wGlobal.GetValue(j);
                        opswap.Process(swapOffset + swapOffsetN + j * strideN, swapOffset + swapOffsetN + p * strideN,
                                       realBlockN);
                    }
                }
                if (~swapIdx && swapIdx >= 8)
                {
                    int swapTotalN = offset / limK * limK;
                    int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
                    int swapOffsetN = swapBlockN * (swapIdx - 8);
                    int realBlockN = min(swapBlockN, swapTotalN - swapOffsetN);
                    if (realBlockN > 0)
                    {
                        for (int j = startIndex; j < endIndex; ++j)
                        {
                            int p = wGlobal.GetValue(j);
                            opswap.Process(swapOffsetN + j * strideN, swapOffsetN + p * strideN, realBlockN);
                        }
                    }
                }
            }
        }
        if (blockIdx >= 8 && M - offset > 512)
        {
            CrossCoreWaitFlag(0x6);
            CrossCoreWaitFlag(0x6);
        }
    }

    if (t > limK && t % limK) CrossCoreWaitFlag(0x9);  // wait for big gemm2 to finish
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wGlobal[t - blockN]);
    if (t % limK)
    {
        if (~swapIdx && swapIdx < 8 && N > M)
        {  // swaps from last big block to next big block
            int swapOffset = t / limK * limK + limK;
            int swapTotalN = N - swapOffset;
            int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
            int swapOffsetN = swapBlockN * swapIdx;
            int realBlockN = min(swapBlockN, swapTotalN - swapOffsetN);
            if (realBlockN > 0)
            {
                for (int j = t / limK * limK; j < min(t / limK * limK + limK, min(orgM, orgN)); ++j)
                {
                    int p = wGlobal.GetValue(j);
                    opswap.Process(swapOffset + swapOffsetN + j * strideN, swapOffset + swapOffsetN + p * strideN,
                                   realBlockN);
                }
            }
        }
    }
    if (t > limK && t % limK)
    {
        if (coreIdx >= 4 && blockIdx & 1)
        {
            int negOffset = (t - blockN) / limK * limK;
            int negBlockM = limK / (blockNum - 4);
            opng.SetMatrix(aGlobalReal[negOffset + (negOffset - limK + (coreIdx - 4) * negBlockM) * strideN], strideN);
            opng.Process(negBlockM, N - negOffset);
        }
        if (~swapIdx && swapIdx >= 8)
        {  // swaps from last big block to prev big block
            int swapTotalN = t / limK * limK;
            int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
            int swapOffsetN = swapBlockN * (swapIdx - 8);
            int realBlockN = min(swapBlockN, swapTotalN - swapOffsetN);
            if (realBlockN > 0)
            {
                for (int j = t / limK * limK; j < min(t / limK * limK + limK, min(orgM, orgN)); ++j)
                {
                    int p = wGlobal.GetValue(j);
                    opswap.Process(swapOffsetN + j * strideN, swapOffsetN + p * strideN, realBlockN);
                }
            }
        }
    }
    if (t % limK)
    {
        if (~swapIdx && swapIdx < 8)
        {  // swaps from last small block to next small blocks
            int swapTotalN = min(t / limK * limK + limK - t, N - t);
            int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
            int swapOffsetN = swapBlockN * swapIdx;
            int realBlockN = max(0, min(swapBlockN, swapTotalN - swapOffsetN));
            for (int j = t - blockN; j < min(orgM, orgN); ++j)
            {
                int p = wGlobal.GetValue(j);
                opswap.Process(t + swapOffsetN + j * strideN, t + swapOffsetN + p * strideN, realBlockN);
            }
        }
    }
    if (t % limK != blockN)
    {
        if (~swapIdx && swapIdx >= 8)
        {  // swaps from last small block to prev small blocks
            int swapTotalN = t % limK - blockN;
            int swapOffset = t / limK * limK;
            if (t % limK == 0)
            {
                swapTotalN += limK;
                swapOffset -= limK;
            }
            int swapBlockN = LU_CEIL_ALIGN(swapTotalN / 8, 1024);
            int swapOffsetN = swapBlockN * (swapIdx - 8);
            int realBlockN = min(swapBlockN, swapTotalN - swapOffsetN);
            if (realBlockN > 0)
            {
                for (int j = t - blockN; j < min(orgM, orgN); ++j)
                {
                    int p = wGlobal.GetValue(j);
                    opswap.Process(swapOffset + swapOffsetN + j * strideN, swapOffset + swapOffsetN + p * strideN,
                                   realBlockN);
                }
            }
        }
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    if (N > M)
    {
        if (M % limK)
        {
            int trsmTotalN = N - M;
            int trsmM = blockM;
            int trsmBlocks = blockNum - 4;
            int trsmBlockN = LU_CEIL_ALIGN((trsmTotalN + trsmBlocks - 1) / trsmBlocks, 64);
            int trsmOffsetN = trsmBlockN * (coreIdx - 4);
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            if (coreIdx >= 4 && trsmOffsetN < trsmTotalN)
            {
                solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                                 aGlobalReal[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalImag[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalImagNeg[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalReal[M - trsmM + (M - trsmM) * strideN],
                                 aGlobalImag[M - trsmM + (M - trsmM) * strideN]);
                CrossCoreSetFlag<0x1, PIPE_MTE3>(0x7);
                CrossCoreWaitFlag(0x7);
                if (blockIdx & 1)
                {
                    opng.SetMatrix(aGlobalReal[trsmOffsetN + M + (M - trsmM) * strideN], strideN);
                    opng.Process(trsmM, realBlockN);
                }
            }
        }
        else
        {
            int trsmTotalN = N - M;
            int trsmM = limK;
            int trsmBlocks = blockNum - 4;
            int trsmBlockN = LU_CEIL_ALIGN((trsmTotalN + trsmBlocks - 1) / trsmBlocks, 64);
            int trsmOffsetN = trsmBlockN * (coreIdx - 4);
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            if (coreIdx >= 4 && trsmOffsetN < trsmTotalN)
            {
                solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                                 aGlobalReal[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalImag[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalImagNeg[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalReal[M - trsmM + (M - trsmM) * strideN],
                                 aGlobalImag[M - trsmM + (M - trsmM) * strideN]);
                CrossCoreSetFlag<0x1, PIPE_MTE3>(0x7);
                CrossCoreWaitFlag(0x7);
                if (blockIdx & 1)
                {
                    opng.SetMatrix(aGlobalReal[trsmOffsetN + M + (M - trsmM) * strideN], strideN);
                    opng.Process(trsmM, realBlockN);
                }
            }
        }
    }

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    tbufPool.Reset();

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
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    opmm.Init(&pipe);
    SetAtomicAdd<float>();
    static constexpr int gemm1BlockNum = 8;
    static constexpr int gemm2BlockNum = 20 - gemm1BlockNum;
    if (blockIdx >= gemm1BlockNum)
    {
        for (int i = blockN; i < limK; i += blockN)
        {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
        }
    }
    for (int i = 1, offset = blockN; i < cnt; ++i, offset += blockN)
    {
        if (offset % limK == 0)
        {
            int trsmTotalN = N - offset;
            int trsmM = limK;
            int trsmBlocks = blockNum - 4;
            int trsmBlockN = LU_CEIL_ALIGN((trsmTotalN + trsmBlocks - 1) / trsmBlocks, 128);
            int trsmOffsetN = trsmBlockN * (blockIdx - 4);
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            if (blockIdx >= 4 && trsmOffsetN < trsmTotalN)
            {
                solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                                 aGlobalReal[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalImag[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalImagNeg[trsmOffsetN + offset + (offset - trsmM) * strideN],
                                 aGlobalReal[offset - trsmM + (offset - trsmM) * strideN],
                                 aGlobalImag[offset - trsmM + (offset - trsmM) * strideN]);
            }
        }
        CrossCoreWaitFlag(0x3);  // wait for trsm to finish
        opmm.SetMatrix(aGlobalReal, aGlobalImag, aGlobalReal, aGlobalImagNeg, aGlobalImag, aGlobalReal, aGlobalImag,
                       strideN, strideN);

        if (offset % limK)
        {
            if (blockIdx >= gemm1BlockNum) continue;
            int gemmTotalM = M - offset;
            int gemmBlockM = LU_CEIL_ALIGN((gemmTotalM + gemm1BlockNum - 1) / gemm1BlockNum, 128);
            int gemmBlockN = min(N - offset, limK - offset % limK);
            int gemmOffsetM = gemmBlockM * blockIdx;
            int gemmRealM = min(gemmBlockM, gemmTotalM - gemmOffsetM);
            if (blockIdx < gemm1BlockNum && gemmRealM > 0)
            {  // gemm1
                opmm.Process(offset - blockN + (offset + gemmOffsetM) * strideN, offset + (offset - blockN) * strideN,
                             offset + (offset + gemmOffsetM) * strideN, gemmRealM, blockN, blockN);
            }
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
            if (blockIdx < gemm1BlockNum && gemmBlockN > blockN && gemmRealM > 0)
            {  // gemm2
                opmm.Process(
                    offset - blockN + (offset + gemmOffsetM) * strideN, offset + blockN + (offset - blockN) * strideN,
                    offset + blockN + (offset + gemmOffsetM) * strideN, gemmRealM, gemmBlockN - blockN, blockN);
            }
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
        }
        else
        {
            int gemmTotalM = M - offset;
            int gemmTotalN = N - offset;
            static constexpr int gemmBlockM = 128;
            static constexpr int gemmBlockN = 128;
            int mBlocks = (gemmTotalM + gemmBlockM - 1) / gemmBlockM;
            int nBlocks = (gemmTotalN + gemmBlockN - 1) / gemmBlockN;
            int TailM = gemmTotalM % gemmBlockM;
            if (!TailM) TailM = gemmBlockM;
            int TailN = gemmTotalN % gemmBlockN;
            if (!TailN) TailN = gemmBlockN;
            int x = 0, y = blockIdx;
            for (; y < mBlocks; y += blockNum)
            {
                opmm.Process(offset - limK + (offset + y * gemmBlockM) * strideN,
                             offset + x * gemmBlockN + (offset - limK) * strideN,
                             offset + x * gemmBlockN + (offset + y * gemmBlockM) * strideN,
                             y + 1 == mBlocks ? TailM : gemmBlockM, x + 1 == nBlocks ? TailN : gemmBlockN, limK);
            }
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x9);
            if (blockIdx < gemm1BlockNum)
            {
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x9);
                continue;
            }
            for (int i = blockN; i < limK; i += blockN)
            {
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
            }
            y = blockIdx - gemm1BlockNum;
            x = 1 + y / mBlocks;
            y %= mBlocks;
            while (x < nBlocks)
            {
                opmm.Process(offset - limK + (offset + y * gemmBlockM) * strideN,
                             offset + x * gemmBlockN + (offset - limK) * strideN,
                             offset + x * gemmBlockN + (offset + y * gemmBlockM) * strideN,
                             y + 1 == mBlocks ? TailM : gemmBlockM, x + 1 == nBlocks ? TailN : gemmBlockN, limK);
                y += gemm2BlockNum;
                x += y / mBlocks;
                y %= mBlocks;
            }
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x9);
        }
    }
    if (N > M)
    {
        if (M % limK == 0)
        {
            int trsmTotalN = N - M;
            int trsmM = limK;
            int trsmBlocks = blockNum - 4;
            int trsmBlockN = LU_CEIL_ALIGN((trsmTotalN + trsmBlocks - 1) / trsmBlocks, 64);
            int trsmOffsetN = trsmBlockN * (blockIdx - 4);
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            if (blockIdx >= 4 && trsmOffsetN < trsmTotalN)
            {
                solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                                 aGlobalReal[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalImag[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalImagNeg[trsmOffsetN + M + (M - trsmM) * strideN],
                                 aGlobalReal[M - trsmM + (M - trsmM) * strideN],
                                 aGlobalImag[M - trsmM + (M - trsmM) * strideN]);
            }
        }
    }
#endif
}

__global__ __aicore__ void cgetrf_kernel(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM,
                                         GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm,
                                         GM_ADDR gather2_gm, GM_ADDR gather3_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    custom_lu(orgM, orgN, blockM, blockN, tileM, A_org, A_work, W, work_gm, gather1_gm, gather2_gm, gather3_gm);
}

void cgetrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                      GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                      GM_ADDR gather3_gm, uint32_t numBlocks, void *stream)
{
    cgetrf_kernel<<<numBlocks, nullptr, stream>>>(sync, orgM, orgN, blockM, blockN, tileM, A_org, A_work, W, work_gm,
                                                  gather1_gm, gather2_gm, gather3_gm);
}
#endif  // CGETRF_MIX_H
