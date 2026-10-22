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
 * \file sgetrf_kernel.cpp
 * \brief
 */

#ifndef SGETRF_MIX_H
#define SGETRF_MIX_H

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "../utils/kernel/c32/pad.hpp"
#include "../utils/kernel/c32/trsm.hpp"
#include "./kernel/getf2.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class GTRF2Solver
{
    TransPose<T> optranspose;
    LUCustom1<T> opLU1;
    LUCustom3<T> opLU3;
    GlobalTensor<T> aGlobal;
    GlobalTensor<T> workGlobal;
    GlobalTensor<uint32_t> wGlobal;
    GlobalTensor<uint32_t> gather1, gather2;
    TQue<QuePosition::VECIN, 1> inQueueSrc;
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
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> aGlobal,
                                GlobalTensor<T> workGlobal, GlobalTensor<uint32_t> wGlobal,
                                GlobalTensor<uint32_t> gather1, GlobalTensor<uint32_t> gather2, int M, int orgM, int N,
                                int blockN, int workM, int tileM)
    {
        this->pipe = pipe;
        this->aGlobal = aGlobal;
        this->workGlobal = workGlobal;
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
                pipe->InitBuffer(inQueueSrc, 1, 8192 * sizeof(T));
                pipe->InitBuffer(outQueueDst, 1, 8192 * sizeof(T));
                optranspose.Init(pipe, inQueueSrc, outQueueDst, aGlobal, workGlobal, gather1, gather2, workM, N, tileM,
                                 blockN);
                opLU3.Init(pipe, inQueueSrc, workGlobal, wGlobal, workM, blockN, blockN);
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
                pipe->InitBuffer(inQueueSrc, 1, 8192 * sizeof(T));
                pipe->InitBuffer(outQueueDst, 1, 8192 * sizeof(T));
                optranspose.Init(pipe, inQueueSrc, outQueueDst, aGlobal, workGlobal, gather1, gather2, workM, N, tileM,
                                 blockN);
                opLU1.Init(pipe, inQueueSrc, workGlobal, wGlobal, workM, blockN, blockN);
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

__aicore__ inline void custom_lu(int orgM, int orgN, int blockN, int tileM, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W,
                                 GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm)
{
    int M = (orgM + 15) / 16 * 16;
    int strideN = (orgN + 127) / 128 * 128;
    int N = (orgN + 15) / 16 * 16;
    int t = min(M, N);
    int cnt = t / blockN;
    int blockM = 16;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int coreIdx = blockIdx >> 1;
    int workM = (M + tileM - 1) / tileM * tileM;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;
    GlobalTensor<uint32_t> wGlobal;
    GlobalTensor<float> workGlobal;
    GlobalTensor<uint32_t> gather1;
    GlobalTensor<uint32_t> gather2;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(W));
    workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(work_gm));
    gather1.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather1_gm));
    gather2.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather2_gm));

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    MatmulCustom<float> opmm;
    SolveTrsm<float> optrsm;

#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    pipe.InitBufPool(tbufPool, 192 * 1024);

    PadCustom<float> oppad;
    oppad.Init(&tbufPool);
    oppad.SetMatrix(aGlobalOrg, aGlobal, orgN, orgN, strideN);
    oppad.PadCustomWork(M, blockNum << 1, blockIdx);
    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    GTRF2Solver<float> getf2solver;
    getf2solver.Init(&tbufPool, aGlobal, workGlobal, wGlobal, gather1, gather2, M, orgM, strideN, blockN, workM, tileM);
    getf2solver.Process(0, 0, M, min(orgN, blockN));

    SwapRows<float> opswap;
    if (blockIdx == 8) opswap.Init(&tbufPool, aGlobal);

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
        if (blockIdx == 8)
        {
            for (int j = offset - blockN; j < offset; ++j)
            {
                int p = wGlobal.GetValue(j);
                opswap.Process(offset + j * strideN, offset + p * strideN, N - offset);
                opswap.Process(j * strideN, p * strideN, offset - blockN);
            }
        }
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
        CrossCoreWaitFlag(0x2);
        int trsmTotalN = min((i & -i) * blockN, t - offset);
        int trsmM = (i & -i) * blockN;
        int trsmBlocks = min(blockNum - 4, (trsmTotalN + 15) / 16);
        int trsmBlockN = ((trsmTotalN + trsmBlocks - 1) / trsmBlocks + 15) / 16 * 16;
        int trsmOffsetN = trsmBlockN * (coreIdx - 4);
        int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
        if (coreIdx >= 4 && trsmOffsetN < trsmTotalN)
        {
            solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                             aGlobal[trsmOffsetN + offset + (offset - trsmM) * strideN],
                             aGlobal[offset - trsmM + (offset - trsmM) * strideN]);
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
        }
        CrossCoreWaitFlag(0x5);
        getf2solver.Process(offset, offset, M - offset, min(orgN - offset, blockN));
        if (coreIdx >= 4 && blockIdx & 1 && trsmOffsetN < trsmTotalN)
        {
            opng.SetMatrix(aGlobal[trsmOffsetN + offset + (offset - trsmM) * strideN], strideN);
            opng.Process(trsmM, realBlockN);
        }
        if (blockIdx >= 8 && M - offset > 512)
        {
            CrossCoreWaitFlag(0x6);
            CrossCoreWaitFlag(0x6);
        }
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wGlobal[t - blockN]);
    if (blockIdx == 8)
    {
        for (int j = t - blockN; j < min(orgM, orgN); ++j)
        {
            int p = wGlobal.GetValue(j);
            opswap.Process(t + j * strideN, t + p * strideN, N - t);
            opswap.Process(j * strideN, p * strideN, t - blockN);
        }
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);

    if (N > M)
    {
        int trsmTotalN = N - M;
        int trsmM = M;
        int trsmBlocks = min(blockNum - 4, (trsmTotalN + 15) / 16);
        int trsmBlockN = ((trsmTotalN + trsmBlocks - 1) / trsmBlocks + 15) / 16 * 16;
        int trsmOffsetN = trsmBlockN * (coreIdx - 4);
        int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
        if (coreIdx >= 4 && trsmOffsetN < trsmTotalN)
        {
            solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM, aGlobal[M + trsmOffsetN], aGlobal[0]);
        }
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
        CrossCoreWaitFlag(0x2);
        if (coreIdx >= 4 && blockIdx & 1 && trsmOffsetN < trsmTotalN)
        {
            opng.SetMatrix(aGlobal[M + trsmOffsetN], strideN);
            opng.Process(trsmM, realBlockN);
        }
    }

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
    tbufPool.Reset();
    oppad.Init(&tbufPool);
    oppad.SetMatrix(aGlobal, aGlobalOrg, orgN, strideN, orgN);
    oppad.PadCustomWork(orgM, blockNum << 1, blockIdx);

#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    opmm.Init(&pipe);
    SetAtomicAdd<float>();
    for (int i = 1, offset = blockN; i < cnt; ++i, offset += blockN)
    {
        int trsmTotalN = min((i & -i) * blockN, t - offset);
        int trsmM = (i & -i) * blockN;
        int trsmBlocks = min(blockNum - 4, (trsmTotalN + 15) / 16);
        int trsmBlockN = ((trsmTotalN + trsmBlocks - 1) / trsmBlocks + 15) / 16 * 16;
        int trsmOffsetN = trsmBlockN * (blockIdx - 4);
        if (blockIdx >= 4 && trsmOffsetN < trsmTotalN)
        {
            int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
            solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM,
                             aGlobal[trsmOffsetN + offset + (offset - trsmM) * strideN],
                             aGlobal[offset - trsmM + (offset - trsmM) * strideN]);
            CrossCoreWaitFlag(0x3);
            opmm.SetMatrix(aGlobal[offset - trsmM + offset * strideN],
                           aGlobal[trsmOffsetN + offset + (offset - trsmM) * strideN],
                           aGlobal[trsmOffsetN + offset + offset * strideN], strideN, strideN);
            opmm.Process(0, 0, 0, M - offset, realBlockN, trsmM);
        }
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x4);
        CrossCoreWaitFlag(0x4);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x5);
    }
    if (N > M)
    {
        int trsmTotalN = N - M;
        int trsmM = M;
        int trsmBlocks = min(blockNum - 4, (trsmTotalN + 15) / 16);
        int trsmBlockN = ((trsmTotalN + trsmBlocks - 1) / trsmBlocks + 15) / 16 * 16;
        int trsmOffsetN = trsmBlockN * (blockIdx - 4);
        int realBlockN = min(trsmTotalN - trsmOffsetN, trsmBlockN);
        if (blockIdx >= 4 && trsmOffsetN < trsmTotalN)
        {
            solveTrsm<float>(optrsm, opmm, trsmM, realBlockN, strideN, blockM, aGlobal[M + trsmOffsetN], aGlobal[0]);
        }
    }
#endif
}

__global__ __aicore__ void sgetrf_kernel(GM_ADDR sync, int orgM, int orgN, int blockN, int tileM, GM_ADDR A_org,
                                         GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm,
                                         GM_ADDR gather2_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    custom_lu(orgM, orgN, blockN, tileM, A_org, A_work, W, work_gm, gather1_gm, gather2_gm);
}

void sgetrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockN, int tileM, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR W,
                      GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm, uint32_t numBlocks, void *stream)
{
    sgetrf_kernel<<<numBlocks, nullptr, stream>>>(sync, orgM, orgN, blockN, tileM, A_org, A_work, W, work_gm,
                                                  gather1_gm, gather2_gm);
}
#endif  // SGETRF_MIX_H
