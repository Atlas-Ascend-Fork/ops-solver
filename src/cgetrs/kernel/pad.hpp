/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>

using namespace AscendC;
using namespace matmul;

#define VEC_CUBE_FLAGID 0x0
#define CUBE_VEC_FLAGID 0x1
#define ALL_VEC_FLAGID 0x2
#define ALL_CUBE_FLAGID 0x3
#define CACHE_LINE_SIZE_FP32 128 // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096 // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率
#define ceil(x, y) (((x) + (y) - 1) / (y) * (y))

template <typename T>
class SplitRealImag {
    static const int TILE_LENGTH = 8192;
    GlobalTensor<T> srcGlobal, dstGlobalReal, dstGlobalImag;
    int N;
    int srcN;
    int dstN;
    int padN;
    int alignedN;
    int linesPerIter;
    int elementsPerBlock;
    int trans;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueueReal, outQueueImag;
    LocalTensor<T> realLocal, imagLocal;
public:
    __aicore__ inline SplitRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *tbufPool, int trans) {
        tbufPool->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        tbufPool->InitBuffer(outQueueReal, 2, TILE_LENGTH * sizeof(T));
        tbufPool->InitBuffer(outQueueImag, 2, TILE_LENGTH * sizeof(T));
        elementsPerBlock = 32 / sizeof(T);

        this->trans = trans;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobalReal, GlobalTensor<T> dstGlobalImag, int N, int srcN, int dstN) {
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
    __aicore__ inline void SplitRealImagWork(int M, int p, int id) {
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN * 2, (k * id + min(id, r)) * dstN, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines) {
        int loopCount = lines / linesPerIter;
        srcOffset += loopCount * linesPerIter * srcN * 2;
        dstOffset += loopCount * linesPerIter * dstN;
        int tailLines = lines - linesPerIter * loopCount;
        if (tailLines) {
            if (N << 1 <= TILE_LENGTH) {
                CopyIn(srcOffset, tailLines / 2);
                Compute(0, 0, tailLines / 2 * alignedN * 2);
                CopyIn(srcOffset + tailLines / 2 * srcN * 2, (tailLines + 1) / 2);
                Compute(1, tailLines / 2 * alignedN, (tailLines + 1) / 2 * alignedN * 2);
            } else {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, tailLines);
        }
        srcOffset -= linesPerIter * srcN * 2;
        dstOffset -= linesPerIter * dstN;
        for (int i = 0; i < loopCount; ++i) {
            if (N << 1 <= TILE_LENGTH) {
                CopyIn(srcOffset, linesPerIter / 2);
                Compute(0, 0, linesPerIter / 2 * alignedN * 2);
                CopyIn(srcOffset + linesPerIter / 2 * srcN * 2, (linesPerIter + 1) / 2);
                Compute(1, linesPerIter / 2 * alignedN, (linesPerIter + 1) / 2 * alignedN * 2);
            } else {
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
    __aicore__ inline void CopyIn(int offset, int lines) {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        if (N << 1 <= TILE_LENGTH) {
            DataCopyExtParams intriParams {
                static_cast<uint16_t>(lines),
                static_cast<uint16_t>(N * 2 * sizeof(T)),
                static_cast<uint16_t>((srcN * 2 - N * 2) * sizeof(T)),
                static_cast<uint16_t>(N % elementsPerBlock && N % elementsPerBlock <= elementsPerBlock / 2),
                0
            };
            DataCopyPadExtParams<T> padParams {true, 0, static_cast<uint8_t>(padN), 0};
            DataCopyPad(srcLocal, srcGlobal[offset], intriParams, padParams);
        } else {
            DataCopy(srcLocal, srcGlobal[offset], lines);
        }
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int progress, int calcedLength, int length) {
        if (!progress) {
            realLocal = outQueueReal.AllocTensor<T>();
            imagLocal = outQueueImag.AllocTensor<T>();
        }
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint64_t rsvdCnt; 
        GatherMask(realLocal[calcedLength], srcLocal, 1, false, 0, {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        GatherMask(imagLocal[calcedLength], srcLocal, 2, false, 0, {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        if (trans == 2) {
            Muls(imagLocal[calcedLength], imagLocal[calcedLength], -1.f, length / 2);
        }
        inQueue.FreeTensor(srcLocal);
        if (progress) {
            outQueueReal.EnQue(realLocal);
            outQueueImag.EnQue(imagLocal);
        }
    }
    __aicore__ inline void CopyOut(int offset, int lines) {
        LocalTensor<T> realLocal = outQueueReal.DeQue<T>();
        LocalTensor<T> imagLocal = outQueueImag.DeQue<T>();
        DataCopyExtParams intriParams{
            static_cast<uint16_t>(lines),
            static_cast<uint16_t>(N * sizeof(T)),
            0,
            static_cast<uint16_t>((dstN - N) * sizeof(T)),
            0
        };
        DataCopyPad(dstGlobalReal[offset], realLocal, intriParams);
        DataCopyPad(dstGlobalImag[offset], imagLocal, intriParams);
        outQueueReal.FreeTensor(realLocal);
        outQueueImag.FreeTensor(imagLocal);
    }
};

template <typename T>
class MergeRealImag {
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
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *tbufPool, GlobalTensor<uint32_t> aGather) {
        tbufPool->InitBuffer(inQueue, 2, TILE_LENGTH * sizeof(T));
        tbufPool->InitBuffer(outQueue, 2, TILE_LENGTH * sizeof(T));
        tbufPool->InitBuffer(gatherQueue, 1, TILE_LENGTH * sizeof(uint32_t));
        elementsPerBlock = 32 / sizeof(T);
        this->aGather = aGather;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal, GlobalTensor<T> aGlobalImag, int N, int srcN, int dstN) {
        this->aOrgGlobal = aOrgGlobal;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->N = N;
        this->alignedN = ceil(N, elementsPerBlock);
        this->srcN = srcN;
        this->dstN = dstN;
        linesPerIter = max(1, TILE_LENGTH / ceil(dstN, 16) / 2);
    }
    __aicore__ inline void MergeRealImagWork(int M, int p, int id) {
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN, (k * id + min(id, r)) * dstN * 2, k + (id < r));
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines) {
        LoadGather();
        int loopCount = lines / linesPerIter;
        for (int i = 0; i < loopCount; ++i) {
            CopyIn(srcOffset, linesPerIter, alignedN);
            Compute(linesPerIter * alignedN * 2);
            CopyOut(dstOffset, linesPerIter, N);
            srcOffset += linesPerIter * srcN;
            dstOffset += linesPerIter * dstN * 2;
        }
        lines %= linesPerIter;
        if (!lines) {
            gatherQueue.FreeTensor(gatherLocal);
            return;
        }
        CopyIn(srcOffset, lines, alignedN);
        Compute(lines * alignedN * 2);
        CopyOut(dstOffset, lines, N);
        gatherQueue.FreeTensor(gatherLocal);
    }
    __aicore__ inline void LoadGather() {
        gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, linesPerIter * alignedN * 2);
        gatherQueue.EnQue(gatherLocal);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }
    __aicore__ inline void CopyIn(int offset, int blockM, int blockN) {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();

        DataCopyParams copyInParams{
            static_cast<uint16_t>(blockM),
            static_cast<uint16_t>(blockN / elementsPerBlock),
            static_cast<uint16_t>((srcN - blockN) / elementsPerBlock),
            0
        };
        DataCopy(srcLocal, aGlobalReal[offset], copyInParams);
        DataCopy(srcLocal[TILE_LENGTH / 2], aGlobalImag[offset], copyInParams);

        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int length) {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut(int offset, int blockM, int blockN) {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();

        DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(blockM),
            static_cast<uint16_t>(blockN * 2 * sizeof(T)),
            static_cast<uint16_t>((alignedN * 2 - blockN * 2) / elementsPerBlock),
            static_cast<uint16_t>((dstN * 2 - blockN * 2) * sizeof(T)),
            0
        };
        DataCopyPad(aOrgGlobal[offset], dstLocal, copyOutParams);

        outQueue.FreeTensor(dstLocal);
    }
};

__aicore__ inline void pad(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, int srcN, int trans, GM_ADDR A_org, GM_ADDR A_work) {
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int resizedM = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));

    SplitRealImag<float> opsp;
    opsp.Init(&tbufPool, trans);
    opsp.SetMatrix(aGlobalOrg, aGlobal, aGlobal[resizedM * strideN], orgN, srcN, strideN);
    opsp.SplitRealImagWork(orgM, blockNum << 1, blockIdx);
    tbufPool.Reset();

    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);
}

__aicore__ inline void restore(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, int dstN, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR gather_gm) {
    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);

    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int resizedM = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;
    GlobalTensor<uint32_t> gather;

    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    gather.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_gm));

    MergeRealImag<float> opmg;
    static constexpr int TILE_LENGTH = 4096;
    opmg.Init(&tbufPool, gather);
    opmg.SetMatrix(aGlobalOrg, aGlobal, aGlobal[resizedM * strideN], min(orgN, TILE_LENGTH), strideN, dstN);
    opmg.MergeRealImagWork(orgM, blockNum << 1, blockIdx);
    if (orgN > TILE_LENGTH) {
        opmg.SetMatrix(aGlobalOrg[TILE_LENGTH * 2], aGlobal[TILE_LENGTH], aGlobal[TILE_LENGTH + resizedM * strideN], orgN - TILE_LENGTH, strideN, dstN);
        opmg.MergeRealImagWork(orgM, blockNum << 1, blockIdx);
    }
    tbufPool.Reset();
}
