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

template<typename T>
class laswpCustom {
    static constexpr int TILE_LENGTH = 8192;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobalReal, dstGlobalImag;
    GlobalTensor<uint32_t> pGlobal;
    TQue<QuePosition::VECIN, 2> inQueueSrc;
    TQue<QuePosition::VECOUT, 2> outQueueDst;
    LocalTensor<T> realLocal, imagLocal;
    LocalTensor<uint32_t> gatherLocal;
    int elementsPerBlock;
public:
    __aicore__ inline laswpCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobalReal, GlobalTensor<T> dstGlobalImag, GlobalTensor<uint32_t> pGlobal, int inv) {
        if (inv == 0) {
            pipe->InitBuffer(inQueueSrc, 2, TILE_LENGTH * sizeof(T));
            pipe->InitBuffer(outQueueDst, 4, TILE_LENGTH * sizeof(T));
        } else {
            pipe->InitBuffer(inQueueSrc, 4, TILE_LENGTH * sizeof(T));
            pipe->InitBuffer(outQueueDst, 2, TILE_LENGTH * sizeof(T));
        }

        this->srcGlobal = srcGlobal;
        this->dstGlobalReal = dstGlobalReal;
        this->dstGlobalImag = dstGlobalImag;
        this->pGlobal = pGlobal;
        this->elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void LoadGather(GlobalTensor<uint32_t> aGather) {
        gatherLocal = inQueueSrc.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, TILE_LENGTH);
        inQueueSrc.EnQue(gatherLocal);
        gatherLocal = inQueueSrc.DeQue<uint32_t>();
    }
    __aicore__ inline void FreeGather() {
        inQueueSrc.FreeTensor(gatherLocal);
    }
    __aicore__ inline void Process(int M, int N, int srcN, int dstN, int inv, int blockNum, int blockIdx) {
        int alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        int padN = (elementsPerBlock - (N * 2 % elementsPerBlock)) % elementsPerBlock;
        int averageNum = M / blockNum;
        int remainderNum = M % blockNum;
        int startIdx = averageNum * blockIdx + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pGlobal[startIdx]);
        for (int i = startIdx; i < startIdx + calcNum; ++i) {
            if (i % 16 == 0) DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pGlobal[i]);
            int p = pGlobal.GetValue(i);

            if (inv == 0) {
                if (N * 2 <= TILE_LENGTH) {
                    CopyIn(p * srcN * 2, alignedN * 2);
                    Compute(1, 1, 0, alignedN * 2);
                } else {
                    CopyIn(p * srcN * 2, TILE_LENGTH);
                    Compute(1, 0, 0, TILE_LENGTH);
                    CopyIn(p * srcN * 2 + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                    Compute(0, 1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
                }
                CopyOut(i * dstN, N);
            } else {
                for (int j = 0; j < N; j += TILE_LENGTH / 2) {
                    int curN = min(N - j, TILE_LENGTH / 2);
                    int alignedCurN = (curN + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
                    CopyIn2(i * dstN + j, alignedCurN);
                    Compute2(alignedCurN * 2);
                    CopyOut2((p * srcN + j) * 2, curN * 2);
                }
            }
        }

    }
    __aicore__ inline void WtoP(GlobalTensor<uint32_t> wGlobal, GlobalTensor<uint32_t> pGlobal, int N) {
        LocalTensor<uint32_t> pLocal = outQueueDst.AllocTensor<uint32_t>();
        for (int i = 0; i < N; ++i) {
            pLocal.SetValue(i, i);
        }

        for (int i = 0; i < N; ++i) {
            int p = wGlobal.GetValue(i);
            if (p != i) {
                uint32_t temp = pLocal.GetValue(p);
                pLocal.SetValue(p, pLocal.GetValue(i));
                pLocal.SetValue(i, temp);
            }
        }

        int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);

        outQueueDst.EnQue(pLocal);
        pLocal = outQueueDst.DeQue<uint32_t>();

        DataCopyExtParams copyOutParams{
            1,
            static_cast<uint16_t>(N * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(pGlobal, pLocal, copyOutParams);
        outQueueDst.FreeTensor(pLocal);
    }
    __aicore__ inline void CopyIn(int offset, int length) {
        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal[offset], length);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int first, int last, int calcedLength, int length) {
        if (first) {
            realLocal = outQueueDst.AllocTensor<T>();
            imagLocal = outQueueDst.AllocTensor<T>();
        }
        LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        uint64_t rsvdCnt; 
        GatherMask(realLocal[calcedLength], srcLocal, 1, false, 0, {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        GatherMask(imagLocal[calcedLength], srcLocal, 2, false, 0, {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        inQueueSrc.FreeTensor(srcLocal);
        if (last) {
            outQueueDst.EnQue(realLocal);
            outQueueDst.EnQue(imagLocal);
        }
    }
    __aicore__ inline void CopyOut(int offset, int N) {
        LocalTensor<T> realLocal = outQueueDst.DeQue<T>();
        LocalTensor<T> imagLocal = outQueueDst.DeQue<T>();
        DataCopyExtParams intriParams{
            static_cast<uint16_t>(1),
            static_cast<uint16_t>(N * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(dstGlobalReal[offset], realLocal, intriParams);
        DataCopyPad(dstGlobalImag[offset], imagLocal, intriParams);
        outQueueDst.FreeTensor(realLocal);
        outQueueDst.FreeTensor(imagLocal);
    }
    __aicore__ inline void CopyIn2(int offset, int length) {
        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopy(srcLocal, dstGlobalReal[offset], length);
        DataCopy(srcLocal[TILE_LENGTH / 2], dstGlobalImag[offset], length);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute2(int length) {
        LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        inQueueSrc.FreeTensor(srcLocal);
        outQueueDst.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut2(int offset, int N) {
        LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        DataCopyExtParams intriParams{
            static_cast<uint16_t>(1),
            static_cast<uint16_t>(N * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(srcGlobal[offset], dstLocal, intriParams);
        outQueueDst.FreeTensor(dstLocal);
    }
};

// parallen on row
__aicore__ inline void laswp(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, int ldb, int inv, GM_ADDR A_org, GM_ADDR A, GM_ADDR w, GM_ADDR p, GM_ADDR gather_gm) {
    int blockNum = GetBlockNum() * 2;
    int blockIdx = GetBlockIdx();
    int resizedM = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int resizedN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<uint32_t> wGlobal;
    GlobalTensor<uint32_t> pGlobal;
    GlobalTensor<uint32_t> gather;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(w));
    pGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(p));
    gather.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_gm));
    
    laswpCustom<float> opswp;
    opswp.Init(&tbufPool, aGlobalOrg, aGlobal, aGlobal[resizedM * strideN], pGlobal, inv);

    if (!blockIdx) {
        opswp.WtoP(wGlobal, pGlobal, orgM);
    }


    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);

    if (inv) {
        opswp.LoadGather(gather);
    }
    opswp.Process(orgM, orgN, ldb, strideN, inv, blockNum, blockIdx);
    if (inv) {
        opswp.FreeGather();
    }

    tbufPool.Reset();
    
    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);
}
