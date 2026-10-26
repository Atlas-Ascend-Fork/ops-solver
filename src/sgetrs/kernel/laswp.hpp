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
class SwapRows {
    static constexpr int TILE_LENGTH = 8192;
    GlobalTensor<T> aGlobal;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind2;
    int elementsPerBlock;
public:
    __aicore__ inline SwapRows() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> aGlobal) {
        pipe->InitBuffer(queBind1, 1, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(queBind2, 1, TILE_LENGTH * sizeof(T));
        this->aGlobal = aGlobal;
        this->elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int N) {
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
};

template<typename T>
class laswpCustom {
    static constexpr int TILE_LENGTH = 8192;
    GlobalTensor<T> scrGlobal;
    GlobalTensor<T> dstGlobal;
    GlobalTensor<uint32_t> pGlobal;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    int elementsPerBlock;
public:
    __aicore__ inline laswpCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> scrGlobal, GlobalTensor<T> dstGlobal, GlobalTensor<uint32_t> pGlobal) {
        pipe->InitBuffer(queBind, 2, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueDst, 1, TILE_LENGTH * sizeof(T));
        this->scrGlobal = scrGlobal;
        this->dstGlobal = dstGlobal;
        this->pGlobal = pGlobal;
        this->elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void Process(int M, int N, int srcN, int dstN, int inv, int blockNum, int blockIdx) {
        int padN = (elementsPerBlock - (N % elementsPerBlock)) % elementsPerBlock;
        int averageNum = M / blockNum;
        int remainderNum = M % blockNum;
        int startIdx = averageNum * blockIdx + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pGlobal[startIdx]);
        for (int i = startIdx; i < startIdx + calcNum; ++i) {
            if (i % 16 == 0) DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pGlobal[i]);
            int p = pGlobal.GetValue(i);
            LocalTensor<T> bindLocal = queBind.AllocTensor<T>();
            DataCopyExtParams copyInParams {
                1,
                static_cast<uint16_t>(N * sizeof(T)),
                static_cast<uint16_t>((srcN - N) * sizeof(T)),
                0,
                0
            };
            DataCopyPadExtParams<T> padParams {false, 0, static_cast<uint8_t>(padN), 0};
            DataCopyPad(bindLocal, scrGlobal[inv ? i * srcN : p * srcN], copyInParams, padParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<T>();
            DataCopyExtParams copyOutParams{
                1,
                static_cast<uint16_t>(N * sizeof(T)),
                0,
                static_cast<uint16_t>((dstN - N) * sizeof(T)),
                0
            };
            DataCopyPad(dstGlobal[inv ? p * dstN : i * dstN], bindLocal, copyOutParams);
            queBind.FreeTensor(bindLocal);
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
};

// parallen on col
__aicore__ inline void laswp(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, GM_ADDR A, GM_ADDR w) {
    int blockNum = GetBlockNum() * 2;
    int blockIdx = GetBlockIdx();
    int resizedN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    GlobalTensor<float> aGlobal;
    GlobalTensor<uint32_t> wGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(w));
    
    int totalNum = resizedN / CACHE_LINE_SIZE_FP32;
    int averageNum = totalNum / blockNum;
    int remainderNum = totalNum % blockNum;
    int startIdx = averageNum * blockIdx + min(blockIdx, remainderNum);
    int calcNum = averageNum + (blockIdx < remainderNum);

    SwapRows<float> opswp;
    opswp.Init(&tbufPool, aGlobal[startIdx * CACHE_LINE_SIZE_FP32]);

    for (int i = 0; i < orgM; ++i) {
        int p = wGlobal.GetValue(i);
        opswp.Process(i * strideN, p * strideN, calcNum * CACHE_LINE_SIZE_FP32);
    }

    tbufPool.Reset();
    
    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);
}

// parallen on row
__aicore__ inline void laswp(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, int ldb, int inv, GM_ADDR A_org, GM_ADDR A, GM_ADDR w, GM_ADDR p) {
    int blockNum = GetBlockNum() * 2;
    int blockIdx = GetBlockIdx();
    int resizedN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<uint32_t> wGlobal;
    GlobalTensor<uint32_t> pGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
    aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(w));
    pGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(p));
    
    laswpCustom<float> opswp;
    if (inv == 0) opswp.Init(&tbufPool, aGlobalOrg, aGlobal, pGlobal);
    else opswp.Init(&tbufPool, aGlobal, aGlobalOrg, pGlobal);

    if (!blockIdx) {
        opswp.WtoP(wGlobal, pGlobal, orgM);
    }


    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);

    if (inv == 0) opswp.Process(orgM, orgN, ldb, strideN, inv, blockNum, blockIdx);
    else opswp.Process(orgM, orgN, strideN, ldb, inv, blockNum, blockIdx);

    tbufPool.Reset();
    
    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);
}
