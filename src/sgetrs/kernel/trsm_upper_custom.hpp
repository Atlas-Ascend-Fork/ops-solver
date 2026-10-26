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
#include "gemm.hpp"

#define VEC_CUBE_FLAGID 0x0
#define CUBE_VEC_FLAGID 0x1
#define ALL_VEC_FLAGID 0x2
#define ALL_CUBE_FLAGID 0x3
#define CACHE_LINE_SIZE_FP32 128 // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096 // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率
#define L2M_SIZE 4096 // L2切分M轴大小
#define L2N_SIZE 8192 // L2切分N轴大小

#include "../../utils/getrs_trsm_helpers.hpp"

using namespace AscendC;
using namespace matmul;

template<typename T>
class SolveTrsmUpper {
    GlobalTensor<T> aGlobal, lGlobal, xGlobal;
    TQue<QuePosition::VECIN, 1> triangularQueue;
    TQue<QuePosition::VECIN, 1> matAQueue;
    TQue<QuePosition::VECOUT, 1> augQueue;
    TBuf<TPosition::VECCALC> calcBuf;
    uint64_t elementsPerBlock;
    int m;
    int resizedM;
    int n;
    int blockM;
    int blockN;
    int trans;
    int unit;
public:
    __aicore__ inline SolveTrsmUpper() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> &aGlobal, GlobalTensor<T> &lGlobal, GlobalTensor<T> &xGlobal, int M, int resizedM, int N, int blockM, int blockN, int trans, int unit) {
        this->m = M;
        this->resizedM = resizedM;
        this->n = N;
        this->blockM = blockM;
        this->blockN = blockN;
        
        this->trans = trans;
        this->unit = unit;

        this->aGlobal = aGlobal;
        this->lGlobal = lGlobal;
        this->xGlobal = xGlobal;

        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(triangularQueue, 1, blockM * blockM * sizeof(T));
        pipe->InitBuffer(matAQueue, 1, blockM * blockN * sizeof(T));
        pipe->InitBuffer(calcBuf, blockM * elementsPerBlock * sizeof(T));
    }
    __aicore__ inline void Process(int offsetM, int mask) {
        uint64_t matABlockOffset = offsetM * n;
        uint64_t matLBlockOffset = offsetM * resizedM + offsetM;

        LocalTensor<T> triangularLocal = triangularQueue.AllocTensor<T>();
        LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();

        DataCopyParams copyParamsMatL {
            static_cast<uint16_t>(blockM),
            static_cast<uint16_t>(blockM / elementsPerBlock),
            static_cast<uint16_t>((resizedM - blockM) / elementsPerBlock),
            0
        };
        DataCopy(triangularLocal, lGlobal[matLBlockOffset], copyParamsMatL);

        DataCopyParams copyParamsMatA {
            static_cast<uint16_t>(blockM), 
            static_cast<uint16_t>(blockN / elementsPerBlock), 
            static_cast<uint16_t>((n - blockN) / elementsPerBlock), 
            0
        };
        DataCopy(matALocal, aGlobal[matABlockOffset], copyParamsMatA);

        triangularQueue.EnQue(triangularLocal);
        matAQueue.EnQue(matALocal);
        triangularLocal = triangularQueue.DeQue<T>();
        matALocal = matAQueue.DeQue<T>();
        LocalTensor<T> tmpLocal = calcBuf.Get<T>();

        for (int rowIdx = blockM - 1; rowIdx >= 0; --rowIdx) {
            if (rowIdx >= mask) {
                Duplicate(matALocal[blockN * rowIdx], 0.f, blockN);
                continue;
            }
            Brcb(tmpLocal, triangularLocal[rowIdx * blockM], blockM / 8, {1, 8});
            if (trans == 1) {
                T tempScalar = unit ? -1.f : -1.f / triangularLocal.GetValue(rowIdx * blockM + rowIdx);
                Muls(matALocal[blockN * rowIdx], matALocal[blockN * rowIdx], tempScalar, blockN);
            }
            for (int i = 0; i < blockN; i += 64) {
                if (trans == 0) MulAddDst(matALocal[rowIdx * blockN + i], matALocal[(rowIdx + 1) * blockN + i], tmpLocal[(rowIdx + 1) * elementsPerBlock], min(64, blockN - i), blockM - 1 - rowIdx, {1, 1, 0, 0, static_cast<uint8_t>(blockN / 8), 1});
                else if (trans == 1) MulAddDst(matALocal[i], matALocal[rowIdx * blockN + i], tmpLocal, min(64, blockN - i), rowIdx, {1, 1, 0, static_cast<uint8_t>(blockN / 8), 0, 1});
            }
            if (trans == 0) {
                T tempScalar = unit ? -1.f : -1.f / triangularLocal.GetValue(rowIdx * blockM + rowIdx);
                Muls(matALocal[blockN * rowIdx], matALocal[blockN * rowIdx], tempScalar, blockN);
            }
        }

        triangularQueue.FreeTensor(triangularLocal);

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopyParams copyParamsMatOut {
            static_cast<uint16_t>(blockM), 
            static_cast<uint16_t>(blockN / elementsPerBlock), 
            0, 
            static_cast<uint16_t>((n - blockN) / elementsPerBlock)
        };
        DataCopy(xGlobal[matABlockOffset], matALocal, copyParamsMatOut);
        matAQueue.FreeTensor(matALocal);
    }
};

__aicore__ inline void custom_trsm_upper_vec(TBufPool<TPosition::VECCALC, 16> &pipe, GlobalTensor<float> &aGlobal,
    GlobalTensor<float> &lGlobal, GM_ADDR A_work, int trsmOffsetN, int realBlockN, int blockIdx, int m, int strideM,
    int strideN, int blockM, int totalBlocks, int trans, int unit, int orgM) {
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work + trsmOffsetN * sizeof(float)));
    int trsmWork = blockIdx & 1 ? (realBlockN / 2 + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32 : realBlockN / 2 / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    SolveTrsmUpper<float> optrsm;
    auto trsmAGlobal = aGlobal[(blockIdx & 1) * (realBlockN / 2 / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32)];
    optrsm.Init(&pipe, trsmAGlobal, lGlobal, trsmAGlobal, m, strideM, strideN, blockM, trsmWork, trans, unit);

    getrs_trsm_upper_vec_loop(pipe, optrsm, totalBlocks, blockM, m, orgM);
}

__aicore__ inline void custom_trsm_upper(TBufPool<TPosition::VECCALC, 16> &pipe, MatmulCustom<float> &opmm, int orgM, int orgN, int blockM, int trans, int unit, GM_ADDR A_work, GM_ADDR L_work) {
    int L2M = L2M_SIZE;
    int L2N = L2N_SIZE;
    int L2K = L2M;
    
    int m = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int n = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideM = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideM & -strideM) >= ADDITIONAL_PAD_SIZE) strideM += CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    int baseM = CACHE_LINE_SIZE_FP32;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int totalBlocks = m / blockM;
    int totalGemmBlock = min(m, L2M) / baseM;
    int lastGemmBoundary = 0;
    int nextGemmBoundary = totalGemmBlock * baseM;
    int coreIdx = blockIdx;
    #ifdef __DAV_C220_VEC__
        coreIdx >>= 1;
    #endif

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> lGlobal;

    lGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(L_work));

    int blockN = (n + blockNum - 1) / blockNum / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int totalNum = n / CACHE_LINE_SIZE_FP32;
    int averageNum = totalNum / blockNum;
    int remainderNum = totalNum % blockNum;
    int startIdx = averageNum * coreIdx + min(coreIdx, remainderNum);
    int calcNum = averageNum + (coreIdx < remainderNum);

    int trsmOffsetN = startIdx * CACHE_LINE_SIZE_FP32;
    int realBlockN = calcNum * CACHE_LINE_SIZE_FP32;

    #ifdef __DAV_C220_VEC__
        custom_trsm_upper_vec(pipe, aGlobal, lGlobal, A_work, trsmOffsetN, realBlockN, blockIdx, m, strideM,
            strideN, blockM, totalBlocks, trans, unit, orgM);

    #endif

    #ifdef  __DAV_C220_CUBE__

        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
        opmm.SetMatrix(lGlobal, aGlobal, aGlobal, strideN, strideM);
        SetAtomicAdd<float>();
        getrs_trsm_upper_cube(opmm, m, n, blockM, L2M, L2N, L2K, strideM, strideN, blockNum, blockIdx,
            totalBlocks, baseM, trsmOffsetN, realBlockN, trans, totalGemmBlock, lastGemmBoundary, nextGemmBoundary);
    #endif
}
