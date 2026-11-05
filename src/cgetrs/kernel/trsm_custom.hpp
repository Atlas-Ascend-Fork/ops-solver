/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "../../utils/getrs_defs.h"
#include "gemm.hpp"
#include "kernel_operator.h"
#include "trsm_column_ops.hpp"
#define L2M_SIZE 4096  // L2切分M轴大小
#define L2N_SIZE 8192  // L2切分N轴大小

#include "../../utils/getrs_trsm_helpers.hpp"

using namespace AscendC;
using namespace matmul;

template <typename T>
class SolveTrsm : public TrsmPanelBase<T>
{
   public:
    __aicore__ inline SolveTrsm() {}
    using TrsmPanelBase<T>::aGlobalReal;
    using TrsmPanelBase<T>::aGlobalImag;
    using TrsmPanelBase<T>::lGlobalReal;
    using TrsmPanelBase<T>::lGlobalImag;
    using TrsmPanelBase<T>::xGlobalReal;
    using TrsmPanelBase<T>::xGlobalImag;
    using TrsmPanelBase<T>::xGlobalImagNeg;
    using TrsmPanelBase<T>::triangularQueue;
    using TrsmPanelBase<T>::matAQueue;
    using TrsmPanelBase<T>::calcBuf;
    using TrsmPanelBase<T>::elementsPerBlock;
    using TrsmPanelBase<T>::m;
    using TrsmPanelBase<T>::resizedM;
    using TrsmPanelBase<T>::n;
    using TrsmPanelBase<T>::blockM;
    using TrsmPanelBase<T>::blockN;
    using TrsmPanelBase<T>::trans;
    using TrsmPanelBase<T>::unit;
    using TrsmPanelBase<T>::CopyInPanel;
    using TrsmPanelBase<T>::ScaleDiagRow;

    // 列方向消元更新（下三角：按 rowIdx 正向取邻行）
    __aicore__ inline void ApplyColumnUpdates(int rowIdx, LocalTensor<T> &triangularLocalReal,
                                              LocalTensor<T> &triangularLocalImag, LocalTensor<T> &matALocalReal,
                                              LocalTensor<T> &matALocalImag, LocalTensor<T> &matALocalImagNeg,
                                              LocalTensor<T> &tmpLocalReal, LocalTensor<T> &tmpLocalImag,
                                              LocalTensor<T> &tmpLocalImagNeg)
    {
        for (int i = 0; i < blockN; i += 64)
        {
            if (trans == 0)
            {
                MulAddDst(matALocalReal[rowIdx * blockN + i], matALocalReal[i], tmpLocalReal, min(64, blockN - i),
                          rowIdx, {1, 1, 0, 0, static_cast<uint8_t>(blockN / 8), 1});
                MulAddDst(matALocalReal[rowIdx * blockN + i], matALocalImag[i], tmpLocalImagNeg, min(64, blockN - i),
                          rowIdx, {1, 1, 0, 0, static_cast<uint8_t>(blockN / 8), 1});
                MulAddDst(matALocalImag[rowIdx * blockN + i], matALocalReal[i], tmpLocalImag, min(64, blockN - i),
                          rowIdx, {1, 1, 0, 0, static_cast<uint8_t>(blockN / 8), 1});
                MulAddDst(matALocalImag[rowIdx * blockN + i], matALocalImag[i], tmpLocalReal, min(64, blockN - i),
                          rowIdx, {1, 1, 0, 0, static_cast<uint8_t>(blockN / 8), 1});
            }
            else
            {
                MulAddDst(matALocalReal[(rowIdx + 1) * blockN + i], matALocalReal[rowIdx * blockN + i],
                          tmpLocalReal[(rowIdx + 1) * elementsPerBlock], min(64, blockN - i), blockM - 1 - rowIdx,
                          {1, 1, 0, static_cast<uint8_t>(blockN / 8), 0, 1});
                MulAddDst(matALocalReal[(rowIdx + 1) * blockN + i], matALocalImag[rowIdx * blockN + i],
                          tmpLocalImagNeg[(rowIdx + 1) * elementsPerBlock], min(64, blockN - i), blockM - 1 - rowIdx,
                          {1, 1, 0, static_cast<uint8_t>(blockN / 8), 0, 1});
                MulAddDst(matALocalImag[(rowIdx + 1) * blockN + i], matALocalReal[rowIdx * blockN + i],
                          tmpLocalImag[(rowIdx + 1) * elementsPerBlock], min(64, blockN - i), blockM - 1 - rowIdx,
                          {1, 1, 0, static_cast<uint8_t>(blockN / 8), 0, 1});
                MulAddDst(matALocalImag[(rowIdx + 1) * blockN + i], matALocalImag[rowIdx * blockN + i],
                          tmpLocalReal[(rowIdx + 1) * elementsPerBlock], min(64, blockN - i), blockM - 1 - rowIdx,
                          {1, 1, 0, static_cast<uint8_t>(blockN / 8), 0, 1});
            }
        }
    }

    __aicore__ inline void Process(int offsetM)
    {
        uint64_t matABlockOffset = offsetM * n;
        uint64_t matLBlockOffset = offsetM * resizedM + offsetM;

        CopyInPanel(matABlockOffset, matLBlockOffset);

        // Compute
        LocalTensor<T> triangularLocalReal = triangularQueue.template DeQue<T>();
        LocalTensor<T> triangularLocalImag = triangularQueue.template DeQue<T>();
        LocalTensor<T> matALocalReal = matAQueue.template DeQue<T>();
        LocalTensor<T> matALocalImag = matAQueue.template DeQue<T>();
        LocalTensor<T> matALocalImagNeg = matAQueue.template AllocTensor<T>();
        LocalTensor<T> tmpLocalReal = calcBuf.template Get<T>();
        auto tmpLocalImag = tmpLocalReal[blockM * elementsPerBlock];
        auto tmpLocalImagNeg = tmpLocalImag[blockM * elementsPerBlock];

        for (int rowIdx = 0; rowIdx < blockM; ++rowIdx)
        {
            Brcb(tmpLocalReal, triangularLocalReal[rowIdx * blockM], blockM / 8, {1, 8});
            Brcb(tmpLocalImag, triangularLocalImag[rowIdx * blockM], blockM / 8, {1, 8});
            Muls(tmpLocalImagNeg, tmpLocalImag, T(-1), blockM * elementsPerBlock);
            ScaleDiagRow(rowIdx, triangularLocalReal, triangularLocalImag, matALocalReal, matALocalImag,
                         matALocalImagNeg);
            ApplyColumnUpdates(rowIdx, triangularLocalReal, triangularLocalImag, matALocalReal, matALocalImag,
                               matALocalImagNeg, tmpLocalReal, tmpLocalImag, tmpLocalImagNeg);
        }

        Muls(matALocalImagNeg, matALocalImag, -1.f, blockM * blockN);

        triangularQueue.FreeTensor(triangularLocalReal);
        triangularQueue.FreeTensor(triangularLocalImag);

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        // CopyOut
        DataCopyParams copyParamsMatOut{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                        0, static_cast<uint16_t>((n - blockN) / elementsPerBlock)};
        DataCopy(xGlobalReal[matABlockOffset], matALocalReal, copyParamsMatOut);
        DataCopy(xGlobalImag[matABlockOffset], matALocalImag, copyParamsMatOut);
        DataCopy(xGlobalImagNeg[matABlockOffset], matALocalImagNeg, copyParamsMatOut);
        matAQueue.FreeTensor(matALocalReal);
        matAQueue.FreeTensor(matALocalImag);
        matAQueue.FreeTensor(matALocalImagNeg);
    }
};

__aicore__ inline void cgetrs_trsm_vec(TBufPool<TPosition::VECCALC, 16> &pipe, GlobalTensor<float> &aGlobal,
                                       GlobalTensor<float> &lGlobalReal, GlobalTensor<float> &lGlobalImag,
                                       GM_ADDR A_work, int trsmOffsetN, int realBlockN, int blockIdx, int m,
                                       int strideM, int strideN, int blockM, int totalBlocks, int trans, int unit,
                                       int resizedM)
{
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work + trsmOffsetN * sizeof(float)));
    auto aGlobalReal = aGlobal;
    auto aGlobalImag = aGlobal[resizedM * strideN];
    auto aGlobalImagNeg = aGlobal[resizedM * strideN * 2];

    int trsmWork = blockIdx & 1
                       ? (realBlockN / 2 + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32
                       : realBlockN / 2 / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    SolveTrsm<float> optrsm;
    auto trsmAGlobalReal = aGlobalReal[(blockIdx & 1) * (realBlockN / 2 / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32)];
    auto trsmAGlobalImag = aGlobalImag[(blockIdx & 1) * (realBlockN / 2 / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32)];
    auto trsmAGlobalImagNeg =
        aGlobalImagNeg[(blockIdx & 1) * (realBlockN / 2 / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32)];
    optrsm.Init(&pipe, trsmAGlobalReal, trsmAGlobalImag, lGlobalReal, lGlobalImag, trsmAGlobalReal, trsmAGlobalImag,
                trsmAGlobalImagNeg, m, strideM, strideN, blockM, trsmWork, trans, unit);

    getrs_trsm_vec_loop(pipe, optrsm, totalBlocks, blockM);
}

__aicore__ inline void custom_trsm(TBufPool<TPosition::VECCALC, 16> &pipe, CMatmulCustom<float> &opmm, int orgM,
                                   int orgN, int blockM, int trans, int unit, GM_ADDR A_work, GM_ADDR L_work)
{
    int L2M = L2M_SIZE;
    int L2N = L2N_SIZE;
    int L2K = L2M;

    int m = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int n = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideM = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideM & -strideM) >= ADDITIONAL_PAD_SIZE) strideM += CACHE_LINE_SIZE_FP32;
    int strideN = (orgN + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;
    int resizedM = (orgM + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;

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
    auto lGlobalReal = lGlobal;
    auto lGlobalImag = lGlobal[resizedM * strideM];

    int trsmOffsetN = 0;
    int realBlockN = 0;
    SplitTrsmColumns(n, blockNum, coreIdx, trsmOffsetN, realBlockN);

#ifdef __DAV_C220_VEC__
    cgetrs_trsm_vec(pipe, aGlobal, lGlobalReal, lGlobalImag, A_work, trsmOffsetN, realBlockN, blockIdx, m, strideM,
                    strideN, blockM, totalBlocks, trans, unit, resizedM);

#endif

#ifdef __DAV_C220_CUBE__

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    auto aGlobalReal = aGlobal;
    auto aGlobalImag = aGlobal[resizedM * strideN];
    auto aGlobalImagNeg = aGlobal[resizedM * strideN * 2];

    opmm.SetMatrix(lGlobalReal, lGlobalImag, aGlobalReal, aGlobalImag, aGlobalImagNeg, aGlobalReal, aGlobalImag,
                   strideN, strideM);
    SetAtomicAdd<float>();
    getrs_trsm_cube(opmm, m, n, blockM, L2M, L2N, L2K, strideM, strideN, blockNum, blockIdx, totalBlocks, baseM,
                    trsmOffsetN, realBlockN, trans, totalGemmBlock, lastGemmBoundary, nextGemmBoundary);
#endif
}
