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
 * \file trsm_column_ops.hpp
 * \brief cgetrs trsm 两份实现（trsm_custom 与 trsm_upper_custom）共用的面板载入、对角行缩放
 * 逻辑与列拆分计算（单一来源），消除跨文件重复代码块（!155 CodeCheck 整改）。
 * 注意：本头文件不得依赖包含方的 using 声明，所有 AscendC 符号显式限定命名空间。
 */

#ifndef CGETRS_TRSM_COLUMN_OPS_H
#define CGETRS_TRSM_COLUMN_OPS_H

#include "../../utils/getrs_defs.h"
#include "kernel_operator.h"

// 对角行缩放：unit == 1 时单位对角直接取负；否则按复数标量 (sReal, sImag) 求 1/s 并以
// (tempScalarReal, tempScalarImag) 缩放实虚行，结果回写 matA 对角行。
// 原实现中 if (trans) 与 if (trans == 0) 两个分支体逐字相同且互补覆盖，故此处不再按 trans 分叉。
template <typename T>
__aicore__ inline void TrsmScaleDiagRow(int rowIdx, int blockN, int blockM, int unit,
                                        AscendC::LocalTensor<T> &triangularLocalReal,
                                        AscendC::LocalTensor<T> &triangularLocalImag,
                                        AscendC::LocalTensor<T> &matALocalReal, AscendC::LocalTensor<T> &matALocalImag,
                                        AscendC::LocalTensor<T> &matALocalImagNeg)
{
    if (unit == 1)
    {
        Muls(matALocalReal[blockN * rowIdx], matALocalReal[blockN * rowIdx], -1.f, blockN);
        Muls(matALocalImag[blockN * rowIdx], matALocalImag[blockN * rowIdx], -1.f, blockN);
    }
    else
    {
        T sReal = triangularLocalReal.GetValue(rowIdx * blockM + rowIdx);
        T sImag = triangularLocalImag.GetValue(rowIdx * blockM + rowIdx);
        T tempScalarReal = -sReal / (sReal * sReal + sImag * sImag);
        T tempScalarImag = sImag / (sReal * sReal + sImag * sImag);

        Muls(matALocalImagNeg[blockN * rowIdx], matALocalReal[blockN * rowIdx], tempScalarImag, blockN);
        Axpy(matALocalImagNeg[blockN * rowIdx], matALocalImag[blockN * rowIdx], tempScalarReal, blockN);
        Muls(matALocalReal[blockN * rowIdx], matALocalReal[blockN * rowIdx], tempScalarReal, blockN);
        Axpy(matALocalReal[blockN * rowIdx], matALocalImag[blockN * rowIdx], -tempScalarImag, blockN);

        Muls(matALocalImag[blockN * rowIdx], matALocalImagNeg[blockN * rowIdx], 1.f, blockN);
    }
}

// 按核数将 N 维 cache line 数均分，得出当前核的 trsm 起始列偏移与块宽；
// 核数异常时按零宽处理并直接返回，由调用方既有链路安全退化（G.EXP.22）
__aicore__ inline void SplitTrsmColumns(int n, int blockNum, int coreIdx, int &trsmOffsetN, int &realBlockN)
{
    if (blockNum <= 0)
    {
        trsmOffsetN = 0;
        realBlockN = 0;
        return;
    }
    int totalNum = n / CACHE_LINE_SIZE_FP32;
    int averageNum = totalNum / blockNum;
    int remainderNum = totalNum % blockNum;
    int lo = coreIdx < remainderNum ? coreIdx : remainderNum;
    int startIdx = averageNum * coreIdx + lo;
    int calcNum = averageNum + (coreIdx < remainderNum);
    trsmOffsetN = startIdx * CACHE_LINE_SIZE_FP32;
    realBlockN = calcNum * CACHE_LINE_SIZE_FP32;
}

// trsm 面板处理的公共基类：成员状态、Init、CopyIn 与对角行缩放为两份实现完全一致的部位
template <typename T>
class TrsmPanelBase
{
   protected:
    AscendC::GlobalTensor<T> aGlobalReal, aGlobalImag, lGlobalReal, lGlobalImag, xGlobalReal, xGlobalImag,
        xGlobalImagNeg;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> triangularQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 3> matAQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    uint64_t elementsPerBlock;
    int m;
    int resizedM;
    int n;
    int blockM;
    int blockN;
    int trans;
    int unit;

   public:
    __aicore__ inline TrsmPanelBase() {}
    __aicore__ inline void Init(AscendC::TBufPool<AscendC::TPosition::VECCALC, 16> *pipe,
                                AscendC::GlobalTensor<T> &aGlobalReal, AscendC::GlobalTensor<T> &aGlobalImag,
                                AscendC::GlobalTensor<T> &lGlobalReal, AscendC::GlobalTensor<T> &lGlobalImag,
                                AscendC::GlobalTensor<T> &xGlobalReal, AscendC::GlobalTensor<T> &xGlobalImag,
                                AscendC::GlobalTensor<T> &xGlobalImagNeg, int M, int resizedM, int N, int blockM,
                                int blockN, int trans, int unit)
    {
        this->m = M;
        this->resizedM = resizedM;
        this->n = N;
        this->blockM = blockM;
        this->blockN = blockN;

        this->trans = trans;
        this->unit = unit;

        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->lGlobalReal = lGlobalReal;
        this->lGlobalImag = lGlobalImag;
        this->xGlobalReal = xGlobalReal;
        this->xGlobalImag = xGlobalImag;
        this->xGlobalImagNeg = xGlobalImagNeg;

        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(triangularQueue, 2, blockM * blockM * sizeof(T));
        pipe->InitBuffer(matAQueue, 3, blockM * blockN * sizeof(T));

        pipe->InitBuffer(calcBuf, 3 * blockM * elementsPerBlock * sizeof(T));
    }

    // 从 GM 载入面板的三角矩阵与 A 分块并做队列入队（matALocalImagNeg 由调用方分配作工作区）
    __aicore__ inline void CopyInPanel(uint64_t matABlockOffset, uint64_t matLBlockOffset)
    {
        AscendC::LocalTensor<T> triangularLocalReal = triangularQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> triangularLocalImag = triangularQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> matALocalReal = matAQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> matALocalImag = matAQueue.AllocTensor<T>();

        // copy L
        AscendC::DataCopyParams copyParamsMatL{static_cast<uint16_t>(blockM),
                                               static_cast<uint16_t>(blockM / elementsPerBlock),
                                               static_cast<uint16_t>((resizedM - blockM) / elementsPerBlock), 0};
        DataCopy(triangularLocalReal, lGlobalReal[matLBlockOffset], copyParamsMatL);
        DataCopy(triangularLocalImag, lGlobalImag[matLBlockOffset], copyParamsMatL);

        // copy A
        AscendC::DataCopyParams copyParamsMatA{static_cast<uint16_t>(blockM),
                                               static_cast<uint16_t>(blockN / elementsPerBlock),
                                               static_cast<uint16_t>((n - blockN) / elementsPerBlock), 0};
        DataCopy(matALocalReal, aGlobalReal[matABlockOffset], copyParamsMatA);
        DataCopy(matALocalImag, aGlobalImag[matABlockOffset], copyParamsMatA);

        triangularQueue.EnQue(triangularLocalReal);
        triangularQueue.EnQue(triangularLocalImag);
        matAQueue.EnQue(matALocalReal);
        matAQueue.EnQue(matALocalImag);
    }

    // 对角行缩放（原 if (trans) / if (trans == 0) 两个互补分支的公共体）
    __aicore__ inline void ScaleDiagRow(int rowIdx, AscendC::LocalTensor<T> &triangularLocalReal,
                                        AscendC::LocalTensor<T> &triangularLocalImag,
                                        AscendC::LocalTensor<T> &matALocalReal, AscendC::LocalTensor<T> &matALocalImag,
                                        AscendC::LocalTensor<T> &matALocalImagNeg)
    {
        TrsmScaleDiagRow(rowIdx, blockN, blockM, unit, triangularLocalReal, triangularLocalImag, matALocalReal,
                         matALocalImag, matALocalImagNeg);
    }
};

#endif  // CGETRS_TRSM_COLUMN_OPS_H
