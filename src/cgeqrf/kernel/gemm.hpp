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
 * \file gemm.hpp
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

enum class L0CWriteType
{
    REPLACE,
    ADD,
    FIRST,
    TAIL,
};
template <typename T>
class CMatmulCustom
{
    GlobalTensor<T> aRealGlobal, bRealGlobal, cRealGlobal;
    GlobalTensor<T> aImagGlobal, bImagGlobal, cImagGlobal;
    GlobalTensor<T> extGlobal;
    TQue<QuePosition::A1, 2> inQueueA1Real, inQueueA1Imag;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1Real, inQueueB1Imag;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1Real;
    TBuf<TPosition::CO1> outBufCO1Imag;
    int strideA, strideB, strideC;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int baseM, baseN, baseK;
    int tailM, tailN, tailK;
    int stepM, stepN, stepK;
    int tailMAligned, tailNAligned;
    int alignedK;
    int totalK, totalKALigned;
    bool syncFlag;
    int blockIdx;
    int mBlocks, nBlocks, kBlocks;
    int cnt;

   public:
    __aicore__ inline CMatmulCustom() {}
    __aicore__ inline void Init(TPipe* pipe)
    {
        this->stepK = 2;
        pipe->InitBuffer(inQueueA1Real, 2, 65536);
        pipe->InitBuffer(inQueueA1Imag, 2, 65536);
        pipe->InitBuffer(inQueueB1Real, 2, 65536);
        pipe->InitBuffer(inQueueB1Imag, 2, 65536);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1Imag, 65536);
        pipe->InitBuffer(outBufCO1Real, 65536);
        this->blockIdx = GetBlockIdx();
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aRealGlobal, GlobalTensor<T> aImagGlobal,
                                     GlobalTensor<T> bRealGlobal, GlobalTensor<T> bImagGlobal,
                                     GlobalTensor<T> extGlobal, GlobalTensor<T> cRealGlobal,
                                     GlobalTensor<T> cImagGlobal, int strideA, int strideB, int strideC)
    {
        this->strideA = strideA;
        this->strideB = strideB;
        this->strideC = strideC;
        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->extGlobal = extGlobal;
        this->bRealGlobal = bRealGlobal;
        this->bImagGlobal = bImagGlobal;
        this->cRealGlobal = cRealGlobal;
        this->cImagGlobal = cImagGlobal;
    }
    template <bool trans_A = false, bool trans_B = false, L0CWriteType writeType = L0CWriteType::REPLACE>
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK,
                                   bool firstWrite = false)
    {
        // extGlobal == bImagNegGlobal
        SetBlockArgs(offsetA, offsetB, offsetC, blockM, blockN, blockK);
        // default: (128, 128, 64)
        SetBaseTiles();
        SetBlockCounts();

        for (int i = 0; i < nBlocks; ++i)
        {
            for (int j = 0; j < mBlocks; ++j)
            {
                SetTails(i, j);
                SetFMatrixCal<trans_A>();
                int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
                LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
                LocalTensor<T> c1ImagLocal = outBufCO1Imag.Get<T>();
                RunKBlocks<trans_A, trans_B>(i, j, c1RealLocal, c1ImagLocal);
                WriteTile<writeType>(offsetC_1, c1RealLocal, c1ImagLocal);
            }
        }
    }
    template <bool trans_A = false, bool trans_B = false, L0CWriteType writeType = L0CWriteType::REPLACE>
    __aicore__ inline void ProcessReal(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK,
                                       bool firstWrite = false)
    {
        // extGlobal == bImagNegGlobal
        SetBlockArgs(offsetA, offsetB, offsetC, blockM, blockN, blockK);
        // default: (128, 128, 64)
        SetBaseTiles();
        SetBlockCounts();

        int32_t eventIDFIXToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);

        for (int i = 0; i < nBlocks; ++i)
        {
            for (int j = 0; j < mBlocks; ++j)
            {
                SetTails(i, j);
                SetFMatrixCal<trans_A>();
                int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
                LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
                RunKBlocksReal<trans_A, trans_B>(i, j, c1RealLocal);
                WriteTileReal<writeType>(offsetC_1, c1RealLocal);
            }
        }
    }
    template <bool trans_A = false, bool trans_B = false, L0CWriteType writeType = L0CWriteType::REPLACE>
    __aicore__ inline void Process2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK)
    {
        // extGlobal == bImagNegGlobal
        SetBlockArgs(offsetA, offsetB, offsetC, blockM, blockN, blockK);
        // default: (128, 128, 64)
        SetBaseTiles();
        SetBlockCounts();

        for (int i = 0; i < nBlocks; ++i)
        {
            for (int j = 0; j < mBlocks; ++j)
            {
                SetTails(i, j);
                SetFMatrixCal<trans_A>();
                int offsetC_1 = offsetC + j * baseM * strideC + i * baseN;
                LocalTensor<T> c1RealLocal = outBufCO1Real.Get<T>();
                LocalTensor<T> c1ImagLocal = outBufCO1Imag.Get<T>();
                RunKBlocksParity<trans_A, trans_B>(i, j, c1RealLocal, c1ImagLocal);
                WriteTileParity<writeType>(offsetC_1, c1RealLocal, c1ImagLocal);
            }
        }
    }

   private:
    // ---- Process / ProcessReal / Process2 共用的骨架（连续切片抽取，未重排语句） ----
    __aicore__ inline void SetBlockArgs(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK)
    {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;
        this->syncFlag = true;
    }
    __aicore__ inline int Pow2AtLeast(int start, int bound)
    {
        int v = start;
        while (v < bound) v <<= 1;
        return v;
    }
    __aicore__ inline void SetBaseTiles()
    {
        baseN = min((singleCoreN + 15) / 16 * 16, 128);
        int nn = Pow2AtLeast(64, baseN);
        if (nn == 0)
        {
            return;
        }
        baseM = min((singleCoreM + 15) / 16 * 16, 16384 / nn);
        int mm = Pow2AtLeast(16, baseM);
        if (mm == 0)
        {
            return;
        }
        baseK = min((singleCoreK + 7) / 8 * 8, min(8192 / nn, 8192 / mm));
    }
    __aicore__ inline void SetBlockCounts()
    {
        mBlocks = (singleCoreM + baseM - 1) / baseM;
        nBlocks = (singleCoreN + baseN - 1) / baseN;
        kBlocks = (singleCoreK + baseK - 1) / baseK;
    }
    __aicore__ inline void SetTails(int i, int j)
    {
        tailN = min(singleCoreN - i * baseN, baseN);
        tailNAligned = (tailN + 15) / 16 * 16;
        tailM = min(singleCoreM - j * baseM, baseM);
        tailMAligned = (tailM + 15) / 16 * 16;
    }
    template <bool trans_A>
    __aicore__ inline void SetFMatrixCal()
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        if constexpr (!trans_A) Load3DSetFMatrixCal(1, tailMAligned, padList);
    }
    __aicore__ inline void SetKChunk(int k)
    {
        totalK = min(singleCoreK - k * baseK, baseK * stepK);
        totalKALigned = (totalK + 7) / 8 * 8;
        cnt = min(kBlocks - k, stepK);
    }
    __aicore__ inline void SetKInner(int k, int x)
    {
        tailK = min(singleCoreK - (k + x) * baseK, baseK);
        alignedK = (tailK + 7) / 8 * 8;
    }
    template <bool trans_A, bool trans_B>
    __aicore__ inline void SetTileOffsets(int k, int i, int j, int& offsetA_1, int& offsetB_1)
    {
        if constexpr (trans_A)
            offsetA_1 = offsetA + k * baseK * strideA + j * baseM;
        else
            offsetA_1 = offsetA + j * baseM * strideA + k * baseK;
        if constexpr (trans_B)
            offsetB_1 = offsetB + i * baseN * strideB + k * baseK;
        else
            offsetB_1 = offsetB + k * baseK * strideB + i * baseN;
    }
    __aicore__ inline void SyncFixFlag();
    template <L0CWriteType writeType>
    __aicore__ inline void WriteHead();
    template <L0CWriteType writeType>
    __aicore__ inline void WriteTail();
    template <L0CWriteType writeType>
    __aicore__ inline void WriteTile(int offsetC_1, const LocalTensor<T>& c1RealLocal,
                                     const LocalTensor<T>& c1ImagLocal);
    template <L0CWriteType writeType>
    __aicore__ inline void WriteTileReal(int offsetC_1, const LocalTensor<T>& c1RealLocal);
    template <L0CWriteType writeType>
    __aicore__ inline void WriteTileParity(int offsetC_1, const LocalTensor<T>& c1RealLocal,
                                           const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CopyInComplex(int offsetA_1, int offsetB_1, int cnt);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CopyInComplexParity(int offsetA_1, int offsetB_1, int cnt);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacStepRealImag(int k, int x, const LocalTensor<T>& a1RealLocal,
                                           const LocalTensor<T>& b1RealLocal, const LocalTensor<T>& b1ImagLocal,
                                           const LocalTensor<T>& c1RealLocal, const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacStepImagReal(int k, int x, const LocalTensor<T>& a1ImagLocal,
                                           const LocalTensor<T>& b1RealLocal, const LocalTensor<T>& b1ImagLocal,
                                           const LocalTensor<T>& c1RealLocal, const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacStepParityReal(int k, int x, const LocalTensor<T>& a1RealLocal,
                                             const LocalTensor<T>& b1RealLocal, const LocalTensor<T>& b1ImagLocal,
                                             const LocalTensor<T>& c1RealLocal, const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacStepParityImag(int k, int x, const LocalTensor<T>& a1ImagLocal,
                                             const LocalTensor<T>& b1RealLocal, const LocalTensor<T>& b1ImagLocal,
                                             const LocalTensor<T>& c1RealLocal, const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacComplex(int k, int cnt, const LocalTensor<T>& c1RealLocal,
                                      const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacComplexParity(int k, int cnt, const LocalTensor<T>& c1RealLocal,
                                            const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void MacReal(int k, int cnt, const LocalTensor<T>& c1RealLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void BeginKTile(int k, int i, int j, int& offsetA_1, int& offsetB_1);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void RunKBlocks(int i, int j, const LocalTensor<T>& c1RealLocal,
                                      const LocalTensor<T>& c1ImagLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void RunKBlocksReal(int i, int j, const LocalTensor<T>& c1RealLocal);
    template <bool trans_A, bool trans_B>
    __aicore__ inline void RunKBlocksParity(int i, int j, const LocalTensor<T>& c1RealLocal,
                                            const LocalTensor<T>& c1ImagLocal);
    template <bool trans, bool neg>
    __aicore__ inline void CopyInATile(int offsetA_1, const GlobalTensor<T>& srcGlobal,
                                       TQue<QuePosition::A1, 2>& inQueue);
    template <bool trans, bool neg>
    __aicore__ inline void CopyInBTile(int offsetB_1, const GlobalTensor<T>& srcGlobal,
                                       TQue<QuePosition::B1, 2>& inQueue);
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInAReal(int offsetA_1, int cnt);
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInAImag(int offsetA_1, int cnt);
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInBReal(int offsetB_1, int cnt);
    template <bool trans = false, bool neg = false>
    __aicore__ inline void CopyInBImag(int offsetB_1, int cnt);
    template <bool trans = false>
    __aicore__ inline void SplitA(const LocalTensor<T>& a1Local, int prevK);
    template <bool trans = false>
    __aicore__ inline void SplitB(const LocalTensor<T>& b1Local, int prevK);
    __aicore__ inline void Compute(const LocalTensor<T>& a2Local, const LocalTensor<T>& c1Local, int first,
                                   const int unitFlag);
    template <bool imag = false>
    __aicore__ inline void CopyOut(const LocalTensor<T>& c1Local, int offsetC_1);
};
