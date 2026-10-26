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
 * \file householder_cube.hpp
 * \brief CUBE 侧：面板驱动与 T/W/Y 更新
 */

#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include "gemm.hpp"
#include "qr_common.hpp"

namespace AscendC
{
class HouseholderQR_CUBE
{
   private:
    CMatmulCustom<float>* opmm;

    GlobalTensor<float> aGlobalReal, aGlobalImag;
    GlobalTensor<float> tGlobalReal, tGlobalImag, tGlobalImagNeg;
    GlobalTensor<float> wGlobalReal, wGlobalImag;
    GlobalTensor<float> yGlobalReal, yGlobalImag, yGlobalImagNeg;
    GlobalTensor<float> eyeGlobal;

    int M, N, padM, coreNum;
    int offsetF, lenL, lenR, stepNum;
    int totalLenBlock, totalMBlock;
    int offsetAF, offsetYF, offsetTF;
    int globalIdx = 0;
    int workNum = 0;
    int workIdx = 0;

   public:
    __aicore__ inline HouseholderQR_CUBE() : M(0), N(0), padM(0) {}

    __aicore__ inline void Init(CMatmulCustom<float>* opmm_, GlobalTensor<float> AReal, GlobalTensor<float> AImag,
                                GlobalTensor<float> TReal, GlobalTensor<float> TImag, GlobalTensor<float> TImagNeg,
                                GlobalTensor<float> WReal, GlobalTensor<float> WImag, GlobalTensor<float> YReal,
                                GlobalTensor<float> YImag, GlobalTensor<float> YImagNeg, GlobalTensor<float> eyeGlobal,
                                int M_, int padM_, int N_, int blockDim_)
    {
        this->opmm = opmm_;

        this->aGlobalReal = AReal;
        this->aGlobalImag = AImag;

        this->tGlobalReal = TReal;
        this->tGlobalImag = TImag;
        this->tGlobalImagNeg = TImagNeg;

        this->wGlobalReal = WReal;
        this->wGlobalImag = WImag;

        this->yGlobalReal = YReal;
        this->yGlobalImag = YImag;
        this->yGlobalImagNeg = YImagNeg;

        this->eyeGlobal = eyeGlobal;

        this->M = M_;
        this->N = N_;
        this->padM = padM_;
        this->coreNum = blockDim_;
    }
    // 把 total 个块按 num 份均分，返回第 idx 份的起点与块数
    __aicore__ inline void BlockRange(int total, int num, int idx, int& start, int& cnt)
    {
        if (num == 0)
        {
            start = 0;
            cnt = 0;
            return;
        }
        int p = total / num;
        int r = total % num;
        start = p * idx + min(idx, r);
        cnt = p + (idx < r);
    }

    // 面板级 FIX 同步：Cube 内部各核的固定旗标对（0x3）
    __aicore__ inline void SyncFix()
    {
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
        CrossCoreWaitFlag(0x3);
    }

    // _T 后缀表明这是一个列主序的矩阵
    __aicore__ inline void ConstructT(int offsetT1, int offset, int coreIdx, int i, int len, int panel_size = 64)
    {
        // panel_size 用作除数（128 / panel_size），默认 64，退化入参直接返回避免除零
        if (panel_size == 0)
        {
            return;
        }
        int gap = 128 / panel_size;
        int offsetInBlock = offset & 127, preOffset = offset - offsetInBlock;
        int offsetT2 = offsetT1 + offsetInBlock * padM + offsetInBlock;
        int offsetT12 = offsetT1 + offsetInBlock;
        int offsetY1 = ((i / gap) * MAX_PANEL_SIZE * padM) + offsetInBlock;
        int offsetY2 = ((i / gap) * MAX_PANEL_SIZE * padM) + offsetInBlock * (padM + 1);
        if (coreIdx == coreNum - 1)
        {
            int32_t eventIDFIXToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
            opmm->SetMatrix(yGlobalReal, yGlobalImag, yGlobalReal, yGlobalImagNeg, yGlobalImag, tGlobalReal,
                            tGlobalImagNeg, padM, padM, padM);
            opmm->Process<false, true>(offsetY1, offsetY2, offsetT12, offsetInBlock, len, M - offset);

            opmm->SetMatrix(tGlobalImagNeg, tGlobalReal, eyeGlobal, eyeGlobal, eyeGlobal, tGlobalImag, tGlobalReal,
                            padM, 64, padM);
            opmm->ProcessReal<false, false>(offsetT12, 0, offsetT12, offsetInBlock, len, len);

            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);

            opmm->SetMatrix(tGlobalReal, tGlobalImagNeg, tGlobalReal, tGlobalImagNeg, tGlobalImag, tGlobalReal,
                            tGlobalImagNeg, padM, padM, padM);
            opmm->Process<false, false>(offsetT1, offsetT12, offsetT12, offsetInBlock, len, offsetInBlock);

            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIXToMTE2);

            opmm->SetMatrix(tGlobalReal, tGlobalImagNeg, tGlobalReal, tGlobalImagNeg, tGlobalImag, tGlobalReal,
                            tGlobalImagNeg, padM, padM, padM);
            opmm->Process<false, false>(offsetT12, offsetT2, offsetT12, offsetInBlock, len, len);

            opmm->SetMatrix(tGlobalImagNeg, tGlobalReal, eyeGlobal, eyeGlobal, eyeGlobal, tGlobalImag, tGlobalReal,
                            padM, 64, padM);
            opmm->ProcessReal<false, false>(offsetT12, 0, offsetT12, offsetInBlock, len, len);
        }
    }
    __aicore__ inline void CurrentWork(GlobalTensor<float> AReal_T, GlobalTensor<float> AImag_T, int m,
                                       int n,  // A^T[m x n]
                                       GlobalTensor<float> YReal_T, GlobalTensor<float> YImag_T,
                                       GlobalTensor<float> YImagNeg_T, int B,  // Y^T[B x n]
                                       GlobalTensor<float> TReal, GlobalTensor<float> TImag,
                                       GlobalTensor<float> TImagNeg, int coreIdx)
    {  // T[B x B]

        int totalNum = (n + 127) / 128;
        int Num = coreNum / 2;
        int Idx = coreIdx / 2;
        int startIdx = 0, calcNum = 0;
        BlockRange(totalNum, Num - 1, Idx - 1, startIdx, calcNum);

        opmm->SetMatrix(AReal_T, AImag_T, YReal_T, YImagNeg_T, YImag_T, wGlobalReal, wGlobalImag, padM, padM, 128);
        if (Idx)
        {
            SyncFix();
        }
        else
        {
            opmm->Process2<false, true>(0, 0, 0, m, B, n);
            SyncFix();
        }

        if (coreIdx)
        {
            SyncFix();
        }

        else
        {
            opmm->SetMatrix(wGlobalReal, wGlobalImag, TReal, TImagNeg, TImag, wGlobalReal, wGlobalImag, 128, padM, 128);
            opmm->Process<false, false>(0, 0, 0, m, B, B);

            SyncFix();
        }

        int cntm = (m + 127) / 128, cntn = (n + 127) / 128;
        totalNum = cntm * cntn;
        SetAtomicAdd<float>();
        opmm->SetMatrix(wGlobalReal, wGlobalImag, YReal_T, YImag_T, YImagNeg_T, AReal_T, AImag_T, 128, padM, padM);

        BlockRange(totalNum, Num, Idx, startIdx, calcNum);

        for (int i = startIdx; i < startIdx + calcNum; ++i)
        {
            int im = i / cntn, in = i - im * cntn;
            int calcm = min(128, m - im * 128);
            int calcn = min(128, n - in * 128);

            opmm->Process2<false, false>(im * 128 * 128, in * 128, im * 128 * padM + in * 128, calcm, calcn, B);
        }

        SyncFix();

        SetAtomicNone();
    }

    __aicore__ inline void CurrentWork2(GlobalTensor<float> AReal_T, GlobalTensor<float> AImag_T, int m,
                                        int n,  // A^T[m x n]
                                        GlobalTensor<float> YReal_T, GlobalTensor<float> YImag_T,
                                        GlobalTensor<float> YImagNeg_T, int B,  // Y^T[B x n]
                                        GlobalTensor<float> TReal, GlobalTensor<float> TImag,
                                        GlobalTensor<float> TImagNeg, int coreIdx)
    {  // T[B x B]
        int cntm = (m + 127) / 128, cntn = 1;

        int totalNum = (n + 127) / 128;
        int Num = coreNum / 2;
        int Idx = coreIdx / 2;
        int startIdx = 0, calcNum = 0;
        BlockRange(cntm, Num, Idx, startIdx, calcNum);

        int calcM = min(m - 128 * startIdx, calcNum * 128);
        opmm->SetMatrix(AReal_T, AImag_T, YReal_T, YImagNeg_T, YImag_T, wGlobalReal, wGlobalImag, padM, padM, 128);
        if (calcNum)
        {
            opmm->Process2<false, true>(128 * startIdx * padM, 0, 128 * startIdx * 128, calcM, B, n);
        }

        SyncFix();

        Num = coreNum;
        Idx = coreIdx;
        BlockRange(cntm, Num, Idx, startIdx, calcNum);
        calcM = min(m - 128 * startIdx, calcNum * 128);
        if (calcNum)
        {
            opmm->SetMatrix(wGlobalReal, wGlobalImag, TReal, TImagNeg, TImag, wGlobalReal, wGlobalImag, 128, padM, 128);
            opmm->Process<false, false>(128 * startIdx * 128, 0, 128 * startIdx * 128, calcM, B, B);
        }
        SyncFix();

        totalNum = cntm * cntn;
        SetAtomicAdd<float>();
        opmm->SetMatrix(wGlobalReal, wGlobalImag, YReal_T, YImag_T, YImagNeg_T, AReal_T, AImag_T, 128, padM, padM);

        BlockRange(totalNum, Num, Idx, startIdx, calcNum);

        for (int i = startIdx; i < startIdx + calcNum; ++i)
        {
            int im = i / cntn, in = i - im * cntn;
            int calcm = min(128, m - im * 128);
            int calcn = min(128, n - in * 128);

            opmm->Process<false, false>(im * 128 * 128, in * 128, im * 128 * padM + in * 128, calcm, calcn, B);
        }

        SyncFix();

        SetAtomicNone();
    }
    // FutureWork 第一步：W = Y^H A 的一段（按 128 列切分到各核）
    __aicore__ inline void FutureWorkW(int len, int B, int coreIdx, int& startIdx, int& p)
    {
        int Idx = coreIdx / 2;
        int Num = coreNum / 2;
        opmm->SetMatrix(aGlobalReal[offsetAF], aGlobalImag[offsetAF], yGlobalReal[offsetYF], yGlobalImagNeg[offsetYF],
                        yGlobalImag[offsetYF], wGlobalReal, wGlobalImag, padM, padM, 128);
        BlockRange(len, Num, Idx, startIdx, p);
        if (p)
        {
            opmm->Process2<false, true>(startIdx * 128 * padM, 0, startIdx * 128 * 128, p * 128, B, B);
        }
        SyncFix();
    }

    // 错峰取块（第二步）：返回 false 表示块数退化、本核无块可做
    __aicore__ inline bool PickZeroBlock(int t, int cntM, int cntN, int& i, int& j)
    {
        if (cntM == 0)
        {
            return false;
        }
        int iCount = cntM - 1;
        if (iCount == 0)
        {
            return false;
        }
        if (cntN == 0)
        {
            return false;
        }
        int ii = t % iCount;               // 0-based 快变轴 -> 相邻核拿到不同 i
        i = ii + 1;                        // 还原到 [1, cntM) 区间
        j = (t / iCount + ii) % cntN;      // 对角错峰 -> 相邻核同时拿到不同 j
        return true;
    }
    // FutureWork 第二步：首行块之外的 A 块原子加进 W（相邻核错峰取块）
    __aicore__ inline void FutureWorkZeroBlock(int cntM, int cntN, int B, int coreIdx)
    {
        SetAtomicAdd<float>();
        int totalBlocks = (cntM - 1) * cntN;
        for (int t = coreIdx; t < totalBlocks; t += coreNum)
        {
            int i, j;
            if (!PickZeroBlock(t, cntM, cntN, i, j))
            {
                break;
            }
            opmm->Process<false, true>(j * 128 * padM + i * 128, i * 128, j * 128 * 128, B, B, B);
        }

        SyncFix();
        SetAtomicNone();
    }

    // FutureWork 第三步：W·T（本核那一部分）
    __aicore__ inline void FutureWorkApplyT(int startIdx, int p, int B)
    {
        if (p)
        {
            opmm->SetMatrix(wGlobalReal, wGlobalImag, tGlobalReal[offsetTF], tGlobalImagNeg[offsetTF],
                            tGlobalImag[offsetTF], wGlobalReal, wGlobalImag, 128, padM, 128);

            opmm->Process<false, false>(startIdx * 128 * 128, 0, startIdx * 128 * 128, p * 128, B, B);
            SyncFix();
        }
        else
        {
            SyncFix();
        }
    }

    // 错峰取块（第四步）：返回 false 表示块数退化、本核无块可做
    __aicore__ inline bool PickSubtractBlock(int t, int cntM, int cntN, int& i, int& j)
    {
        if (cntM == 0)
        {
            return false;
        }
        if (cntN == 0)
        {
            return false;
        }
        i = t % cntM;                      // 快变轴 -> 相邻核拿到不同的 i
        j = (t / cntM + i) % cntN;         // 对角错峰 -> 相邻核同时也拿到不同 j
        return true;
    }
    // FutureWork 第四步：A -= W·Y（原子累加）
    __aicore__ inline void FutureWorkSubtract(int cntM, int cntN, int B, int coreIdx)
    {
        SetAtomicAdd<float>();

        opmm->SetMatrix(wGlobalReal, wGlobalImag, yGlobalReal[offsetYF], yGlobalImag[offsetYF],
                        yGlobalImagNeg[offsetYF], aGlobalReal[offsetAF], aGlobalImag[offsetAF], 128, padM, padM);
        int totalBlocks = cntM * cntN;

        for (int t = coreIdx; t < totalBlocks; t += coreNum)
        {
            int i, j;
            if (!PickSubtractBlock(t, cntM, cntN, i, j))
            {
                break;
            }
            opmm->Process<false, false>(j * 128 * 128, i * 128, j * 128 * padM + i * 128, 128, 128, 128);
        }

        SyncFix();
        SetAtomicNone();
    }

    __aicore__ inline void FutureWork(int panelIdx, int updateIdx, int len, int coreIdx, int B = 128)
    {
        // 2 0 6
        if (len <= 0 || updateIdx < 0)
        {
            return;
        }
        offsetAF = panelIdx * 128 * padM + updateIdx * 128;
        offsetYF = updateIdx * 128 * padM;
        offsetTF = updateIdx * 128 * padM;
        int cntM = padM / 128 - updateIdx;
        int cntN = len;

        int startIdx = 0;
        int p = 0;
        FutureWorkW(len, B, coreIdx, startIdx, p);
        FutureWorkZeroBlock(cntM, cntN, B, coreIdx);
        FutureWorkApplyT(startIdx, p, B);
        // A - WY
        FutureWorkSubtract(cntM, cntN, B, coreIdx);
    }

    // panel_size = 32 的奇数面板（i & 3）：先算 nxt 段，再算同一 128 块里的 resid 段
    __aicore__ inline void WorkOddPanel32(int offset, int curM, int i, int nEnd, int& len, int& t, int& start,
                                          int& update, int blockIdx)
    {
        int preOffsetInBlock = (offset - 32) & 127;
        int offsetY = ((i / 4) * MAX_PANEL_SIZE * padM) + preOffsetInBlock * (padM + 1);
        int offsetT = ((i / 4) * MAX_PANEL_SIZE * padM) + preOffsetInBlock * (padM + 1);
        int offsetA = offset * padM + (offset - 32);
        int lenAnxt = min(32, nEnd - offset);

        CurrentWork(aGlobalReal[offsetA], aGlobalImag[offsetA], lenAnxt, curM, yGlobalReal[offsetY],
                    yGlobalImag[offsetY], yGlobalImagNeg[offsetY], 32, tGlobalReal[offsetT], tGlobalImag[offsetT],
                    tGlobalImagNeg[offsetT], blockIdx);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
        int offsetInBlock = offset & 127;
        int lenAres = min(128 - offsetInBlock - lenAnxt, nEnd - offset - lenAnxt);
        int offsetAres = (offset + lenAnxt) * padM + offset - 32;

        CurrentWork(aGlobalReal[offsetAres], aGlobalImag[offsetAres], lenAres, curM, yGlobalReal[offsetY],
                    yGlobalImag[offsetY], yGlobalImagNeg[offsetY], 32, tGlobalReal[offsetT], tGlobalImag[offsetT],
                    tGlobalImagNeg[offsetT], blockIdx);

        t = len / (4 - i & 3);
        FutureWork(start, update, t, blockIdx);
        start += t;
        len = len - t;
        CrossCoreWaitFlag(0x1);
        int offsetTpre = (i / 4) * MAX_PANEL_SIZE * padM;

        ConstructT(offsetTpre, offset, blockIdx, i, lenAnxt, 32);
        SyncFix();
    }

    // panel_size = 32 的偶数面板（每 4 个一组的第一块）：整段 128 行，并重置 len/t/start/update
    __aicore__ inline void WorkEvenPanel32(int offset, int curM, int i, int nEnd, int base, int& len, int& t,
                                           int& start, int& update, int blockIdx)
    {
        int offsetA = offset * padM + offset - 128;
        int offsetY = ((i / 4 - 1) * MAX_PANEL_SIZE * padM);
        int offsetT = ((i / 4 - 1) * MAX_PANEL_SIZE * padM);
        int lenAnxt = min(128, nEnd - offset);

        CurrentWork(aGlobalReal[offsetA], aGlobalImag[offsetA], lenAnxt, curM + 96, yGlobalReal[offsetY],
                    yGlobalImag[offsetY], yGlobalImagNeg[offsetY], 128, tGlobalReal[offsetT], tGlobalImag[offsetT],
                    tGlobalImagNeg[offsetT], blockIdx);

        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
        len = base / 128 - 1;
        t = len / 4;
        start = offset / 128 + 1;
        update = offset / 128 - 1;
        FutureWork(start, update, t, blockIdx);
        start += t;
        len -= t;
        CrossCoreWaitFlag(0x1);
    }

    __aicore__ inline void Process2(int blockIdx)
    {
        offsetF = lenL = lenR = stepNum = 0;
        int panelNum = (min(M - CHANGE, N) + 31) / 32;
        int nEnd = min(M, N);
        int offset = 32, curM = M;

        CrossCoreWaitFlag(0x1);
        int len = 0, t = 0, start = 0, update = 0;
        for (int i = 1; i < panelNum; ++i, offset += 32, curM -= 32)
        {
            if (i & 3)
                WorkOddPanel32(offset, curM, i, nEnd, len, t, start, update, blockIdx);
            else
                WorkEvenPanel32(offset, curM, i, nEnd, N - offset % N, len, t, start, update, blockIdx);
        }
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
    }

    // Process4 收尾：补最后一个不满 4 的组的 FutureWork，以及 N > M 时的尾部列块
    __aicore__ inline void FinishPanels4(int offset, int len, int start, int update, int blockIdx)
    {
        if (len) FutureWork(start, update, len, blockIdx);
        if (N > M)
            OverflowTail(offset, blockIdx);
    }

    // N > M 时多出来的列块（只用 CurrentWork / CurrentWork2，不带 T 递推）
    __aicore__ inline void OverflowTail(int offset, int blockIdx)
    {
        int offset2 = (offset - 32) / 128 * 128;
        int offsetA = padM * padM + offset2;
        int b = M - offset2;
        int offsetY = ((offset2 / 128) * MAX_PANEL_SIZE * padM);
        int offsetT = ((offset2 / 128) * MAX_PANEL_SIZE * padM);

        int lenAnxt = N - padM;
        CurrentWork2(aGlobalReal[offsetA], aGlobalImag[offsetA], lenAnxt, b, yGlobalReal[offsetY], yGlobalImag[offsetY],
                     yGlobalImagNeg[offsetY], b, tGlobalReal[offsetT], tGlobalImag[offsetT], tGlobalImagNeg[offsetT],
                     blockIdx);

        offset2 = max(offset - 32, 0);
        offsetA = M * padM + offset2;
        b = M - offset2;
        int preOffsetInBlock = offset2 & 127;
        offsetY = ((offset2 / 128) * MAX_PANEL_SIZE * padM) + preOffsetInBlock * (padM + 1);
        offsetT = ((offset2 / 128) * MAX_PANEL_SIZE * padM) + preOffsetInBlock * (padM + 1);

        lenAnxt = padM - M;
        CurrentWork(aGlobalReal[offsetA], aGlobalImag[offsetA], lenAnxt, b, yGlobalReal[offsetY], yGlobalImag[offsetY],
                    yGlobalImagNeg[offsetY], b, tGlobalReal[offsetT], tGlobalImag[offsetT], tGlobalImagNeg[offsetT],
                    blockIdx);
    }

    __aicore__ inline void Process4(int blockIdx)
    {
        offsetF = lenL = lenR = stepNum = 0;
        int panelNum = (min(M, N) + 31) / 32;
        int padN = (N + 127) / 128 * 128;
        int offset = 32, curM = M;

        CrossCoreWaitFlag(0x1);
        int len = 0, t = 0, start = 0, update = 0;
        for (int i = 1; i < panelNum; ++i, offset += 32, curM -= 32)
        {
            if (i & 3)
                WorkOddPanel32(offset, curM, i, N, len, t, start, update, blockIdx);
            else
                WorkEvenPanel32(offset, curM, i, N, padN - offset, len, t, start, update, blockIdx);
        }
        FinishPanels4(offset, len, start, update, blockIdx);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
    }
    // panel_size = 64 时的奇数面板（i & 1）：算 nxt 段并喂给 ConstructT
    __aicore__ inline void WorkOddPanel64(int offset, int curM, int i, int len, int t, int blockIdx)
    {
        const int panel_size = 64;
        const int gap = 128 / panel_size;
        int minMN = min(M, N);
        int preOffsetInBlock = (offset - panel_size) & 127;
        int offsetA = offset * padM + (offset - panel_size);
        int offsetY = ((i / gap) * MAX_PANEL_SIZE * padM) + preOffsetInBlock * (padM + 1);
        int offsetT = ((i / gap) * MAX_PANEL_SIZE * padM) + preOffsetInBlock * (padM + 1);
        int lenAnxt = min(panel_size, minMN - offset);
        CurrentWork(aGlobalReal[offsetA], aGlobalImag[offsetA], lenAnxt, curM, yGlobalReal[offsetY],
                    yGlobalImag[offsetY], yGlobalImagNeg[offsetY], panel_size, tGlobalReal[offsetT],
                    tGlobalImag[offsetT], tGlobalImagNeg[offsetT], blockIdx);

        CrossCoreSetFlag<0x2, PIPE_FIX>(0x9);
        FutureWork(offset / 128 + 1 + t, offset / 128 - 1, len - t, blockIdx);
        CrossCoreWaitFlag(0x8);
        int offsetTpre = (i / gap) * MAX_PANEL_SIZE * padM;
        ConstructT(offsetTpre, offset, blockIdx, i, lenAnxt);
        SyncFix();
    }

    // panel_size = 64 时的偶数面板：算整段 128 行并顺带算出下一轮的 len / t
    __aicore__ inline void WorkEvenPanel64(int offset, int curM, int i, int& len, int& t, int blockIdx)
    {
        const int gap = 128 / 64;
        int minMN = min(M, N);
        int offsetA = offset * padM + offset - 128;
        int offsetY = ((i / gap - 1) * MAX_PANEL_SIZE * padM);
        int offsetT = ((i / gap - 1) * MAX_PANEL_SIZE * padM);
        int lenAnxt = min(128, minMN - offset);

        CurrentWork(aGlobalReal[offsetA], aGlobalImag[offsetA], lenAnxt, curM + 64, yGlobalReal[offsetY],
                    yGlobalImag[offsetY], yGlobalImagNeg[offsetY], 128, tGlobalReal[offsetT], tGlobalImag[offsetT],
                    tGlobalImagNeg[offsetT], blockIdx);

        CrossCoreSetFlag<0x2, PIPE_FIX>(0x9);
        len = (N - offset) / 128 - 1;
        t = len - len / 2;
        FutureWork(offset / 128 + 1, offset / 128 - 1, t, blockIdx);
        CrossCoreWaitFlag(0x8);
    }

    // Process3 / Process 共用的面板推进循环，只有初值与起始下标不同
    __aicore__ inline void RunPanels64(int offset0, int curM0, int startIdx, int blockIdx)
    {
        const int panel_size = 64;
        int panelNum = (min(M, N) + panel_size - 1) / panel_size;
        int offset = offset0, curM = curM0;
        int len = 0, t = 0;
        for (int i = startIdx; i < panelNum; ++i, offset += panel_size, curM -= panel_size)
        {
            if (i & 1)
                WorkOddPanel64(offset, curM, i, len, t, blockIdx);
            else
                WorkEvenPanel64(offset, curM, i, len, t, blockIdx);
        }
        if (N > M)
        {
            len = (N - offset) / 128;
            FutureWork(offset / 128, offset / 128 - 1, len, blockIdx);
        }
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
    }

    __aicore__ inline void Process(int blockIdx)
    {
        int offset = min(M - CHANGE, N);
        RunPanels64(offset, CHANGE + 64, offset / 64, blockIdx);
    }

    __aicore__ inline void Process3(int blockIdx) { RunPanels64(64, M, 1, blockIdx); }
};
}  // namespace AscendC
