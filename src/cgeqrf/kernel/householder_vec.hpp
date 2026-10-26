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
 * \file householder_vec.hpp
 * \brief VEC 侧：面板/行分块反射器与 T 生成（本文件保留类内定义）
 */

#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include "qr_common.hpp"

namespace AscendC
{
class HouseholderQR_VEC : public HouseholderQRCommon
{
   private:
    TBufPool<TPosition::VECCALC, 16>* pipe;
    int panelYSize, panelTSize;
    const int chunkSize = 8192;
    TQue<QuePosition::VECIN, 1> aQueueIn;       // 256 * 32 * 2 = 2 * 8192
    TQue<QuePosition::VECIN, 1> aQueueOut;      // 256 * 3 * 2 = 1536
    TQue<QuePosition::VECOUT, 1> wQueueOut;     // 32 * 32 * 2 = 2048
    TQue<QuePosition::VECIN, 1> normQueueIn;    // 64
    TQue<QuePosition::VECOUT, 1> tmpQueueOut;   // 8 * 32 * 3 = 768
    TQue<QuePosition::VECIN, 1> gatherQueueIn;  // gatherN: 64, gatherW: 32
    TBuf<TPosition::VECCALC> workBuf;           // 64 * 32 = 2048

    LocalTensor<uint32_t> gatherNLocal;

   public:
    __aicore__ inline HouseholderQR_VEC() : HouseholderQRCommon() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe) { this->pipe = pipe; }
    __aicore__ inline void InitRow()
    {
        pipe->InitBuffer(aQueueIn, 1, 256 * 32 * sizeof(float) * 2);
        pipe->InitBuffer(aQueueOut, 1, 32 * sizeof(float) * 2);
        pipe->InitBuffer(wQueueOut, 1, 32 * 32 * sizeof(float) * 2);
        pipe->InitBuffer(normQueueIn, 1, 64 * sizeof(float));
        pipe->InitBuffer(tmpQueueOut, 2, 768 * sizeof(uint32_t));
        pipe->InitBuffer(gatherQueueIn, 1, 128 * sizeof(uint32_t));
        pipe->InitBuffer(workBuf, 64 * 32 * sizeof(float));
    }
    __aicore__ inline void InitColumn()
    {
        pipe->Reset();
        pipe->InitBuffer(aQueueIn, 1, 2 * chunkSize * sizeof(float));
        pipe->InitBuffer(normQueueIn, 1, chunkSize * sizeof(float));
        pipe->InitBuffer(aQueueOut, 1, chunkSize * sizeof(float));
        pipe->InitBuffer(tmpQueueOut, 1, 1024 * sizeof(float));
        pipe->InitBuffer(workBuf, (chunkSize + 1024) * sizeof(float));
    }

    __aicore__ inline void SetMatrix(GlobalTensor<float> aReal, GlobalTensor<float> aImag, GlobalTensor<float> tau_,
                                     GlobalTensor<float> tReal, GlobalTensor<float> tImag, GlobalTensor<float> tImagNeg,
                                     GlobalTensor<float> wReal, GlobalTensor<float> wImag, GlobalTensor<float> yReal,
                                     GlobalTensor<float> yImag, GlobalTensor<float> yImagNeg, int M_, int N_, int padM_,
                                     int padN_)
    {
        SetMatrixCommon(aReal, aImag, tau_, tReal, tImag, tImagNeg, wReal, wImag, yReal, yImag, yImagNeg, M_, N_,
                        padM_, padN_);
        InitRow();
    }
    __aicore__ inline void InitGather(GlobalTensor<uint32_t> gatherGlobal, int panel_size = 32)
    {
        gatherNLocal = gatherQueueIn.AllocTensor<uint32_t>();
        DataCopy(gatherNLocal, gatherGlobal, 2 * panel_size);
    }
    __aicore__ inline void FreeGather() { gatherQueueIn.FreeTensor(gatherNLocal); }
    __aicore__ inline void Process(int blockDim, int blockIdx)
    {
        ProcessPanels(blockDim, blockIdx, min(M - CHANGE, N));
    }
    __aicore__ inline void Process2(int blockDim, int blockIdx)
    {
        ProcessPanels(blockDim, blockIdx, min(M, N));
    }
    // 面板循环：limit 之前每 32 列一个面板推进；0 号核处理表头 64 行，其余核按 64 行均分剩余行
    // 面板开头：各核把自己的那一段 W / 范数缓冲清零（前 16 核清 W_real，次 16 核清 W_imag，32 号核清范数）
    __aicore__ inline void ZeroPanelBuffers(int blockIdx)
    {
        LocalTensor<float> tmpZero = tmpQueueOut.AllocTensor<float>();
        Duplicate<float>(tmpZero, 0.0f, 64);
        tmpQueueOut.EnQue(tmpZero);
        tmpZero = tmpQueueOut.DeQue<float>();
        if (blockIdx < 16)
            DataCopy(wGlobalReal[blockIdx * 64], tmpZero, 64);
        else if (blockIdx < 32)
            DataCopy(wGlobalImag[(blockIdx - 16) * 64], tmpZero, 64);
        else if (blockIdx == 32)
            DataCopy(tmpGlobal, tmpZero, 32);

        tmpQueueOut.FreeTensor(tmpZero);
    }

    // 除 0 号核外，其余核按 64 行一段均分第 64 行以后的剩余行
    __aicore__ inline void SplitRows(int blockDim, int blockIdx, int curM, int& startIdx, int& calcLen)
    {
        int remainRows = max(0, curM - 64);
        int totalNum = (remainRows + 63) / 64;
        int averageNum = totalNum / (blockDim - 1);
        int remainNum = totalNum % (blockDim - 1);
        startIdx = 64 + (averageNum * (blockIdx - 1) + min(blockIdx - 1, remainNum)) * 64;
        calcLen = (averageNum + ((blockIdx - 1) < remainNum)) * 64;
        calcLen = startIdx + calcLen > curM ? curM - startIdx : calcLen;
    }

    // 面板循环：limit 之前每 32 列一个面板推进；0 号核处理表头 64 行，其余核按 64 行均分剩余行
    __aicore__ inline void ProcessPanels(int blockDim, int blockIdx, int limit)
    {
        ZeroPanelBuffers(blockIdx);
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);

        panelYSize = MAX_PANEL_SIZE * padM;
        panelTSize = MAX_PANEL_SIZE * padM;
        int offset = 0, curM = M;
        for (int i = 0; offset < limit; ++i, offset += 32, curM -= 32)
        {
            int columns = min(32, limit - offset);
            if (!blockIdx)
                RowSplitProcessHead(blockDim, 0, offset, 0, min(64, curM), columns, i);
            else
            {
                int startIdx = 0, calcLen = 0;
                SplitRows(blockDim, blockIdx, curM, startIdx, calcLen);
                if (calcLen <= 0)
                    EmptyRowProcess(blockDim, blockIdx, offset, columns, i);
                else
                    RowSplitProcess(blockDim, blockIdx, offset, startIdx, calcLen, columns, i);
            }
        }
    }
    __aicore__ inline void DupAndReduce2(LocalTensor<float>& aLocal, LocalTensor<float>& norm_2Local, int& columns,
                                         int& aligned_rows, int& offset, int& stride)
    {
        // 逐列清零尾部补零区，避免 dstRepeatStride(stride / 8) 在 stride >= 2048 时按 8 位回绕
        for (int c = 0; c != columns; ++c)
        {
            Duplicate<float>(aLocal[c * stride + aligned_rows], 0, offset - aligned_rows);
        }
        for (int i = 0; i < offset; i += 64)
        {
            LocalTensor<float> workLocal = workBuf.Get<float>();
            Mul(workLocal, aLocal[i], aLocal[i], 64, columns,
                {1, 1, 1, 8, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
            WholeReduceSum<float>(workLocal, workLocal, 64, columns, 1, 1, 8);
            Add(norm_2Local, norm_2Local, workLocal, PANEL_SIZE);
        }
    }
    // 把实部或虚部的一段行载入 UB，并累加到当前面板的列范数（行数非 8 的整数倍时走 pad 通路）
    __aicore__ inline void LoadRowsAndAccumNorm(GlobalTensor<float>& src, int offsetA, int rows, int columns,
                                                int aligned_rows, int offset, int stride,
                                                LocalTensor<float>& aLocal, LocalTensor<float>& norm_2Local,
                                                int32_t eventIDMTE2ToV)
    {
        if (rows & 7)
        {
            DataCopyExtParams aDataCopyInParams{static_cast<uint16_t>(columns),
                                                static_cast<uint32_t>(rows * sizeof(float)),
                                                static_cast<uint32_t>((padM - rows) * sizeof(float)),
                                                static_cast<uint32_t>((stride - aligned_rows) / 8), 0};
            DataCopyPadExtParams<float> aPadParams{true, 0, static_cast<uint8_t>(aligned_rows - rows), 0};
            DataCopyPad<float>(aLocal, src[offsetA], aDataCopyInParams, aPadParams);
        }
        else
        {
            DataCopyParams aDataCopyInParams{static_cast<uint16_t>(columns), static_cast<uint16_t>(aligned_rows / 8),
                                             static_cast<uint16_t>((padM - aligned_rows) / 8),
                                             static_cast<uint16_t>((stride - aligned_rows) / 8)};
            DataCopy(aLocal, src[offsetA], aDataCopyInParams);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

        DupAndReduce2(aLocal, norm_2Local, columns, aligned_rows, offset, stride);
    }

    __aicore__ inline void CopyInAndProcessNorm(int& offsetA, int& offsetM, int& rows, int& columns, int& aligned_rows,
                                                int& offset, int& stride, LocalTensor<float>& aLocalReal,
                                                LocalTensor<float>& aLocalImag, int& flag)
    {
        LocalTensor<float> norm_2Local = tmpQueueOut.AllocTensor<float>();
        Duplicate<float>(norm_2Local, 0.0f, PANEL_SIZE);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        LoadRowsAndAccumNorm(aGlobalReal, offsetA, rows, columns, aligned_rows, offset, stride, aLocalReal, norm_2Local,
                             eventIDMTE2ToV);
        LoadRowsAndAccumNorm(aGlobalImag, offsetA, rows, columns, aligned_rows, offset, stride, aLocalImag, norm_2Local,
                             eventIDMTE2ToV);
        tmpQueueOut.EnQue(norm_2Local);
        norm_2Local = tmpQueueOut.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopy(tmpGlobal[(flag & 1) * PANEL_SIZE], norm_2Local, PANEL_SIZE);  // columns % 8 == 0
        SetAtomicNone();
        tmpQueueOut.FreeTensor(norm_2Local);
    }
    // 本核负责的行区间做一次 rank-1 更新（w 由各核原子累加后读回，再作用到整个列块）
    // w = reduce(x * y)：首段直接落到 w（进入时 w 是本面板清零后的缓冲），后续 64 行段累加
    __aicore__ inline void ReduceRowsInitW(LocalTensor<float>& w, LocalTensor<float>& x, LocalTensor<float>& y,
                                           int offset, int columns, int stride)
    {
        for (int i = 0; i < offset; i += 64)
        {
            LocalTensor<float> workLocal = workBuf.Get<float>();
            Mul(workLocal, x[i], y[i], 64, columns, {1, 1, 1, 8, static_cast<uint8_t>(stride / 8), 0});
            if (i)
            {
                WholeReduceSum<float>(workLocal, workLocal, 64, columns, 1, 1, 8);
                Add(w, w, workLocal, PANEL_SIZE);
            }
            else
            {
                WholeReduceSum<float>(w, workLocal, 64, columns, 1, 1, 8);
            }
        }
    }

    // w += / -= reduce(x * y)，按 64 行一段累加
    __aicore__ inline void ReduceRowsAccumW(LocalTensor<float>& w, LocalTensor<float>& x, LocalTensor<float>& y,
                                            int offset, int columns, int stride, bool sub)
    {
        for (int i = 0; i < offset; i += 64)
        {
            LocalTensor<float> workLocal = workBuf.Get<float>();
            Mul(workLocal, x[i], y[i], 64, columns, {1, 1, 1, 8, static_cast<uint8_t>(stride / 8), 0});
            WholeReduceSum<float>(workLocal, workLocal, 64, columns, 1, 1, 8);
            if (sub)
            {
                Sub(w, w, workLocal, PANEL_SIZE);
            }
            else
            {
                Add(w, w, workLocal, PANEL_SIZE);
            }
        }
    }

    // 4 个 MulAddDst 完成一次复数 rank-1 更新（a_r/a_i 各自加两项）
    __aicore__ inline void UpdateRows(int k, int columns, int offset, int stride, LocalTensor<float>& vReal,
                                      LocalTensor<float>& vImag, LocalTensor<float>& aLocalReal,
                                      LocalTensor<float>& aLocalImag, LocalTensor<float>& tmpWReal,
                                      LocalTensor<float>& tmpWImag, LocalTensor<float>& tmpWImagNeg)
    {
        for (int i = 0; i < offset; i += 64)
        {
            AscendC::MulAddDst(aLocalReal[(k + 1) * stride + i], vReal[i], tmpWReal[8 * (k + 1)], 64, columns - (k + 1),
                               {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, 1});  // a_r += w_r * v_r
            AscendC::MulAddDst(aLocalReal[(k + 1) * stride + i], vImag[i], tmpWImagNeg[8 * (k + 1)], 64,
                               columns - (k + 1),
                               {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, 1});  // a_r += -w_i * v_i
            AscendC::MulAddDst(aLocalImag[(k + 1) * stride + i], vImag[i], tmpWReal[8 * (k + 1)], 64, columns - (k + 1),
                               {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, 1});  // a_i += w_r * v_i
            AscendC::MulAddDst(aLocalImag[(k + 1) * stride + i], vReal[i], tmpWImag[8 * (k + 1)], 64, columns - (k + 1),
                               {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, 1});  // a_i += w_i * v_r
        }
    }

    // 读回各核累加好的全量 w，展开成广播副本后作用到本核行区间
    __aicore__ inline void ApplyRankOne(int k, int columns, int offset, int stride, LocalTensor<float>& vReal,
                                        LocalTensor<float>& vImag, LocalTensor<float>& aLocalReal,
                                        LocalTensor<float>& aLocalImag, LocalTensor<float>& wLocalReal,
                                        LocalTensor<float>& wLocalImag, int offsetW, float tau_Real, float tau_Imag)
    {
        LocalTensor<float> tmpWReal = tmpQueueOut.AllocTensor<float>();
        LocalTensor<float> tmpWImag = tmpWReal[8 * PANEL_SIZE];
        LocalTensor<float> tmpWImagNeg = tmpWImag[8 * PANEL_SIZE];

        DataCopy(wLocalReal, wGlobalReal[offsetW], PANEL_SIZE);
        DataCopy(wLocalImag, wGlobalImag[offsetW], PANEL_SIZE);

        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

        Muls(tmpWImagNeg, wLocalReal, tau_Real, PANEL_SIZE);
        Axpy(tmpWImagNeg, wLocalImag, -tau_Imag, PANEL_SIZE);
        Brcb(tmpWReal, tmpWImagNeg, (columns + 7) / 8, {1, 8});
        Muls(tmpWImagNeg, wLocalReal, tau_Imag, PANEL_SIZE);
        Axpy(tmpWImagNeg, wLocalImag, tau_Real, PANEL_SIZE);
        Brcb(tmpWImag, tmpWImagNeg, (columns + 7) / 8, {1, 8});
        Muls(tmpWImagNeg, tmpWImag, -1.0f, 8 * columns);

        UpdateRows(k, columns, offset, stride, vReal, vImag, aLocalReal, aLocalImag, tmpWReal, tmpWImag, tmpWImagNeg);
        tmpQueueOut.FreeTensor(tmpWReal);
    }

    // 本核负责的行区间做一次 rank-1 更新（w 由各核原子累加后读回，再作用到整个列块）
    __aicore__ inline void Update(int k, int offsetM, int columns, int offset, int stride,
                                  LocalTensor<float>& vReal, LocalTensor<float>& vImag, LocalTensor<float>& aLocalReal,
                                  LocalTensor<float>& aLocalImag, LocalTensor<float> wLocalReal,
                                  LocalTensor<float> wLocalImag, float tau_Real, float tau_Imag, int flag)
    {
        tau_Real *= -1;  // here -tau.conj()
        // v^T * a = (v_r, -v_i) * (a_r, a_i)
        int offsetW = (flag & 1) * PANEL_SIZE * PANEL_SIZE + k * PANEL_SIZE;
        ReduceRowsInitW(wLocalReal, aLocalReal, vReal, offset, columns, stride);          // w_real  = a_r * v_r
        ReduceRowsAccumW(wLocalReal, aLocalImag, vImag, offset, columns, stride, false);  // w_real += a_i * v_i
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        SetAtomicAdd<float>();
        DataCopy(wGlobalReal[offsetW], wLocalReal, PANEL_SIZE);

        ReduceRowsInitW(wLocalImag, aLocalImag, vReal, offset, columns, stride);         // w_imag  = a_i * v_r
        ReduceRowsAccumW(wLocalImag, aLocalReal, vImag, offset, columns, stride, true);  // w_imag -= a_r * v_i
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(wGlobalImag[offsetW], wLocalImag, PANEL_SIZE);
        SetAtomicNone();

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);

        ApplyRankOne(k, columns, offset, stride, vReal, vImag, aLocalReal, aLocalImag, wLocalReal, wLocalImag, offsetW,
                     tau_Real, tau_Imag);
    }
    // 行分块分解的上下文：把派生偏移集中在一处，helper 之间只再传少量 tensor
    struct RowCtx
    {
        int offsetM, offsetIdx, rows, columns, flag;
        int offsetA, offsetY, offsetT, alignedRows, offset, stride;
    };

    __aicore__ inline void MakeRowCtx(RowCtx& ctx, int offsetM, int offsetIdx, int rows, int columns, int flag)
    {
        ctx.offsetM = offsetM;
        ctx.offsetIdx = offsetIdx;
        ctx.rows = rows;
        ctx.columns = columns;
        ctx.flag = flag;
        ctx.offsetA = offsetM * padM + offsetM + offsetIdx;
        ctx.offsetY = ((flag / 4) * panelYSize) + (offsetM & 127) * (padM + 1) + offsetIdx;
        ctx.offsetT = ((flag / 4) * panelTSize) + (offsetM & 127) * (padM + 1);
        ctx.alignedRows = (rows + 7) / 8 * 8;
        ctx.offset = (ctx.alignedRows + 63) / 64 * 64;
        ctx.stride = 2 * ctx.offset;
    }

    // 复数 Householder：beta 为对角元，(invReal, invImag) 用于构造 v，tau 为反射子系数
    struct Householder
    {
        float beta, tauReal, tauImag, invReal, invImag;
    };

    __aicore__ inline void MakeHouseholder(Householder& h, float x0Real, float x0Imag, float norm2k)
    {
        h.beta = sqrt(norm2k) * (x0Real >= 0 ? -1 : 1);
        float v0Real = x0Real - h.beta, v0Imag = x0Imag;
        float v02 = v0Real * v0Real + v0Imag * v0Imag;
        h.invReal = v0Real / v02, h.invImag = -v0Imag / v02;
        h.tauReal = -v0Real / h.beta, h.tauImag = -v0Imag / h.beta;
    }

    // 零列判定：一律用增量范数，保证与其它核的分支/flags 数量一致（两条分支的 flag 对数不同）
    __aicore__ inline bool IsZeroColumn(float x0Real, float incNorm)
    {
        if (incNorm == 0)
        {
            return true;
        }
        return incNorm <= 1e-8f || 1 - x0Real * x0Real / incNorm < 1e-6f;
    }

    // 非首面板需先等 Cube 把上一面板的结果写完
    __aicore__ inline void WaitCubePrevPanel(int offsetM)
    {
        if (offsetM)
        {
            CrossCoreWaitFlag(0x2);  // wait Cube Flag
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);
        }
    }

    // Head 面板开头：清零本核负责的 W 与范数缓冲，并写回 GM 的另一半缓冲
    __aicore__ inline void ResetPanelW(LocalTensor<float>& wLocalReal, LocalTensor<float>& wLocalImag,
                                       LocalTensor<float>& norm_2, int flag)
    {
        Duplicate<float>(wLocalReal, 0.0f, 2 * PANEL_SIZE * PANEL_SIZE);
        Duplicate<float>(norm_2, 0.0f, PANEL_SIZE);

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(wGlobalReal[(~flag & 1) * PANEL_SIZE * PANEL_SIZE], wLocalReal, PANEL_SIZE * PANEL_SIZE);
        DataCopy(wGlobalImag[(~flag & 1) * PANEL_SIZE * PANEL_SIZE], wLocalImag, PANEL_SIZE * PANEL_SIZE);
        DataCopy(tmpGlobal[(~flag & 1) * PANEL_SIZE], norm_2, PANEL_SIZE);
    }

    // Head 分支最后一列算完后，把本面板的 tau 一次写回 GM
    __aicore__ inline void StoreTau(int offsetM, int columns, LocalTensor<float>& tauLocal)
    {
        aQueueOut.EnQue(tauLocal);
        tauLocal = aQueueOut.DeQue<float>();
        DataCopyExtParams tauDataCopyParams{static_cast<uint16_t>(1),
                                            static_cast<uint32_t>(2 * columns * sizeof(float)), 0, 0, 0};
        DataCopyPad(tau[offsetM * 2], tauLocal, tauDataCopyParams);
    }

    // 把本列的表头元素广播到 GM（v0Global），供其它行分块核使用
    __aicore__ inline void BroadcastX0(int k, float x0Real, float x0Imag)
    {
        LocalTensor<float> xOut = tmpQueueOut.AllocTensor<float>();
        xOut.SetValue(0, x0Real);
        xOut.SetValue(1, x0Imag);
        tmpQueueOut.EnQue(xOut);
        xOut = tmpQueueOut.DeQue<float>();
        DataCopyExtParams xDataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(2 * sizeof(float)), 0, 0, 0};
        DataCopyPad<float>(v0Global[2 * k], xOut, xDataCopyParams);
        tmpQueueOut.FreeTensor(xOut);
    }

    // 读回本列的尾范数（各核已原子累加进 GM）
    __aicore__ inline void ReloadNorm(int flag, LocalTensor<float>& norm_2)
    {
        DataCopy(norm_2, tmpGlobal[(flag & 1) * PANEL_SIZE], PANEL_SIZE);
        normQueueIn.EnQue(norm_2);
        norm_2 = normQueueIn.DeQue<float>();
    }

    // 本核覆盖整条尾部时（curM <= 64）按 tile 精确重算当前列的尾部范数：增量值多步相减后残留
    // 浮点抵消误差，尾部很小时会放大到 1e-4 以上，使 beta/tau 与真实 Householder 不一致（Q 非正交）
    __aicore__ inline float ExactTailNorm(int k, const RowCtx& ctx, LocalTensor<float>& aLocalReal)
    {
        LocalTensor<float> normWork = workBuf.Get<float>();
        Mul(normWork, aLocalReal[k * ctx.stride], aLocalReal[k * ctx.stride], ctx.stride);
        if (k > 0)
        {
            // 前 k 行已是 R，不参与尾部范数
            Duplicate<float>(normWork, 0.0f, k);
            Duplicate<float>(normWork[ctx.offset], 0.0f, k);
        }
        Add(normWork, normWork, normWork[ctx.offset], ctx.offset);
        ReduceSum<float>(normWork, normWork, normWork, ctx.offset);
        return normWork.GetValue(0);
    }

    // 零列：y 的三份副本清零，并把本列的 W 从 GM 读回 UB
    __aicore__ inline void ZeroYAndLoadW(const RowCtx& ctx, int k, LocalTensor<float>& wLocalReal,
                                         LocalTensor<float>& wLocalImag)
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);

        LocalTensor<float> zeroLocal = tmpQueueOut.AllocTensor<float>();
        Duplicate<float>(zeroLocal, 0.0f, ctx.alignedRows);
        tmpQueueOut.EnQue(zeroLocal);
        zeroLocal = tmpQueueOut.DeQue<float>();
        DataCopy(yGlobalReal[ctx.offsetY + k * padM], zeroLocal, ctx.alignedRows);
        DataCopy(yGlobalImag[ctx.offsetY + k * padM], zeroLocal, ctx.alignedRows);
        DataCopy(yGlobalImagNeg[ctx.offsetY + k * padM], zeroLocal, ctx.alignedRows);
        tmpQueueOut.FreeTensor(zeroLocal);

        CrossCoreWaitFlag(0x0);

        int offsetW = (ctx.flag & 1) * PANEL_SIZE * PANEL_SIZE + k * PANEL_SIZE;
        DataCopy(wLocalReal[k * PANEL_SIZE], wGlobalReal[offsetW], PANEL_SIZE);
        DataCopy(wLocalImag[k * PANEL_SIZE], wGlobalImag[offsetW], PANEL_SIZE);
    }

    // v = (x - beta * e_k) / |v0|^2 的实部与虚部（vImag 紧接在 vReal 之后）
    __aicore__ inline void BuildVector(int k, const RowCtx& ctx, LocalTensor<float>& aLocalReal,
                                       LocalTensor<float>& aLocalImag, const Householder& h,
                                       LocalTensor<float>& vReal)
    {
        LocalTensor<float> vImag = vReal[ctx.offset];
        Muls(vReal, aLocalImag[k * ctx.stride], -h.invImag, ctx.offset);
        Axpy(vReal, aLocalReal[k * ctx.stride], h.invReal, ctx.offset);
        Muls(vImag, aLocalReal[k * ctx.stride], h.invImag, ctx.offset);
        Axpy(vImag, aLocalImag[k * ctx.stride], h.invReal, ctx.offset);
    }

    // Head 分支里本 tile 前 k 行已是 R：v 的对应位置清零，并把 v[k] 固定为 1
    __aicore__ inline void MaskHeadVector(int k, const RowCtx& ctx, LocalTensor<float>& vReal)
    {
        LocalTensor<float> vImag = vReal[ctx.offset];
        Duplicate<float>(vReal, 0, k);
        Duplicate<float>(vImag, 0, k);
        vReal.SetValue(k, 1.0f);
        vImag.SetValue(k, 0.0f);
    }

    // 取负的虚部副本（y_imag_neg 用）；必须在 MaskHeadVector 之后算，否则首行清零会漏掉它
    __aicore__ inline void MakeNegImag(const RowCtx& ctx, LocalTensor<float>& vReal)
    {
        LocalTensor<float> vImag = vReal[ctx.offset];
        LocalTensor<float> vImagNeg = vImag[ctx.offset];
        Muls(vImagNeg, vImag, -1.0f, ctx.offset);
    }

    // v 上 GM 的三份副本：y_real / y_imag / y_imag_neg
    __aicore__ inline void PublishVector(int k, const RowCtx& ctx, LocalTensor<float>& vReal)
    {
        tmpQueueOut.EnQue(vReal);
        vReal = tmpQueueOut.DeQue<float>();
        LocalTensor<float> vImag = vReal[ctx.offset];
        LocalTensor<float> vImagNeg = vImag[ctx.offset];
        DataCopy(yGlobalReal[ctx.offsetY + k * padM], vReal, ctx.alignedRows);
        DataCopy(yGlobalImag[ctx.offsetY + k * padM], vImag, ctx.alignedRows);
        DataCopy(yGlobalImagNeg[ctx.offsetY + k * padM], vImagNeg, ctx.alignedRows);
    }

    // Head 分支的零列：tau 置 0（最后一列顺手写回 GM），y 清零并读回本列 W
    __aicore__ inline void SkipColumnHead(const RowCtx& ctx, int k, LocalTensor<float>& tauLocal,
                                          LocalTensor<float>& wLocalReal, LocalTensor<float>& wLocalImag)
    {
        tauLocal.SetValue(2 * k, 0.0f);
        tauLocal.SetValue(2 * k + 1, 0.0f);
        if (k + 1 >= ctx.columns)
        {
            StoreTau(ctx.offsetM, ctx.columns, tauLocal);
        }
        ZeroYAndLoadW(ctx, k, wLocalReal, wLocalImag);
    }

    // Head 分支的正常列：算 beta/tau，构造 v 并上 GM，再做本核的列更新
    __aicore__ inline void FactorColumnHead(const RowCtx& ctx, int k, float x0Real, float x0Imag, float norm2k,
                                            LocalTensor<float>& aLocalReal, LocalTensor<float>& aLocalImag,
                                            LocalTensor<float>& tauLocal, LocalTensor<float>& wLocalReal,
                                            LocalTensor<float>& wLocalImag);

    // 命中抵消阈值时把该列置 0，并用 norm_2[0] < 0 作为重算信号
    __aicore__ inline void MarkRecalcColumns(int k, const RowCtx& ctx, LocalTensor<float>& norm_2,
                                             LocalTensor<float>& norm_dir);

    // Head 分支的重算：本核按 tile 补算被置零的列，再原子累加回 GM
    __aicore__ inline void RecalcTailNormHead(const RowCtx& ctx, int k, LocalTensor<float>& aLocalReal,
                                              LocalTensor<float>& norm_2, LocalTensor<float>& norm_dir);

    // 用本列 v 的各方向分量更新后面各列的尾部范数
    __aicore__ inline void RefreshTailNorm(const RowCtx& ctx, int k, LocalTensor<float>& aLocalReal,
                                           LocalTensor<float>& norm_2);

    // Head 核的单列：广播表头元素 -> 取回尾范数（必要时精确重算）-> 零列或 Householder -> 更新尾部范数
    __aicore__ inline void ColumnStepHead(const RowCtx& ctx, int k, LocalTensor<float>& aLocalReal,
                                          LocalTensor<float>& aLocalImag, LocalTensor<float>& norm_2,
                                          LocalTensor<float>& tauLocal, LocalTensor<float>& wLocalReal,
                                          LocalTensor<float>& wLocalImag);

    __aicore__ inline void RowSplitProcessHead(int blockDim, int blockIdx, int offsetM, int offsetIdx, int rows,
                                               int columns, int flag);

    // 从 GM 读回 Head 核广播的本列表头元素（需先清 cache）
    __aicore__ inline void LoadPeerX0(int k, float& x0Real, float& x0Imag);

    // 行分块核的重算：本核先算自己那部分，再原子累加回 GM（与其它核配合）
    __aicore__ inline void RecalcTailNormPeer(const RowCtx& ctx, int k, LocalTensor<float>& aLocalReal,
                                              LocalTensor<float>& norm_2);

    // 行分块核的正常列：算 beta/tau，构造 v 并上 GM，再做本核的列更新
    __aicore__ inline void FactorColumnPeer(const RowCtx& ctx, int k, float x0Real, float x0Imag, float norm2k,
                                            LocalTensor<float>& aLocalReal, LocalTensor<float>& aLocalImag,
                                            LocalTensor<float>& tauLocal, LocalTensor<float>& wLocalReal,
                                            LocalTensor<float>& wLocalImag);

    // 行分块核的单列：等范数就绪（必要时协作重算）-> 读回表头元素 -> 零列或 Householder
    __aicore__ inline void ColumnStepPeer(const RowCtx& ctx, int k, LocalTensor<float>& aLocalReal,
                                          LocalTensor<float>& aLocalImag, LocalTensor<float>& tauLocal,
                                          LocalTensor<float>& wLocalReal, LocalTensor<float>& wLocalImag);

    __aicore__ inline void RowSplitProcess(int blockDim, int blockIdx, int offsetM, int offsetIdx, int rows,
                                           int columns, int flag);

    // 由 tau 与已算好的 W 递推本核负责的那一行 T（写回 T = -T^bar 的三份副本）
    __aicore__ inline void BuildT(int offsetT, int blockIdx, int columns, LocalTensor<float>& tauLocal,
                                  LocalTensor<float>& wLocalReal, LocalTensor<float>& wLocalImag);

    // 面板收尾：通知 Cube 本面板矢量部分已就绪，并把 R 写回 GM
    // 通知 Cube：本面板的矢量部分已就绪
    __aicore__ inline void NotifyPanelDone();

    // 面板收尾：通知 Cube 本面板矢量部分已就绪，并把 R 写回 GM
    __aicore__ inline void FinishPanel(const RowCtx& ctx, LocalTensor<float>& aLocalReal,
                                       LocalTensor<float>& aLocalImag);

    // 本核在本面板没有行要算：走完各列的同步旗标，并把 W 读回本地（后面的尾部更新还要用）
    __aicore__ inline void FetchPanelW(int blockIdx, int columns, int flag, LocalTensor<float>& wLocalReal,
                                       LocalTensor<float>& wLocalImag);

    // 从 GM 读回本面板的 tau（本核没有行要算时只能从 GM 拿）
    __aicore__ inline void LoadTau(int offsetM, int columns, LocalTensor<float>& tauLocal);

    __aicore__ inline void EmptyRowProcess(int blockDim, int blockIdx, int offsetM, int columns, int flag);
};
}  // namespace AscendC
