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
 * \file householder_vec_impl.hpp
 * \brief HouseholderQR_VEC 的类外成员定义（行分块与 peer 更新）
 */

#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include "householder_vec.hpp"

namespace AscendC
{
    __aicore__ inline void HouseholderQR_VEC::FactorColumnHead(const RowCtx& ctx, int k, float x0Real,
                                                               float x0Imag, float norm2k,
                                                               LocalTensor<float>& aLocalReal,
                                                               LocalTensor<float>& aLocalImag,
                                                               LocalTensor<float>& tauLocal,
                                                               LocalTensor<float>& wLocalReal,
                                                               LocalTensor<float>& wLocalImag)
    {
        Householder h;
        MakeHouseholder(h, x0Real, x0Imag, max(x0Real * x0Real + x0Imag * x0Imag, norm2k));
        tauLocal.SetValue(2 * k, h.tauReal);
        tauLocal.SetValue(2 * k + 1, h.tauImag);
        if (k + 1 >= ctx.columns)
        {
            StoreTau(ctx.offsetM, ctx.columns, tauLocal);
        }

        LocalTensor<float> vReal = tmpQueueOut.AllocTensor<float>();
        BuildVector(k, ctx, aLocalReal, aLocalImag, h, vReal);
        MaskHeadVector(k, ctx, vReal);
        MakeNegImag(ctx, vReal);
        PublishVector(k, ctx, vReal);

        LocalTensor<float> vImag = vReal[ctx.offset];
        Update(k, ctx.offsetM, ctx.columns, ctx.offset, ctx.stride, vReal, vImag, aLocalReal, aLocalImag,
               wLocalReal[k * PANEL_SIZE], wLocalImag[k * PANEL_SIZE], h.tauReal, h.tauImag, ctx.flag);

        uint64_t mask[2] = {~((1ull << (k + 1)) - 1), 0};
        Copy(aLocalReal[k * ctx.stride], vReal, mask, 2, {1, 1, 8, 8});
        tmpQueueOut.FreeTensor(vReal);
        aLocalReal.SetValue(k * ctx.stride + k, h.beta);
        aLocalImag.SetValue(k * ctx.stride + k, 0.0f);
    }

    __aicore__ inline void HouseholderQR_VEC::MarkRecalcColumns(int k, const RowCtx& ctx,
                                                                LocalTensor<float>& norm_2,
                                                                LocalTensor<float>& norm_dir)
    {
        norm_2.SetValue(0, 1e-8f);
        for (int i = k + 1; i < ctx.columns; ++i)
        {
            float dir = norm_dir.GetValue(i);
            float inc = norm_2.GetValue(i);
            if (dir > 0.0f && inc < 2e-4f / (1.0f - 2e-4f) * dir)
            {
                norm_2.SetValue(i, 0.0f);
                norm_2.SetValue(0, -1e-8f);
            }
        }
    }

    __aicore__ inline void HouseholderQR_VEC::RecalcTailNormHead(const RowCtx& ctx, int k,
                                                                 LocalTensor<float>& aLocalReal,
                                                                 LocalTensor<float>& norm_2,
                                                                 LocalTensor<float>& norm_dir)
    {
        DataCopy(tmpGlobal[(ctx.flag & 1) * PANEL_SIZE], norm_2, PANEL_SIZE);

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreSetFlag<0x0, PIPE_MTE2>(0x4);

        Duplicate<float>(norm_dir, 0.0f, PANEL_SIZE);
        LocalTensor<float> workLocal = workBuf.Get<float>();
        for (int i = k + 1; i < ctx.columns; ++i)
        {
            if (norm_2.GetValue(i) <= 0.0f)
            {
                Mul(workLocal, aLocalReal[i * ctx.stride], aLocalReal[i * ctx.stride], ctx.stride);
                // 只统计第 k+1 行起的剩余部分（前 k+1 行已是 R，不参与范数）
                Duplicate<float>(workLocal, 0.0f, k + 1);
                Duplicate<float>(workLocal[ctx.offset], 0.0f, k + 1);
                Add(workLocal, workLocal, workLocal[ctx.offset], ctx.offset);
                ReduceSum<float>(workLocal, workLocal, workLocal, ctx.offset);
                norm_dir.SetValue(i, workLocal.GetValue(0));
            }
        }
        CrossCoreWaitFlag(0x0);
        CrossCoreWaitFlag(0x4);

        tmpQueueOut.EnQue(norm_dir);
        norm_dir = tmpQueueOut.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopy(tmpGlobal[(ctx.flag & 1) * PANEL_SIZE], norm_dir, PANEL_SIZE);
        SetAtomicNone();
    }

    __aicore__ inline void HouseholderQR_VEC::RefreshTailNorm(const RowCtx& ctx, int k,
                                                              LocalTensor<float>& aLocalReal,
                                                              LocalTensor<float>& norm_2)
    {
        LocalTensor<float> norm_dir = tmpQueueOut.AllocTensor<float>();
        Gather(norm_dir, aLocalReal, gatherNLocal, k * sizeof(float), 2 * PANEL_SIZE);
        Mul(norm_dir, norm_dir, norm_dir, 2 * PANEL_SIZE);
        Add(norm_dir, norm_dir, norm_dir[PANEL_SIZE], PANEL_SIZE);
        Sub(norm_2, norm_2, norm_dir, PANEL_SIZE);

        MarkRecalcColumns(k, ctx, norm_2, norm_dir);
        if (norm_2.GetValue(0) < 0.0f)
        {
            // let 0 position be the flag, if < 0, then re-calc
            RecalcTailNormHead(ctx, k, aLocalReal, norm_2, norm_dir);
        }
        else
        {
            DataCopy(tmpGlobal[(ctx.flag & 1) * PANEL_SIZE], norm_2, PANEL_SIZE);
        }
        tmpQueueOut.FreeTensor(norm_dir);
    }

    __aicore__ inline void HouseholderQR_VEC::ColumnStepHead(const RowCtx& ctx, int k,
                                                             LocalTensor<float>& aLocalReal,
                                                             LocalTensor<float>& aLocalImag,
                                                             LocalTensor<float>& norm_2,
                                                             LocalTensor<float>& tauLocal,
                                                             LocalTensor<float>& wLocalReal,
                                                             LocalTensor<float>& wLocalImag)
    {
        float x0Real = aLocalReal.GetValue(k * ctx.stride + k - ctx.offsetIdx);
        float x0Imag = aLocalImag.GetValue(k * ctx.stride + k - ctx.offsetIdx);
        BroadcastX0(k, x0Real, x0Imag);

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);  // norm process fin
        CrossCoreWaitFlag(0x0);

        ReloadNorm(ctx.flag, norm_2);
        float norm_2_k = norm_2.GetValue(k);
        if (ctx.rows >= M - ctx.offsetM)
        {
            norm_2_k = ExactTailNorm(k, ctx, aLocalReal);
        }
        if (IsZeroColumn(x0Real, norm_2.GetValue(k)))
        {
            SkipColumnHead(ctx, k, tauLocal, wLocalReal, wLocalImag);
        }
        else
        {
            FactorColumnHead(ctx, k, x0Real, x0Imag, norm_2_k, aLocalReal, aLocalImag, tauLocal, wLocalReal,
                             wLocalImag);
        }
        RefreshTailNorm(ctx, k, aLocalReal, norm_2);
    }

    __aicore__ inline void HouseholderQR_VEC::RowSplitProcessHead(int blockDim, int blockIdx, int offsetM,
                                                                  int offsetIdx, int rows, int columns, int flag)
    {
        // assert(rows <= 64)
        RowCtx ctx;
        MakeRowCtx(ctx, offsetM, offsetIdx, rows, columns, flag);

        LocalTensor<float> aLocalReal = aQueueIn.AllocTensor<float>();
        LocalTensor<float> aLocalImag = aLocalReal[ctx.offset];
        LocalTensor<float> wLocalReal = wQueueOut.AllocTensor<float>();
        LocalTensor<float> wLocalImag = wLocalReal[PANEL_SIZE * PANEL_SIZE];
        LocalTensor<float> norm_2 = normQueueIn.AllocTensor<float>();
        LocalTensor<float> tauLocal = aQueueOut.AllocTensor<float>();

        ResetPanelW(wLocalReal, wLocalImag, norm_2, flag);
        WaitCubePrevPanel(offsetM);
        CopyInAndProcessNorm(ctx.offsetA, offsetM, rows, columns, ctx.alignedRows, ctx.offset, ctx.stride, aLocalReal,
                             aLocalImag, flag);

        for (int k = 0; k < columns; ++k)
        {
            ColumnStepHead(ctx, k, aLocalReal, aLocalImag, norm_2, tauLocal, wLocalReal, wLocalImag);
        }
        normQueueIn.FreeTensor(norm_2);

        if (blockIdx < columns)
        {  // Process T
            BuildT(ctx.offsetT, blockIdx, columns, tauLocal, wLocalReal, wLocalImag);
        }
        wQueueOut.FreeTensor(wLocalReal);
        aQueueOut.FreeTensor(tauLocal);

        FinishPanel(ctx, aLocalReal, aLocalImag);
        aQueueIn.FreeTensor(aLocalReal);
    }

    __aicore__ inline void HouseholderQR_VEC::LoadPeerX0(int k, float& x0Real, float& x0Imag)
    {
        DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            v0Global[2 * k]);
        x0Real = v0Global.GetValue(2 * k);
        x0Imag = v0Global.GetValue(2 * k + 1);
    }

    __aicore__ inline void HouseholderQR_VEC::RecalcTailNormPeer(const RowCtx& ctx, int k,
                                                                 LocalTensor<float>& aLocalReal,
                                                                 LocalTensor<float>& norm_2)
    {
        LocalTensor<float> norm_dir = tmpQueueOut.AllocTensor<float>();
        CrossCoreSetFlag<0x0, PIPE_MTE2>(0x4);

        Duplicate<float>(norm_dir, 0.0f, PANEL_SIZE);
        LocalTensor<float> workLocal = workBuf.Get<float>();
        for (int i = k + 1; i < ctx.columns; ++i)
        {
            if (norm_2.GetValue(i) <= 0.0f)
            {
                Mul(workLocal, aLocalReal[i * ctx.stride], aLocalReal[i * ctx.stride], ctx.stride);
                ReduceSum<float>(workLocal, workLocal, workLocal, ctx.stride);
                norm_dir.SetValue(i, workLocal.GetValue(0));
            }
        }
        normQueueIn.FreeTensor(norm_2);
        CrossCoreWaitFlag(0x4);

        tmpQueueOut.EnQue(norm_dir);
        norm_dir = tmpQueueOut.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopy(tmpGlobal[(ctx.flag & 1) * PANEL_SIZE], norm_dir, PANEL_SIZE);
        SetAtomicNone();
        tmpQueueOut.FreeTensor(norm_dir);

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);

        norm_2 = normQueueIn.AllocTensor<float>();
        ReloadNorm(ctx.flag, norm_2);
    }

    __aicore__ inline void HouseholderQR_VEC::FactorColumnPeer(const RowCtx& ctx, int k, float x0Real,
                                                               float x0Imag, float norm2k,
                                                               LocalTensor<float>& aLocalReal,
                                                               LocalTensor<float>& aLocalImag,
                                                               LocalTensor<float>& tauLocal,
                                                               LocalTensor<float>& wLocalReal,
                                                               LocalTensor<float>& wLocalImag)
    {
        Householder h;
        MakeHouseholder(h, x0Real, x0Imag, max(x0Real * x0Real + x0Imag * x0Imag, norm2k));
        tauLocal.SetValue(k * 2, h.tauReal);
        tauLocal.SetValue(k * 2 + 1, h.tauImag);

        LocalTensor<float> vReal = tmpQueueOut.AllocTensor<float>();
        BuildVector(k, ctx, aLocalReal, aLocalImag, h, vReal);
        MakeNegImag(ctx, vReal);
        PublishVector(k, ctx, vReal);

        LocalTensor<float> vImag = vReal[ctx.offset];
        Update(k, ctx.offsetM, ctx.columns, ctx.offset, ctx.stride, vReal, vImag, aLocalReal, aLocalImag,
               wLocalReal[k * PANEL_SIZE], wLocalImag[k * PANEL_SIZE], h.tauReal, h.tauImag, ctx.flag);
        Copy(aLocalReal[k * ctx.stride], vReal, 64, ctx.stride / 64, {1, 1, 8, 8});
        tmpQueueOut.FreeTensor(vReal);
    }

    __aicore__ inline void HouseholderQR_VEC::ColumnStepPeer(const RowCtx& ctx, int k,
                                                             LocalTensor<float>& aLocalReal,
                                                             LocalTensor<float>& aLocalImag,
                                                             LocalTensor<float>& tauLocal,
                                                             LocalTensor<float>& wLocalReal,
                                                             LocalTensor<float>& wLocalImag)
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);  // wait write norms
        CrossCoreWaitFlag(0x0);

        LocalTensor<float> norm_2 = normQueueIn.AllocTensor<float>();
        ReloadNorm(ctx.flag, norm_2);
        if (norm_2.GetValue(0) < 0.0f)
        {  // re-calc norm
            RecalcTailNormPeer(ctx, k, aLocalReal, norm_2);
        }

        float x0Real = 0.0f, x0Imag = 0.0f;
        LoadPeerX0(k, x0Real, x0Imag);
        if (IsZeroColumn(x0Real, norm_2.GetValue(k)))
        {
            // skip to next col
            ZeroYAndLoadW(ctx, k, wLocalReal, wLocalImag);
        }
        else
        {
            FactorColumnPeer(ctx, k, x0Real, x0Imag, norm_2.GetValue(k), aLocalReal, aLocalImag, tauLocal, wLocalReal,
                             wLocalImag);
        }
        normQueueIn.FreeTensor(norm_2);
    }

    __aicore__ inline void HouseholderQR_VEC::RowSplitProcess(int blockDim, int blockIdx, int offsetM,
                                                              int offsetIdx, int rows, int columns, int flag)
    {
        RowCtx ctx;
        MakeRowCtx(ctx, offsetM, offsetIdx, rows, columns, flag);

        LocalTensor<float> aLocalReal = aQueueIn.AllocTensor<float>();
        LocalTensor<float> aLocalImag = aLocalReal[ctx.offset];
        LocalTensor<float> wLocalReal = wQueueOut.AllocTensor<float>();
        LocalTensor<float> wLocalImag = wLocalReal[PANEL_SIZE * PANEL_SIZE];
        LocalTensor<float> tauLocal = aQueueOut.AllocTensor<float>();
        Duplicate<float>(wLocalReal, 0.0f, 2 * PANEL_SIZE * PANEL_SIZE);

        WaitCubePrevPanel(offsetM);
        CopyInAndProcessNorm(ctx.offsetA, offsetM, rows, columns, ctx.alignedRows, ctx.offset, ctx.stride, aLocalReal,
                             aLocalImag, flag);

        for (int k = 0; k < columns; ++k)
        {
            ColumnStepPeer(ctx, k, aLocalReal, aLocalImag, tauLocal, wLocalReal, wLocalImag);
        }
        aQueueOut.EnQue(tauLocal);
        tauLocal = aQueueOut.DeQue<float>();

        if (blockIdx < columns)
        {  // Process T
            BuildT(ctx.offsetT, blockIdx, columns, tauLocal, wLocalReal, wLocalImag);
        }
        wQueueOut.FreeTensor(wLocalReal);
        aQueueOut.FreeTensor(tauLocal);

        FinishPanel(ctx, aLocalReal, aLocalImag);
        aQueueIn.FreeTensor(aLocalReal);  // FIX: 若不释放，下一 panel 的 AllocTensor 会永久阻塞
    }

    __aicore__ inline void HouseholderQR_VEC::BuildT(int offsetT, int blockIdx, int columns,
                                                     LocalTensor<float>& tauLocal,
                                                     LocalTensor<float>& wLocalReal,
                                                     LocalTensor<float>& wLocalImag)
    {
        LocalTensor<float> tLocalReal = tmpQueueOut.AllocTensor<float>();
        LocalTensor<float> tLocalImag = tLocalReal[PANEL_SIZE];
        LocalTensor<float> tLocalImagNeg = tLocalImag[PANEL_SIZE];
        Duplicate<float>(tLocalReal, 0, 2 * PANEL_SIZE);
        tLocalReal.SetValue(blockIdx, tauLocal.GetValue(blockIdx * 2));
        tLocalImag.SetValue(blockIdx, tauLocal.GetValue(blockIdx * 2 + 1));
        for (int i = blockIdx + 1; i < columns; ++i)
        {
            // w[i,j] = v_i^H * v_j, need v_j^H * v_i = w[i,j].conj
            float tauReal = tauLocal.GetValue(i * 2);
            float tauImag = tauLocal.GetValue(i * 2 + 1);
            LocalTensor<float> workLocal = workBuf.Get<float>();
            Mul(workLocal, tLocalReal, wLocalReal[i * PANEL_SIZE], PANEL_SIZE);
            MulAddDst(workLocal, tLocalImag, wLocalImag[i * PANEL_SIZE], PANEL_SIZE);  // -1 comes from w_imag
            ReduceSum<float>(workLocal, workLocal, workLocal, columns);
            float sReal = workLocal.GetValue(0);
            Mul(workLocal, tLocalReal, wLocalImag[i * PANEL_SIZE], PANEL_SIZE);
            Muls(workLocal, workLocal, -1.0f, PANEL_SIZE);  // -1 comes from w_imag
            MulAddDst(workLocal, tLocalImag, wLocalReal[i * PANEL_SIZE], PANEL_SIZE);
            ReduceSum<float>(workLocal, workLocal, workLocal, columns);
            float sImag = workLocal.GetValue(0);
            tLocalReal.SetValue(i, sImag * tauImag - sReal * tauReal);
            tLocalImag.SetValue(i, -sImag * tauReal - sReal * tauImag);
        }
        Muls(tLocalReal, tLocalReal, -1.0f, PANEL_SIZE);
        Muls(tLocalImagNeg, tLocalImag, -1.0f, PANEL_SIZE);
        tmpQueueOut.EnQue(tLocalReal);
        tLocalReal = tmpQueueOut.DeQue<float>();
        tLocalImag = tLocalReal[PANEL_SIZE];
        tLocalImagNeg = tLocalImag[PANEL_SIZE];
        DataCopy(tGlobalReal[offsetT + blockIdx * padM], tLocalReal, PANEL_SIZE);
        DataCopy(tGlobalImagNeg[offsetT + blockIdx * padM], tLocalImag, PANEL_SIZE);
        DataCopy(tGlobalImag[offsetT + blockIdx * padM], tLocalImagNeg, PANEL_SIZE);  // T = -T^bar
        tmpQueueOut.FreeTensor(tLocalReal);
    }

    __aicore__ inline void HouseholderQR_VEC::NotifyPanelDone()
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
    }

    __aicore__ inline void HouseholderQR_VEC::FinishPanel(const RowCtx& ctx, LocalTensor<float>& aLocalReal,
                                                          LocalTensor<float>& aLocalImag)
    {
        NotifyPanelDone();

        DataCopyParams aDataCopyOutParams{static_cast<uint16_t>(ctx.columns),
                                          static_cast<uint16_t>(ctx.alignedRows / 8),
                                          static_cast<uint16_t>((ctx.stride - ctx.alignedRows) / 8),
                                          static_cast<uint16_t>((padM - ctx.rows) / 8)};
        DataCopy(aGlobalReal[ctx.offsetA], aLocalReal, aDataCopyOutParams);
        DataCopy(aGlobalImag[ctx.offsetA], aLocalImag, aDataCopyOutParams);

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
    }

    __aicore__ inline void HouseholderQR_VEC::FetchPanelW(int blockIdx, int columns, int flag,
                                                          LocalTensor<float>& wLocalReal,
                                                          LocalTensor<float>& wLocalImag)
    {
        for (int k = 0; k < columns; ++k)
        {
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);
            LocalTensor<float> norm_2 = normQueueIn.AllocTensor<float>();
            ReloadNorm(flag, norm_2);
            if (norm_2.GetValue(0) < 0)
            {
                CrossCoreSetFlag<0x0, PIPE_MTE2>(0x4);
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                CrossCoreWaitFlag(0x4);
                CrossCoreWaitFlag(0x0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);

            if (blockIdx < columns)
            {
                int offsetW = (flag & 1) * PANEL_SIZE * PANEL_SIZE + k * PANEL_SIZE;
                DataCopy(wLocalReal[k * PANEL_SIZE], wGlobalReal[offsetW], PANEL_SIZE);
                DataCopy(wLocalImag[k * PANEL_SIZE], wGlobalImag[offsetW], PANEL_SIZE);
            }
            normQueueIn.FreeTensor(norm_2);
        }
    }

    __aicore__ inline void HouseholderQR_VEC::LoadTau(int offsetM, int columns, LocalTensor<float>& tauLocal)
    {
        DataCopyExtParams tauDataCopyParams{static_cast<uint16_t>(1),
                                            static_cast<uint32_t>(2 * columns * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> tauPadParams{0, 0, 0, 0};
        DataCopyPad<float>(tauLocal, tau[2 * offsetM], tauDataCopyParams, tauPadParams);
        normQueueIn.EnQue(tauLocal);
        tauLocal = normQueueIn.DeQue<float>();
    }

    __aicore__ inline void HouseholderQR_VEC::EmptyRowProcess(int blockDim, int blockIdx, int offsetM,
                                                              int columns, int flag)
    {
        int offsetT = ((flag / 4) * panelTSize) + (offsetM & 127) * (padM + 1);
        LocalTensor<float> wLocalReal = wQueueOut.AllocTensor<float>();
        LocalTensor<float> wLocalImag = wLocalReal[PANEL_SIZE * PANEL_SIZE];

        WaitCubePrevPanel(offsetM);
        FetchPanelW(blockIdx, columns, flag, wLocalReal, wLocalImag);

        if (blockIdx < columns)
        {  // Process T
            LocalTensor<float> tauLocal = normQueueIn.AllocTensor<float>();
            LoadTau(offsetM, columns, tauLocal);
            BuildT(offsetT, blockIdx, columns, tauLocal, wLocalReal, wLocalImag);
            normQueueIn.FreeTensor(tauLocal);
        }
        wQueueOut.FreeTensor(wLocalReal);

        NotifyPanelDone();
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
    }

}  // namespace AscendC
