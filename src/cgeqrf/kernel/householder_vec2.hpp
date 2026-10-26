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
 * \file householder_vec2.hpp
 * \brief VEC2 侧：一次一列的 Householder QR（本文件保留类内定义）
 */

#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include "qr_common.hpp"

namespace AscendC
{
class HouseholderQR_VEC2 : public HouseholderQRCommon
{
   public:
    __aicore__ inline HouseholderQR_VEC2() : HouseholderQRCommon() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe)
    {
        pipe->InitBuffer(aQueueIn, 1, 2 * (chunkSize + 1024) * sizeof(float));
        pipe->InitBuffer(bQueueIn, 1, (chunkSize) * sizeof(float));
        pipe->InitBuffer(aQueueOut, 1, (chunkSize) * sizeof(float));
        pipe->InitBuffer(tQueueDst, 1, 2 * 1024 * sizeof(float));
        pipe->InitBuffer(tauQueueOut, 2, 128 * sizeof(float));
        pipe->InitBuffer(workQueue, (chunkSize + 3 * 1024) * sizeof(float));
        pipe->InitBuffer(gatherQueueIn, 1, 256 * sizeof(uint32_t));
    }
    __aicore__ inline void InitGather(GlobalTensor<uint32_t> gatherGlobal, int panel_size = 64)
    {
        gatherNLocal = gatherQueueIn.AllocTensor<uint32_t>();
        DataCopy(gatherNLocal, gatherGlobal, 2 * panel_size);
    }
    __aicore__ inline void FreeGather() { gatherQueueIn.FreeTensor(gatherNLocal); }
    __aicore__ inline void SetMatrix(GlobalTensor<float> aReal, GlobalTensor<float> aImag, GlobalTensor<float> tau_,
                                     GlobalTensor<float> tReal, GlobalTensor<float> tImag, GlobalTensor<float> tImagNeg,
                                     GlobalTensor<float> wReal, GlobalTensor<float> wImag, GlobalTensor<float> yReal,
                                     GlobalTensor<float> yImag, GlobalTensor<float> yImagNeg, int M_, int N_, int padM_,
                                     int padN_)
    {
        SetMatrixCommon(aReal, aImag, tau_, tReal, tImag, tImagNeg, wReal, wImag, yReal, yImag, yImagNeg, M_, N_,
                        padM_, padN_);
    }
    // 一个面板的发单：前 align_blockdim 个核真正计算，其余核只做全核同步；
    // 之后翻转 flag、推进 offset 并按剩余行数重算 columns（顺序不能改）
    __aicore__ inline void PanelStep(int blockDim, int blockIdx, int align_blockdim, int panel_size, int& offset, int& flag,
                                     int& columns)
    {
        if (blockIdx < align_blockdim)
        {
            Compute(align_blockdim, blockIdx, padM * offset + offset, M - offset, offset, flag, columns);
        }
        else
        {
            syn_all(align_blockdim, 0);
        }
        flag = flag ^ 1;
        offset += panel_size;
        columns = M - offset <= M_threshold ? 4 : 2;
    }
    __aicore__ inline void ComputeAll(int blockDim, int blockIdx)
    {
        int minMN = min(M, N);
        int offset = min(M - CHANGE, N);  // offset % 32 ?= 0
        int flag = 0;
        int columns = padM <= M_threshold ? 4 : 2;
        int align_blockdim = min(32, blockDim);
        int panel_size = align_blockdim * columns;

        for (; offset + panel_size <= minMN;)
        {
            PanelStep(blockDim, blockIdx, align_blockdim, panel_size, offset, flag, columns);
        }
    }
    __aicore__ inline void ComputeAll2(int blockDim, int blockIdx)
    {
        int minMN = min(M, N);
        int offset = 0;
        int flag = 0;
        int columns = 2;
        int align_blockdim = min(32, minMN);
        int panel_size = 64;
        // panel_size = min
        minMN = (minMN + panel_size - 1) / panel_size * panel_size;

        for (; offset + panel_size <= minMN;)
        {
            PanelStep(blockDim, blockIdx, align_blockdim, panel_size, offset, flag, columns);
        }
        for (; offset + panel_size <= min(padM, padN);)
        {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
            CrossCoreWaitFlag(0x9);
            offset += panel_size;
        }
    }
    __aicore__ inline void syn_all(int blockDim, int blockIdx)
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        for (int k = blockIdx + 1; k != blockDim; k++)
        {
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            // 读取中间结果
            CrossCoreWaitFlag(0x0);
        }
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x9);
    }

    // 面板分解的上下文：把派生量集中在一处，helper 之间只需再传少量 tensor
    struct PanelCtx
    {
        int blockDim, blockIdx, curM, offsetM, columns;
        int offset, offsetY, aligned_M;
    };

    __aicore__ inline void Compute(int blockDim, int blockIdx, int offsetA, int curM, int offsetM, int flag,
                                   int columns)
    {
        PanelCtx ctx{blockDim, blockIdx, curM, offsetM, columns, 0, 0, 0};
        ctx.offset = align_to(curM, 64);
        ctx.offsetY = offsetM * padM + offsetM % 128;
        ctx.aligned_M = align_to(curM, 8);
        // workLocal辅助计算 a保存当前列 householder_v用于拷贝到GM T_local存储blockidx行T矩阵
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> a = aQueueIn.AllocTensor<float>();
        LocalTensor<float> householder_v = aQueueOut.AllocTensor<float>();
        LocalTensor<float> T_local = tQueueDst.AllocTensor<float>();
        LocalTensor<float> tauLocal = tauQueueOut.AllocTensor<float>();
        LocalTensor<float> v0Local = tauQueueOut.AllocTensor<float>();
        float alpha[8];

        LoadPanel(ctx, a, offsetA);
        ApplyPrevReflectors(ctx, a, workLocal);
        FactorizeColumns(ctx, a, householder_v, workLocal, tauLocal, v0Local, alpha);

        // 等待当前列v向量拷贝过去
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        BuildW(ctx, a, householder_v, workLocal);
        BuildT(ctx, householder_v, workLocal, T_local, tauLocal);

        // 等待后一列v向量完成
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        UpdatePeers(ctx, a, householder_v, workLocal, T_local);
        StoreT(ctx, T_local);
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);

        tQueueDst.FreeTensor<float>(T_local);
        aQueueOut.FreeTensor<float>(householder_v);
        tauQueueOut.FreeTensor(tauLocal);
        tauQueueOut.FreeTensor(v0Local);

        CrossCoreWaitFlag(0x0);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
        StorePanel(ctx, a, alpha, offsetA);
        CrossCoreWaitFlag(0x9);
    }

    // A：按当前列块把实部/虚部载入 UB，并清掉尾部补零区
    __aicore__ inline void LoadPanel(const PanelCtx& ctx, LocalTensor<float>& a, int offsetA)
    {
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        DataCopyParams aCopyParams{static_cast<uint16_t>(ctx.columns), static_cast<uint16_t>(ctx.curM * sizeof(float)),
                                   static_cast<uint16_t>((padM - ctx.curM) * sizeof(float)),
                                   static_cast<uint16_t>((ctx.offset - ctx.aligned_M) / 8)};
        DataCopyPadParams aPadParams{true, 0, static_cast<uint8_t>(ctx.aligned_M - ctx.curM), 0};
        DataCopyPad(a, aGlobalReal[offsetA + padM * ctx.blockIdx * ctx.columns], aCopyParams, aPadParams);
        DataCopyPad(a[8192], aGlobalImag[offsetA + padM * ctx.blockIdx * ctx.columns], aCopyParams, aPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        // 逐列清零尾部补零区：原先用 6 参 Duplicate 带 dstRepeatStride(offset / 8)，
        // offset >= 2048 时该字段按 8 位回绕，补零区清不干净，残留数据会经 W/Y 更新污染 C
        for (int c = 0; c != ctx.columns; c++)
        {
            Duplicate<float>(a[c * ctx.offset + ctx.aligned_M], 0, ctx.offset - ctx.aligned_M);
            Duplicate<float>(a[c * ctx.offset + ctx.aligned_M + 8192], 0, ctx.offset - ctx.aligned_M);
        }
    }

    // 一个反射子（tau0, tau1）作用到第 j 列：先算两个内积，再实部/虚部各 2 次 Axpy
    __aicore__ inline void ApplyReflectorToColumn(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& b,
                                                 LocalTensor<float>& workLocal, int j, float tau0, float tau1)
    {
        Mul(workLocal, a[ctx.offset * j], b, ctx.aligned_M);
        Mul(workLocal[4096], a[ctx.offset * j + 8192], b[4096], ctx.aligned_M);
        ReduceSum(workLocal, workLocal, workLocal, ctx.aligned_M);
        ReduceSum(workLocal[4096], workLocal[4096], workLocal[4096], ctx.aligned_M);
        float scalar0 = workLocal.GetValue(0) + workLocal.GetValue(4096);
        Mul(workLocal, a[ctx.offset * j + 8192], b, ctx.aligned_M);
        Mul(workLocal[4096], a[ctx.offset * j], b[4096], ctx.aligned_M);
        ReduceSum(workLocal, workLocal, workLocal, ctx.aligned_M);
        ReduceSum(workLocal[4096], workLocal[4096], workLocal[4096], ctx.aligned_M);
        float scalar1 = workLocal.GetValue(0) - workLocal.GetValue(4096);
        Axpy(a[ctx.offset * j], b, -(tau0 * scalar0 - tau1 * scalar1), ctx.aligned_M);
        Axpy(a[ctx.offset * j], b[4096], (tau1 * scalar0 + tau0 * scalar1), ctx.aligned_M);
        Axpy(a[ctx.offset * j + 8192], b[4096], -(tau0 * scalar0 - tau1 * scalar1), ctx.aligned_M);
        Axpy(a[ctx.offset * j + 8192], b, -(tau1 * scalar0 + tau0 * scalar1), ctx.aligned_M);
    }

    // B：把前面各列块（k < blockIdx）的反射子作用到当前面板
    __aicore__ inline void ApplyPrevReflectorRow(const PanelCtx& ctx, LocalTensor<float>& a,
                                                 LocalTensor<float>& workLocal, int k, int i)
    {
        LocalTensor<float> b = bQueueIn.AllocTensor<float>();
        float tau0 = tau.GetValue(2 * (k * ctx.columns + i + ctx.offsetM));
        float tau1 = -tau.GetValue(2 * (k * ctx.columns + i + ctx.offsetM) + 1);
        DataCopy(b, yGlobalReal[ctx.offsetY + padM * k * ctx.columns + padM * i], ctx.aligned_M);
        DataCopy(b[4096], yGlobalImag[ctx.offsetY + padM * k * ctx.columns + padM * i], ctx.aligned_M);
        bQueueIn.EnQue(b);
        b = bQueueIn.DeQue<float>();
        for (int j = 0; j != ctx.columns; j++)
        {
            ApplyReflectorToColumn(ctx, a, b, workLocal, j, tau0, tau1);
        }
        bQueueIn.FreeTensor<float>(b);
    }
    __aicore__ inline void ApplyPrevReflectors(const PanelCtx& ctx, LocalTensor<float>& a,
                                               LocalTensor<float>& workLocal)
    {
        for (int k = 0; k != ctx.blockIdx; k++)
        {
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                tau[2 * (k * ctx.columns + ctx.offsetM)]);
            for (int i = 0; i != ctx.columns; i++)
            {
                ApplyPrevReflectorRow(ctx, a, workLocal, k, i);
            }
        }
    }

    // C：逐列做 Householder 分解，v 写回 UB 并同步到 Y，tau 实时写 GM
    __aicore__ inline void FactorizeColumns(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v,
                                            LocalTensor<float>& workLocal, LocalTensor<float>& tauLocal,
                                            LocalTensor<float>& v0Local, float* alpha)
    {
        int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));

        for (int i = 0; i != ctx.columns; i++)
        {
            float norm_2 = 0;
            float x0 = 0;
            ColumnNorm(ctx, a, workLocal, i, norm_2, x0);
            if (norm_2 < x0 * x0 + float(1e-6))
            {
                ZeroColumn(ctx, householder_v, tauLocal, v0Local, alpha, i, eventIDVToMTE3, eventIDMTE3ToS);
                continue;
            }
            FactorizeOne(ctx, a, householder_v, workLocal, tauLocal, v0Local, alpha, i, norm_2, x0, eventIDVToMTE3,
                         eventIDMTE3ToV);
        }
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            tau[2 * (ctx.blockIdx * ctx.columns + ctx.offsetM)]);
    }

    // C1：当前列在 blockIdx 之后的实虚平方和与首元素（用于判零列与求 beta）
    __aicore__ inline void ColumnNorm(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& workLocal, int i,
                                      float& norm_2, float& x0)
    {
        Mul(workLocal, a[ctx.offset * i], a[ctx.offset * i], ctx.aligned_M);
        Mul(workLocal[4096], a[ctx.offset * i + 8192], a[ctx.offset * i + 8192], ctx.aligned_M);
        Duplicate<float>(workLocal, 0, ctx.blockIdx * ctx.columns + i);
        Duplicate<float>(workLocal[4096], 0, ctx.blockIdx * ctx.columns + i);
        ReduceSum<float>(workLocal, workLocal, workLocal, ctx.aligned_M);
        ReduceSum<float>(workLocal[4096], workLocal[4096], workLocal[4096], ctx.aligned_M);
        norm_2 = workLocal.GetValue(0) + workLocal.GetValue(4096);
        x0 = a.GetValue(ctx.blockIdx * ctx.columns + i + ctx.offset * i);
    }

    // C2：退化列（已在下三角内）直接把 tau/v 置零并清 Y 的三块
    __aicore__ inline void ZeroColumn(const PanelCtx& ctx, LocalTensor<float>& householder_v, LocalTensor<float>& tauLocal,
                                      LocalTensor<float>& v0Local, float* alpha, int i, int32_t eventIDVToMTE3,
                                      int32_t eventIDMTE3ToS)
    {
        alpha[i] = 0;
        tauLocal.SetValue(2 * i, 0.0f);
        tauLocal.SetValue(2 * i + 1, 0.0f);

        v0Local.SetValue(i, 0.0f);
        Duplicate<float>(householder_v, 0, ctx.offset);
        tau.SetValue(2 * (ctx.blockIdx * ctx.columns + i + ctx.offsetM), 0.0f);
        tau.SetValue(2 * (ctx.blockIdx * ctx.columns + i + ctx.offsetM) + 1, 0.0f);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(yGlobalReal[ctx.offsetY + padM * (ctx.blockIdx * ctx.columns + i)], householder_v, ctx.offset);
        DataCopy(yGlobalImag[ctx.offsetY + padM * (ctx.blockIdx * ctx.columns + i)], householder_v, ctx.offset);
        DataCopy(yGlobalImagNeg[ctx.offsetY + padM * (ctx.blockIdx * ctx.columns + i)], householder_v, ctx.offset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    // C3：正常列：算 v/tau、写 Y、对右侧列做秩一更新、把 v 掩码写回 a
    __aicore__ inline void FactorizeOne(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v,
                                        LocalTensor<float>& workLocal, LocalTensor<float>& tauLocal,
                                        LocalTensor<float>& v0Local, float* alpha, int i, float norm_2, float x0,
                                        int32_t eventIDVToMTE3, int32_t eventIDMTE3ToV)
    {
        float x1 = a.GetValue(ctx.blockIdx * ctx.columns + i + ctx.offset * i + 8192);
        float norm_x = sqrt(norm_2);
        alpha[i] = x0 < 0.0f ? -norm_x : norm_x;
        v0Local.SetValue(i, x0 + alpha[i]);
        a.SetValue(ctx.blockIdx * ctx.columns + i + ctx.offset * i, v0Local.GetValue(i));
        float sum = v0Local.GetValue(i) * v0Local.GetValue(i) + x1 * x1;
        float tau0 = v0Local.GetValue(i) / alpha[i];
        float tau1 = x1 / alpha[i];

        // 将当前列v的tau实时更新到GM
        Copy(householder_v, a[ctx.offset * i], 64, ctx.offset / 64, {1, 1, 8, 8});
        Copy(householder_v[4096], a[ctx.offset * i + 8192], 64, ctx.offset / 64, {1, 1, 8, 8});
        tauLocal.SetValue(2 * i, tau0);
        tauLocal.SetValue(2 * i + 1, tau1);
        tau.SetValue(2 * (ctx.blockIdx * ctx.columns + i + ctx.offsetM), tau0);
        tau.SetValue(2 * (ctx.blockIdx * ctx.columns + i + ctx.offsetM) + 1, tau1);

        Muls(householder_v, householder_v, v0Local.GetValue(i) / sum, ctx.aligned_M);
        Muls(householder_v[4096], householder_v[4096], v0Local.GetValue(i) / sum, ctx.aligned_M);
        Axpy(householder_v, a[ctx.offset * i + 8192], x1 / sum, ctx.aligned_M);
        Axpy(householder_v[4096], a[ctx.offset * i], -x1 / sum, ctx.aligned_M);

        Duplicate<float>(householder_v, 0, ctx.blockIdx * ctx.columns + i);
        Duplicate<float>(householder_v[4096], 0, ctx.blockIdx * ctx.columns + i);
        RankOneUpdate(ctx, a, householder_v, workLocal, tau0, tau1, i);
        CommitColumn(ctx, a, householder_v, i, eventIDVToMTE3, eventIDMTE3ToV);
    }

    // C4：把当前列的反射子作用到同面板右侧各列
    __aicore__ inline void RankOneUpdate(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v,
                                         LocalTensor<float>& workLocal, float tau0, float tau1, int i)
    {
        for (int j = i + 1; j != ctx.columns; j++)
        {
            Mul(workLocal, a[ctx.offset * j], householder_v, ctx.aligned_M);
            Mul(workLocal[4096], a[ctx.offset * j + 8192], householder_v[4096], ctx.aligned_M);
            ReduceSum(workLocal, workLocal, workLocal, ctx.aligned_M);
            ReduceSum(workLocal[4096], workLocal[4096], workLocal[4096], ctx.aligned_M);
            float scalar0 = workLocal.GetValue(0) + workLocal.GetValue(4096);
            Mul(workLocal, a[ctx.offset * j + 8192], householder_v, ctx.aligned_M);
            Mul(workLocal[4096], a[ctx.offset * j], householder_v[4096], ctx.aligned_M);
            ReduceSum(workLocal, workLocal, workLocal, ctx.aligned_M);
            ReduceSum(workLocal[4096], workLocal[4096], workLocal[4096], ctx.aligned_M);
            float scalar1 = workLocal.GetValue(0) - workLocal.GetValue(4096);
            Axpy(a[ctx.offset * j], householder_v, -(tau0 * scalar0 + tau1 * scalar1), ctx.aligned_M);
            Axpy(a[ctx.offset * j], householder_v[4096], (-tau1 * scalar0 + tau0 * scalar1), ctx.aligned_M);
            Axpy(a[ctx.offset * j + 8192], householder_v[4096], -(tau0 * scalar0 + tau1 * scalar1), ctx.aligned_M);
            Axpy(a[ctx.offset * j + 8192], householder_v, -(-tau1 * scalar0 + tau0 * scalar1), ctx.aligned_M);
        }
    }

    // C5：把 v 的掩码段写回 a，并把实部/虚部/-虚部同步到 Y
    __aicore__ inline void CommitColumn(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v, int i,
                                        int32_t eventIDVToMTE3, int32_t eventIDMTE3ToV)
    {
        int startM = ctx.blockIdx * ctx.columns / 64 * 64;
        uint64_t mask_update[2] = {(~0ULL) ^ ((1ULL << (ctx.blockIdx * ctx.columns + i - startM)) - 1), 0};

        if (startM + 64 < ctx.aligned_M)
        {
            DataCopy(a[i * ctx.offset + startM + 64], householder_v[startM + 64], ctx.aligned_M - startM - 64);
            DataCopy(a[i * ctx.offset + startM + 64 + 8192], householder_v[startM + 64 + 4096],
                     ctx.aligned_M - startM - 64);
        }
        Copy(a[i * ctx.offset + startM], householder_v[startM], mask_update, 1, {1, 1, 8, 8});
        Copy(a[i * ctx.offset + startM + 8192], householder_v[startM + 4096], mask_update, 1, {1, 1, 8, 8});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        // 确保拷贝过去的v向量是没有脏元素的
        DataCopy(yGlobalImag[ctx.offsetY + padM * (ctx.blockIdx * ctx.columns + i)], householder_v[4096], ctx.aligned_M);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        Muls(householder_v[4096], householder_v[4096], -1.0f, ctx.aligned_M);

        DataCopy(yGlobalReal[ctx.offsetY + padM * (ctx.blockIdx * ctx.columns + i)], householder_v, ctx.aligned_M);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(yGlobalImagNeg[ctx.offsetY + padM * (ctx.blockIdx * ctx.columns + i)], householder_v[4096],
                 ctx.aligned_M);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    }

    // D：本核这一列块内部两两内积，得到 W（写到 W GM）
    __aicore__ inline void BuildW(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v,
                                  LocalTensor<float>& workLocal);

    // E：由 W 递推本核负责的 T 行（实部在前 1024，虚部在后 1024）
    __aicore__ inline void BuildT(const PanelCtx& ctx, LocalTensor<float>& householder_v, LocalTensor<float>& workLocal,
                                  LocalTensor<float>& T_local, LocalTensor<float>& tauLocal);

    // F：对后面各列块（k > blockIdx）计算 W 中间结果并递推对应的 T 行
    __aicore__ inline void UpdatePeers(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v,
                                       LocalTensor<float>& workLocal, LocalTensor<float>& T_local);

    // F1：本核对第 k 列块算 W 的中间结果（实部/虚部分开写，避免连续地址写冲突）
    __aicore__ inline void WritePeerW(const PanelCtx& ctx, LocalTensor<float>& a, LocalTensor<float>& householder_v,
                                      LocalTensor<float>& workLocal, int k);

    // F2：读回中间结果 W，递推本核负责的 T 行
    __aicore__ inline void UpdatePeerT(const PanelCtx& ctx, LocalTensor<float>& workLocal, LocalTensor<float>& T_local,
                                       int k);

    // G：把没有脏元素的 T 矩阵写回（imag / -real / imagNeg 三块）
    __aicore__ inline void StoreT(const PanelCtx& ctx, LocalTensor<float>& T_local);

    // H：对角写回 -alpha，并把本列块的 a 写回 GM
    __aicore__ inline void StorePanel(const PanelCtx& ctx, LocalTensor<float>& a, float* alpha, int offsetA);
   private:
    int panelYSize, panelTSize;
    const int chunkSize = 8192;
    TQue<QuePosition::VECIN, 1> aQueueIn;
    TQue<QuePosition::VECIN, 1> bQueueIn;
    TQue<QuePosition::VECOUT, 1> aQueueOut;
    TBuf<QuePosition::VECCALC> workQueue;
    TQue<QuePosition::VECOUT, 1> tQueueDst;
    TQue<QuePosition::VECOUT, 2> tauQueueOut;
    TQue<QuePosition::VECIN, 1> gatherQueueIn;  // gatherN: 64, gatherW: 32

    LocalTensor<uint32_t> gatherNLocal;
};
}  // namespace AscendC
