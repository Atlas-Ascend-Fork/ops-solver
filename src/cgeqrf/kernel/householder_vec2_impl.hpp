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
 * \file householder_vec2_impl.hpp
 * \brief HouseholderQR_VEC2 的类外成员定义（W/T 构造与 peer 更新）
 */

#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include "householder_vec2.hpp"

namespace AscendC
{
    __aicore__ inline void HouseholderQR_VEC2::BuildW(const PanelCtx& ctx, LocalTensor<float>& a,
                                                      LocalTensor<float>& householder_v,
                                                      LocalTensor<float>& workLocal)
    {
        Duplicate<float>(householder_v, 0, 8 * ctx.columns);
        Duplicate<float>(householder_v[4096], 0, 8 * ctx.columns);
        float scalar = 0;
        for (int i = ctx.columns - 1; i >= 0; i--)
        {
            for (int j = i - 1; j >= 0; j--)
            {
                Mul(workLocal[1024], a[ctx.offset * i], a[ctx.offset * j], ctx.aligned_M);
                Mul(workLocal[5120], a[ctx.offset * i + 8192], a[ctx.offset * j + 8192], ctx.aligned_M);
                Duplicate<float>(workLocal[1024], 0, ctx.blockIdx * ctx.columns + i);
                Duplicate<float>(workLocal[5120], 0, ctx.blockIdx * ctx.columns + i);
                ReduceSum(workLocal[1024], workLocal[1024], workLocal[1024], ctx.aligned_M);
                ReduceSum(workLocal[5120], workLocal[5120], workLocal[5120], ctx.aligned_M);
                scalar = workLocal.GetValue(1024) + workLocal.GetValue(5120);
                householder_v.SetValue(i * 8 + j, scalar);
            }
            DataCopy(wGlobalReal[ctx.blockDim * 8 * i + ctx.blockIdx * 8 + (ctx.blockIdx % 2) * 2048],
                     householder_v[i * 8], 8);
            for (int j = i - 1; j >= 0; j--)
            {
                Mul(workLocal[1024], a[ctx.offset * i], a[ctx.offset * j + 8192], ctx.aligned_M);
                Mul(workLocal[5120], a[ctx.offset * i + 8192], a[ctx.offset * j], ctx.aligned_M);
                Duplicate<float>(workLocal[1024], 0, ctx.blockIdx * ctx.columns + i);
                Duplicate<float>(workLocal[5120], 0, ctx.blockIdx * ctx.columns + i);
                ReduceSum(workLocal[1024], workLocal[1024], workLocal[1024], ctx.aligned_M);
                ReduceSum(workLocal[5120], workLocal[5120], workLocal[5120], ctx.aligned_M);
                scalar = workLocal.GetValue(5120) - workLocal.GetValue(1024);
                householder_v.SetValue(i * 8 + j + 4096, scalar);
            }
            DataCopy(wGlobalImag[ctx.blockDim * 8 * i + ctx.blockIdx * 8 + (ctx.blockIdx % 2) * 2048],
                     householder_v[i * 8 + 4096], 8);
        }
    }

    __aicore__ inline void HouseholderQR_VEC2::BuildT(const PanelCtx& ctx, LocalTensor<float>& householder_v,
                                                      LocalTensor<float>& workLocal,
                                                      LocalTensor<float>& T_local, LocalTensor<float>& tauLocal)
    {
        Duplicate<float>(T_local, 0, ctx.blockDim * ctx.columns * ctx.columns);
        Duplicate<float>(workLocal, 0, ctx.blockDim * ctx.columns * 8);  // 64 * 8
        Duplicate<float>(T_local[1024], 0, ctx.blockDim * ctx.columns * ctx.columns);
        Duplicate<float>(workLocal[1024], 0, ctx.blockDim * ctx.columns * 8);
        // 设置为负，从而算出-T，workLocal存储“分散”的当前T的一行
        for (int i = 0; i != ctx.columns; i++)
        {
            float tau0 = -tauLocal.GetValue(2 * i);
            float tau1 = -tauLocal.GetValue(2 * i + 1);

            float scalar0, scalar1;
            T_local.SetValue(ctx.blockDim * ctx.columns * i + i + ctx.blockIdx * ctx.columns, tau0);
            workLocal.SetValue((ctx.blockDim * i * 8) + ctx.blockIdx * 8 + i, tau0);
            T_local.SetValue(ctx.blockDim * ctx.columns * i + i + ctx.blockIdx * ctx.columns + 1024, tau1);
            workLocal.SetValue((ctx.blockDim * i * 8) + ctx.blockIdx * 8 + i + 1024, tau1);
            for (int j = i + 1; j != ctx.columns; j++)
            {
                tau0 = -tauLocal.GetValue(2 * j);
                tau1 = -tauLocal.GetValue(2 * j + 1);
                Mul(workLocal[2048], workLocal[ctx.blockDim * i * 8 + ctx.blockIdx * 8], householder_v[j * 8], 8);
                Mul(workLocal[2048 + 4096], workLocal[ctx.blockDim * i * 8 + ctx.blockIdx * 8 + 1024],
                    householder_v[j * 8 + 4096], 8);
                ReduceSum(workLocal[2048], workLocal[2048], workLocal[2048], 8);
                ReduceSum(workLocal[2048 + 4096], workLocal[2048 + 4096], workLocal[2048 + 4096], 8);
                scalar0 = workLocal(2048) - workLocal(2048 + 4096);

                Mul(workLocal[2048], workLocal[ctx.blockDim * i * 8 + ctx.blockIdx * 8], householder_v[j * 8 + 4096], 8);
                Mul(workLocal[2048 + 4096], workLocal[ctx.blockDim * i * 8 + ctx.blockIdx * 8 + 1024],
                    householder_v[j * 8], 8);
                ReduceSum(workLocal[2048], workLocal[2048], workLocal[2048], 8);
                ReduceSum(workLocal[2048 + 4096], workLocal[2048 + 4096], workLocal[2048 + 4096], 8);
                scalar1 = workLocal(2048) + workLocal(2048 + 4096);

                T_local.SetValue(ctx.blockDim * ctx.columns * i + j + ctx.blockIdx * ctx.columns,
                                 (tau0 * scalar0 - tau1 * scalar1));
                workLocal.SetValue(ctx.blockDim * i * 8 + ctx.blockIdx * 8 + j, (tau0 * scalar0 - tau1 * scalar1));
                T_local.SetValue(ctx.blockDim * ctx.columns * i + j + ctx.blockIdx * ctx.columns + 1024,
                                 (tau0 * scalar1 + tau1 * scalar0));
                workLocal.SetValue(ctx.blockDim * i * 8 + ctx.blockIdx * 8 + j + 1024, (tau0 * scalar1 + tau1 * scalar0));
            }
        }
    }

    __aicore__ inline void HouseholderQR_VEC2::UpdatePeers(const PanelCtx& ctx, LocalTensor<float>& a,
                                                           LocalTensor<float>& householder_v,
                                                           LocalTensor<float>& workLocal,
                                                           LocalTensor<float>& T_local)
    {
        Duplicate<float>(householder_v, 0, 8 * ctx.columns);
        Duplicate<float>(householder_v[4096], 0, 8 * ctx.columns);

        for (int k = ctx.blockIdx + 1; k != ctx.blockDim; k++)
        {
            WritePeerW(ctx, a, householder_v, workLocal, k);

            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            // 读取中间结果w
            CrossCoreWaitFlag(0x0);

            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                tau[2 * (k * ctx.columns + ctx.offsetM)]);
            UpdatePeerT(ctx, workLocal, T_local, k);
        }
    }

    __aicore__ inline void HouseholderQR_VEC2::WritePeerW(const PanelCtx& ctx, LocalTensor<float>& a,
                                                          LocalTensor<float>& householder_v,
                                                          LocalTensor<float>& workLocal, int k)
    {
        for (int i = 0; i != ctx.columns; i++)
        {
            // 拷贝后面列的v向量
            LocalTensor<float> b = bQueueIn.AllocTensor<float>();
            DataCopy(b, yGlobalReal[ctx.offsetY + padM * k * ctx.columns + padM * i], ctx.aligned_M);
            DataCopy(b[4096], yGlobalImag[ctx.offsetY + padM * k * ctx.columns + padM * i], ctx.aligned_M);
            bQueueIn.EnQue(b);
            bQueueIn.DeQue<float>();
            for (int j = 0; j != ctx.columns; j++)
            {
                Mul(workLocal[2048], a[j * ctx.offset], b, ctx.aligned_M);
                Mul(workLocal[2048 + 4096], a[j * ctx.offset + 8192], b[4096], ctx.aligned_M);

                Duplicate<float>(workLocal[2048], 0, k * ctx.columns + i);
                Duplicate<float>(workLocal[2048 + 4096], 0, k * ctx.columns + i);

                ReduceSum(workLocal[2048], workLocal[2048], workLocal[2048], ctx.aligned_M);
                ReduceSum(workLocal[2048 + 4096], workLocal[2048 + 4096], workLocal[2048 + 4096], ctx.aligned_M);
                // 写回中间结果w 只能分开写回，写到连续地址写冲突太严重
                householder_v.SetValue(i * 8 + j, workLocal.GetValue(2048) + workLocal.GetValue(2048 + 4096));
            }
            DataCopy(wGlobalReal[ctx.blockDim * 8 * i + ctx.blockIdx * 8 + (k % 2) * 2048], householder_v[i * 8], 8);
            for (int j = 0; j != ctx.columns; j++)
            {
                Mul(workLocal[2048], a[j * ctx.offset], b[4096], ctx.aligned_M);
                Mul(workLocal[2048 + 4096], a[j * ctx.offset + 8192], b, ctx.aligned_M);

                Duplicate<float>(workLocal[2048], 0, k * ctx.columns + i);
                Duplicate<float>(workLocal[2048 + 4096], 0, k * ctx.columns + i);

                ReduceSum(workLocal[2048], workLocal[2048], workLocal[2048], ctx.aligned_M);
                ReduceSum(workLocal[2048 + 4096], workLocal[2048 + 4096], workLocal[2048 + 4096], ctx.aligned_M);
                // 写回中间结果w 只能分开写回，写到连续地址写冲突太严重
                householder_v.SetValue(i * 8 + j + 4096, workLocal.GetValue(2048) - workLocal.GetValue(2048 + 4096));
            }
            DataCopy(wGlobalImag[ctx.blockDim * 8 * i + ctx.blockIdx * 8 + (k % 2) * 2048], householder_v[i * 8 + 4096], 8);
            bQueueIn.FreeTensor<float>(b);
        }
    }

    __aicore__ inline void HouseholderQR_VEC2::UpdatePeerT(const PanelCtx& ctx, LocalTensor<float>& workLocal,
                                                           LocalTensor<float>& T_local, int k)
    {
        for (int i = 0; i != ctx.columns; i++)
        {
            LocalTensor<float> b = bQueueIn.AllocTensor<float>();
            DataCopy(b, wGlobalReal[ctx.blockDim * 8 * i + (k % 2) * 2048], align_to(k * 8 + 8, 64));
            DataCopy(b[4096], wGlobalImag[ctx.blockDim * 8 * i + (k % 2) * 2048], align_to(k * 8 + 8, 64));

            // 计算出当前核负责的T行的下一位
            bQueueIn.EnQue(b);
            b = bQueueIn.DeQue<float>();
            float tau0 = -tau.GetValue(2 * (k * ctx.columns + i + ctx.offsetM));
            float tau1 = -tau.GetValue(2 * (k * ctx.columns + i + ctx.offsetM) + 1);

            float scalar0, scalar1;
            for (int j = 0; j != ctx.columns; j++)
            {
                Mul(workLocal[2048], b, workLocal[ctx.blockDim * 8 * j], k * 8 + 8);
                Mul(workLocal[2048 + 4096], b[4096], workLocal[ctx.blockDim * 8 * j + 1024], k * 8 + 8);
                ReduceSum(workLocal[2048], workLocal[2048], workLocal[2048], k * 8 + 8);
                ReduceSum(workLocal[2048 + 4096], workLocal[2048 + 4096], workLocal[2048 + 4096], k * 8 + 8);
                scalar0 = workLocal.GetValue(2048) - workLocal.GetValue(2048 + 4096);
                Mul(workLocal[2048], b[4096], workLocal[ctx.blockDim * 8 * j], k * 8 + 8);
                Mul(workLocal[2048 + 4096], b, workLocal[ctx.blockDim * 8 * j + 1024], k * 8 + 8);
                ReduceSum(workLocal[2048], workLocal[2048], workLocal[2048], k * 8 + 8);
                ReduceSum(workLocal[2048 + 4096], workLocal[2048 + 4096], workLocal[2048 + 4096], k * 8 + 8);
                scalar1 = workLocal.GetValue(2048) + workLocal.GetValue(2048 + 4096);

                T_local.SetValue(ctx.blockDim * ctx.columns * j + i + k * ctx.columns, (tau0 * scalar0 - tau1 * scalar1));
                workLocal.SetValue(ctx.blockDim * 8 * j + k * 8 + i, (tau0 * scalar0 - tau1 * scalar1));
                T_local.SetValue(ctx.blockDim * ctx.columns * j + i + k * ctx.columns + 1024,
                                 (tau1 * scalar0 + tau0 * scalar1));
                workLocal.SetValue(ctx.blockDim * 8 * j + k * 8 + i + 1024, (tau1 * scalar0 + tau0 * scalar1));
            }
            bQueueIn.FreeTensor<float>(b);
        }
    }

    __aicore__ inline void HouseholderQR_VEC2::StoreT(const PanelCtx& ctx, LocalTensor<float>& T_local)
    {
        int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        DataCopy(tGlobalImag[ctx.columns * padM * ctx.blockIdx + ctx.offsetY], T_local[1024],
                 {static_cast<uint16_t>(ctx.columns), static_cast<uint16_t>(ctx.blockDim * ctx.columns / 8), 0,
                  static_cast<uint16_t>((padM - ctx.blockDim * ctx.columns) / 8)});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        Muls(T_local[1024], T_local[1024], -1.0f, ctx.blockDim * ctx.columns * ctx.columns);
        DataCopy(tGlobalReal[ctx.columns * padM * ctx.blockIdx + ctx.offsetY], T_local,
                 {static_cast<uint16_t>(ctx.columns), static_cast<uint16_t>(ctx.blockDim * ctx.columns / 8), 0,
                  static_cast<uint16_t>((padM - ctx.blockDim * ctx.columns) / 8)});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(tGlobalImagNeg[ctx.columns * padM * ctx.blockIdx + ctx.offsetY], T_local[1024],
                 {static_cast<uint16_t>(ctx.columns), static_cast<uint16_t>(ctx.blockDim * ctx.columns / 8), 0,
                  static_cast<uint16_t>((padM - ctx.blockDim * ctx.columns) / 8)});
    }

    __aicore__ inline void HouseholderQR_VEC2::StorePanel(const PanelCtx& ctx, LocalTensor<float>& a,
                                                          float* alpha, int offsetA)
    {
        for (int i = 0; i != ctx.columns; i++)
        {
            a.SetValue(i * ctx.offset + ctx.blockIdx * ctx.columns + i, -alpha[i]);
            a.SetValue(i * ctx.offset + ctx.blockIdx * ctx.columns + i + 8192, 0);
        }

        int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);

        DataCopyParams aCopyParams2{static_cast<uint16_t>(ctx.columns), static_cast<uint16_t>(ctx.aligned_M / 8),
                                    static_cast<uint16_t>((ctx.offset - ctx.aligned_M) / 8),
                                    static_cast<uint16_t>((padM - ctx.aligned_M) / 8)};
        DataCopy(aGlobalReal[offsetA + padM * ctx.blockIdx * ctx.columns], a, aCopyParams2);
        DataCopy(aGlobalImag[offsetA + padM * ctx.blockIdx * ctx.columns], a[8192], aCopyParams2);
        aQueueIn.FreeTensor<float>(a);
    }

}  // namespace AscendC
