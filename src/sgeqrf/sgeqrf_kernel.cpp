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
 * \file sgeqrf_kernel.cpp
 * \brief
 */

#ifndef SGEQRF_MIX_H
#define SGEQRF_MIX_H

#include "kernel_operator.h"
#include "./kernel/gemm.hpp"
#include "sgeqrf_launchers.hpp"

using namespace AscendC;
static constexpr int M_threshold = 0;
__aicore__ inline int align_to(int a, int b)
{
    return (a + b - 1) / b * b;
}
__aicore__ inline int gcd(int a, int b)
{
    // 显式判零：调用方可能传入 0（如分块数为 0 的退化形状），此时 gcd 退化为 |a|，
    // 后续 blockNum / g、m_cubes / g 才有确定的非零除数。
    a = a < 0 ? -a : a;
    if (b == 0)
    {
        return a;
    }
    b = b < 0 ? -b : b;
    while (b != 0)
    {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

namespace AscendC
{
    class singleCoreHouseholderQR
    {
    private:
        int M, N, aligned_M, strideM;
        const int chunkSize = 2 * 64 * 64;
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> aQueue;
        TBuf<TPosition::VECCALC> vBuf;
        TBuf<TPosition::VECCALC> workBuf;
        TQue<QuePosition::VECOUT, 1> tauQueueDst;

        GlobalTensor<float> A;
        GlobalTensor<float> Apad;
        GlobalTensor<float> tau;

    public:
        __aicore__ inline singleCoreHouseholderQR() : M(0), N(0)
        {
        }

        __aicore__ inline void Init()
        {
            pipe.InitBuffer(aQueue, 1, chunkSize * sizeof(float));
            pipe.InitBuffer(vBuf, chunkSize * sizeof(float));
            pipe.InitBuffer(workBuf, chunkSize * sizeof(float));
            pipe.InitBuffer(tauQueueDst, 1, 64 * sizeof(float));
        }

        __aicore__ inline void SetMatrix(
            __gm__ uint8_t* A_, __gm__ uint8_t* Apad_, __gm__ uint8_t* tau_, int M_, int N_, int strideM_)
        {
            A.SetGlobalBuffer((__gm__ float*)A_);
            Apad.SetGlobalBuffer((__gm__ float*)Apad_);
            tau.SetGlobalBuffer((__gm__ float*)tau_);
            M = M_;
            N = N_;
            strideM = strideM_;
            aligned_M = align_to(M_, 8);
        }

        __aicore__ inline void Compute()
        {
            int minMN = M < N ? M : N;
            LocalTensor<float> a = aQueue.AllocTensor<float>();
            DataCopyParams copyParams{
                static_cast<uint16_t>(N), static_cast<uint16_t>(M * sizeof(float)), 0,
                static_cast<uint16_t>((strideM - aligned_M) / 8)};
            DataCopyPadParams padParams{true, 0, static_cast<uint8_t>(aligned_M - M), 0};
            DataCopyPad(a, A, copyParams, padParams);
            aQueue.EnQue(a);

            uint64_t mask[2] = {0, 0};
            if (M >= 64)
                mask[0] = ~0ULL;
            else
                mask[0] = (1ULL << M) - 1;

            LocalTensor<float> v = vBuf.Get<float>();
            LocalTensor<float> workLocal = workBuf.Get<float>();
            LocalTensor<float> Tau = tauQueueDst.AllocTensor<float>();

            a = aQueue.DeQue<float>();
            FactorizeColumns(a, v, workLocal, Tau, mask, minMN);
            tauQueueDst.EnQue(Tau);
            Tau = tauQueueDst.DeQue<float>();

            DataCopyParams copyParams2{
                static_cast<uint16_t>(N), static_cast<uint16_t>(M * sizeof(float)),
                static_cast<uint16_t>((strideM - aligned_M) / 8), static_cast<uint16_t>((strideM - M) * sizeof(float))};
            DataCopyPad(Apad, a, copyParams2);
            aQueue.FreeTensor<float>(a);

            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint16_t>(minMN * sizeof(float));
            DataCopyPad(tau, Tau, copyParams);
            tauQueueDst.FreeTensor<float>(Tau);
        }

    private:
        // 逐列做 Householder 分解：v 为当前列工作区，tau 与归一化后的 v 就地写回 a
        __aicore__ inline void FactorizeColumns(LocalTensor<float>& a, LocalTensor<float>& v,
                                                LocalTensor<float>& workLocal, LocalTensor<float>& Tau,
                                                uint64_t* mask, int minMN)
        {
            for (int k = 0; k < minMN; ++k)
            {
                Copy(v, a[k * strideM], 64, strideM / 64, {1, 1, 8, 8});
                Duplicate<float>(v, 0, k);
                Mul(workLocal, v, v, aligned_M);
                ReduceSum(workLocal, workLocal, workLocal, aligned_M);
                float x0 = v.GetValue(k);
                float norm_2 = workLocal.GetValue(0);
                if (norm_2 == x0 * x0)
                {
                    Tau.SetValue(k, 0.0f);
                    continue;
                }
                float norm_x = sqrt(norm_2);
                float beta = x0 < 0 ? norm_x : -norm_x;
                float tauK = (beta - x0) / beta;
                float v0 = x0 - beta;
                v.SetValue(k, v0);
                ApplyReflector(k, tauK / (v0 * v0), a, v, workLocal);
                mask[0] ^= (1ull << k);
                Muls(a[k * strideM], v, 1.0f / v0, mask, 1, {1, 1, 8, 8});
                if (aligned_M > 64)
                {
                    Muls(a[k * strideM + 64], v[64], 1.0f / v0, aligned_M - 64);
                }
                a.SetValue(k * strideM + k, beta);
                Tau.SetValue(k, tauK);
            }
        }

        // 把第 k 列的反射子作用到其右侧所有列
        __aicore__ inline void ApplyReflector(int k, float tauPrime, LocalTensor<float>& a,
                                              const LocalTensor<float>& v, const LocalTensor<float>& workLocal)
        {
            for (int j = k + 1; j < N; ++j)
            {
                Mul(workLocal, v, a[j * strideM], aligned_M);
                ReduceSum(workLocal, workLocal, workLocal, aligned_M);
                Axpy(a[j * strideM], v, -tauPrime * workLocal.GetValue(0), aligned_M);
            }
        }
    };
}  // namespace AscendC

namespace AscendC
{
    class HouseholderQR
    {
    public:
        __aicore__ inline HouseholderQR() : M(0), N(0), padM(0), padN(0)
        {
        }

        __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe)
        {
            pipe->InitBuffer(aQueueIn, 1, 2 * chunkSize * sizeof(float));
            pipe->InitBuffer(bQueueIn, 1, chunkSize * sizeof(float));
            pipe->InitBuffer(aQueueOut, 1, chunkSize * sizeof(float));
            pipe->InitBuffer(tQueueDst, 1, 1024 * sizeof(float));
            pipe->InitBuffer(tauQueueOut, 2, 32 * sizeof(float));
            pipe->InitBuffer(workQueue, (chunkSize + 1024) * sizeof(float));
        }
        __aicore__ inline void Pad(
            TBufPool<TPosition::VECCALC, 16>* pipe, int srcM, int srcN, int padM, int padN, int blockIdx, int blockNum)
        {
            // 下方按 blockNum 均分行块、按 padN 计算单次搬运行数，二者都必须非零；
            // 退化入参直接返回，避免除零（实际调用点 blockNum >= 2、padN >= 64，仅为静态检查兜底）。
            if (blockNum == 0)
            {
                return;
            }
            if (padN == 0)
            {
                return;
            }
            pipe->InitBuffer(padQue, 1, 4 * chunkSize * sizeof(float));
            LocalTensor<float> local = padQue.AllocTensor<float>();
            int p = srcM / blockNum;
            int r = srcM % blockNum;
            p = p + (blockIdx < r);
            int start = blockIdx * p + ((blockIdx < r) ? 0 : r);
            int k = 3 * chunkSize / padN;
            // padN 超过 3 * chunkSize 时上式向下取整为 0，后续 p / k 与 DataCopy 的参数都会失效
            if (k == 0)
            {
                k = 1;
            }
            int t = p / k;
            r = p % k;
            PadCopyRows(local, start, k, t, srcN, padN);
            PadCopyTail(local, start + t * k, r, srcN, padN);
            padQue.FreeTensor(local);
        }

        // 搬运 t 个满 k 行的块：A -> APad，每块单独做 MTE2/MTE3 同步
        __aicore__ inline void PadCopyRows(LocalTensor<float>& local, int start, int k, int rows, int srcN, int padN)
        {
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(k), static_cast<uint16_t>(srcN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, static_cast<uint8_t>(padN), 0};
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(k), static_cast<uint16_t>(srcN * sizeof(float)), 0,
                static_cast<uint16_t>((padN - srcN) * sizeof(float)), 0};
            int32_t eventIDMTE3ToMTE2 =
                static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            int32_t eventIDMTE2ToMTE3 =
                static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            for (int i = 0; i < rows; ++i)
            {
                DataCopyPad(local, A[(start + i * k) * srcN], copyInParams, padParams);  // MTE2
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                DataCopyPad(APad[(start + i * k) * padN], local, copyOutParams);  // MTE3
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
        }

        // 搬运不足 k 行的尾部剩余
        __aicore__ inline void PadCopyTail(LocalTensor<float>& local, int start, int rows, int srcN, int padN)
        {
            if (rows <= 0)
            {
                return;
            }
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(srcN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, static_cast<uint8_t>(padN), 0};
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(srcN * sizeof(float)), 0,
                static_cast<uint16_t>((padN - srcN) * sizeof(float)), 0};
            int32_t eventIDMTE2ToMTE3 =
                static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            DataCopyPad(local, A[start * srcN], copyInParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            DataCopyPad(APad[start * padN], local, copyOutParams);
        }

        __aicore__ inline void SetMatrix(
            __gm__ uint8_t* A_, __gm__ uint8_t* APad_, __gm__ uint8_t* tau_, __gm__ uint8_t* T_, __gm__ uint8_t* W_,
            __gm__ uint8_t* Y_, int M_, int N_, int padM_, int padN_)
        {
            APad.SetGlobalBuffer((__gm__ float*)APad_);
            A.SetGlobalBuffer((__gm__ float*)A_);
            tau.SetGlobalBuffer((__gm__ float*)tau_);
            T.SetGlobalBuffer((__gm__ float*)T_);
            W.SetGlobalBuffer((__gm__ float*)W_);
            Y.SetGlobalBuffer((__gm__ float*)Y_);
            M = M_;
            N = N_;
            padM = padM_;
            padN = padN_;
        }
        __aicore__ inline void ComputeAll(int blockDim, int blockIdx)
        {
            int offset = 0;
            int minMN = min(M, N);
            int flag = 0;
            int columns = padM <= M_threshold ? 4 : 2;
            int align_blockdim = min(32, blockDim);
            panelYSize = align_blockdim * padM * 4;
            panelTSize = align_blockdim * align_blockdim * 16;

            for (; offset + align_blockdim * columns <= minMN;)
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
                offset += align_blockdim * columns;
                columns = M - offset <= M_threshold ? 4 : 2;
            }
            int tail = minMN - offset;
            columns = M - offset <= M_threshold ? 4 : 2;
            int tailblocks = (tail + columns - 1) / columns;
            if (offset != minMN)
            {
                if (blockIdx < tailblocks)
                {
                    Compute_tail(
                        tailblocks, blockIdx, padM * offset + offset, M - offset, offset, flag, columns,
                        tail % columns);
                }
                else
                {
                    syn_all(tailblocks, 0);
                }
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
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
            CrossCoreWaitFlag(0x2);
        }
        __aicore__ inline void Compute(
            int blockDim, int blockIdx, int offsetA, int curM, int offsetM, int flag, int columns)
        {
            ComputePanel(MakePanelArgs(blockDim, blockIdx, offsetA, curM, offsetM, flag, columns, columns, 0));
        }
        __aicore__ inline void Compute_tail(
            int blockDim, int blockIdx, int offsetA, int curM, int offsetM, int flag, int columns, int len)
        {
            int cur_columns = (blockIdx + 1) == blockDim ? len : columns;
            cur_columns = cur_columns == 0 ? columns : cur_columns;
            ComputePanel(MakePanelArgs(blockDim, blockIdx, offsetA, curM, offsetM, flag, columns, cur_columns, len));
        }

    private:
        // 面板计算的全部标量参数：整面板与尾面板共用同一套描述，避免两份实现漂移
        struct PanelArgs
        {
            int blockDim;
            int blockIdx;
            int offsetA;
            int curM;
            int offsetM;
            int flag;
            int columns;     // 每个核负责的列块宽度
            int curColumns;  // 当前面板实际参与计算的列数（尾面板末核可能小于 columns）
            int len;         // 尾面板末核剩余列数，0 表示整面板
            int offset;      // align_to(curM, 64)
            int offsetY;     // flag * panelYSize
            int alignedM;    // align_to(curM, 8)
            int lenT;        // T 矩阵写回的有效行数
        };

        __aicore__ inline PanelArgs MakePanelArgs(int blockDim, int blockIdx, int offsetA, int curM, int offsetM,
                                                  int flag, int columns, int curColumns, int len)
        {
            PanelArgs p;
            p.blockDim = blockDim;
            p.blockIdx = blockIdx;
            p.offsetA = offsetA;
            p.curM = curM;
            p.offsetM = offsetM;
            p.flag = flag;
            p.columns = columns;
            p.curColumns = curColumns;
            p.len = len;
            p.offset = align_to(curM, 64);
            p.offsetY = flag * panelYSize;
            p.alignedM = align_to(curM, 8);
            p.lenT = blockDim * columns - (len == 0 ? 0 : columns - len);
            return p;
        }

        // 从 APad 载入当前面板，并清零 64 对齐引入的尾部补零区
        __aicore__ inline void LoadPanel(const PanelArgs& p, LocalTensor<float>& a)
        {
            DataCopyParams aCopyParams{
                static_cast<uint16_t>(p.curColumns), static_cast<uint16_t>(p.curM * sizeof(float)),
                static_cast<uint16_t>((padM - p.curM) * sizeof(float)),
                static_cast<uint16_t>((p.offset - p.alignedM) / 8)};
            DataCopyPadParams aPadParams{true, 0, static_cast<uint8_t>(p.alignedM - p.curM), 0};
            DataCopyPad(a, APad[p.offsetA + padM * p.blockIdx * p.columns], aCopyParams, aPadParams);
            aQueueIn.EnQue(a);
            a = aQueueIn.DeQue<float>();
            // 逐列清零尾部补零区，避免 5 参 Duplicate 的 dstRepeatStride(offset / 8) 在 offset >= 2048 时按 8 位回绕
            for (int c = 0; c != p.curColumns; c++)
            {
                Duplicate<float>(a[c * p.offset + p.alignedM], 0, p.offset - p.alignedM);
            }
        }

        // 用前面所有核已发布的反射子依次作用到当前面板
        __aicore__ inline void ApplyPreviousReflectors(const PanelArgs& p, const LocalTensor<float>& a,
                                                       const LocalTensor<float>& workLocal)
        {
            for (int k = 0; k != p.blockIdx; k++)
            {
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                CrossCoreWaitFlag(0x0);
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    tau[k * p.columns + p.offsetM]);
                for (int i = 0; i != p.columns; i++)
                {
                    LocalTensor<float> b = bQueueIn.AllocTensor<float>();
                    DataCopy(b, Y[p.offsetY + padM * k * p.columns + padM * i], p.alignedM);
                    bQueueIn.EnQue(b);
                    b = bQueueIn.DeQue<float>();
                    float tauK = tau.GetValue(k * p.columns + i + p.offsetM);
                    for (int j = 0; j != p.curColumns; j++)
                    {
                        Mul(workLocal, a[p.offset * j], b, p.alignedM);
                        ReduceSum(workLocal, workLocal, workLocal, p.alignedM);
                        Axpy(a[p.offset * j], b, -tauK * workLocal.GetValue(0), p.alignedM);
                    }
                    bQueueIn.FreeTensor<float>(b);
                }
            }
        }

        // 把 householder 向量写回 Y；EnQue/DeQue 同时完成 V->MTE3 与缓冲区复用保护
        __aicore__ inline void StoreToY(const PanelArgs& p, LocalTensor<float>& householderV, int i)
        {
            aQueueOut.EnQue(householderV);
            householderV = aQueueOut.DeQue<float>();
            DataCopy(Y[p.offsetY + padM * (p.blockIdx * p.columns + i)], householderV, p.offset);
            aQueueOut.FreeTensor<float>(householderV);
            householderV = aQueueOut.AllocTensor<float>();
        }

        // 对第 i 列做 Householder 变换，更新 tau / v0 / alpha，并把 v 写回 Y
        __aicore__ inline void FactorizeColumn(const PanelArgs& p, LocalTensor<float>& a,
                                               LocalTensor<float>& householderV, const LocalTensor<float>& workLocal,
                                               const LocalTensor<float>& tauLocal,
                                               const LocalTensor<float>& v0Local, float* alpha, int i)
        {
            Mul(workLocal, a[p.offset * i], a[p.offset * i], p.alignedM);
            Duplicate<float>(workLocal, 0, p.blockIdx * p.columns + i);
            ReduceSum<float>(workLocal, workLocal, workLocal, p.alignedM);
            float x0 = a.GetValue(p.blockIdx * p.columns + i + p.offset * i);
            float norm2 = workLocal.GetValue(0);
            if (norm2 == x0 * x0)
            {
                alpha[i] = -x0;
                tauLocal.SetValue(i, 0.0f);
                v0Local.SetValue(i, 1.0f);
                tau.SetValue(p.blockIdx * p.columns + i + p.offsetM, 0.0f);
                Duplicate<float>(householderV, 0, p.offset);
                StoreToY(p, householderV, i);
                return;
            }
            float normX = sqrt(norm2);
            alpha[i] = x0 < 0 ? -normX : normX;
            v0Local.SetValue(i, x0 + alpha[i]);
            a.SetValue(p.blockIdx * p.columns + i + p.offset * i, v0Local.GetValue(i));
            tauLocal.SetValue(i, 1.0f / (norm2 + x0 * alpha[i]));
            // 把当前列的 v 拷进 householderV，并抹掉前 i 个已消去的分量
            Copy(householderV, a[p.offset * i], 64, p.offset / 64, {1, 1, 8, 8});
            Duplicate<float>(householderV, 0, p.blockIdx * p.columns + i);
            for (int j = i + 1; j != p.curColumns; j++)
            {
                Mul(workLocal, a[p.offset * j], householderV, p.alignedM);
                ReduceSum(workLocal, workLocal, workLocal, p.alignedM);
                Axpy(a[p.offset * j], householderV, -tauLocal.GetValue(i) * workLocal.GetValue(0), p.alignedM);
            }
            tau.SetValue(p.blockIdx * p.columns + i + p.offsetM, tauLocal.GetValue(i));
            StoreToY(p, householderV, i);
        }

        // 计算 W = Y^T * a 的核内分块，按 i 行、j 列写回 W（奇偶两片轮换）
        __aicore__ inline void FormW(const PanelArgs& p, const LocalTensor<float>& a,
                                     LocalTensor<float>& householderV, const LocalTensor<float>& workLocal)
        {
            Duplicate<float>(householderV, 0, 8 * p.columns);
            for (int i = p.curColumns - 1; i >= 0; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    Mul(workLocal, a[p.offset * i], a[p.offset * j], p.alignedM);
                    Duplicate<float>(workLocal, 0, p.blockIdx * p.columns + i);
                    ReduceSum(workLocal, workLocal, workLocal, p.alignedM);
                    householderV.SetValue(i * 8 + j, workLocal.GetValue(0));
                }
                DataCopy(W[p.blockDim * 8 * i + p.blockIdx * 8 + (p.blockIdx % 2) * 1024], householderV[i * 8], 8);
            }
        }

        // 由本核对角块与前序核的 W 递推 T：先算本核三角块，再按 k 递增吸收 peer 的 w
        __aicore__ inline void FormTDiagonal(const PanelArgs& p, LocalTensor<float>& householderV,
                                             const LocalTensor<float>& workLocal, const LocalTensor<float>& T_local,
                                             const LocalTensor<float>& tauLocal)
        {
            Duplicate<float>(T_local, 0, p.lenT * p.columns);
            Duplicate<float>(workLocal, 0, p.blockDim * p.columns * 8);
            // 设置为负，从而算出 -T，workLocal 存储“分散”的当前 T 的一行
            for (int i = 0; i != p.curColumns; i++)
            {
                float scalar = -tauLocal.GetValue(i);
                T_local.SetValue(p.lenT * i + i + p.blockIdx * p.columns, scalar);
                workLocal.SetValue(p.blockDim * i * 8 + p.blockIdx * 8 + i, scalar);
                for (int j = i + 1; j != p.curColumns; j++)
                {
                    Mul(workLocal[1024], workLocal[p.blockDim * i * 8 + p.blockIdx * 8], householderV[j * 8], 8);
                    ReduceSum(workLocal[1024], workLocal[1024], workLocal[1024], 8);
                    float tValue = -tauLocal.GetValue(j) * workLocal(1024);
                    T_local.SetValue(p.lenT * i + j + p.blockIdx * p.columns, tValue);
                    workLocal.SetValue(p.blockDim * i * 8 + p.blockIdx * 8 + j, tValue);
                }
            }
            Duplicate<float>(householderV, 0, 8 * p.columns);
        }

        // 吸收 blockIdx 之后各核的反射子，把 T 的其余行补齐
        __aicore__ inline void FormTFromPeers(const PanelArgs& p, const LocalTensor<float>& a,
                                              LocalTensor<float>& householderV, const LocalTensor<float>& workLocal,
                                              const LocalTensor<float>& T_local)
        {
            for (int k = p.blockIdx + 1; k != p.blockDim; k++)
            {
                int kColumns = (p.len != 0 && k == p.blockDim - 1) ? p.len : p.columns;
                // 拷贝后面列的 v 向量并算出对应的 w
                for (int i = 0; i != kColumns; i++)
                {
                    LocalTensor<float> b = bQueueIn.AllocTensor<float>();
                    DataCopy(b, Y[p.offsetY + padM * k * p.columns + padM * i], p.alignedM);
                    bQueueIn.EnQue(b);
                    b = bQueueIn.DeQue<float>();
                    for (int j = 0; j != p.curColumns; j++)
                    {
                        Mul(workLocal[1024], a[j * p.offset], b, p.alignedM);
                        ReduceSum(workLocal[1024], workLocal[1024], workLocal[1024], p.alignedM);
                        // 写回中间结果 w 只能分开写回，写到连续地址写冲突太严重
                        householderV.SetValue(i * 8 + j, workLocal.GetValue(1024));
                    }
                    DataCopy(W[p.blockDim * 8 * i + p.blockIdx * 8 + (k % 2) * 1024], householderV[i * 8], 8);
                    bQueueIn.FreeTensor<float>(b);
                }

                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
                // 读取中间结果 w
                CrossCoreWaitFlag(0x0);

                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    tau[k * p.columns + p.offsetM]);
                for (int i = 0; i != kColumns; i++)
                {
                    LocalTensor<float> b = bQueueIn.AllocTensor<float>();
                    DataCopy(b, W[p.blockDim * 8 * i + (k % 2) * 1024], align_to(k * 8 + 8, 64));
                    // 计算出当前核负责的 T 行的下一位
                    bQueueIn.EnQue(b);
                    b = bQueueIn.DeQue<float>();
                    for (int j = 0; j != p.curColumns; j++)
                    {
                        Mul(workLocal[1024], b, workLocal[p.blockDim * 8 * j], k * 8 + 8);
                        ReduceSum(workLocal[1024], workLocal[1024], workLocal[1024], k * 8 + 8);
                        float tValue = -tau.GetValue(k * p.columns + i + p.offsetM) * workLocal.GetValue(1024);
                        T_local.SetValue(p.lenT * j + i + k * p.columns, tValue);
                        workLocal.SetValue(p.blockDim * 8 * j + k * 8 + i, tValue);
                    }
                    bQueueIn.FreeTensor<float>(b);
                }
            }
        }

        // T 只写回有效行 lenT；整面板时 lenT * columns 与 blockDim * columns * columns 等价
        __aicore__ inline void StoreT(const PanelArgs& p, const LocalTensor<float>& T_local)
        {
            DataCopyExtParams tCopyParams{1, static_cast<uint32_t>(p.lenT * p.columns * sizeof(float)), 0, 0, 0};
            DataCopyPad(T[p.lenT * p.blockIdx * p.columns + p.flag * panelTSize], T_local, tCopyParams);
        }

        // 把 tau（含 v0 缩放）写回 GM
        __aicore__ inline void WriteTau(const PanelArgs& p, LocalTensor<float>& tauLocal,
                                        const LocalTensor<float>& v0Local)
        {
            Mul(tauLocal, tauLocal, v0Local, 8);
            Mul(tauLocal, tauLocal, v0Local, 8);
            tauQueueOut.EnQue(tauLocal);
            tauLocal = tauQueueOut.DeQue<float>();
            DataCopyParams tauCopyParams{
                static_cast<uint16_t>(1), static_cast<uint16_t>(p.curColumns * sizeof(float)), 0, 0};
            DataCopyPad(tau[p.blockIdx * p.columns + p.offsetM], tauLocal, tauCopyParams);
            tauQueueOut.FreeTensor(tauLocal);
        }

        // 把 R 与归一化后的 v 写回 APad
        __aicore__ inline void NormalizeAndStore(const PanelArgs& p, LocalTensor<float>& a,
                                                 const LocalTensor<float>& v0Local, const float* alpha)
        {
            int startM = p.blockIdx * p.columns / 64 * 64;
            uint64_t maskUpdate[2] = {(~0ULL) ^ ((1ULL << (p.blockIdx * p.columns - startM)) - 1), 0};
            for (int i = 0; i != p.curColumns; i++)
            {  // 64 % columns = 0
                if (startM + 64 < p.alignedM)
                {
                    Muls(a[i * p.offset + startM + 64], a[i * p.offset + startM + 64], 1.0f / v0Local.GetValue(i),
                         p.alignedM - startM - 64);
                }
                maskUpdate[0] <<= 1;
                Muls(a[i * p.offset + startM], a[i * p.offset + startM], 1.0f / v0Local.GetValue(i), maskUpdate, 1,
                     {1, 1, 8, 8});
                a.SetValue(i * p.offset + p.blockIdx * p.columns + i, -alpha[i]);
            }
        }

        // 面板写回：先等 V/S 流水收敛，再按 padM 间距写回 APad，最后等下一个列块
        __aicore__ inline void StorePanel(const PanelArgs& p, LocalTensor<float>& a)
        {
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

            int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);

            DataCopyParams aCopyParams2{
                static_cast<uint16_t>(p.curColumns), static_cast<uint16_t>(p.alignedM / 8),
                static_cast<uint16_t>((p.offset - p.alignedM) / 8), static_cast<uint16_t>((padM - p.alignedM) / 8)};
            DataCopy(APad[p.offsetA + padM * p.blockIdx * p.columns], a, aCopyParams2);
            aQueueIn.FreeTensor<float>(a);
            // 等待下一个列块更新完毕
            CrossCoreWaitFlag(0x2);
        }

        // 合并后的面板驱动：整面板（len == 0）与尾面板共用同一条路径
        __aicore__ inline void ComputePanel(const PanelArgs& p)
        {
            LocalTensor<float> workLocal = workQueue.Get<float>();
            LocalTensor<float> a = aQueueIn.AllocTensor<float>();
            LocalTensor<float> householderV = aQueueOut.AllocTensor<float>();
            LocalTensor<float> T_local = tQueueDst.AllocTensor<float>();
            LocalTensor<float> tauLocal = tauQueueOut.AllocTensor<float>();
            LocalTensor<float> v0Local = tauQueueOut.AllocTensor<float>();
            float alpha[8];

            LoadPanel(p, a);
            ApplyPreviousReflectors(p, a, workLocal);
            for (int i = 0; i != p.curColumns; i++)
            {
                FactorizeColumn(p, a, householderV, workLocal, tauLocal, v0Local, alpha, i);
            }

            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                tau[p.blockIdx * p.columns + p.offsetM]);
            // 等待当前列 v 向量拷贝过去
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);

            FormW(p, a, householderV, workLocal);
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
            CrossCoreWaitFlag(0x0);

            FormTDiagonal(p, householderV, workLocal, T_local, tauLocal);
            FormTFromPeers(p, a, householderV, workLocal, T_local);
            // 写回没有脏元素的 T 矩阵
            StoreT(p, T_local);
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);

            tQueueDst.FreeTensor<float>(T_local);
            aQueueOut.FreeTensor<float>(householderV);

            CrossCoreWaitFlag(0x0);
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);

            WriteTau(p, tauLocal, v0Local);
            NormalizeAndStore(p, a, v0Local, alpha);
            tauQueueOut.FreeTensor(v0Local);
            StorePanel(p, a);
        }

    private:
        int M, N, padM, padN, panelYSize, panelTSize;
        const int chunkSize = 8192;
        TQue<QuePosition::VECIN, 1> aQueueIn;
        TQue<QuePosition::VECIN, 1> bQueueIn;
        TQue<QuePosition::VECOUT, 1> aQueueOut;
        TBuf<QuePosition::VECCALC> workQueue;
        TQue<QuePosition::VECOUT, 1> tQueueDst;
        TQue<QuePosition::VECOUT, 2> tauQueueOut;
        TQue<QuePosition::VECIN, 2> padQue;

        GlobalTensor<float> APad;
        GlobalTensor<float> A;
        GlobalTensor<float> tau;
        GlobalTensor<float> T;
        GlobalTensor<float> W;
        GlobalTensor<float> Y;
    };
}  // namespace AscendC

// C = C - Y * W 的更新循环参数与状态
struct UpdateCtrl
{
    int loop;            // 剩余触发次数
    int changeLoop;      // 切换到 128 列宽的迭代分界
    int vectorBlockdim;  // 当前列块宽度
    int mTail;           // 末次列宽
    int Cstart;          // 当前 C 在 A 中的起始偏移
    int remainingCols;   // 剩余待更新列数
    int MtoCompute;      // 剩余待更新行数
    int flag;            // panelT / panelY 奇偶片
};

// 一个核在 64 行（列）块方向上的均分结果
struct CubeSplit
{
    int cubes;    // 总块数
    int myStart;  // 本核起始块
    int myNum;    // 本核块数
    int myBlock;  // 本核负责的行（列）数
};

// 128 对齐分块发单的序号映射
struct TileDispatch
{
    int mCubes;          // 行块数
    int uCubes;          // 列块数
    int stride;          // 行列块宽（128）
    int vectorBlockdim;  // K 方向宽度
    int memStride;       // A 的行间距
    int updateCols;      // 待更新列数
    int total;           // 总块数
    int start;           // 本核负责的全局序号下界
    int end;             // 本核负责的全局序号上界
    int groupStart;      // 本组全局序号下界
    int groupEnd;        // 本组全局序号上界
    int groupOffset;     // 组内轮转偏移
    int groupblocks;     // 组内块数
};

// 三段 matmul 共享的全局缓冲与标量环境
struct UpdateEnv
{
    GlobalTensor<float> A;
    GlobalTensor<float> T;
    GlobalTensor<float> W;
    GlobalTensor<float> Y;
    int orgM;
    int M;
    int N;
    int blockDim;
    int blockIdx;
    int panelTSize;
    int panelYSize;
};

// 初始化更新循环：按 M_threshold 选择 64/128 两种列宽切换策略
__aicore__ inline void CalcUpdateLoopParam(int orgM, int M, int N, UpdateCtrl& ctrl)
{
    int minMN = min(orgM, N);
    if (M <= M_threshold)
    {
        ctrl.vectorBlockdim = 128;
        ctrl.loop = (minMN - 1) / ctrl.vectorBlockdim + (orgM < N);
        ctrl.changeLoop = -1;
        ctrl.mTail = (orgM % ctrl.vectorBlockdim) ? (orgM % ctrl.vectorBlockdim) : ctrl.vectorBlockdim;
    }
    else
    {
        ctrl.vectorBlockdim = 64;
        ctrl.changeLoop = (M - M_threshold + ctrl.vectorBlockdim - 1) / ctrl.vectorBlockdim;
        if (minMN - ctrl.changeLoop * ctrl.vectorBlockdim <= 0)
        {
            ctrl.loop = (minMN - 1) / ctrl.vectorBlockdim + (orgM < N);
            ctrl.mTail = (orgM % ctrl.vectorBlockdim) ? (orgM % ctrl.vectorBlockdim) : ctrl.vectorBlockdim;
            ctrl.changeLoop = -1;
        }
        else
        {
            ctrl.loop = ctrl.changeLoop + (minMN - ctrl.changeLoop * ctrl.vectorBlockdim - 1) / 128 + (orgM < N);
            ctrl.mTail = ((orgM - ctrl.changeLoop * ctrl.vectorBlockdim) % 128);
            if (!ctrl.mTail)
            {
                ctrl.mTail = 128;
            }
            ctrl.changeLoop = ctrl.loop - ctrl.changeLoop;
        }
    }
    ctrl.Cstart = min(ctrl.vectorBlockdim, minMN) * M;
    ctrl.remainingCols = N - min(ctrl.vectorBlockdim, minMN);
    ctrl.MtoCompute = orgM;
    ctrl.flag = 0;
}

// 把 total 按 64 行（列）块在 blockDim 个核间均分
__aicore__ inline CubeSplit SplitCubes(int total, int blockDim, int blockIdx, int limit)
{
    CubeSplit s;
    s.cubes = 0;
    s.myStart = 0;
    s.myNum = 0;
    s.myBlock = 0;
    // blockDim 为 0 时直接返回空分块（实际调用点 GetBlockNum() >= 1，仅为静态检查兜底）
    if (blockDim == 0)
    {
        return s;
    }
    s.cubes = (total + 63) / 64;
    int base = s.cubes / blockDim;
    int remaining = s.cubes % blockDim;
    s.myStart = blockIdx * base + min(blockIdx, remaining);
    s.myNum = base + (blockIdx < remaining ? 1 : 0);
    s.myBlock = min(s.myNum * 64, limit - 64 * s.myStart);
    return s;
}

// 由 gcd 轮转分配推导本核/本组的全局序号区间
__aicore__ inline TileDispatch MakeTileDispatch(int blockDim, int blockIdx, int mCubes, int uCubes, int stride,
                                                int vectorBlockdim, int updateCols, int memStride)
{
    TileDispatch d;
    d.mCubes = mCubes;
    d.uCubes = uCubes;
    d.stride = stride;
    d.vectorBlockdim = vectorBlockdim;
    d.updateCols = updateCols;
    d.memStride = memStride;
    d.total = mCubes * uCubes;
    // 预置"无 tile"语义：end = -1 让上层发单循环自然空转
    d.start = 0;
    d.end = -1;
    d.groupStart = 0;
    d.groupEnd = -1;
    d.groupOffset = 0;
    d.groupblocks = 0;
    // blockDim 为 0 时返回空映射（实际调用点 GetBlockNum() >= 1，仅为静态检查兜底）
    if (blockDim == 0)
    {
        return d;
    }
    d.start = (d.total * blockIdx) / blockDim;
    d.end = (d.total * (blockIdx + 1)) / blockDim - 1;
    int g = gcd(uCubes, blockDim);
    // gcd 只在 (0, 0) 输入下返回 0，此时同样返回空映射，避免下面两个除法除零
    if (g == 0)
    {
        return d;
    }
    int groupNum = blockDim / g;
    int groupIdx = blockIdx / groupNum;
    int k = mCubes / g;
    d.groupOffset = (k * groupIdx) / groupNum;
    d.groupStart = (d.total * groupIdx * groupNum) / blockDim;
    d.groupEnd = (d.total * (groupIdx + 1) * groupNum) / blockDim - 1;
    d.groupblocks = d.groupEnd - d.groupStart + 1;
    return d;
}

// 全局序号 idx 是否由本核发射；idx2 为轮转后的本核执行序号
__aicore__ inline bool OwnedTile(const TileDispatch& d, int idx, int& idx2)
{
    if (d.groupblocks <= 0 || idx < d.groupStart || idx > d.groupEnd)
    {
        return false;
    }
    // idx2 -> 全局 idx 的逆轮转（与分配公式互为反函数，集合完全一致）
    idx2 = d.groupStart + ((idx - d.groupStart + d.groupblocks - d.groupOffset) % d.groupblocks);
    return idx2 >= d.start && idx2 <= d.end;
}

// 发射第 i 列块上 jBegin 起 jCount 个 tile，blockN 为单块列宽
__aicore__ inline void IssueTileColumn(MatmulCustom<float, float, float>& opmm, const TileDispatch& d, int i,
                                       int jBegin, int jCount, int blockN, int& idx)
{
    for (int j = jBegin; j < jBegin + jCount; ++j)
    {
        ++idx;
        int idx2 = 0;
        if (OwnedTile(d, idx, idx2))
        {
            // 列方向按剩余列裁剪，避免 128 对齐把最后一档铺到 padN 之外
            int blkM = min(d.stride, d.updateCols - i * d.stride);
            opmm.Process2(i * d.stride * d.vectorBlockdim, j * d.stride, i * d.stride * d.memStride + j * d.stride,
                          blkM, blockN, d.vectorBlockdim, -1, false, true);
        }
    }
}

// 按末列是否单独用 64 列宽处理，分发 (i, j) 两方向的所有 tile
__aicore__ inline void IssueUpdateTiles(MatmulCustom<float, float, float>& opmm, const TileDispatch& d,
                                        bool splitLast)
{
    if (d.groupblocks <= 0 || d.mCubes <= 0)
    {
        return;
    }
    int idx = -1;
    int full = splitLast ? d.mCubes - 1 : d.mCubes;
    for (int i = 0; i < d.uCubes; ++i)
    {
        IssueTileColumn(opmm, d, i, 0, full, d.stride, idx);
    }
    if (!splitLast)
    {
        return;
    }
    for (int i = 0; i < d.uCubes; ++i)
    {
        IssueTileColumn(opmm, d, i, d.mCubes - 1, 1, 64, idx);
    }
}

// 第一段：计算 W = Y^T * C（按 64 行块在核间均分），rows 返回本核的行分块
__aicore__ inline void ComputeWYtC(MatmulCustom<float, float, float>& opmm, UpdateEnv& env, const UpdateCtrl& ctrl,
                                   int updateCols, CubeSplit& rows)
{
    opmm.SetMatrix(env.A[ctrl.Cstart], env.Y[ctrl.flag * env.panelYSize], env.W, ctrl.vectorBlockdim, env.M);
    rows = SplitCubes(ctrl.MtoCompute, env.blockDim, env.blockIdx, ctrl.MtoCompute);
    if (rows.myNum)
    {
        opmm.Process(rows.myStart * 64, rows.myStart * 64, 0, updateCols, ctrl.vectorBlockdim, rows.myBlock,
                     env.blockIdx, true, true);
    }
    if (rows.cubes < env.blockDim && env.blockIdx >= rows.cubes)
    {
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
        CrossCoreWaitFlag(0x3);
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
    CrossCoreWaitFlag(0x3);
}

// 计算 W = T * W（只需 0 号核做）
__aicore__ inline void ApplyPanelT(MatmulCustom<float, float, float>& opmm, UpdateEnv& env, const UpdateCtrl& ctrl,
                                   int updateCols)
{
    if (env.blockIdx == 0)
    {
        opmm.SetMatrix(env.W, env.T[ctrl.flag * env.panelTSize], env.W, ctrl.vectorBlockdim, ctrl.vectorBlockdim);
        opmm.Process(0, 0, 0, updateCols, ctrl.vectorBlockdim, ctrl.vectorBlockdim, -1, false, false);
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
    CrossCoreWaitFlag(0x3);
}

// 第一段：计算 C = C - Y * W（按 64 行块在核间均分）
__aicore__ inline void UpdateCByYW(MatmulCustom<float, float, float>& opmm, UpdateEnv& env, const UpdateCtrl& ctrl,
                                   int updateCols, const CubeSplit& rows)
{
    opmm.SetMatrix(env.W, env.Y[ctrl.flag * env.panelYSize], env.A[ctrl.Cstart], env.M, ctrl.vectorBlockdim);
    if (rows.myNum)
    {
        opmm.Process(0, rows.myStart * 64, rows.myStart * 64, updateCols, rows.myBlock, ctrl.vectorBlockdim, -1,
                     false, true);
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
    CrossCoreWaitFlag(0x3);
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
}

// 第二段：计算 W = Y^T * C（按 64 列块在核间均分），cols 返回本核的列分块
__aicore__ inline void ComputeWYtCTiles(MatmulCustom<float, float, float>& opmm, UpdateEnv& env,
                                        const UpdateCtrl& ctrl, int updateCols, int mCubes, CubeSplit& cols,
                                        int& align)
{
    cols = SplitCubes(updateCols, env.blockDim, env.blockIdx, updateCols);
    opmm.SetMatrix(env.A[ctrl.Cstart], env.Y[ctrl.flag * env.panelYSize], env.W, ctrl.vectorBlockdim, env.M);
    align = (ctrl.MtoCompute % 128) ? 64 : 128;
    if (cols.myNum)
    {
        opmm.Process(cols.myStart * 64 * env.M, 0, cols.myStart * 64 * ctrl.vectorBlockdim, cols.myBlock,
                     ctrl.vectorBlockdim, mCubes == 1 ? ctrl.MtoCompute : align, -1, true, false);
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
    CrossCoreWaitFlag(0x3);
}

// 发射一段连续 tile（同一行、列块相邻）
__aicore__ inline void IssueSegment(MatmulCustom<float, float, float>& opmm, UpdateEnv& env, const UpdateCtrl& ctrl,
                                    int updateCols, int align, int runI, int runJ0, int runJ1)
{
    int stride = 128;
    // 列方向按剩余列裁剪，避免 128 对齐把最后一档铺到 padN 之外
    int blkM = min(stride, updateCols - runI * stride);
    opmm.Process(runI * stride * env.M + runJ0 * stride + align,  // C 偏移（段首）
                 runJ0 * stride + align,                          // A 偏移（连续）
                 runI * stride * ctrl.vectorBlockdim,             // B 偏移
                 blkM, ctrl.vectorBlockdim,
                 (runJ1 - runJ0 + 1) * stride,  // N 尺寸
                 -1, true, true);
}

// 第二段：C = C - Y * W 的第一阶段（128 对齐、按连续段合并发单）
__aicore__ inline void IssuePanelUpdates(MatmulCustom<float, float, float>& opmm, UpdateEnv& env,
                                         const UpdateCtrl& ctrl, int updateCols, int align)
{
    int stride = 128;
    // 行方向按 128 对齐后的块数（与 align 配套，最多比 64 分块少一块）
    int mCubes = (ctrl.MtoCompute - align + stride - 1) / stride;
    TileDispatch d = MakeTileDispatch(env.blockDim, env.blockIdx, mCubes, (updateCols + stride - 1) / stride, stride,
                                      ctrl.vectorBlockdim, updateCols, env.M);
    if (d.groupblocks <= 0 || d.mCubes <= 0)
    {
        return;
    }
    int runI = -1, runJ0 = -1, runJ1 = -1;  // 当前未发射的连续段
    for (int t = d.start; t <= d.end; ++t)
    {
        // idx2 -> 全局 idx 的逆轮转（与分配公式互为反函数，集合完全一致）
        int idx = d.groupStart + ((t - d.groupStart + d.groupOffset) % d.groupblocks);
        int i = idx / d.mCubes;
        int j = idx % d.mCubes;
        // 同一行、且列块相邻才合并（换行/跨缝都会自然断段）
        if (runI >= 0 && i == runI && j == runJ1 + 1)
        {
            runJ1 = j;  // 延续当前段
            continue;
        }
        if (runI >= 0)
        {  // 断段：先发上一段
            IssueSegment(opmm, env, ctrl, updateCols, align, runI, runJ0, runJ1);
        }
        runI = i;
        runJ0 = j;
        runJ1 = j;  // 开新段
    }
    if (runI >= 0)
    {  // 收尾
        IssueSegment(opmm, env, ctrl, updateCols, align, runI, runJ0, runJ1);
    }
}

// 第二段：计算 W = T * W
__aicore__ inline void ApplyPanelTTiles(MatmulCustom<float, float, float>& opmm, UpdateEnv& env,
                                        const UpdateCtrl& ctrl, const CubeSplit& cols)
{
    opmm.SetMatrix(env.W, env.T[ctrl.flag * env.panelTSize], env.W, ctrl.vectorBlockdim, ctrl.vectorBlockdim);
    if (cols.myNum)
    {
        opmm.Process(cols.myStart * 64 * ctrl.vectorBlockdim, 0, cols.myStart * 64 * ctrl.vectorBlockdim, cols.myBlock,
                     ctrl.vectorBlockdim, ctrl.vectorBlockdim, -1, false, false);
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
    CrossCoreWaitFlag(0x3);
}

// 第二段：计算 C = C - Y * W（末列按 64 列宽单独发单）
__aicore__ inline void UpdateCByYWTiles(MatmulCustom<float, float, float>& opmm, UpdateEnv& env,
                                        const UpdateCtrl& ctrl, int updateCols)
{
    int stride = 128;
    opmm.SetMatrix(env.W, env.Y[ctrl.flag * env.panelYSize], env.A[ctrl.Cstart], env.M, ctrl.vectorBlockdim);
    TileDispatch d = MakeTileDispatch(env.blockDim, env.blockIdx, (ctrl.MtoCompute + stride - 1) / stride,
                                      (updateCols + stride - 1) / stride, stride, ctrl.vectorBlockdim, updateCols,
                                      env.M);
    IssueUpdateTiles(opmm, d, (ctrl.MtoCompute % stride) != 0);
}

// 面板推进：减去已完成的行/列块，切换列宽与 T/Y 的奇偶片
__aicore__ inline void AdvanceCtrl(UpdateCtrl& ctrl, int M, int N)
{
    ctrl.MtoCompute -= ctrl.vectorBlockdim;
    if (ctrl.loop <= ctrl.changeLoop)
    {
        ctrl.vectorBlockdim = 128;
    }
    ctrl.remainingCols -= ctrl.vectorBlockdim;
    ctrl.Cstart = (N - ctrl.remainingCols) * (M + 1) - ctrl.vectorBlockdim;
    ctrl.flag = ctrl.flag ^ 1;
}

// 绑定 6 个 GM 缓冲并记录核数与 panel 步长
__aicore__ inline void InitUpdateEnv(UpdateEnv& env, GM_ADDR a, GM_ADDR tau, GM_ADDR T, GM_ADDR w, GM_ADDR y,
                                     int orgM, int M, int N)
{
    env.A.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    env.T.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T));
    env.W.SetGlobalBuffer((reinterpret_cast<__gm__ float*>(w)) + 2048);
    env.Y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    env.orgM = orgM;
    env.M = M;
    env.N = N;
    env.blockDim = GetBlockNum();
    env.blockIdx = GetBlockIdx();
    env.panelTSize = 32 * 32 * 16;
    env.panelYSize = 32 * M * 4;
}

// 本轮列块宽度：整 128 取 128，尾部不足 128 时退化为 64
__aicore__ inline int NextUpdateCols(const UpdateCtrl& ctrl)
{
    int updateCols = (ctrl.remainingCols > 128) ? 128 : ctrl.remainingCols;
    if (ctrl.remainingCols % 128)
    {
        updateCols = 64;
    }
    return updateCols;
}

__aicore__ inline void update_c(GM_ADDR a, GM_ADDR tau, GM_ADDR T, GM_ADDR w, GM_ADDR y, int orgM, int M, int N)
{
    TPipe pipe;
    MatmulCustom<float, float, float> opmm;
    UpdateEnv env;
    InitUpdateEnv(env, a, tau, T, w, y, orgM, M, N);
    opmm.Init(&pipe);

    UpdateCtrl ctrl;
    CalcUpdateLoopParam(orgM, M, N, ctrl);

    while (ctrl.loop--)
    {
        CrossCoreWaitFlag(0x1);
        if (orgM < N && ctrl.loop == 0)
        {
            ctrl.vectorBlockdim = ctrl.mTail;
        }
        int updateCols = NextUpdateCols(ctrl);

        CubeSplit rows;
        ComputeWYtC(opmm, env, ctrl, updateCols, rows);
        ApplyPanelT(opmm, env, ctrl, updateCols);
        UpdateCByYW(opmm, env, ctrl, updateCols, rows);

        // 更新剩余矩阵
        ctrl.Cstart += updateCols * env.M;
        int restCols = ctrl.remainingCols - updateCols;
        if (restCols == 0)
        {
            AdvanceCtrl(ctrl, M, N);
            continue;
        }

        CubeSplit cols;
        int align = 128;
        ComputeWYtCTiles(opmm, env, ctrl, restCols, rows.cubes, cols, align);
        IssuePanelUpdates(opmm, env, ctrl, restCols, align);
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
        CrossCoreWaitFlag(0x3);
        ApplyPanelTTiles(opmm, env, ctrl, cols);
        UpdateCByYWTiles(opmm, env, ctrl, restCols);
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);

        AdvanceCtrl(ctrl, M, N);
        CrossCoreWaitFlag(0x3);
    }
    if (orgM >= N)
    {
        CrossCoreWaitFlag(0x1);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
}

// 小矩阵路径：单核 Householder QR（N * align_to(M, 64) <= 8192）
__aicore__ inline void qr_single_core(int M, int N, int padM, GM_ADDR a, GM_ADDR aPad, GM_ADDR tau)
{
#ifdef __DAV_C220_VEC__
    if (GetBlockIdx())
    {
        return;
    }
    AscendC::singleCoreHouseholderQR op;
    op.Init();
    op.SetMatrix(a, aPad, tau, M, N, padM);
    op.Compute();
#elif __DAV_C220_CUBE__
    return;
#endif
}

// 面板路径：AIV 做 panel 分解与 T/W 生成，AIC 做 C 的批量更新
__aicore__ inline void qr_panel(int M, int N, int padM, int padN, GM_ADDR a, GM_ADDR aPad, GM_ADDR tau,
                                GM_ADDR work_gm)
{
    // work 区布局与 paco 侧 T / W / Y 三块独立缓冲区一致
    constexpr int MAX_PANEL_SIZE = 128;
    GM_ADDR T = work_gm;
    GM_ADDR W = T + (MAX_PANEL_SIZE * MAX_PANEL_SIZE * 2) * sizeof(float);
    GM_ADDR Y = W + (padM * padN + 2048) * sizeof(float);
#ifdef __DAV_C220_VEC__
    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    pipe.InitBufPool(tbufPool, 192 * 1024);
    AscendC::HouseholderQR op;

    if (M < 64)
    {
        op.SetMatrix(a, aPad, tau, T, W, Y, M, N, padM, padN);
        op.Pad(&tbufPool, N, M, padN, padM, GetBlockIdx(), GetBlockNum() << 1);
    }
    else
    {
        op.SetMatrix(a, aPad, tau, T, W, Y, padM, padN, padM, padN);
        op.Pad(&tbufPool, N, M, padN, padM, GetBlockIdx(), GetBlockNum() << 1);
    }

    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x3);
    CrossCoreWaitFlag(0x3);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
    tbufPool.Reset();
    op.Init(&tbufPool);

    op.ComputeAll(GetBlockNum() << 1, GetBlockIdx());
#elif __DAV_C220_CUBE__
    CrossCoreWaitFlag(0x4);
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
    CrossCoreWaitFlag(0x3);
    if (M < 64)
    {
        update_c(aPad, tau, T, W, Y, M, padM, N);
    }
    else
    {
        update_c(aPad, tau, T, W, Y, padM, padM, padN);
    }
#endif
}

__aicore__ inline void qr_custom(
    int M, int N, int padM, int padN, GM_ADDR a, GM_ADDR aPad, GM_ADDR tau, GM_ADDR work_gm)
{
    if (N * align_to(M, 64) <= 8192)
    {
        qr_single_core(M, N, padM, a, aPad, tau);
        return;
    }
    qr_panel(M, N, padM, padN, a, aPad, tau, work_gm);
}

__global__ __aicore__ void sgeqrf_kernel(
    GM_ADDR sync, int orgM, int orgN, int padM, int padN, GM_ADDR A, GM_ADDR A_pad, GM_ADDR tau, GM_ADDR work_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    qr_custom(orgM, orgN, padM, padN, A, A_pad, tau, work_gm);
}

void sgeqrf_kernel_do(
    GM_ADDR sync, int orgM, int orgN, int padM, int padN, GM_ADDR A, GM_ADDR A_pad, GM_ADDR tau, GM_ADDR work_gm,
    uint32_t numBlocks, void* stream)
{
    sgeqrf_kernel<<<numBlocks, nullptr, stream>>>(sync, orgM, orgN, padM, padN, A, A_pad, tau, work_gm);
}

#endif  // SGEQRF_MIX_H
