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
 * \file qr_common.hpp
 * \brief 单核路径与 VEC/VEC2 共用基类
 */

#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include "kernel_operator.h"

using namespace AscendC;
static constexpr int M_threshold = -1;  // 2048
static constexpr int MAX_PANEL_SIZE = 128;
static constexpr int PANEL_SIZE = 32;
static constexpr int CHANGE = 4096;

#define VV_SYNC_FLAG 0x0
#define VC_SYNC_FLAG 0x1
#define CV_SYNC_FLAG 0x2
#define CC_SYNC_FLAG 0x3

__aicore__ inline int align_to(int a, int b)
{
    if (b == 0)
    {
        return a;
    }
    return (a + b - 1) / b * b;
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

    GlobalTensor<float> Areal;
    GlobalTensor<float> AImag;
    GlobalTensor<float> tau;

   public:
    __aicore__ inline singleCoreHouseholderQR() : M(0), N(0) {}

    __aicore__ inline void Init()
    {
        pipe.InitBuffer(aQueue, 1, chunkSize * sizeof(float));
        pipe.InitBuffer(vBuf, chunkSize * sizeof(float));
        pipe.InitBuffer(workBuf, chunkSize * sizeof(float));
        pipe.InitBuffer(tauQueueDst, 1, 128 * sizeof(float));
    }

    __aicore__ inline void SetMatrix(__gm__ uint8_t* A1_, __gm__ uint8_t* A2_, __gm__ uint8_t* tau_, int M_, int N_,
                                     int strideM_)
    {
        Areal.SetGlobalBuffer((__gm__ float*)A1_);
        AImag.SetGlobalBuffer((__gm__ float*)A2_);
        tau.SetGlobalBuffer((__gm__ float*)tau_);
        M = M_;
        N = N_;
        strideM = strideM_;
        aligned_M = align_to(M_, 8);
    }

    // 整段矩阵进 UB（实部 offset 4096 之后接虚部）
    __aicore__ inline void LoadInput()
    {
        LocalTensor<float> a = aQueue.AllocTensor<float>();
        DataCopyParams copyParams{static_cast<uint16_t>(N), static_cast<uint16_t>(M * sizeof(float)),
                                  // static_cast<uint16_t>((strideM - M) * sizeof(float)),
                                  0, static_cast<uint16_t>((strideM - aligned_M) / 8)};
        DataCopyPadParams padParams{true, 0, static_cast<uint8_t>(aligned_M - M), 0};
        DataCopyPad(a, Areal, copyParams, padParams);
        DataCopyPad(a[4096], AImag, copyParams, padParams);
        aQueue.EnQue(a);
    }

    // 取第 k 列到 v，并把已经是 R 的前 k 行清零
    __aicore__ inline void MakeColumnV(int k, LocalTensor<float>& a, LocalTensor<float>& v)
    {
        Copy(v, a[k * strideM], 64, strideM / 64, {1, 1, 8, 8});
        Copy(v[4096], a[k * strideM + 4096], 64, strideM / 64, {1, 1, 8, 8});
        Duplicate<float>(v, 0, k);
        Duplicate<float>(v[4096], 0, k);
    }

    // v 的实部/虚部平方和落到 workLocal（读结果用 ColumnNormSum）
    __aicore__ inline void ColumnNormPrepare(LocalTensor<float>& v, LocalTensor<float>& workLocal)
    {
        Mul(workLocal, v, v, aligned_M);
        Mul(workLocal[4096], v[4096], v[4096], aligned_M);
        ReduceSum(workLocal, workLocal, workLocal, aligned_M);
        ReduceSum(workLocal[4096], workLocal[4096], workLocal[4096], aligned_M);
    }

    // 二范数平方 = 实部 + 虚部
    __aicore__ inline float ColumnNormSum(LocalTensor<float>& workLocal)
    {
        return workLocal.GetValue(0) + workLocal.GetValue(4096);
    }

    // 把第 k 个反射子作用到后面的列上
    __aicore__ inline void ApplyTrailing(int k, float tau_prime, LocalTensor<float>& v, LocalTensor<float>& a,
                                         LocalTensor<float>& workLocal)
    {
        for (int j = k + 1; j < N; ++j)
        {
            Mul(workLocal, v, a[j * strideM], aligned_M);
            ReduceSum(workLocal, workLocal, workLocal, aligned_M);
            Axpy(a[j * strideM], v, -tau_prime * workLocal.GetValue(0), aligned_M);
        }
    }

    // Muls 用的行掩码：前 M 行有效
    __aicore__ inline void InitMask(uint64_t* mask)
    {
        if (M >= 64)
            mask[0] = ~0ULL;
        else
            mask[0] = (1ULL << M) - 1;
    }

    // 复数 Householder 的标量部分：beta 为对角元，v0 为 v 的 k 元，tau_prime 为 1/|v0|^2 归一化系数
    struct Reflector
    {
        float beta, tauK, v0, tauPrime;
    };

    __aicore__ inline void MakeReflector(float x0, float norm_2, Reflector& r)
    {
        float norm_x = sqrt(norm_2);
        r.beta = x0 < 0 ? norm_x : -norm_x;
        r.tauK = (r.beta - x0) / r.beta;
        r.v0 = x0 - r.beta;
        r.tauPrime = r.tauK / (r.v0 * r.v0);
    }

    // 反射子写回 a：存 v / v0，对角元写 beta
    __aicore__ inline void StoreReflector(int k, float beta, float v0, LocalTensor<float>& a, LocalTensor<float>& v,
                                          uint64_t* mask)
    {
        if (v0 == 0)
        {
            return;
        }
        mask[0] ^= (1ull << k);
        Muls(a[k * strideM], v, 1.0f / v0, mask, 1, {1, 1, 8, 8});
        if (aligned_M > 64) Muls(a[k * strideM + 64], v[64], 1.0f / v0, aligned_M - 64);
        a.SetValue(k * strideM + k, beta);
    }

    // A 写回 GM
    __aicore__ inline void StoreMatrix(LocalTensor<float>& a)
    {
        DataCopyParams copyParams{static_cast<uint16_t>(N), static_cast<uint16_t>(M * sizeof(float)),
                                  static_cast<uint16_t>((strideM - aligned_M) / 8),
                                  static_cast<uint16_t>((strideM - M) * sizeof(float))};
        DataCopyPad(Areal, a, copyParams);
        DataCopyPad(AImag, a[4096], copyParams);
    }

    __aicore__ inline void Compute()
    {
        int minMN = M < N ? M : N;
        LoadInput();

        uint64_t mask[2] = {0, 0};
        InitMask(mask);

        LocalTensor<float> v = vBuf.Get<float>();
        LocalTensor<float> workLocal = workBuf.Get<float>();
        LocalTensor<float> Tau = tauQueueDst.AllocTensor<float>();

        LocalTensor<float> a = aQueue.DeQue<float>();
        for (int k = 0; k < minMN; ++k)
        {
            MakeColumnV(k, a, v);
            ColumnNormPrepare(v, workLocal);
            float x0 = v.GetValue(k);
            float norm_2 = ColumnNormSum(workLocal);
            if (norm_2 < float(1e-6))
            {
                Tau.SetValue(2 * k, 0.0f);
                Tau.SetValue(2 * k + 1, 0.0f);
                continue;
            }
            Reflector r;
            MakeReflector(x0, norm_2, r);
            v.SetValue(k, r.v0);

            ApplyTrailing(k, r.tauPrime, v, a, workLocal);
            StoreReflector(k, r.beta, r.v0, a, v, mask);

            Tau.SetValue(k, r.tauK);
        }
        tauQueueDst.EnQue(Tau);
        Tau = tauQueueDst.DeQue<float>();

        StoreMatrix(a);
        aQueue.FreeTensor<float>(a);

        DataCopyParams copyParams{static_cast<uint16_t>(N), static_cast<uint16_t>(M * sizeof(float)),
                                  // static_cast<uint16_t>((strideM - M) * sizeof(float)),
                                  0, static_cast<uint16_t>((strideM - aligned_M) / 8)};
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(minMN * sizeof(float));
        DataCopyPad(tau, Tau, copyParams);
        tauQueueDst.FreeTensor<float>(Tau);
    }
};
}  // namespace AscendC

namespace AscendC
{
// VEC / VEC2 共用的 GM 张量句柄与矩阵形状：SetMatrix 的公共部分只实现一次
class HouseholderQRCommon
{
   protected:
    int M, N, padM, padN;
    GlobalTensor<float> aGlobalReal, aGlobalImag;
    GlobalTensor<float> tau;
    GlobalTensor<float> tGlobalReal, tGlobalImag, tGlobalImagNeg;
    GlobalTensor<float> wGlobalReal, wGlobalImag;
    GlobalTensor<float> yGlobalReal, yGlobalImag, yGlobalImagNeg;
    GlobalTensor<float> tmpGlobal, v0Global;

    __aicore__ inline HouseholderQRCommon() : M(0), N(0), padM(0), padN(0) {}

    __aicore__ inline void SetMatrixCommon(GlobalTensor<float> aReal, GlobalTensor<float> aImag,
                                           GlobalTensor<float> tau_, GlobalTensor<float> tReal,
                                           GlobalTensor<float> tImag, GlobalTensor<float> tImagNeg,
                                           GlobalTensor<float> wReal, GlobalTensor<float> wImag,
                                           GlobalTensor<float> yReal, GlobalTensor<float> yImag,
                                           GlobalTensor<float> yImagNeg, int M_, int N_, int padM_, int padN_)
    {
        this->aGlobalReal = aReal, this->aGlobalImag = aImag;
        this->tau = tau_;
        this->tGlobalReal = tReal, this->tGlobalImag = tImag, this->tGlobalImagNeg = tImagNeg;
        this->wGlobalReal = wReal, this->wGlobalImag = wImag;
        this->yGlobalReal = yReal, this->yGlobalImag = yImag, this->yGlobalImagNeg = yImagNeg;
        this->tmpGlobal = wGlobalReal[2048];  // 64
        this->v0Global = wGlobalImag[2048];   // 64
        this->M = M_;
        this->N = N_;
        this->padM = padM_;
        this->padN = padN_;
    }
};
}  // namespace AscendC
