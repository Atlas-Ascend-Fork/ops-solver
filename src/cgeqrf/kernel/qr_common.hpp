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
