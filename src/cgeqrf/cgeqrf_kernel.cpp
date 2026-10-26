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
 * \file cgeqrf_kernel.cpp
 * \brief
 */
#ifndef CGEQRF_MIX_H
#define CGEQRF_MIX_H

#include "kernel/gemm.hpp"
#include "kernel/gemm_impl.hpp"
#include "kernel/gemm_tile.hpp"
#include "kernel/householder_cube.hpp"
#include "kernel/householder_vec.hpp"
#include "kernel/householder_vec_impl.hpp"
#include "kernel/householder_vec2.hpp"
#include "kernel/householder_vec2_impl.hpp"
#include "kernel/pad.hpp"
#include "kernel/qr_common.hpp"
#include "cgeqrf_launchers.hpp"

// 内核主体与 paco cgeqrf 保持一致（含 A2/A3 修复）。host 侧已把 A 打包成 M 间距，
// 故内核内部一律按 M 取步长；lda 只作用于 host 的 H2D/D2H pitch。
// GM 句柄与形状打包一处，各分支只做一次 SetMatrix/Init，不再逐个重复 14 个实参
struct QrEnv
{
    GlobalTensor<float> aOrgGlobal, aRealGlobal, aImagGlobal;
    GlobalTensor<float> tauGlobal, eyeGlobal;
    GlobalTensor<uint32_t> gatherN, gatherBuf;
    GlobalTensor<float> tRealGlobal, tImagGlobal, tImagNegGlobal;
    GlobalTensor<float> wRealGlobal, wImagGlobal;
    GlobalTensor<float> yRealGlobal, yImagGlobal, yImagNegGlobal;
    int M, N, padM, padN;
    uint32_t blockDim, blockIdx;
};
__aicore__ inline void MakeQrEnv(QrEnv& env, int M, int N, int padM, GM_ADDR A, GM_ADDR aPad, GM_ADDR tau,
                                 GM_ADDR work_gm, GM_ADDR gather_gm, GM_ADDR eye_gm)
{
    env.blockDim = GetBlockNum();
    env.blockIdx = GetBlockIdx();
    env.M = M;
    env.N = N;
    env.padM = padM;
    env.aOrgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    env.aRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPad));
    env.padN = (N + 127) / 128 * 128;
    env.aImagGlobal = env.aRealGlobal[padM * env.padN];
    env.tauGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tau));
    env.gatherN.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gather_gm));
    env.eyeGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye_gm));
    env.gatherBuf = env.gatherN[128];
    int k = (N + 127) / 128;
    env.tRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm));
    env.tImagGlobal = env.tRealGlobal[padM * env.padN];
    env.tImagNegGlobal = env.tImagGlobal[padM * env.padN];
    env.wRealGlobal = env.tImagNegGlobal[padM * env.padN];
    env.wImagGlobal = env.wRealGlobal[max(padM, env.padN) * 128 + 4096 + 64];
    env.yRealGlobal = env.wImagGlobal[max(padM, env.padN) * 128 + 4096 + 64];
    env.yImagGlobal = env.yRealGlobal[padM * env.padN];
    env.yImagNegGlobal = env.yImagGlobal[padM * env.padN];
}
// VEC / CUBE 两侧共用的形状判据：小规模走 32 面板，超大高瘦列走另一套驱动
__aicore__ inline bool IsSmallShape(const QrEnv& env)
{
    if (env.M <= 256) return true;
    if (env.N <= 256) return true;
    return env.M / env.N >= 8;
}
#ifdef __DAV_C220_VEC__
__aicore__ inline void RunSplitComplex(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    SplitRealImag<float> opsp;
    opsp.Init(&tbufPool);
    opsp.SetMatrix(env.aOrgGlobal, env.aRealGlobal, env.aImagGlobal, env.M, env.M, env.padM);
    opsp.SplitRealImagWork(env.N, env.blockDim << 1, env.blockIdx);
    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);
}
// 返回 true 表示本核提前结束（不再做尾部的实虚合并）
__aicore__ inline bool RunSingleCore(const QrEnv& env)
{
    if (env.blockIdx) return true;
    AscendC::singleCoreHouseholderQR op;
    op.Init();
    op.Compute();
    return false;
}
__aicore__ inline void RunVecSmall(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    AscendC::HouseholderQR_VEC op;
    op.Init(&tbufPool);
    op.SetMatrix(env.aRealGlobal, env.aImagGlobal, env.tauGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                 env.wRealGlobal, env.wImagGlobal, env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal, env.M, env.N,
                 env.padM, env.N);
    op.InitGather(env.gatherN);
    op.Process2(env.blockDim << 1, env.blockIdx);
    op.FreeGather();
    tbufPool.Reset();
    CrossCoreWaitFlag(0x2);
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);
}
__aicore__ inline void RunVecBigVec(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    AscendC::HouseholderQR_VEC op;
    op.Init(&tbufPool);
    op.SetMatrix(env.aRealGlobal, env.aImagGlobal, env.tauGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                 env.wRealGlobal, env.wImagGlobal, env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal, env.padM,
                 env.padN, env.padM, env.padN);
    op.InitGather(env.gatherN);
    op.Process(env.blockDim << 1, env.blockIdx);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
    op.FreeGather();
    tbufPool.Reset();
}
__aicore__ inline void RunVecBigVec2(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    AscendC::HouseholderQR_VEC2 op2;
    op2.Init(&tbufPool);
    op2.SetMatrix(env.aRealGlobal, env.aImagGlobal, env.tauGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                  env.wRealGlobal, env.wImagGlobal, env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal, env.M, env.N,
                  env.padM, env.padN);
    CrossCoreWaitFlag(0x9);
    op2.ComputeAll(env.blockDim << 1, env.blockIdx);
    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);
}
__aicore__ inline void RunVecDefault(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    AscendC::HouseholderQR_VEC2 op2;
    op2.Init(&tbufPool);
    op2.SetMatrix(env.aRealGlobal, env.aImagGlobal, env.tauGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                  env.wRealGlobal, env.wImagGlobal, env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal, env.M, env.N,
                  env.padM, env.padN);
    op2.ComputeAll2(env.blockDim << 1, env.blockIdx);
    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);
}
__aicore__ inline void RunVec(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    if (IsSmallShape(env))
    {
        RunVecSmall(env, tbufPool);
    }
    else if (env.M > CHANGE)
    {
        RunVecBigVec(env, tbufPool);
        RunVecBigVec2(env, tbufPool);
    }
    else
    {
        RunVecDefault(env, tbufPool);
    }
}
__aicore__ inline void RunMergeRealImag(QrEnv& env, TBufPool<TPosition::VECCALC, 16>& tbufPool)
{
    MergeRealImag<float> opmg;
    static constexpr int TILE_LENGTH = 4096;
    opmg.Init(&tbufPool, env.gatherBuf);
    opmg.SetMatrix(env.aOrgGlobal, env.aRealGlobal, env.aImagGlobal, min(env.M, TILE_LENGTH), env.padM, env.M);
    opmg.MergeRealImagWork(env.N, env.blockDim << 1, env.blockIdx);
    if (env.M > TILE_LENGTH)
    {
        opmg.SetMatrix(env.aOrgGlobal[TILE_LENGTH * 2], env.aRealGlobal[TILE_LENGTH], env.aImagGlobal[TILE_LENGTH],
                       env.M - TILE_LENGTH, env.padM, env.M);
        opmg.MergeRealImagWork(env.N, env.blockDim << 1, env.blockIdx);
    }
    tbufPool.Reset();
}
#endif
#ifdef __DAV_C220_CUBE__
__aicore__ inline void RunCube(QrEnv& env, TPipe& pipe)
{
    CMatmulCustom<float> opmm;
    opmm.Init(&pipe);
    AscendC::HouseholderQR_CUBE op;

    int minMN = min(env.M, env.N);
    if (IsSmallShape(env))
    {
        op.Init(&opmm, env.aRealGlobal, env.aImagGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                env.wRealGlobal[4160], env.wImagGlobal[4160], env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal,
                env.eyeGlobal, env.M, env.padM, env.N, env.blockDim);
        op.Process4(env.blockIdx);
    }
    else if (env.M > CHANGE)
    {
        op.Init(&opmm, env.aRealGlobal, env.aImagGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                env.wRealGlobal[4160], env.wImagGlobal[4160], env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal,
                env.eyeGlobal, env.padM, env.padM, env.padN, env.blockDim);
        op.Process2(env.blockIdx);
        op.Process(env.blockIdx);
    }
    else
    {
        op.Init(&opmm, env.aRealGlobal, env.aImagGlobal, env.tRealGlobal, env.tImagGlobal, env.tImagNegGlobal,
                env.wRealGlobal[4160], env.wImagGlobal[4160], env.yRealGlobal, env.yImagGlobal, env.yImagNegGlobal,
                env.eyeGlobal, env.padM, env.padM, env.padN, env.blockDim);
        CrossCoreWaitFlag(0x8);
        op.Process3(env.blockIdx);
    }
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);
}
#endif
__aicore__ inline void qr_custom(int M, int N, int lda, int padM, GM_ADDR A, GM_ADDR aPad, GM_ADDR tau, GM_ADDR work_gm,
                                 GM_ADDR gather_gm, GM_ADDR eye_gm)
{
    (void)lda;
    QrEnv env;
    MakeQrEnv(env, M, N, padM, A, aPad, tau, work_gm, gather_gm, eye_gm);
    TPipe pipe;
#ifdef __DAV_C220_VEC__
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    pipe.InitBufPool(tbufPool, 192 * 1024 * 4);
    RunSplitComplex(env, tbufPool);
#endif
    if (env.N * align_to(env.M, 64) <= 0)
    {
#ifdef __DAV_C220_VEC__
        if (RunSingleCore(env)) return;
#elif __DAV_C220_CUBE__
        return;
#endif
    }
    else
    {
#ifdef __DAV_C220_VEC__
        RunVec(env, tbufPool);
#elif __DAV_C220_CUBE__
        RunCube(env, pipe);
#endif
    }
#ifdef __DAV_C220_VEC__
    RunMergeRealImag(env, tbufPool);
#endif
}

__global__ __aicore__ void cgeqrf_kernel(GM_ADDR sync, int orgM, int orgN, int lda, int padM, GM_ADDR A, GM_ADDR A_pad,
                                         GM_ADDR tau, GM_ADDR work_gm, GM_ADDR gather_gm, GM_ADDR eye_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    qr_custom(orgM, orgN, lda, padM, A, A_pad, tau, work_gm, gather_gm, eye_gm);
}

void cgeqrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int lda, int padM, GM_ADDR A, GM_ADDR A_pad, GM_ADDR tau,
                      GM_ADDR work_gm, GM_ADDR gather_gm, GM_ADDR eye_gm, uint32_t numBlocks, void* stream)
{
    cgeqrf_kernel<<<numBlocks, nullptr, stream>>>(sync, orgM, orgN, lda, padM, A, A_pad, tau, work_gm, gather_gm,
                                                  eye_gm);
}
#endif  // CGEQRF_MIX_H