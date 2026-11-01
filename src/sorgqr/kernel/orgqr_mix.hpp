/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file orgqr_mix.hpp
 * \brief 由 geqrf 的 (v, tau) 生成显式 Q 的前 n 列（融合 MIX kernel，Ascend950PR / fp32）
 *
 * 核心恒等式（省掉 LAPACK dorg2r 的顺序循环）：
 *   块反射子 Q_blk = I - V*T*V^T，org2r 要的是它的前 ib 列 Q_blk*E（E = [I_ib; 0]），
 *   而 V^T*E = (V(1:ib,1:ib))^T = L^T ⇒ **先把本块自己的列置成单位阵的对应列，
 *   再跑一遍标准 LARFB，出来的就是 org2r 的结果** ⇒ 本块展开与右侧列更新合并成同一次 LARFB。
 *
 * 块序**从后往前**：V 在被本块覆盖之前就已读进 wsY，所以可以原地做。
 * FLOP 正好是 LAPACK 的结构利用计数（约 4/3·n^3），不是 2mnk 的稠密计数。
 *
 */

#pragma once

// 五个 mix kernel 的公共件（CATLASS include、类型别名、跨核同步、分 K 调度、
// householder/larfb_plan、以及标准库那几个头）都由这一个头提供。
#include "../../sgeqrf/kernel/mix_common.hpp"


constexpr uint32_t MAX_NB = 128;

struct OrgqrParams {
    GM_ADDR A;    // m x n 列主序 ld=lda：输入 geqrf 的输出，输出 Q 的前 n 列
    GM_ADDR tau;  // k 个
    GM_ADDR wsY;  // 物化的 V：ldy x nb
    GM_ADDR wsT;  // -T：nb x nb（行主序 ld=jb）
    GM_ADDR wsG;  // G = V^T V：nb x nb（行主序）
    GM_ADDR wsW;  // LARFB 的 W / W2（复用同一块；jb <= 256 时安全）
    GM_ADDR wsP;  // 分 K 的 partial 区：nChunks 份，每份 (plan.m x plan.n) 连续存放
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldy;
    uint32_t nb;
    GM_ADDR devInfo;  // device 侧 int32：成功时由 kernel 自己写 0，可为空
    uint64_t capP;  // wsP 的容量（元素）。分 K 份数的硬上限，**host 是唯一来源**：
                    // kernel 只拿它做 clamp，不自己再推一遍容量公式（两份公式漂移过）
};


/// 第 b 块的形状：起始列 j、块宽 jb、参与的行数 rows、LARFB 作用的列数 cols
/// （cols = 本块自己 + 右侧全部）。
struct OrgqrBlockShape {
    uint32_t j, jb, rows, cols;
};

__aicore__ inline OrgqrBlockShape OrgqrBlockAt(uint32_t b, uint32_t m, uint32_t n, uint32_t k, uint32_t nb)
{
    const uint32_t j = b * nb;
    const uint32_t jb = ((k - j) < nb) ? (k - j) : nb;
    return OrgqrBlockShape{j, jb, m - j, n - j};
}

/// 排 LARFB 的三段 plan，并决定第一段要不要分 K。返回份数（1 = 不分）。
///
/// T 是行主序存的，而 PlanLarfb* 的视图标志按"T 为列主序"定义 ⇒ 这里要的 Q（NoTrans）
/// 对应传 Trans::Trans。（geqrf 要 Q^T 时传 NoTrans，正好相反。）
/// lp[0] 的行主序视图是 (m=cols, n=jb, k=rows)：N 方向只有 1 个 tile、M 方向
/// ceil(cols/256) 个 ⇒ cols=765 时只有 3 个核在读整个 C（12.5 MB）。分 K 把它摊开。
__aicore__ inline uint32_t OrgqrPlanLarfb(uint32_t rows, uint32_t cols, uint32_t jb, uint32_t ldy, uint32_t lda,
                                          uint32_t j, uint64_t capP, QrCore::GemmPlan* lp, uint32_t* wChunkLen)
{
    const int64_t cOff = static_cast<int64_t>(j) * lda + j;
    QrCore::PlanLarfbLeft(rows, cols, jb, ldy, lda, jb, jb, jb, 0, cOff, lp, QrCore::Trans::Trans);
    return KSplitChunks(lp[0].m, lp[0].n, lp[0].k, AscendC::GetBlockNum(), capP, wChunkLen);
}

/// kernel 要用到的 GM 视图，一次绑好
struct OrgqrGm {
    AscendC::GlobalTensor<float> A, tau, Y, T, G, W, P;

    __aicore__ inline void Init(const OrgqrParams& params)
    {
        A.SetGlobalBuffer((__gm__ float*)params.A);
        tau.SetGlobalBuffer((__gm__ float*)params.tau);
        Y.SetGlobalBuffer((__gm__ float*)params.wsY);
        T.SetGlobalBuffer((__gm__ float*)params.wsT);
        W.SetGlobalBuffer((__gm__ float*)params.wsW);
        P.SetGlobalBuffer((__gm__ float*)params.wsP);
        G.SetGlobalBuffer((__gm__ float*)params.wsG);
    }
};

#ifdef __DAV_VEC__
/// AIV 侧的 UB 分块。TQue / TBuf 必须活到 kernel 结束，所以一起放在这个结构体里，
/// 由 kernel 在自己的作用域里声明一个实例。
struct OrgqrUb {
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qIn, qG, qTau;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufW, bufT, bufRed;
    AscendC::LocalTensor<float> work, tl, red;

    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(qIn, 1, MAX_ROWS * sizeof(float));
        pipe.InitBuffer(qG, 1, MAX_NB * MAX_NB * sizeof(float));
        pipe.InitBuffer(qTau, 1, MAX_NB * sizeof(float));
        pipe.InitBuffer(bufW, MAX_ROWS * sizeof(float));
        pipe.InitBuffer(bufT, MAX_NB * MAX_NB * sizeof(float));
        pipe.InitBuffer(bufRed, RED_CAP * sizeof(float));
        work = bufW.Get<float>();
        tl = bufT.Get<float>();
        red = bufRed.Get<float>();
    }
};

/// 把 A 的第 c 列整列置成单位阵的第 c 列（全 0，仅第 c 行为 1）
__aicore__ inline void OrgqrSetIdentityCol(const AscendC::GlobalTensor<float>& A,
                                           const AscendC::LocalTensor<float>& work, uint32_t m, uint32_t lda,
                                           uint32_t c)
{
    AscendC::Duplicate(work, 0.0f, AlignUp32(m, 8));
    AscendC::PipeBarrier<PIPE_ALL>();
    if (c < m) {
        work.SetValue(c, 1.0f);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(m * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(A[static_cast<uint64_t>(c) * lda], work, cp);
    AscendC::PipeBarrier<PIPE_ALL>();
}

/// T 的三角递推并写出 -T（LARFB 要的是 -T）。
///
#endif  // __DAV_VEC__

#ifdef __DAV_CUBE__
/// G = V^T V（cube 侧）。计算与跨核发布都在共享的 GemmGBlock 里，ormqr 用的是同一份。
__aicore__ inline void OrgqrGemmG(const OrgqrParams& params, uint32_t rows, uint32_t jb, uint32_t ldy,
                                  uint32_t gChunks, uint32_t gChunkLen)
{
    GemmGBlock(params.wsY, params.wsG, params.wsP, rows, jb, ldy, gChunks, gChunkLen);
}

/// LARFB 的三次 GEMM（cube 侧）。分 K 时中间要把 partial 交给 vec 归约再回来做 lp[1]，
/// 所以这里的 Notify/Wait 必须与 OrgqrLarfbVec 一一对应。
__aicore__ inline void OrgqrLarfbCube(const OrgqrParams& params, const QrCore::GemmPlan* lp, uint32_t wChunks,
                                      uint32_t wChunkLen)
{
    if (wChunks > 1) {
        RunPlanOnCubeSplitK<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[0], params.A, params.wsY, params.wsP,
                                                                          wChunks, wChunkLen);
        CubeBarrier();
        CubeNotifyVec();  // partial 齐了，交给 vec 归约
        CubeWaitVec();    // 等归约完再做 lp[1]
    } else {
        RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[0], params.A, params.wsY, params.wsW);
    }
    CubeBarrier();
    if (lp[1].bColMajorView) {
        RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[1], params.wsW, params.wsT, params.wsW);
    } else {
        RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[1], params.wsW, params.wsT, params.wsW);
    }
    CubeBarrier();
    RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[2], params.wsW, params.wsY, params.A);
    CubeBarrier();
    CubeNotifyVec();
}
#endif  // __DAV_CUBE__

#ifdef __DAV_VEC__
/// LARFB 期间 vec 侧要做的事：分 K 时归约 lp[0] 的 partial，然后等 cube 把本块做完。
__aicore__ inline void OrgqrLarfbVec(const AscendC::GlobalTensor<float>& W, const AscendC::GlobalTensor<float>& P,
                                     const QrCore::GemmPlan* lp, const AscendC::LocalTensor<float>& work,
                                     const AscendC::LocalTensor<float>& red, uint32_t wChunks, uint32_t coreId,
                                     uint32_t coreNum)
{
    LarfbReduceVec(W, P, lp, work, red, wChunks, coreId, coreNum);
    VecWaitCube();  // 本块做完
}
/// 块循环之前的 vec 侧准备：devInfo 置 0，再把第 k..n-1 列置成单位阵的对应列
/// （它们会被所有块的反射子依次作用）。
__aicore__ inline void OrgqrPrologueVec(const OrgqrParams& params, const OrgqrGm& gm, OrgqrUb& ub,
                                        uint32_t m, uint32_t n, uint32_t k, uint32_t lda, uint32_t coreId,
                                        uint32_t coreNum)
{
    ZeroDevInfo(params.devInfo, ub.work, coreId);
    for (uint32_t c = k + coreId; c < n; c += coreNum) {
        OrgqrSetIdentityCol(gm.A, ub.work, m, lda, c);
    }
    VecBarrier();
}

/// 一块的 vec 侧中段：等 cube 出 G -> 归约 partial -> T 递推 -> 本块列置成单位阵列。
///
/// 把本块列置成单位阵列**必须**在 V 已经物化之后（前面有 VecBarrier 保证），否则会把 V 冲掉。
__aicore__ inline void OrgqrBlockVecMid(const OrgqrGm& gm, OrgqrUb& ub, uint32_t m, uint32_t lda, uint32_t j,
                                        uint32_t jb, uint32_t gChunks, uint32_t coreId, uint32_t coreNum)
{
    VecWaitCube();
    // G 的 partial 归约。vec 侧本来就要等 cube 之后才读 G，所以这一步不额外加同步点。
    if (gChunks > 1) {
        ReducePartials(gm.G, gm.P, static_cast<uint64_t>(jb) * jb, gChunks, ub.work, ub.red, RED_CAP, coreId,
                       coreNum);
        VecBarrier();
    }
    BuildMinusTBlock(gm.T, gm.G, gm.tau, ub.qG, ub.qTau, ub.tl, j, jb, coreId, coreNum);
    for (uint32_t c = j + coreId; c < j + jb; c += coreNum) {
        OrgqrSetIdentityCol(gm.A, ub.work, m, lda, c);
    }
    VecBarrier();
}
#endif  // __DAV_VEC__

extern "C" __global__ __aicore__ void sorgqr_mix_kernel(OrgqrParams params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const uint32_t m = params.m, n = params.n, k = params.k;
    const uint32_t lda = params.lda, ldy = params.ldy, nb = params.nb;
    const uint32_t nBlocks = CeilDivU(k, nb);

#ifdef __DAV_VEC__
    AscendC::TPipe pipe;
    OrgqrUb ub;
    ub.Init(pipe);

    OrgqrGm gm;
    gm.Init(params);

    // MIX 的 AIV 侧：GetBlockIdx() 已是 AIV 核号（不乘 2），GetBlockNum() 是 AIC 数（要乘 2）
    const uint32_t coreId = AscendC::GetBlockIdx();
    const uint32_t coreNum = AscendC::GetBlockNum() * 2;

    OrgqrPrologueVec(params, gm, ub, m, n, k, lda, coreId, coreNum);
#endif

    // 块序从后往前：这样每块的 V 在被本块覆盖之前就已经读进 wsY，可以原地做
    for (uint32_t bIdx = 0; bIdx < nBlocks; ++bIdx) {
        const OrgqrBlockShape bs = OrgqrBlockAt(nBlocks - 1u - bIdx, m, n, k, nb);
        const uint32_t j = bs.j, jb = bs.jb, rows = bs.rows, cols = bs.cols;

#ifdef __DAV_VEC__
        // ---- 1. 物化 V 到 wsY（对角 1、上方 0）----
        MaterializeVBlock(gm.A, gm.Y, ub.qIn, ub.work, j, jb, rows, lda, ldy, coreId, coreNum);
        VecBarrier();
        VecNotifyCube();
#elif defined(__DAV_CUBE__)
        CubeWaitVec();
#endif

        // ---- 2. G = V^T V（cube）----
        // gp[0] 的行主序视图是 (m=jb, n=jb, k=rows)：M/N 方向只有 1 个 tile ⇒ 不分 K
        // 就只有 1 个核在串着读 2*jb*rows 个元素。rows=4096 时实测约 20 us/块。
        uint32_t gChunkLen = 0;
        const uint32_t gChunks = KSplitChunks(jb, jb, rows, AscendC::GetBlockNum(), params.capP, &gChunkLen);
#ifdef __DAV_CUBE__
        OrgqrGemmG(params, rows, jb, ldy, gChunks, gChunkLen);
#endif

#ifdef __DAV_VEC__
        // ---- 3/4. G 的归约 -> T 递推 -> 本块列置成单位阵列 ----
        OrgqrBlockVecMid(gm, ub, m, lda, j, jb, gChunks, coreId, coreNum);
        VecNotifyCube();
#elif defined(__DAV_CUBE__)
        CubeWaitVec();
#endif

        // ---- 5. LARFB(Left, NoTrans) 作用到 A(j:m, j:n)：本块自己 + 右侧全部 ----
        QrCore::GemmPlan lp[3];
        uint32_t wChunkLen = 0;
        const uint32_t wChunks = OrgqrPlanLarfb(rows, cols, jb, ldy, lda, j, params.capP, lp, &wChunkLen);
#ifdef __DAV_CUBE__
        OrgqrLarfbCube(params, lp, wChunks, wChunkLen);
#endif
#ifdef __DAV_VEC__
        OrgqrLarfbVec(gm.W, gm.P, lp, ub.work, ub.red, wChunks, coreId, coreNum);
#endif
    }
}
