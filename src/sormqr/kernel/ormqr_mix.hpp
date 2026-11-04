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
 * \file ormqr_mix.hpp
 * \brief 把 geqrf 的 (v, tau) 表示的 op(Q) 左乘/右乘到 C（融合 MIX kernel，950PR / fp32）
 *
 * 块反射子引擎：每块 nb 列 ⇒ 物化 V 到 wsY → `G = V^T V`(cube，按 K 分核)
 * → T 三角递推(AIV，**按行分核、核间零通信**) → LARFB 三次 GEMM。
 * 四种 (side, trans) 组合由 params.side / params.trans 选，块序按 LarfbReverseOrder 决定。
 *
 * ⚠ T 按行分核后是**行主序**存的，而 PlanLarfb* 的视图标志按"T 为列主序"定义 ⇒
 * 传给 plan 的 trans 必须取反（kernel 里是 planTrans）。
 *
 * `W = V^T C` 在行主序视图下 M 或 N 方向恒只有 1 个 tile，不分 K 就只有 1~3 个核在读
 * 整个 C ⇒ 按 K 切块写私有 partial、再由 AIV **定序**归约（不能用 SetAtomicAdd，
 * 原子加顺序不定会破坏逐位可复现）。
 *
 */

#pragma once

// 五个 mix kernel 的公共件（CATLASS include、类型别名、跨核同步、分 K 调度、
// householder/larfb_plan、以及标准库那几个头）都由这一个头提供。
#include "../../sgeqrf/kernel/mix_common.hpp"

using MmadCR = MmadOf<layout::ColumnMajor, layout::RowMajor>;

constexpr uint32_t MAX_NB = 128;

struct EngineParams
{
    GM_ADDR V;    // 反射子：LEFT 时 m x k，RIGHT 时 n x k（列主序，ldv）
    GM_ADDR tau;  // k 个
    GM_ADDR C;    // m x n（列主序，ldc），原地更新
    GM_ADDR wsY;  // 物化的 V 块：ldy x nb
    GM_ADDR wsT;  // nb x nb，存 -T；T 按行分核回写为行主序（ldt = nb，见文件头说明），
                  // PlanLarfb* 的视图标志按"T 为列主序"定义，故传入 plan 的 trans 需取反
    GM_ADDR wsG;  // nb x nb（行主序）
    GM_ADDR wsW;  // nb x max(m,n)
    GM_ADDR wsW2;
    GM_ADDR wsP;      // 分 K 的 partial 区：nChunks 份，每份 (plan.m x plan.n) 连续存放
    GM_ADDR devInfo;  // device 侧 int32：成功时由 kernel 自己写 0（见下面那段注释），可为空
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t ldv;
    uint32_t ldc;
    uint32_t nb;
    uint32_t ldy;
    uint32_t side;      // 0 = LEFT, 1 = RIGHT
    uint32_t trans;     // 0 = NoTrans, 1 = Trans
    uint64_t capP = 0;  // wsP 容量（元素）。**host 是唯一来源**，kernel 只拿它 clamp 份数
};

/// kernel 要用到的 GM 视图，一次绑好。T 与 T2 是同一块：按行写入时用 T2 这个名字。
struct OrmqrGm
{
    AscendC::GlobalTensor<float> V, tau, Y, T, T2, G, W, P;

    __aicore__ inline void Init(const EngineParams& params)
    {
        V.SetGlobalBuffer((__gm__ float*)params.V);
        tau.SetGlobalBuffer((__gm__ float*)params.tau);
        Y.SetGlobalBuffer((__gm__ float*)params.wsY);
        T.SetGlobalBuffer((__gm__ float*)params.wsT);
        T2.SetGlobalBuffer((__gm__ float*)params.wsT);
        G.SetGlobalBuffer((__gm__ float*)params.wsG);
        W.SetGlobalBuffer((__gm__ float*)params.wsW);
        P.SetGlobalBuffer((__gm__ float*)params.wsP);
    }
};

/// 从 params 解出来的标量（含块序方向与块数），kernel 一开头算一次
struct OrmqrShape
{
    uint32_t m, n, k, ldv, ldc, nb, ldy, vRows, nBlocks;
    bool isLeft, reverse;
    QrCore::Trans trans;
};

__aicore__ inline OrmqrShape OrmqrShapeOf(const EngineParams& params)
{
    OrmqrShape sh{};
    sh.m = params.m;
    sh.n = params.n;
    sh.k = params.k;
    sh.ldv = params.ldv;
    sh.ldc = params.ldc;
    sh.nb = params.nb;
    sh.ldy = params.ldy;
    sh.isLeft = (params.side == 0u);
    sh.trans = (params.trans == 0u) ? QrCore::Trans::NoTrans : QrCore::Trans::Trans;
    const QrCore::Side side = sh.isLeft ? QrCore::Side::Left : QrCore::Side::Right;
    sh.reverse = QrCore::LarfbReverseOrder(side, sh.trans);
    sh.vRows = sh.isLeft ? sh.m : sh.n;  // V 的行数
    sh.nBlocks = CeilDivU(sh.k, sh.nb);
    return sh;
}

/// 第 b 块的形状：起始列 j、块宽 jb、本块 V 的高度 rows
struct OrmqrBlockShape
{
    uint32_t j, jb, rows;
};

__aicore__ inline OrmqrBlockShape OrmqrBlockAt(uint32_t b, uint32_t k, uint32_t nb, uint32_t vRows)
{
    const uint32_t j = b * nb;
    const uint32_t jb = ((k - j) < nb) ? (k - j) : nb;
    return OrmqrBlockShape{j, jb, vRows - j};
}

/// 排 LARFB 的三段 plan，并决定第一段要不要分 K。返回份数（1 = 不分）。
///
/// ★ T 现在是**行主序**存的（按行分核的必然结果），而 PlanLarfb* 的视图标志
///   是按"T 缓冲为列主序"定义的 ⇒ 两者的 trans 含义正好相反，这里必须取反。
///   （probe 100 的 geqrf 已经经验证实：行主序 T + 要 Q^T 时传的正是 NoTrans。）
///
/// lp[0] 的输出 tile 数 = ceil(lp[0].m/256)*ceil(lp[0].n/256)：LEFT 下
/// (m=n列数, n=jb) ⇒ jb<=256 时 N 方向恒为 1；RIGHT 下 (m=jb, n=M) ⇒ M 方向恒为 1。
/// 两种情形都可能只有 1~3 个核在读整个 C，所以按 K 切。
__aicore__ inline uint32_t OrmqrPlanApply(bool isLeft, QrCore::Trans trans, uint32_t m, uint32_t n, uint32_t rows,
                                          uint32_t jb, uint32_t ldy, uint32_t ldc, uint32_t j, uint64_t capP,
                                          QrCore::GemmPlan* lp, uint32_t* wChunkLen)
{
    const QrCore::Trans planTrans = (trans == QrCore::Trans::NoTrans) ? QrCore::Trans::Trans : QrCore::Trans::NoTrans;
    if (isLeft)
    {
        // C 块 = C[j:m, 0:n]，列主序偏移 = 行 j
        QrCore::PlanLarfbLeft(rows, n, jb, ldy, ldc, jb, jb, jb, 0, static_cast<int64_t>(j), lp, planTrans);
    }
    else
    {
        // C 块 = C[0:m, j:n]，列主序偏移 = 列 j
        QrCore::PlanLarfbRight(m, rows, jb, ldy, ldc, m, m, jb, 0, static_cast<int64_t>(j) * static_cast<int64_t>(ldc),
                               lp, planTrans);
    }
    return KSplitChunks(lp[0].m, lp[0].n, lp[0].k, AscendC::GetBlockNum(), capP, wChunkLen);
}

#ifdef __DAV_CUBE__
/// G = V^T V（cube 侧）。计算与跨核发布都在共享的 GemmGBlock 里，ormqr 用的是同一份。
__aicore__ inline void OrmqrGemmG(const EngineParams& params, uint32_t rows, uint32_t jb, uint32_t ldy,
                                  uint32_t gChunks, uint32_t gChunkLen)
{
    GemmGBlock(params.wsY, params.wsG, params.wsP, rows, jb, ldy, gChunks, gChunkLen);
}

/// LEFT：C <- op(Q) C，三次 GEMM。分 K 时中间把 partial 交给 vec 归约再回来做 lp[1]。
__aicore__ inline void OrmqrApplyCubeLeft(const EngineParams& params, const QrCore::GemmPlan* lp, uint32_t wChunks,
                                          uint32_t wChunkLen)
{
    if (wChunks > 1)
    {
        RunPlanOnCubeSplitK<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[0], params.C, params.wsY, params.wsP,
                                                                           wChunks, wChunkLen);
        CubeBarrier();
        CubeNotifyVec();  // partial 齐了，交给 vec 归约
        CubeWaitVec();    // 等归约完再做 lp[1]
    }
    else
    {
        RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[0], params.C, params.wsY, params.wsW);
    }
    CubeBarrier();
    if (lp[1].bColMajorView)
    {
        RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[1], params.wsW, params.wsT, params.wsW2);
    }
    else
    {
        RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[1], params.wsW, params.wsT, params.wsW2);
    }
    CubeBarrier();
    RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[2], params.wsW2, params.wsY, params.C);
    CubeBarrier();
}

/// RIGHT：C <- C op(Q)，三次 GEMM
__aicore__ inline void OrmqrApplyCubeRight(const EngineParams& params, const QrCore::GemmPlan* lp, uint32_t wChunks,
                                           uint32_t wChunkLen)
{
    if (wChunks > 1)
    {
        RunPlanOnCubeSplitK<MmadRR, layout::RowMajor, layout::RowMajor>(lp[0], params.wsY, params.C, params.wsP,
                                                                        wChunks, wChunkLen);
        CubeBarrier();
        CubeNotifyVec();
        CubeWaitVec();
    }
    else
    {
        RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[0], params.wsY, params.C, params.wsW);
    }
    CubeBarrier();
    if (lp[1].aColMajorView)
    {
        RunPlanOnCube<MmadCR, layout::ColumnMajor, layout::RowMajor>(lp[1], params.wsT, params.wsW, params.wsW2);
    }
    else
    {
        RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[1], params.wsT, params.wsW, params.wsW2);
    }
    CubeBarrier();
    RunPlanOnCube<MmadCR, layout::ColumnMajor, layout::RowMajor>(lp[2], params.wsY, params.wsW2, params.C);
    CubeBarrier();
}

/// 按 side 分派到 LEFT / RIGHT 两条路径
__aicore__ inline void OrmqrApplyCube(const EngineParams& params, bool isLeft, const QrCore::GemmPlan* lp,
                                      uint32_t wChunks, uint32_t wChunkLen)
{
    if (isLeft)
    {
        OrmqrApplyCubeLeft(params, lp, wChunks, wChunkLen);
    }
    else
    {
        OrmqrApplyCubeRight(params, lp, wChunks, wChunkLen);
    }
}
#endif  // __DAV_CUBE__

#ifdef __DAV_VEC__
/// AIV 侧的 UB 分块。TQue / TBuf 必须活到 kernel 结束，所以一起放在这个结构体里。
struct OrmqrUb
{
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qIn, qG, qTau;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufW, bufT, bufA, bufRed;
    AscendC::LocalTensor<float> work, tl, acc, red;

    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(qIn, 1, MAX_ROWS * sizeof(float));
        pipe.InitBuffer(qG, 1, MAX_NB * MAX_NB * sizeof(float));
        pipe.InitBuffer(qTau, 1, MAX_NB * sizeof(float));
        pipe.InitBuffer(bufW, MAX_ROWS * sizeof(float));
        pipe.InitBuffer(bufT, MAX_NB * MAX_NB * sizeof(float));
        pipe.InitBuffer(bufA, MAX_NB * sizeof(float));
        pipe.InitBuffer(bufRed, RED_CAP * sizeof(float));
        work = bufW.Get<float>();
        tl = bufT.Get<float>();
        acc = bufA.Get<float>();
        red = bufRed.Get<float>();
    }
};

/// 一块的 vec 侧中段：G 的 partial 归约 -> T 递推。
/// vec 侧本来就要等 cube 之后才读 G，所以归约这一步不额外加同步点。
__aicore__ inline void OrmqrBlockVecMid(const OrmqrGm& gm, OrmqrUb& ub, uint32_t j, uint32_t jb, uint32_t gChunks,
                                        uint32_t coreId, uint32_t coreNum)
{
    VecWaitCube();
    if (gChunks > 1)
    {
        ReducePartials(gm.G, gm.P, static_cast<uint64_t>(jb) * jb, gChunks, ub.work, ub.red, RED_CAP, coreId, coreNum);
        VecBarrier();
    }
    BuildMinusTBlock(gm.T2, gm.G, gm.tau, ub.qG, ub.qTau, ub.tl, j, jb, coreId, coreNum);
    VecBarrier();
    VecNotifyCube();
}

/// 施加块反射子期间 vec 侧要做的事：分 K 时归约 lp[0] 的 partial
__aicore__ inline void OrmqrApplyVec(const AscendC::GlobalTensor<float>& W, const AscendC::GlobalTensor<float>& P,
                                     const QrCore::GemmPlan* lp, const AscendC::LocalTensor<float>& work,
                                     const AscendC::LocalTensor<float>& red, uint32_t wChunks, uint32_t coreId,
                                     uint32_t coreNum)
{
    LarfbReduceVec(W, P, lp, work, red, wChunks, coreId, coreNum);
}
/// AIV 侧的一整套家当。TPipe 必须活过整个 kernel，所以只能打成结构体放在入口的栈上，
/// 不能抽成函数（函数返回时 TPipe 析构，UB 分配全部失效）。
struct OrmqrVecCtx
{
    AscendC::TPipe pipe;
    OrmqrUb ub;
    OrmqrGm gm;
    uint32_t coreId = 0;
    uint32_t coreNum = 0;

    __aicore__ inline void Init(const EngineParams& params)
    {
        ub.Init(pipe);
        gm.Init(params);
        // MIX 的 AIV 侧：GetBlockIdx() 已是 AIV 核号（不乘 2），GetBlockNum() 是 AIC 数（要乘 2）
        coreId = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum() * 2;
        ZeroDevInfo(params.devInfo, ub.work, coreId);
    }
};

/// 物化 V 块，并把 wsY 发布给 cube。
/// 发布这一步原先藏在 OrmqrMaterializeV 的结尾；统一到共享的 MaterializeVBlock 时
/// 漏掉过一次（cube 侧 CubeWaitVec 等不到 flag，内核不返回），所以这里显式留一处。
__aicore__ inline void OrmqrMaterializeAndPublish(OrmqrVecCtx& vc, uint32_t j, uint32_t jb, uint32_t rows, uint32_t ldv,
                                                  uint32_t ldy)
{
    MaterializeVBlock(vc.gm.V, vc.gm.Y, vc.ub.qIn, vc.ub.work, j, jb, rows, ldv, ldy, vc.coreId, vc.coreNum);
    VecBarrier();
    VecNotifyCube();
}
#endif  // __DAV_VEC__

extern "C" __global__ __aicore__ void sormqr_mix_kernel(EngineParams params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const OrmqrShape sh = OrmqrShapeOf(params);
    const uint32_t m = sh.m, n = sh.n, k = sh.k, ldv = sh.ldv, ldc = sh.ldc, nb = sh.nb, ldy = sh.ldy;

#ifdef __DAV_VEC__
    OrmqrVecCtx vc;
    vc.Init(params);
#endif

    for (uint32_t bIdx = 0; bIdx < sh.nBlocks; ++bIdx)
    {
        const OrmqrBlockShape bs = OrmqrBlockAt(sh.reverse ? (sh.nBlocks - 1u - bIdx) : bIdx, k, nb, sh.vRows);
        const uint32_t j = bs.j, jb = bs.jb, rows = bs.rows;

#ifdef __DAV_VEC__
        // ---- 1. 物化 V 块到 wsY（对角 1、上方 0）----
        OrmqrMaterializeAndPublish(vc, j, jb, rows, ldv, ldy);
#elif defined(__DAV_CUBE__)
        CubeWaitVec();
#endif

        // ---- 2. G = V^T V（cube）----
        // 行主序视图是 (m=jb, n=jb, k=rows)：M/N 方向只有 1 个输出 tile ⇒ 不分 K 就只有
        // 1 个核把 2*jb*rows 个元素串着读完（rows=4096 时实测约 20 us/块）。
        uint32_t gChunkLen = 0;
        const uint32_t gChunks = KSplitChunks(jb, jb, rows, AscendC::GetBlockNum(), params.capP, &gChunkLen);
#ifdef __DAV_CUBE__
        OrmqrGemmG(params, rows, jb, ldy, gChunks, gChunkLen);
#endif

#ifdef __DAV_VEC__
        // ---- 3. G 的归约 + T 的三角递推（按行分核，核间零通信）----
        OrmqrBlockVecMid(vc.gm, vc.ub, j, jb, gChunks, vc.coreId, vc.coreNum);
#elif defined(__DAV_CUBE__)
        CubeWaitVec();
#endif

        // ---- 4. 施加块反射子（cube，三次 GEMM）----
        QrCore::GemmPlan lp[3];
        uint32_t wChunkLen = 0;
        const uint32_t wChunks =
            OrmqrPlanApply(sh.isLeft, sh.trans, m, n, rows, jb, ldy, ldc, j, params.capP, lp, &wChunkLen);
#ifdef __DAV_CUBE__
        OrmqrApplyCube(params, sh.isLeft, lp, wChunks, wChunkLen);
#endif
#ifdef __DAV_VEC__
        OrmqrApplyVec(vc.gm.W, vc.gm.P, lp, vc.ub.work, vc.ub.red, wChunks, vc.coreId, vc.coreNum);
#endif
#ifdef __DAV_CUBE__
        CubeNotifyVec();
#elif defined(__DAV_VEC__)
        VecWaitCube();
#endif
    }
}
