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
 * \file xlarft_mix.hpp
 * \brief 由 (V, tau) 直接算块反射子的三角因子 T（融合 MIX kernel，950PR / fp32）
 *
 *   AIV : 物化 V 到 workspace（FORWARD 下对角 1、上方 0；BACKWARD 下第 n-k+i 行 1、下方 0）
 *   cube: G = V^T V（按 K 分核，份数取 makespan 最优）
 *   AIV : T 的三角递推（**与 ormqr 引擎共用** larfb 的那一份按行分核实现，设计文档 D6）
 *   AIV : 把本核那几行按列主序 ldt 写进用户的 T，**只写被引用的那半个三角**（对齐 LAPACK slarft）
 *
 * ★ k=256 时整块 G 要 256 KB、放不进单核 UB ⇒ 递推走 StreamG：每个 i 只取 G 的第 i 行
 *   （用 G = V^T V 的对称性把"取第 i 列"换成连续读；列在行主序里是跨步的，
 *   DataCopyPad 的 4 字节块会退化成逐元素搬）。
 *
 */

#pragma once

// 五个 mix kernel 的公共件（CATLASS include、类型别名、跨核同步、分 K 调度、
// householder/larfb_plan、以及标准库那几个头）都由这一个头提供。
#include "../../sgeqrf/kernel/mix_common.hpp"

constexpr uint32_t MAX_N = 4096;     // 单列整列驻留 UB
constexpr uint32_t MAX_K = 256;      // 任务书 §2.6：k <= min(n, 256)
constexpr uint32_t MAX_T_ROWS = 32;  // 本核最多拥有的 T 行数：k=256 时要求 AIV 核数 >= 8

struct XlarftParams
{
    GM_ADDR V;    // n x k 列主序 ldv：第 i 列的隐含 1 之外是反射子
    GM_ADDR tau;  // k 个
    GM_ADDR T;    // k x k 列主序 ldt（只写被引用的那半个三角）
    GM_ADDR wsY;  // 物化后的 V：ldy x k
    GM_ADDR wsG;  // G = V^T V：k x k 行主序 ld=k
    GM_ADDR wsP;  // 分 K 的 partial 区
    uint32_t n;
    uint32_t k;
    uint32_t ldv;
    uint32_t ldt;
    uint32_t ldy;
    uint32_t backward;  // 0 = FORWARD, 1 = BACKWARD
    uint64_t capP;      // wsP 容量（元素）。**host 是唯一来源**，kernel 只拿它 clamp 份数
};

/// kernel 要用到的 GM 视图，一次绑好
struct XlarftGm
{
    AscendC::GlobalTensor<float> V, tau, Tg, Y, G, P;

    __aicore__ inline void Init(const XlarftParams& params)
    {
        V.SetGlobalBuffer((__gm__ float*)params.V);
        tau.SetGlobalBuffer((__gm__ float*)params.tau);
        Tg.SetGlobalBuffer((__gm__ float*)params.T);
        Y.SetGlobalBuffer((__gm__ float*)params.wsY);
        G.SetGlobalBuffer((__gm__ float*)params.wsG);
        P.SetGlobalBuffer((__gm__ float*)params.wsP);
    }
};

#ifdef __DAV_VEC__
/// AIV 侧的 UB 分块。TBuf 必须活到 kernel 结束，所以连 TBuf 一起放在这个结构体里，
/// 由 kernel 在自己的作用域里声明一个实例。
struct XlarftUb
{
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCol, bufG, bufTau, bufT, bufSp, bufRed;
    AscendC::LocalTensor<float> col, gl, taul, tl, sp, red;

    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(bufCol, MAX_N * sizeof(float));
        pipe.InitBuffer(bufG, MAX_K * sizeof(float));
        pipe.InitBuffer(bufTau, MAX_K * sizeof(float));
        pipe.InitBuffer(bufT, MAX_T_ROWS * MAX_K * sizeof(float));
        pipe.InitBuffer(bufSp, MAX_K * 8 * sizeof(float));
        pipe.InitBuffer(bufRed, RED_CAP * sizeof(float));
        col = bufCol.Get<float>();
        gl = bufG.Get<float>();
        taul = bufTau.Get<float>();
        tl = bufT.Get<float>();
        sp = bufSp.Get<float>();
        red = bufRed.Get<float>();
    }
};

/// 物化 V 到 wsY：按 direct 把该置 0 的那一半清掉、隐含的 1 写进去。
///
/// 先把整列原样读进 UB（偏移 0，对齐没问题），再把该置 0/1 的位置改掉。
/// ★ 不要"把 GM 的有效段直接搬到 UB 的第 d+1 个位置"：UB 偏移非 8 的倍数时
///   DataCopyPad **静默什么都不搬**。
__aicore__ inline void XlarftMaterializeV(const AscendC::GlobalTensor<float>& V, const AscendC::GlobalTensor<float>& Y,
                                          const AscendC::LocalTensor<float>& col, uint32_t n, uint32_t k, uint32_t ldv,
                                          uint32_t ldy, bool backward, uint32_t coreId, uint32_t coreNum)
{
    const uint32_t nAligned = AlignUp32(n, 8);
    for (uint32_t c = coreId; c < k; c += coreNum)
    {
        const uint32_t d = backward ? (n - k + c) : c;  // 隐含 1 的那一行
        {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
            AscendC::DataCopyPad(col, V[static_cast<uint64_t>(c) * ldv], cp, pd);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (!backward)
        {
            if (d > 0)
            {
                AscendC::Duplicate(col, 0.0f, d);  // 前缀置 0：基址 0，天然对齐
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        else
        {
            // 后缀 (d, n) 置 0：对齐的那段用向量，前面零散的几个用标量
            const uint32_t tailAligned = AlignUp32(d + 1u, 8);
            if (tailAligned < nAligned)
            {
                AscendC::Duplicate(col[tailAligned], 0.0f, nAligned - tailAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t r = d + 1u; r < tailAligned && r < n; ++r)
            {
                col.SetValue(r, 0.0f);
            }
        }
        col.SetValue(d, 1.0f);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(Y[static_cast<uint64_t>(c) * ldy], col, cp);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

/// tau 整段读进 UB（后面 T 的递推逐个取标量，所以要同步到 S 流水）
__aicore__ inline void XlarftLoadTau(const AscendC::GlobalTensor<float>& tau, const AscendC::LocalTensor<float>& taul,
                                     uint32_t k)
{
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(k * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
    AscendC::DataCopyPad(taul, tau, cp, pd);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
}

/// 写回 T：T 是 k x k 列主序 ldt，本核的"行 r"在列主序里是跨步的。
/// 只写**被引用的那半个三角**（FORWARD 上三角、BACKWARD 下三角），对齐 LAPACK slarft
/// （它不碰另一半）。DataCopyPad 的 UB 侧每块独占 32B ⇒ 先把该行摊成 8 个 float 的步长。
__aicore__ inline void XlarftStoreT(const AscendC::GlobalTensor<float>& Tg, const AscendC::LocalTensor<float>& tl,
                                    const AscendC::LocalTensor<float>& sp, uint32_t k, uint32_t ldt, uint32_t ldtUb,
                                    bool backward, uint32_t coreId, uint32_t coreNum)
{
    uint32_t q = 0;
    for (uint32_t r = coreId; r < k; r += coreNum, ++q)
    {
        const uint32_t lo = backward ? 0u : r;
        const uint32_t hi = backward ? (r + 1u) : k;
        const uint32_t cnt = hi - lo;
        for (uint32_t t = 0; t < cnt; ++t)
        {
            sp.SetValue(t * 8u, tl.GetValue(q * ldtUb + lo + t));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams cpT{static_cast<uint16_t>(cnt), static_cast<uint32_t>(sizeof(float)), 0,
                                       (ldt - 1u) * static_cast<uint32_t>(sizeof(float)), 0};
        AscendC::DataCopyPad(Tg[static_cast<uint64_t>(lo) * ldt + r], sp, cpT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
}
#endif  // __DAV_VEC__

#ifdef __DAV_CUBE__
/// G = V^T V（cube 侧）。窄 GEMM（输出 tile 数 < AIC 核数）时走确定性分 K：
/// 每份写独立 partial，之后由 AIV 定序归约 —— 不用原子加，保证逐位可复现。
__aicore__ inline void XlarftGemmG(const XlarftParams& params, uint32_t n, uint32_t k, uint32_t ldy, uint32_t gChunks,
                                   uint32_t gChunkLen)
{
    QrCore::GemmPlan gp[3];
    QrCore::PlanLarfbLeft(n, k, k, ldy, ldy, k, k, k, 0, 0, gp);
    if (gChunks > 1)
    {
        RunPlanOnCubeSplitK<MmadRC, layout::RowMajor, layout::ColumnMajor>(gp[0], params.wsY, params.wsY, params.wsP,
                                                                           gChunks, gChunkLen);
    }
    else
    {
        RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(gp[0], params.wsY, params.wsY, params.wsG);
    }
    CubeBarrier();
    CubeNotifyVec();
}
#endif  // __DAV_CUBE__

extern "C" __global__ __aicore__ void xlarft_mix_kernel(XlarftParams params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const uint32_t n = params.n, k = params.k, ldv = params.ldv, ldt = params.ldt, ldy = params.ldy;
    const bool backward = (params.backward != 0u);

#ifdef __DAV_VEC__
    AscendC::TPipe pipe;
    XlarftUb ub;
    ub.Init(pipe);
    const AscendC::LocalTensor<float>& col = ub.col;
    const AscendC::LocalTensor<float>& gl = ub.gl;
    const AscendC::LocalTensor<float>& taul = ub.taul;
    const AscendC::LocalTensor<float>& tl = ub.tl;
    const AscendC::LocalTensor<float>& red = ub.red;

    XlarftGm gm;
    gm.Init(params);
    const AscendC::GlobalTensor<float>& G = gm.G;
    const AscendC::GlobalTensor<float>& P = gm.P;

    // MIX 的 AIV 侧：GetBlockIdx() 已是 AIV 核号（不乘 2），GetBlockNum() 是 AIC 数（要乘 2）
    const uint32_t coreId = AscendC::GetBlockIdx();
    const uint32_t coreNum = AscendC::GetBlockNum() * 2;

    // ---- 1. 物化 V 到 wsY ----
    XlarftMaterializeV(gm.V, gm.Y, col, n, k, ldv, ldy, backward, coreId, coreNum);
    VecBarrier();
    VecNotifyCube();
#elif defined(__DAV_CUBE__)
    CubeWaitVec();
#endif

    // ---- 2. G = V^T V（cube）----
    uint32_t gChunkLen = 0;
    const uint32_t gChunks = KSplitChunks(k, k, n, AscendC::GetBlockNum(), params.capP, &gChunkLen);
#ifdef __DAV_CUBE__
    XlarftGemmG(params, n, k, ldy, gChunks, gChunkLen);
#endif

#ifdef __DAV_VEC__
    VecWaitCube();
    if (gChunks > 1)
    {
        ReducePartials(G, P, static_cast<uint64_t>(k) * k, gChunks, col, red, RED_CAP, coreId, coreNum);
        VecBarrier();
    }

    // ---- 3. tau 进 UB ----
    XlarftLoadTau(gm.tau, taul, k);

    // ---- 4. T 的三角递推（与引擎共用 qr_core/t_rowparallel.h）----
    // StreamG=true：k=256 时整块 G 要 256 KB 放不进 UB，所以每个 i 只取 G 的第 i 行
    // （G = V^T V 对称 ⇒ 第 i 行等于第 i 列，而列在行主序里是跨步的，DataCopyPad
    //   的 4 字节块会退化成逐元素搬）。
    const uint32_t ldtUb = AlignUp32(k, 8);
    const uint32_t myRows = backward
                                ? QrCore::BuildTRowParallel<true, true>(tl, ldtUb, gl, k, G, taul, k, coreId, coreNum)
                                : QrCore::BuildTRowParallel<false, true>(tl, ldtUb, gl, k, G, taul, k, coreId, coreNum);
    (void)myRows;

    // ---- 5. 写回 T ----
    XlarftStoreT(gm.Tg, tl, ub.sp, k, ldt, ldtUb, backward, coreId, coreNum);
    VecBarrier();
#endif
}
