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
 * \file geqrf_mix.hpp
 * \brief 分块 Householder QR 的融合 MIX kernel（Ascend950PR / fp32）
 *
 * 一次调用一个融合 MIX kernel，面板结构照参考仓 paco：
 *   for j = 0; j < min(m,n); j += panelWidth:          // panelWidth = alignBD * columns = 64
 *     AIV : 每核独占 columns(=2) 条相邻列、**全程驻留 UB**；核间三角依赖
 *           （for k<blockIdx 吸收前序核的反射子；for k>blockIdx 服务后继核），
 *           **T 在服务环节顺带算出**（每核只算自己那几行），未归一化的 v 写进 Y
 *     cube: LARFB 以三次 GEMM 更新尾随矩阵 A[j:m, j+jb:n]
 *
 * 约定（刻意保留）：Y 存**未归一化**的 v、配 tau'；T 存 **-T**、行主序 ld=jb；
 * A/tau 最后才转成 LAPACK 约定。于是尾随更新是 PlanLarfbLeft 的 **NoTrans** 形式
 * （行主序 T 让视图标志的含义相对"列主序 T"取反）。
 *
 * 只用 alignBD=32 个 AIV 核做面板，其余核跑 syn_all 把 set/wait 配对数补齐：
 * 最忙核的工作量 ∝ 面板宽度、与怎么切核无关，多给核只会增加会合次数。
 *
 */

#pragma once

// 五个 mix kernel 的公共件（CATLASS include、类型别名、跨核同步、分 K 调度、
// householder/larfb_plan、以及标准库那几个头）都由这一个头提供。
#include "./mix_common.hpp"


constexpr int CHUNK = 8192;  // 单核 UB 基本块（float 个数），同参考仓

struct Geqrf2Params {
    GM_ADDR A;
    GM_ADDR tau;
    GM_ADDR wsY;   // 未归一化 v：ldy x panelWidth（列主序）
    GM_ADDR wsT;   // -T：panelWidth x panelWidth（行主序，ld = jb）
    GM_ADDR wsW;   // 面板内 T 行交换（前 4096 float）+ 尾随更新的 W（之后）
    GM_ADDR wsW2;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldy;
    uint32_t columns;
    uint32_t alignBD;
    GM_ADDR devInfo;  // device 侧 int32：成功时由 kernel 自己写 0，可为空
};

namespace {
__aicore__ inline int AlignTo(int a, int b)
{
    return (b == 0) ? a : (a + b - 1) / b * b;
}

/// 非工作核把配对数补齐（工作核总配对数 = blockDim + 2）
__aicore__ inline void SynAll(int blockDim)
{
    VecBarrier();
    VecBarrier();
    for (int k = 1; k != blockDim; ++k) { VecBarrier(); }
    VecBarrier();
}

}  // namespace

/// kernel 要用到的 GM 视图，一次绑好
struct GeqrfGm {
    AscendC::GlobalTensor<float> A, tau, Y, T, W;

    __aicore__ inline void Init(const Geqrf2Params& params)
    {
        A.SetGlobalBuffer((__gm__ float*)params.A);
        tau.SetGlobalBuffer((__gm__ float*)params.tau);
        Y.SetGlobalBuffer((__gm__ float*)params.wsY);
        T.SetGlobalBuffer((__gm__ float*)params.wsT);
        W.SetGlobalBuffer((__gm__ float*)params.wsW);
    }
};

#ifdef __DAV_VEC__
/// AIV 侧的 UB 分块。TQue / TBuf 必须活到 kernel 结束，所以一起放在这个结构体里。
struct GeqrfUb {
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> aQueueIn, bQueueIn, aQueueOut, tQueueDst;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf, scratchBuf;
    AscendC::LocalTensor<float> workLocal, scratch;

    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(aQueueIn, 1, 2 * CHUNK * sizeof(float));
        pipe.InitBuffer(bQueueIn, 1, CHUNK * sizeof(float));
        pipe.InitBuffer(aQueueOut, 1, CHUNK * sizeof(float));
        pipe.InitBuffer(tQueueDst, 1, 1024 * sizeof(float));
        pipe.InitBuffer(workBuf, (CHUNK + 1024) * sizeof(float));
        pipe.InitBuffer(scratchBuf, 32 * sizeof(float));
        workLocal = workBuf.Get<float>();
        scratch = scratchBuf.Get<float>();
    }
};

/// 一个面板里**一个核**的全部上下文：UB 张量、本核负责的列区间、以及 Householder
/// 的标量缓存。八个相位要在它们之间来回传，收成一个盒子比挂十几个形参清楚。
///
/// alpha / v0 / tauPrime / diagSave 都按 columns <= 8 开，与 host 的分列上限一致。
struct PanelCtx {
    AscendC::LocalTensor<float> a, hv, T_local, workLocal, scratch;
    AscendC::DataCopyExtParams cpCol{}, cp1{}, cp8{};
    AscendC::DataCopyPadExtParams<float> pd{};
    int j = 0, jb = 0, bd = 0, columns = 0, blockIdx = 0;
    int myBase = 0, myCols = 0, curM = 0, offset = 0, lda = 0, ldy = 0, ldtUb = 0;
    uint64_t copyMask = 64;
    uint8_t repeatTimes = 0;
    uint64_t maskUpdate[2] = {0, 0};
    float alpha[8] = {0.0f}, v0[8] = {0.0f}, tauPrime[8] = {0.0f}, diagSave[8] = {0.0f};
};

/// 本核在这个面板上的起点：算出列区间，申请 UB，把自己负责的列读进来
__aicore__ inline void GeqrfPanelInit(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    cx.myBase = cx.blockIdx * cx.columns;
    cx.myCols = ((cx.jb - cx.myBase) < cx.columns) ? (cx.jb - cx.myBase) : cx.columns;
    cx.offset = AlignTo(cx.curM, 64);

    cx.a = ub.aQueueIn.AllocTensor<float>();
    cx.hv = ub.aQueueOut.AllocTensor<float>();
    cx.T_local = ub.tQueueDst.AllocTensor<float>();
    cx.cpCol = AscendC::DataCopyExtParams{1, static_cast<uint32_t>(cx.curM * sizeof(float)), 0, 0, 0};
    cx.cp1 = AscendC::DataCopyExtParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    cx.cp8 = AscendC::DataCopyExtParams{1, static_cast<uint32_t>(8 * sizeof(float)), 0, 0, 0};
    cx.pd = AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0};

    for (int i = 0; i != cx.myCols; ++i) {
        AscendC::DataCopyPad(cx.a[cx.offset * i],
                             gm.A[static_cast<uint64_t>(cx.j + cx.myBase + i) * cx.lda + cx.j], cx.cpCol, cx.pd);
    }
    ub.aQueueIn.EnQue(cx.a);
    cx.a = ub.aQueueIn.DeQue<float>();

    cx.copyMask = 64;
    cx.repeatTimes = static_cast<uint8_t>((cx.curM + 63) / 64);

}

/// 吸收前序各核的反射子：把已完成核的 v 作用到本核的列上
__aicore__ inline void GeqrfAbsorbPrior(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ---- 吸收前序各核的反射子 ----
    for (int k = 0; k != cx.blockIdx; ++k) {
        VecBarrier();
        const int kBase = k * cx.columns;
        const int kCols = ((cx.jb - kBase) < cx.columns) ? (cx.jb - kBase) : cx.columns;
        for (int i = 0; i != kCols; ++i) {
            AscendC::LocalTensor<float> b = ub.bQueueIn.AllocTensor<float>();
            AscendC::DataCopyPad(b, gm.Y[static_cast<uint64_t>(kBase + i) * cx.ldy], cx.cpCol, cx.pd);
            ub.bQueueIn.EnQue(b);
            b = ub.bQueueIn.DeQue<float>();
            AscendC::DataCopyPad(cx.scratch, gm.tau[cx.j + kBase + i], cx.cp1, cx.pd);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float tauK = cx.scratch.GetValue(0);
            for (int q = 0; q != cx.myCols; ++q) {
                AscendC::Mul(cx.workLocal, cx.a[cx.offset * q], b, cx.curM);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::ReduceSum<float>(cx.workLocal, cx.workLocal, cx.workLocal, cx.curM);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Axpy(cx.a[cx.offset * q], b, -tauK * cx.workLocal.GetValue(0), cx.curM);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            ub.bQueueIn.FreeTensor(b);
        }
    }

}

/// 极端量级列的定标重算 —— LAPACK slarfg 里 snrm2 定标在这里的等价物。
///
/// **稳态不会进来**：只有 `norm2` 溢出成 Inf/NaN，或平方整体下溢导致
/// `sumSqBelow <= 0` 时才调用，所以正常数据的指令序列与数值都不变。
/// 不定标时 fp32 下的两类错误输入（真机可复现）：
///   · |元素| 约 >= 1.8e19：`norm2` 溢出 ⇒ alpha = ±Inf ⇒ v0 = ±Inf 被写进 A、tau 写 0；
///   · |元素| 约 <= 1e-19：平方落进次正规/被 FTZ 清零 ⇒ norm2 只剩 diag^2 ⇒ 判成退化、
///     该列根本不消元，而真实范数（例如 m=100、元素 1e-20 时约 1e-19）本是可表示的
///     ⇒ QR 恒等式不成立。
///
/// 返回 false 表示定标后对角以下仍然全零（该列确实退化）⇒ 调用方维持 tau = 0、对角原样。
__aicore__ inline bool GeqrfScaledReflector(PanelCtx& cx, int i, float diag, float* alphaOut, float* v0Out,
                                            float* tauPrimeOut)
{
    // cmax = max |a(myBase+i .. curM-1, i)|；hv 此刻还没被写入反射子，可当暂存
    AscendC::Abs(cx.hv, cx.a[cx.offset * i], cx.curM);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Duplicate<float>(cx.hv, 0.0f, cx.myBase + i);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::ReduceMax<float>(cx.hv, cx.hv, cx.hv, cx.curM, true);
    AscendC::PipeBarrier<PIPE_ALL>();
    const float cmax = cx.hv.GetValue(0);
    if (!(cmax > 0.0f)) {
        return false;  // 整列（含对角）全零
    }
    const float inv = 1.0f / cmax;
    AscendC::Muls(cx.workLocal, cx.a[cx.offset * i], inv, cx.curM);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Duplicate<float>(cx.workLocal, 0.0f, cx.myBase + i);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Mul(cx.workLocal, cx.workLocal, cx.workLocal, cx.curM);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::ReduceSum<float>(cx.workLocal, cx.workLocal, cx.workLocal, cx.curM);
    AscendC::PipeBarrier<PIPE_ALL>();
    const float s2 = cx.workLocal.GetValue(0);  // = (norm/cmax)^2，落在 [1, curM]，不会溢出
    const float d = diag * inv;                 // |d| <= 1
    if (s2 - d * d <= 0.0f) {
        return false;  // 定标后对角以下仍为 0 ⇒ 真退化
    }
    const float sq = sqrt(s2);
    const float as = (d <= 0.0f) ? -sq : sq;
    *alphaOut = cmax * as;
    *v0Out = cmax * (d + as);
    // tauPrime = 1/(norm2 + diag*alpha) = inv*inv/(s2 + d*as)。
    // 拆成 inv*(inv/den) 三步算：直接写 cmax*cmax 在 cmax >= 2e19 时会溢出。
    const float den = s2 + d * as;
    *tauPrimeOut = (den != 0.0f) ? (inv * (inv / den)) : 0.0f;
    return true;
}

/// 定下第 i 列的反射子三元组 (alpha, v0, tauPrime)：正常 / 退化 / 定标兜底三条路。
__aicore__ inline void GeqrfColumnReflector(PanelCtx& cx, int i, float norm2, float diag)
{
    // norm2 是平方和，正常情形恒 >= 0；用区间比较同时挡住 Inf 与 NaN
    // （NaN 的所有比较都为假），不写 isfinite 以免被 fast-math 化简掉。
    const bool usable = (norm2 >= 0.0f) && (norm2 <= 3.0e38f);
    // norm2 这里包含对角元（Duplicate 只清掉对角**以上**），所以减一次 diag^2
    // 就是对角以下的平方和，且两边都是同一个 fp32 乘积 ⇒ 退化情形判得精确。
    const float sumSqBelow = norm2 - diag * diag;
    if (usable && sumSqBelow > 0.0f) {
        const float normX = sqrt(norm2);
        cx.alpha[i] = (diag <= 0.0f) ? -normX : normX;
        const float den = norm2 + diag * cx.alpha[i];
        cx.tauPrime[i] = (den != 0.0f) ? 1.0f / den : 0.0f;
        cx.v0[i] = diag + cx.alpha[i];
        return;
    }
    if (GeqrfScaledReflector(cx, i, diag, &cx.alpha[i], &cx.v0[i], &cx.tauPrime[i])) {
        return;
    }
    // 对角**以下**全零（含"尾部只剩对角元"）：LAPACK 的 slarfg 取 tau = 0、对角元原样
    // 保留。若照一般公式走会得到 tau = 2、对角元变号 —— 分解仍然正确（Q 里多一个 -1），
    // 但多引入舍入，m 很小时会把 SQRT01 顶出阈值（2x2 实测 ratio 0.84 vs CPU 0.15，阈值 0.73）。
    cx.alpha[i] = -diag;    // 后面写回的 -cx.alpha 就是原对角元
    cx.tauPrime[i] = 0.0f;
    cx.v0[i] = 1.0f;        // 只用于 1/cx.v0 的缩放，取 1 以免 inf/NaN
}

/// 分解自己那几列：逐列 slarfg，再把该列的反射子作用到本核右边的列
__aicore__ inline void GeqrfFactorOwnCols(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ---- 分解自己那几列 ----
    for (int i = 0; i != cx.myCols; ++i) {
        AscendC::Mul(cx.workLocal, cx.a[cx.offset * i], cx.a[cx.offset * i], cx.curM);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate<float>(cx.workLocal, 0.0f, cx.myBase + i);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::ReduceSum<float>(cx.workLocal, cx.workLocal, cx.workLocal, cx.curM);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float norm2 = cx.workLocal.GetValue(0);
        const float diag = cx.a.GetValue(cx.myBase + i + cx.offset * i);
        cx.diagSave[i] = diag;
        GeqrfColumnReflector(cx, i, norm2, diag);
        cx.a.SetValue(cx.myBase + i + cx.offset * i, cx.v0[i]);
        AscendC::PipeBarrier<PIPE_ALL>();
        cx.scratch.SetValue(0, cx.tauPrime[i]);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(gm.tau[cx.j + cx.myBase + i], cx.scratch, cx.cp1);

        AscendC::Copy(cx.hv, cx.a[cx.offset * i], cx.copyMask, cx.repeatTimes, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate<float>(cx.hv, 0.0f, cx.myBase + i);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int q = i + 1; q != cx.myCols; ++q) {
            AscendC::Mul(cx.workLocal, cx.a[cx.offset * q], cx.hv, cx.curM);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::ReduceSum<float>(cx.workLocal, cx.workLocal, cx.workLocal, cx.curM);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Axpy(cx.a[cx.offset * q], cx.hv, -cx.tauPrime[i] * cx.workLocal.GetValue(0), cx.curM);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        ub.aQueueOut.EnQue(cx.hv);
        cx.hv = ub.aQueueOut.DeQue<float>();
        AscendC::DataCopyPad(gm.Y[static_cast<uint64_t>(cx.myBase + i) * cx.ldy], cx.hv, cx.cpCol);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    VecBarrier();

}

/// 本核内的 w_i：同一核内各列之间的耦合项，写进 wsW 供后继核取用
__aicore__ inline void GeqrfLocalW(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ---- 本核内的 w_i ----
    AscendC::Duplicate<float>(cx.hv, 0.0f, 8 * cx.columns);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = cx.myCols - 1; i >= 0; --i) {
        for (int q = i - 1; q >= 0; --q) {
            AscendC::Mul(cx.workLocal, cx.a[cx.offset * i], cx.a[cx.offset * q], cx.curM);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate<float>(cx.workLocal, 0.0f, cx.myBase + i);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::ReduceSum<float>(cx.workLocal, cx.workLocal, cx.workLocal, cx.curM);
            AscendC::PipeBarrier<PIPE_ALL>();
            cx.hv.SetValue(i * 8 + q, cx.workLocal.GetValue(0));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(gm.W[cx.bd * 8 * i + cx.blockIdx * 8 + (cx.blockIdx % 2) * 1024], cx.hv[i * 8], cx.cp8);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

}

/// T 的自己那几行：对角块 + 核内耦合
__aicore__ inline void GeqrfTOwnRows(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ---- T 的自己那几行：对角块 + 核内耦合 ----
    // ★ cx.T_local 的 UB 行距取 AlignTo(cx.jb,8)：用 cx.bd*cx.columns 时，cx.jb 小（如 4）会让
    //   DataCopyPad 的 UB 源偏移不是 8 的倍数 ⇒ **静默什么都不搬**，T 的奇数行
    //   永远没被写出。cx.jb=64 时恰好是 64 的倍数，所以这个缺陷只在小 cx.jb 上暴露。
    cx.ldtUb = AlignTo(cx.jb, 8);
    AscendC::Duplicate<float>(cx.T_local, 0.0f, cx.columns * cx.ldtUb);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Duplicate<float>(cx.workLocal, 0.0f, cx.bd * cx.columns * 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i != cx.myCols; ++i) {
        cx.T_local.SetValue(cx.ldtUb * i + cx.myBase + i, -cx.tauPrime[i]);
        cx.workLocal.SetValue(cx.bd * i * 8 + cx.blockIdx * 8 + i, -cx.tauPrime[i]);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int q = i + 1; q != cx.myCols; ++q) {
            AscendC::Mul(cx.workLocal[1024], cx.workLocal[cx.bd * i * 8 + cx.blockIdx * 8], cx.hv[q * 8], 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::ReduceSum<float>(cx.workLocal[1024], cx.workLocal[1024], cx.workLocal[1024], 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float sv = cx.workLocal.GetValue(1024);
            cx.T_local.SetValue(cx.ldtUb * i + cx.myBase + q, -cx.tauPrime[q] * sv);
            cx.workLocal.SetValue(cx.bd * i * 8 + cx.blockIdx * 8 + q, -cx.tauPrime[q] * sv);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    AscendC::Duplicate<float>(cx.hv, 0.0f, 8 * cx.columns);
    AscendC::PipeBarrier<PIPE_ALL>();
    VecBarrier();

}

/// 服务后继各核，顺带补完 T 的自己这几行
__aicore__ inline void GeqrfServeLater(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ---- 服务后继各核，顺带补完 T 的自己这几行 ----
    for (int k = cx.blockIdx + 1; k != cx.bd; ++k) {
        const int kBase = k * cx.columns;
        const int kCols = ((cx.jb - kBase) < cx.columns) ? (cx.jb - kBase) : cx.columns;
        for (int i = 0; i != kCols; ++i) {
            AscendC::LocalTensor<float> b = ub.bQueueIn.AllocTensor<float>();
            AscendC::DataCopyPad(b, gm.Y[static_cast<uint64_t>(kBase + i) * cx.ldy], cx.cpCol, cx.pd);
            ub.bQueueIn.EnQue(b);
            b = ub.bQueueIn.DeQue<float>();
            for (int q = 0; q != cx.myCols; ++q) {
                AscendC::Mul(cx.workLocal[1024], cx.a[q * cx.offset], b, cx.curM);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::ReduceSum<float>(cx.workLocal[1024], cx.workLocal[1024], cx.workLocal[1024], cx.curM);
                AscendC::PipeBarrier<PIPE_ALL>();
                cx.hv.SetValue(i * 8 + q, cx.workLocal.GetValue(1024));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gm.W[cx.bd * 8 * i + cx.blockIdx * 8 + (k % 2) * 1024], cx.hv[i * 8], cx.cp8);
            AscendC::PipeBarrier<PIPE_ALL>();
            ub.bQueueIn.FreeTensor(b);
        }

        VecBarrier();

        for (int i = 0; i != kCols; ++i) {
            AscendC::LocalTensor<float> b = ub.bQueueIn.AllocTensor<float>();
            const int len = k * 8 + 8;
            AscendC::DataCopyExtParams cpL{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(b, gm.W[cx.bd * 8 * i + (k % 2) * 1024], cpL, cx.pd);
            ub.bQueueIn.EnQue(b);
            b = ub.bQueueIn.DeQue<float>();
            AscendC::DataCopyPad(cx.scratch, gm.tau[cx.j + kBase + i], cx.cp1, cx.pd);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float tauKi = cx.scratch.GetValue(0);
            for (int q = 0; q != cx.myCols; ++q) {
                AscendC::Mul(cx.workLocal[1024], b, cx.workLocal[cx.bd * 8 * q], len);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::ReduceSum<float>(cx.workLocal[1024], cx.workLocal[1024], cx.workLocal[1024], len);
                AscendC::PipeBarrier<PIPE_ALL>();
                const float sv = cx.workLocal.GetValue(1024);
                cx.T_local.SetValue(cx.ldtUb * q + kBase + i, -tauKi * sv);
                cx.workLocal.SetValue(cx.bd * 8 * q + k * 8 + i, -tauKi * sv);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            ub.bQueueIn.FreeTensor(b);
        }
    }

}

/// 写回 T 的自己那几行（行主序，ld = bd*columns）
__aicore__ inline void GeqrfStoreT(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ---- 写回 T 的自己那几行（行主序，ld = cx.bd*cx.columns）----
    for (int i = 0; i != cx.myCols; ++i) {
        AscendC::DataCopyExtParams cpT{1, static_cast<uint32_t>(cx.jb * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(gm.T[static_cast<uint64_t>(cx.myBase + i) * cx.jb], cx.T_local[cx.ldtUb * i], cpT);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    VecBarrier();

}

/// 本核那几列写回 A，并把这一面板用的 UB 还回去
__aicore__ inline void GeqrfStoreA(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    ub.aQueueIn.EnQue(cx.a);
    cx.a = ub.aQueueIn.DeQue<float>();
    for (int i = 0; i != cx.myCols; ++i) {
        AscendC::DataCopyPad(gm.A[static_cast<uint64_t>(cx.j + cx.myBase + i) * cx.lda + cx.j],
                             cx.a[cx.offset * i], cx.cpCol);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    ub.tQueueDst.FreeTensor(cx.T_local);
    ub.aQueueIn.FreeTensor(cx.a);
    ub.aQueueOut.FreeTensor(cx.hv);
}

/// 归一化并写回 A（转成 LAPACK 约定），然后释放 UB
__aicore__ inline void GeqrfNormalizeStore(PanelCtx& cx, const GeqrfGm& gm, GeqrfUb& ub)
{
    // ★ 必须用位掩码跳过对角以上的元素，不能把指针偏到非 8 倍数位置（507035）。
    //   掩码一次只覆盖 64 个元素，所以 myBase >= 64 时整段往后挪 64 —— 原来这是两段
    //   几乎逐字相同的代码，差别只有这个基址。
    const int base = (cx.myBase >= 64) ? 64 : 0;
    cx.maskUpdate[0] = (~0ULL) - ((1ULL << (cx.myBase - base)) - 1);
    for (int i = 0; i != cx.myCols; ++i) {
        // cx.v0[i] 在退化分支里被置成 1.0f，构造上不会是 0；就地判一次
        // 既满足静态检查，也把原来每轮两次的除法合成一次。
        const float invV0 = (cx.v0[i] != 0.0f) ? (1.0f / cx.v0[i]) : 0.0f;
        AscendC::Muls(cx.a[cx.offset * i + base], cx.a[cx.offset * i + base], invV0, cx.maskUpdate, 1,
                      {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_ALL>();
        cx.maskUpdate[0] <<= 1;
        if (cx.curM > base + 64) {
            AscendC::Muls(cx.a[cx.offset * i + base + 64], cx.a[cx.offset * i + base + 64], invV0,
                          cx.curM - base - 64);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        cx.a.SetValue(cx.offset * i + cx.myBase + i, -cx.alpha[i]);
        AscendC::PipeBarrier<PIPE_ALL>();
        // tau 按 LAPACK slarfg 的式子算：tau = (beta - diag)/beta，beta = -alpha。
        // 与 tauPrime*v0^2 代数等价但少两次舍入 —— m 很小时 SQRT01 的分母
        // (m*||A||_1*eps) 极小，1 ulp 就能把判据顶出阈值（2x2 实测）。
        // 退化情形 alpha = -diag ⇒ beta = diag ⇒ 结果恰为 0，与 tauPrime=0 一致。
        const float betaI = -cx.alpha[i];
        cx.scratch.SetValue(0, (betaI != 0.0f) ? ((betaI - cx.diagSave[i]) / betaI) : 0.0f);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(gm.tau[cx.j + cx.myBase + i], cx.scratch, cx.cp1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    GeqrfStoreA(cx, gm, ub);
}
/// 一个面板的 vec 侧全程：非工作核只补齐同步配对，工作核跑完八个相位。
__aicore__ inline void GeqrfPanelVec(const GeqrfGm& gm, GeqrfUb& ub, int j, int jb, int bd, int columns, int rows,
                                     int lda, int ldy, int blockIdx)
{
    if (blockIdx >= bd) {
        SynAll(bd);
    } else {
        PanelCtx cx;
        cx.j = j;
        cx.jb = jb;
        cx.bd = bd;
        cx.columns = columns;
        cx.blockIdx = blockIdx;
        cx.curM = rows;
        cx.lda = lda;
        cx.ldy = ldy;
        cx.workLocal = ub.workLocal;
        cx.scratch = ub.scratch;
        GeqrfPanelInit(cx, gm, ub);
        GeqrfAbsorbPrior(cx, gm, ub);
        GeqrfFactorOwnCols(cx, gm, ub);
        GeqrfLocalW(cx, gm, ub);
        GeqrfTOwnRows(cx, gm, ub);
        GeqrfServeLater(cx, gm, ub);
        GeqrfStoreT(cx, gm, ub);
        GeqrfNormalizeStore(cx, gm, ub);
    }
    VecBarrier();
    VecNotifyCube();
}

#endif  // __DAV_VEC__

#ifdef __DAV_CUBE__
/// 尾随更新：C += Y * ((-T)^T_rowmajor * (Y^T C))，即 NoTrans 形式的 LARFB
__aicore__ inline void GeqrfTrailingCube(const Geqrf2Params& params, int j, int jb, int rows, int trail, int lda,
                                         int ldy)
{
    if (trail <= 0) {
        return;
    }
    QrCore::GemmPlan lp[3];
    const int64_t cOff = static_cast<int64_t>(j + jb) * lda + j;
    QrCore::PlanLarfbLeft(rows, trail, jb, ldy, lda, jb, jb, jb, 0, cOff, lp, QrCore::Trans::NoTrans);
    RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(lp[0], params.A, params.wsY, params.wsW2);
    CubeBarrier();
    RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[1], params.wsW2, params.wsT, params.wsW2);
    CubeBarrier();
    RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(lp[2], params.wsW2, params.wsY, params.A);
    CubeBarrier();
}
#endif  // __DAV_CUBE__

extern "C" __global__ __aicore__ void sgeqrf_mix_kernel(Geqrf2Params params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const int m = static_cast<int>(params.m);
    const int n = static_cast<int>(params.n);
    const int lda = static_cast<int>(params.lda);
    const int ldy = static_cast<int>(params.ldy);
    const int columns = static_cast<int>(params.columns);
    const int alignBD = static_cast<int>(params.alignBD);
    const int panelWidth = alignBD * columns;
    const int kmax = (m < n) ? m : n;

#ifdef __DAV_VEC__
    AscendC::TPipe pipe;
    GeqrfUb ub;
    ub.Init(pipe);
    GeqrfGm gm;
    gm.Init(params);

    // MIX 的 AIV 侧：GetBlockIdx() 已是 AIV 核号（不乘 2），GetBlockNum() 是 AIC 数（要乘 2）
    const int blockIdx = static_cast<int>(AscendC::GetBlockIdx());

    ZeroDevInfo(params.devInfo, ub.workLocal, static_cast<uint32_t>(blockIdx));
#endif

    for (int j = 0; j < kmax; j += panelWidth) {
        const int jb = ((kmax - j) < panelWidth) ? (kmax - j) : panelWidth;
        const int rows = m - j;
        // 本面板参与的核数；最后一个核可能只有 jb % columns 条真列
        const int bd = (columns == 0) ? 0 : (jb + columns - 1) / columns;

#ifdef __DAV_VEC__
        GeqrfPanelVec(gm, ub, j, jb, bd, columns, rows, lda, ldy, blockIdx);
        VecWaitCube();
#elif defined(__DAV_CUBE__)
        CubeWaitVec();
        // ---- 尾随更新（trail = n - j - jb 列）----
        GeqrfTrailingCube(params, j, jb, rows, n - j - jb, lda, ldy);
        CubeNotifyVec();
#endif
    }
}
