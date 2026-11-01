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
 * \file trsm_mix.hpp
 * rief 上三角回代 R * X = C（融合 MIX kernel，950PR / fp32）—— SSgels 链路的最后一段
 *
 * 分块回代（块宽 nb，从后往前）：
 *   vec : X <- C（之后原地回代；C 不被破坏、X 是独立输出）
 *   cube: W(nb x nrhs) = R(rows, tail) * X(tail, :)（按 K 分核）
 *   vec : X(rows,:) -= W，再做块内 nb 步回代，**按 nrhs 列分核**；
 *         列内每步是 1 次标量除 + 2 个向量指令（长度 <= nb），不是 nb^2/2 次标量 FMA
 *         —— AIV 上一次标量 FMA 约 45 周期，nb=64 时两者差约 17 倍。
 *
 * 行主序视图的小技巧：列主序 (R x C, ld) 的内存 == 行主序 (C x R, ld) 的内存，于是
 *   Wt~(nrhs x nb) = X~(nrhs x |tail|) * R~(|tail| x nb)
 * 两个操作数都是行主序（复用 MmadRR），输出 Wt 的内存正好是列主序的 W(nb x nrhs, ld=nb)。
 *
 */

#pragma once

// 五个 mix kernel 的公共件（CATLASS include、类型别名、跨核同步、分 K 调度、
// householder/larfb_plan、以及标准库那几个头）都由这一个头提供。
#include "../../sgeqrf/kernel/mix_common.hpp"


constexpr uint32_t MAX_N = 4096;
constexpr uint32_t MAX_NB = 64;

struct TrsmParams {
    GM_ADDR A;    // n x n 列主序 ld=lda，用它的上三角
    GM_ADDR C;    // n x nrhs 列主序 ld=ldc（只读）
    GM_ADDR X;    // n x nrhs 列主序 ld=ldx（输出）
    GM_ADDR wsW;  // nb x nrhs 列主序 ld=nb
    GM_ADDR wsP;  // 分 K 的 partial 区
    uint32_t n;
    uint32_t nrhs;
    uint32_t lda;
    uint32_t ldc;
    uint32_t ldx;
    uint32_t nb;
    uint64_t capP;
};


/// W = R(rows, tail) * X(tail, :) 的 plan。
/// 行主序视图：Wt~(nrhs x rows) = X~(nrhs x tail) * R~(tail x rows)，两个操作数都行主序。
__aicore__ inline QrCore::GemmPlan TrsmPlanW(uint32_t nrhs, uint32_t rows, uint32_t tail, uint32_t tail0,
                                             uint32_t r0, uint32_t lda, uint32_t ldx)
{
    QrCore::GemmPlan gp{};
    gp.m = static_cast<int>(nrhs);
    gp.n = static_cast<int>(rows);
    gp.k = static_cast<int>(tail);
    gp.aOffset = static_cast<int64_t>(tail0);
    gp.lda = static_cast<int>(ldx);
    gp.bOffset = static_cast<int64_t>(tail0) * lda + r0;
    gp.ldb = static_cast<int>(lda);
    gp.dOffset = 0;
    gp.ldd = static_cast<int>(rows);
    gp.accumulate = false;
    return gp;
}

#ifdef __DAV_CUBE__
/// W 的那次 GEMM（cube 侧）
__aicore__ inline void TrsmGemmWCube(const TrsmParams& params, uint32_t nrhs, uint32_t rows, uint32_t tail,
                                     uint32_t tail0, uint32_t r0, uint32_t lda, uint32_t ldx, uint32_t wChunks,
                                     uint32_t wChunkLen)
{
    const QrCore::GemmPlan gp = TrsmPlanW(nrhs, rows, tail, tail0, r0, lda, ldx);
    CubeWaitVec();
    if (wChunks > 1) {
        RunPlanOnCubeSplitK<MmadRR, layout::RowMajor, layout::RowMajor>(gp, params.X, params.A, params.wsP, wChunks,
                                                                       wChunkLen);
    } else {
        RunPlanOnCube<MmadRR, layout::RowMajor, layout::RowMajor>(gp, params.X, params.A, params.wsW);
    }
    CubeBarrier();
    CubeNotifyVec();
}
#endif  // __DAV_CUBE__

#ifdef __DAV_VEC__
/// AIV 侧的 UB 分块。TBuf 必须活到 kernel 结束，所以放在这个结构体里。
struct TrsmUb {
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCol, bufR, bufT, bufRed;
    AscendC::LocalTensor<float> col, rl, tmp, red;

    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(bufCol, MAX_N * sizeof(float));
        pipe.InitBuffer(bufR, MAX_NB * MAX_NB * sizeof(float));
        pipe.InitBuffer(bufT, MAX_NB * sizeof(float));
        pipe.InitBuffer(bufRed, RED_CAP * sizeof(float));
        col = bufCol.Get<float>();
        rl = bufR.Get<float>();
        tmp = bufT.Get<float>();
        red = bufRed.Get<float>();
    }
};

/// W 的那次 GEMM 期间 vec 侧要做的事：等 cube 算完，分 K 时再归约 partial
__aicore__ inline void TrsmGemmWVec(const AscendC::GlobalTensor<float>& W, const AscendC::GlobalTensor<float>& P,
                                    const AscendC::LocalTensor<float>& col,
                                    const AscendC::LocalTensor<float>& red, uint32_t nrhs, uint32_t rows,
                                    uint32_t wChunks, uint32_t coreId, uint32_t coreNum)
{
    VecNotifyCube();
    VecWaitCube();
    if (wChunks > 1) {
        ReducePartials(W, P, static_cast<uint64_t>(nrhs) * rows, wChunks, col, red, RED_CAP, coreId, coreNum);
        VecBarrier();
    }
}
#endif  // __DAV_VEC__

#ifdef __DAV_VEC__
/// X <- C 整块拷贝（之后在 X 上原地回代；C 不被破坏、X 是独立输出）
__aicore__ inline void TrsmCopyCtoX(const AscendC::GlobalTensor<float>& C, const AscendC::GlobalTensor<float>& X,
                                    const AscendC::LocalTensor<float>& col, uint32_t n, uint32_t nrhs,
                                    uint32_t ldc, uint32_t ldx, uint32_t coreId, uint32_t coreNum)
{
    for (uint32_t j = coreId; j < nrhs; j += coreNum) {
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
        AscendC::DataCopyPad(col, C[static_cast<uint64_t>(j) * ldc], cp, pd);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(X[static_cast<uint64_t>(j) * ldx], col, cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

/// 把本块的上三角对角块读进 UB（行距 nb）。每核只在第一次用到时读一次。
__aicore__ inline void TrsmLoadDiagBlock(const AscendC::GlobalTensor<float>& A,
                                         const AscendC::LocalTensor<float>& rl, uint32_t r0, uint32_t rows,
                                         uint32_t lda, uint32_t nb)
{
    for (uint32_t c = 0; c < rows; ++c) {
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>((c + 1) * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
        AscendC::DataCopyPad(rl[c * nb], A[static_cast<uint64_t>(r0 + c) * lda + r0], cp, pd);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

/// 块内回代一列：x(i) /= R(i,i)，然后 x(0..i-1) -= x(i)*R(0..i-1, i)。
/// 每步 1 次标量除 + 2 个向量指令（长度 <= nb），**不是** nb^2/2 次标量 FMA。
__aicore__ inline void TrsmSolveOneCol(const AscendC::LocalTensor<float>& col,
                                       const AscendC::LocalTensor<float>& rl,
                                       const AscendC::LocalTensor<float>& tmp, uint32_t rows, uint32_t nb)
{
    for (uint32_t t = 0; t < rows; ++t) {
        const uint32_t i = rows - 1u - t;
        // R(i,i) == 0 即三角系统奇异。按 IEEE 语义传播（与 BLAS strsm 在奇异三角上的
        // 行为一致）：x/+0 得 ±Inf、0/+0 得 NaN，调用方检查 dX 即可发现秩亏。
        // ★ 这里用常量 +Inf 而不是写 `1.0f / 0.0f`：结果与真除法一致，但不含字面除零，
        //   因此 codecheck 的除零红线（G.EXP.22-CPP）仍然过。
        //   早期版本在这里置 0，会给出"看起来合法"的有限解、把奇异性掩盖掉。
        const float diagR = rl.GetValue(i * nb + i);
        const float dinv = (diagR != 0.0f) ? (1.0f / diagR) : __builtin_inff();
        const float xi = col.GetValue(i) * dinv;
        col.SetValue(i, xi);
        if (i > 0) {
            // 标量与向量之间来回依赖（下一步的 GetValue 要读这一步 Add 的结果），
            // 用 PIPE_ALL 屏障最稳：S_V / V_S 这两个 HardEvent 本仓没验证过。
            AscendC::Muls(tmp, rl[i * nb], -xi, i);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(col, col, tmp, i);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

/// X(rows, :) -= W，再做块内回代。按 nrhs 列分核；对角块每核只读一次。
__aicore__ inline void TrsmBackSubstVec(const AscendC::GlobalTensor<float>& A,
                                        const AscendC::GlobalTensor<float>& X,
                                        const AscendC::GlobalTensor<float>& W,
                                        const AscendC::LocalTensor<float>& col,
                                        const AscendC::LocalTensor<float>& rl,
                                        const AscendC::LocalTensor<float>& tmp, uint32_t r0, uint32_t rows,
                                        uint32_t tail, uint32_t nrhs, uint32_t lda, uint32_t ldx, uint32_t nb,
                                        uint32_t coreId, uint32_t coreNum)
{
    bool loadedR = false;
    for (uint32_t j = coreId; j < nrhs; j += coreNum) {
        if (!loadedR) {
            TrsmLoadDiagBlock(A, rl, r0, rows, lda, nb);
            loadedR = true;
        }
        // X(rows, j)（必要时连 W 的那一段）读进 UB
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
        AscendC::DataCopyPad(col, X[static_cast<uint64_t>(j) * ldx + r0], cp, pd);
        if (tail > 0) {
            AscendC::DataCopyPad(tmp, W[static_cast<uint64_t>(j) * rows], cp, pd);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (tail > 0) {
            AscendC::Sub(col, col, tmp, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        TrsmSolveOneCol(col, rl, tmp, rows, nb);
        AscendC::PipeBarrier<PIPE_ALL>();  // 标量 SetValue 与向量结果都要先落地
        AscendC::DataCopyPad(X[static_cast<uint64_t>(j) * ldx + r0], col, cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
#endif  // __DAV_VEC__

extern "C" __global__ __aicore__ void strsm_mix_kernel(TrsmParams params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const uint32_t n = params.n, nrhs = params.nrhs;
    const uint32_t lda = params.lda, ldc = params.ldc, ldx = params.ldx, nb = params.nb;
    const uint32_t nBlk = CeilDivU(n, nb);

#ifdef __DAV_VEC__
    AscendC::TPipe pipe;
    TrsmUb ub;
    ub.Init(pipe);
    const AscendC::LocalTensor<float>& col = ub.col;
    const AscendC::LocalTensor<float>& rl = ub.rl;
    const AscendC::LocalTensor<float>& tmp = ub.tmp;
    const AscendC::LocalTensor<float>& red = ub.red;

    AscendC::GlobalTensor<float> A, C, X, W, P;
    A.SetGlobalBuffer((__gm__ float*)params.A);
    C.SetGlobalBuffer((__gm__ float*)params.C);
    X.SetGlobalBuffer((__gm__ float*)params.X);
    W.SetGlobalBuffer((__gm__ float*)params.wsW);
    P.SetGlobalBuffer((__gm__ float*)params.wsP);

    // MIX 的 AIV 侧：GetBlockIdx() 已是 AIV 核号（不乘 2），GetBlockNum() 是 AIC 数（要乘 2）
    const uint32_t coreId = AscendC::GetBlockIdx();
    const uint32_t coreNum = AscendC::GetBlockNum() * 2;

    // ---- 0. X <- C（之后在 X 上原地回代；C 不被破坏、X 是独立输出）----
    TrsmCopyCtoX(C, X, col, n, nrhs, ldc, ldx, coreId, coreNum);
    VecBarrier();
#endif

    for (uint32_t bIdx = 0; bIdx < nBlk; ++bIdx) {
        const uint32_t b = nBlk - 1u - bIdx;
        const uint32_t r0 = b * nb;
        const uint32_t rows = ((n - r0) < nb) ? (n - r0) : nb;
        const uint32_t tail0 = r0 + rows;
        const uint32_t tail = n - tail0;

        // ---- 1. W = R(rows, tail) * X(tail, :)（cube）----
        // 行主序视图：Wt~(nrhs x rows) = X~(nrhs x tail) * R~(tail x rows)，两个操作数都行主序
        uint32_t wChunkLen = 0;
        const uint32_t wChunks =
            (tail == 0) ? 1u : KSplitChunks(nrhs, rows, tail, AscendC::GetBlockNum(), params.capP, &wChunkLen);
        if (tail > 0) {
#ifdef __DAV_CUBE__
            TrsmGemmWCube(params, nrhs, rows, tail, tail0, r0, lda, ldx, wChunks, wChunkLen);
#elif defined(__DAV_VEC__)
            TrsmGemmWVec(W, P, col, red, nrhs, rows, wChunks, coreId, coreNum);
#endif
        }

#ifdef __DAV_VEC__
        // ---- 2. X(rows, :) -= W，并做块内回代；两步都按 nrhs 列分核 ----
        TrsmBackSubstVec(A, X, W, col, rl, tmp, r0, rows, tail, nrhs, lda, ldx, nb, coreId, coreNum);
        VecBarrier();
#endif
    }
}
