/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * LARFB 的布局计划层（纯算术，不依赖 ACL / AscendC / CATLASS，可在本地 g++ 下自验）
 *
 * 核心恒等式：列主序 (R x C, ld) 的内存 == 行主序 (C x R, ld) 的内存。
 * 因此"转置"是免费的视图切换，而 CATLASS 的 layout tag 正好吃这两种视图。
 *
 * 块反射器更新（side = LEFT, trans = N）：
 *     C := C - V * T * (V^T * C)
 * 把负号折进 T（调用方传入的 T 已含负号），拆成三个 GEMM：
 *     G1:  W  = V^T * C          (k x N)
 *     G2:  W2 = Tneg * W         (k x N)
 *     G3:  C += V * W2           (M x N, 累加)
 *
 * 全部按列主序语义给参数，本层翻译成 CATLASS 的行主序视图：
 *
 *  | 阶段 | A                      | B                        | D                      | 累加 |
 *  |------|------------------------|--------------------------|------------------------|------|
 *  | G1   | RowMajor(N, M, ldc)    | **ColumnMajor(M, k, ldv)** | RowMajor(N, k, ldw)  | 否   |
 *  | G2   | RowMajor(N, k, ldw)    | RowMajor(k, k, ldt)      | RowMajor(N, k, ldw2)   | 否   |
 *  | G3   | RowMajor(N, k, ldw2)   | RowMajor(k, M, ldv)      | RowMajor(N, M, ldc)    | 是   |
 *
 * 只有 G1 的 B 需要列主序视图；其余都是行主序。
 */
#ifndef QR_CORE_LARFB_PLAN_H
#define QR_CORE_LARFB_PLAN_H

#include <cstdint>

// 同一份代码既要在本地 g++ 下自验，也要在 kernel 里直接用，避免公式写两遍漂移
#ifdef __CCE_AICORE__
#define QR_CORE_FN __aicore__ inline
#else
#define QR_CORE_FN inline
#endif

namespace QrCore
{

enum class Side
{
    Left,
    Right
};
enum class Trans
{
    NoTrans,
    Trans
};

/// 一次实数 GEMM 的计划：D(m x n) [+]= A(m x k) * B(k x n)，全部按行主序视图表达
struct GemmPlan
{
    int m = 0;
    int n = 0;
    int k = 0;
    int64_t aOffset = 0;  // 元素偏移（相对各自矩阵的基址）
    int lda = 0;
    bool aColMajorView = false;
    int64_t bOffset = 0;
    int ldb = 0;
    bool bColMajorView = false;
    int64_t dOffset = 0;
    int ldd = 0;
    bool accumulate = false;
};

/// LARFB 的三个阶段
enum class LarfbStage
{
    VtC = 0,
    TW = 1,
    VW = 2
};

/**
 * 生成 side = LEFT 的三个 GEMM 计划。
 *
 * 语义（全部列主序）：
 *   V : M x k，位于反射子矩阵的 (row0, col0) 处，leading dimension ldv
 *   C : M x N，位于被作用矩阵的 (row0, colC) 处，leading dimension ldc
 *   W : k x N（工作区，ldw）；W2 : k x N（工作区，ldw2，可与 W 同址）
 *   T : k x k（已含负号，ldt）
 *
 * row0 / col0 / colC 以元素计，用于在大矩阵里定位子块。
 */
/// trans = NoTrans 施加 Q（C <- (I - V T V^T) C），Trans 施加 Q^T（C <- (I - V T^T V^T) C）。
/// 两者只差 G2 里 T 操作数的视图：T 缓冲按列主序存放，RowMajor 视图即 T^T。
/// geqrf 的尾随更新要的是 Q^T；但 geqrf 侧的 T 按行主序存放（见 geqrf_mix.hpp），
/// 相对本处"列主序 T"的视图定义含义取反，因此实际传 Trans::NoTrans——
/// 数学上等价于列主序 T 下传 Trans::Trans 的效果，勿按本行上方视图定义望文生义。
QR_CORE_FN void PlanLarfbLeft(int M, int N, int k, int ldv, int ldc, int ldw, int ldw2, int ldt, int64_t vOffset,
                              int64_t cOffset, GemmPlan out[3], Trans trans = Trans::NoTrans)
{
    // G1: W = V^T * C   ->   W~(N x k) = C~(N x M, RowMajor ldc) * V~^T(M x k, ColumnMajor ldv)
    out[0].m = N;
    out[0].n = k;
    out[0].k = M;
    out[0].aOffset = cOffset;
    out[0].lda = ldc;
    out[0].aColMajorView = false;
    out[0].bOffset = vOffset;
    out[0].ldb = ldv;
    out[0].bColMajorView = true;
    out[0].dOffset = 0;
    out[0].ldd = ldw;
    out[0].accumulate = false;

    // G2: W2 = Tneg * W  ->  W2~(N x k) = W~(N x k, RowMajor ldw) * Tneg~(k x k, RowMajor ldt)
    out[1].m = N;
    out[1].n = k;
    out[1].k = k;
    out[1].aOffset = 0;
    out[1].lda = ldw;
    out[1].aColMajorView = false;
    out[1].bOffset = 0;
    out[1].ldb = ldt;
    // NoTrans: 操作数取 T^T（RowMajor 视图）-> W2 = T W     -> 施加 Q
    // Trans  : 操作数取 T  （ColMajor 视图）-> W2 = T^T W   -> 施加 Q^T
    out[1].bColMajorView = (trans == Trans::Trans);
    out[1].dOffset = 0;
    out[1].ldd = ldw2;
    out[1].accumulate = false;

    // G3: C += V * W2  ->  C~(N x M) += W2~(N x k, RowMajor ldw2) * V~(k x M, RowMajor ldv)
    out[2].m = N;
    out[2].n = M;
    out[2].k = k;
    out[2].aOffset = 0;
    out[2].lda = ldw2;
    out[2].aColMajorView = false;
    out[2].bOffset = vOffset;
    out[2].ldb = ldv;
    out[2].bColMajorView = false;
    out[2].dOffset = cOffset;
    out[2].ldd = ldc;
    out[2].accumulate = true;
}

/**
 * side = RIGHT：C := C - (C * V) * T * V^T，C 为 M x N，V 为 N x k。
 *   G1: W  = C * V      (M x k)
 *   G2: W2 = W * Tneg   (M x k)
 *   G3: C += W2 * V^T   (M x N)
 *
 * 行主序视图：
 *   G1: W~(k x M) = V~(k x N, RowMajor ldv) * C~(N x M, RowMajor ldc)
 *   G2: W2~(k x M) = Tneg~(k x k, RowMajor ldt) * W~(k x M, RowMajor ldw)
 *   G3: C~(N x M) += V(N x k, ColumnMajor ldv) * W2~(k x M, RowMajor ldw2)
 */
/// trans 的含义同 PlanLarfbLeft（这里 T 在 G2 的 A 侧）。
QR_CORE_FN void PlanLarfbRight(int M, int N, int k, int ldv, int ldc, int ldw, int ldw2, int ldt, int64_t vOffset,
                               int64_t cOffset, GemmPlan out[3], Trans trans = Trans::NoTrans)
{
    out[0].m = k;
    out[0].n = M;
    out[0].k = N;
    out[0].aOffset = vOffset;
    out[0].lda = ldv;
    out[0].aColMajorView = false;
    out[0].bOffset = cOffset;
    out[0].ldb = ldc;
    out[0].bColMajorView = false;
    out[0].dOffset = 0;
    out[0].ldd = ldw;
    out[0].accumulate = false;

    out[1].m = k;
    out[1].n = M;
    out[1].k = k;
    out[1].aOffset = 0;
    // NoTrans: 操作数取 T^T -> W2 = C V T ；Trans: 取 T -> W2 = C V T^T
    out[1].lda = ldt;
    out[1].aColMajorView = (trans == Trans::Trans);
    out[1].bOffset = 0;
    out[1].ldb = ldw;
    out[1].bColMajorView = false;
    out[1].dOffset = 0;
    out[1].ldd = ldw2;
    out[1].accumulate = false;

    out[2].m = N;
    out[2].n = M;
    out[2].k = k;
    out[2].aOffset = vOffset;
    out[2].lda = ldv;
    out[2].aColMajorView = true;
    out[2].bOffset = 0;
    out[2].ldb = ldw2;
    out[2].bColMajorView = false;
    out[2].dOffset = cOffset;
    out[2].ldd = ldc;
    out[2].accumulate = true;
}

/// 块反射器的遍历次序：施加 Q 时块要倒着走，施加 Q^T 时正着走（RIGHT 相反）
QR_CORE_FN bool LarfbReverseOrder(Side side, Trans trans)
{
    const bool isLeft = (side == Side::Left);
    const bool isNoTrans = (trans == Trans::NoTrans);
    return isLeft == isNoTrans;  // LEFT&N -> true, LEFT&T -> false, RIGHT&N -> false, RIGHT&T -> true
}

}  // namespace QrCore

#endif  // QR_CORE_LARFB_PLAN_H
