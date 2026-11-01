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
 * \file sgeqrf_host.cpp
 * \brief aclsolverSgeqrf / aclsolverSgeqrf_bufferSize 的 Host 侧实现
 *
 * 语义对齐 cusolverDnSgeqrf：列主序、任意 lda（禁止要求 lda == m）、workspace 以
 * **元素个数** 计、devInfo 为 device 侧 int32（0 成功 / -i 表示第 i 个参数非法，不计 handle）。
 *
 * 参数序号（任务书 §2.1 第 3 条，不计 handle）：
 *   aclsolverSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
 *                           1  2  3   4    5       6        7       8
 *   aclsolverSgeqrf_bufferSize(handle, m, n, A, lda, Lwork)
 *                                      1  2  3   4     5
 */

// 五个 host 都要的那批头（标准库、acl、assert/gm_addr/align_up、platform）收在这里
#include "../../utils/host_common.h"

#include "../kernel/geqrf_launch.h"

namespace
{
// kernel 侧 UB 的容量约束，必须与 kernel/geqrf_mix.hpp 保持一致。
//
// 面板分解让每个 AIV 核把自己那 columns 条列**全程驻留 UB**，每列按 AlignTo(curM,64) 跨步，
// 而 aQueueIn 只有 2 * QR_CHUNK 个 float ⇒ 硬约束：columns * AlignTo(m,64) <= 2*QR_CHUNK。
// columns=2 时 m <= 8192；columns=4 时 m <= 4096。
constexpr int32_t QR_CHUNK = 8192;
// 面板内每核负责的列数。当前只启用 columns=2 这条已全量验证过的路径；
// columns=4（面板宽 128）留待 S1 评估。
constexpr int32_t QR_COLUMNS = 2;
// 参与面板分解的 AIV 核数。实测：最忙核的工作量 ∝ 面板宽度、与怎么切核无关，
// 多给核不会更快、只会增加核间会合次数；32*2=64 的面板宽度同时匹配 cube 的 K。
constexpr int32_t QR_ALIGN_BD = 32;

// workspace 各区段的起始按 512B（128 个 float）对齐
constexpr int64_t QR_WS_ALIGN = 128;
// Y 的 leading dimension 按 64 行对齐，匹配 L1TileShape 的 M 方向分形
constexpr int32_t QR_LDY_ALIGN = 64;
// 面板内交换 T 行用的暂存区（kernel 侧下标上界约 2 KB，取 4096 个 float 留足余量）
constexpr int64_t QR_PANEL_W_ELEMS = 4096;


/// 本实现支持的最大行数（由上面的 UB 约束推出）
inline int32_t MaxRowsSupported()
{
    return static_cast<int32_t>(2 * QR_CHUNK / QR_COLUMNS / QR_LDY_ALIGN) * QR_LDY_ALIGN;
}

/// 面板宽度 = alignBD * columns，并收敛到 min(m, n)
inline int32_t PanelWidth(int32_t m, int32_t n)
{
    const int32_t k = std::min(m, n);
    return std::max<int32_t>(1, std::min(QR_ALIGN_BD * QR_COLUMNS, k));
}

/// workspace 布局（单位：float 元素个数）。host 算偏移、kernel 只接收偏移不自行推导 ——
/// 两侧各写一份布局公式是这个 kernel 上踩过的坑（越界写进相邻缓冲 = 挂死而非算错）。
struct WsLayout
{
    int32_t panelWidth = 0;
    int32_t ldy = 0;
    int64_t offY = 0;       // 未归一化的 v：ldy x panelWidth（列主序）
    int64_t offT = 0;       // -T：panelWidth x panelWidth（行主序，ld = jb）
    int64_t offPanelW = 0;  // 面板内交换 T 行的暂存
    int64_t offW = 0;       // 尾随更新的 W / W2（复用同一块，见下）
    int64_t total = 0;      // 总元素数，即 Lwork
};

WsLayout PlanWorkspace(int32_t m, int32_t n)
{
    WsLayout ws;
    ws.panelWidth = PanelWidth(m, n);
    ws.ldy = static_cast<int32_t>(AlignUp64(m, QR_LDY_ALIGN));

    int64_t cur = 0;
    auto take = [&cur](int64_t elems) {
        const int64_t off = cur;
        cur += AlignUp64(elems, QR_WS_ALIGN);
        return off;
    };
    ws.offY = take(static_cast<int64_t>(ws.ldy) * ws.panelWidth);
    ws.offT = take(static_cast<int64_t>(ws.panelWidth) * ws.panelWidth);
    ws.offPanelW = take(QR_PANEL_W_ELEMS);
    // 尾随更新的三个 GEMM：W = Y^T C 与 W2 = (-T)W **复用同一块**。
    // 安全前提是 jb <= 256（L1 的 N 方向只有一个 tile ⇒ 每个输出行块由同一个核读一次写一次）。
    // 当前 jb = 64 恒成立；**若将来把面板宽度抬到 > 256，必须拆成两块**。
    ws.offW = take(static_cast<int64_t>(ws.panelWidth) * std::max(m, n));
    ws.total = cur;
    return ws;
}

/// 入参校验失败的统一出口（实现在 utils/host_common.h，五个算子共用）
#define QR_PARAM_CHECK(cond, idx, msg) SOLVER_PARAM_CHECK(devInfo, cond, idx, msg)

/// m / n / lda 的合法性，geqrf 与 _bufferSize 共用（paramBase 用于区分两者的序号）
aclsolverStatus_t CheckShape(int32_t m, int32_t n, int32_t lda, int32_t *badIdx)
{
    if (m < 0)
    {
        *badIdx = 1;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (n < 0)
    {
        *badIdx = 2;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    // LAPACK 约定：lda >= max(1, m)。**不要求** lda == m（任务书 §2.1 第 1 条明文禁止）
    if (lda < std::max<int32_t>(1, m))
    {
        *badIdx = 4;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

} // namespace

aclsolverStatus_t aclsolverSgeqrf_bufferSize(aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork)
{
    (void)handle; // 尺寸查询与 handle/stream 无关，对齐 cuSolver 允许在未绑定 stream 时调用
    // A 不参与尺寸推导（既不读内容也不读地址），因此允许传 nullptr：
    // 调用方常在分配 A 之前先查 workspace 尺寸，强制非空会让这种正常用法失败。
    (void)A;
    SOLVER_ECHECK(Lwork != nullptr, "aclsolverSgeqrf_bufferSize invalid param 5: Lwork is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    int32_t badIdx = 0;
    const aclsolverStatus_t st =
        CheckShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(lda), &badIdx);
    SOLVER_ECHECK(st == ACLSOLVER_STATUS_SUCCESS,
                  "aclsolverSgeqrf_bufferSize invalid param: 需满足 m >= 0、n >= 0、lda >= max(1, m)。", st);
    if (m == 0 || n == 0)
    {
        *Lwork = 0; // 空问题不需要 workspace
        return ACLSOLVER_STATUS_SUCCESS;
    }
    const WsLayout ws = PlanWorkspace(static_cast<int32_t>(m), static_cast<int32_t>(n));
    SOLVER_ECHECK(ws.total <= INT32_MAX, "aclsolverSgeqrf_bufferSize: workspace 元素数超出 int32 范围。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    *Lwork = static_cast<int>(ws.total);
    return ACLSOLVER_STATUS_SUCCESS;
}

aclsolverStatus_t aclsolverSgeqrf(aclsolverHandle_t handle, int m, int n, float *A, int lda, float *TAU,
                                  float *Workspace, int Lwork, int *devInfo)
{
    // devInfo 自身为空时无处写 -i，只能直接报错（任务书参数表：空指针报错）
    SOLVER_ECHECK(devInfo != nullptr, "aclsolverSgeqrf invalid param 8: devInfo is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(handle != nullptr, "aclsolverSgeqrf: handle is nullptr.", ACLSOLVER_STATUS_HANDLE_IS_NULLPTR);

    const int32_t mi = static_cast<int32_t>(m);
    const int32_t ni = static_cast<int32_t>(n);
    const int32_t ldai = static_cast<int32_t>(lda);

    int32_t badIdx = 0;
    if (CheckShape(mi, ni, ldai, &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        (void)WriteInfoSync(devInfo, -badIdx);
        SOLVER_ECHECK(false, "aclsolverSgeqrf invalid param: 需满足 m >= 0、n >= 0、lda >= max(1, m)。",
                      ACLSOLVER_STATUS_INVALID_VALUE);
    }

    // 空问题（m == 0 或 n == 0）按成功返回，info = 0，不下发 kernel（任务书 §2.4）
    if (mi == 0 || ni == 0)
    {
        const aclError rt = WriteInfoSync(devInfo, 0);
        SOLVER_ECHECK(rt == ACL_SUCCESS, "aclsolverSgeqrf: 写 devInfo 失败。", ACLSOLVER_STATUS_MAPPING_ERROR);
        return ACLSOLVER_STATUS_SUCCESS;
    }

    QR_PARAM_CHECK(A != nullptr, 3, "aclsolverSgeqrf invalid param 3: A is nullptr.");
    QR_PARAM_CHECK(TAU != nullptr, 5, "aclsolverSgeqrf invalid param 5: TAU is nullptr.");

    const WsLayout ws = PlanWorkspace(mi, ni);
    QR_PARAM_CHECK(ws.total <= INT32_MAX, 7, "aclsolverSgeqrf: workspace 元素数超出 int32 范围。");
    QR_PARAM_CHECK(Workspace != nullptr, 6, "aclsolverSgeqrf invalid param 6: Workspace is nullptr.");
    QR_PARAM_CHECK(static_cast<int64_t>(Lwork) >= ws.total, 7,
                   "aclsolverSgeqrf invalid param 7: Lwork 小于 _bufferSize 返回的尺寸。");

    // 面板分解要求每核的 columns 条列整列驻留 UB，故 m 有上限；超限明确报不支持，不做静默截断。
    // （全部验收用例 m <= 4096；columns=2 时本实现支持到 8192）
    if (mi > MaxRowsSupported())
    {
        (void)WriteInfoSync(devInfo, -1);
        SOLVER_ECHECK(false, "aclsolverSgeqrf: m 超过当前 kernel 支持的行数上限（见本文件 MaxRowsSupported）。",
                      ACLSOLVER_STATUS_NOT_SUPPORTED);
    }

    aclrtStream stream = nullptr;
    const aclsolverStatus_t stRt = aclsolverGetStream(handle, &stream);
    SOLVER_ECHECK(stRt == ACLSOLVER_STATUS_SUCCESS, "aclsolverSgeqrf: aclsolverGetStream 失败。", stRt);

    // 成功路径上 devInfo 置 0 **交给 kernel 自己写**，不在 host 侧下 aclrtMemsetAsync：
    // 那是流上一条独立任务，实测 min 8.9 us，而小形状的整体地板只有 25 us（36% 都是它）。
    const auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    SOLVER_ECHECK(platform != nullptr, "aclsolverSgeqrf: 取平台信息失败。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    // MIX kernel 的 blockDim 以 AIC 块数为单位（每块配 2 个 AIV）
    const uint32_t numBlocks = platform->GetCoreNumAic();
    SOLVER_ECHECK(numBlocks > 0, "aclsolverSgeqrf: AIC 核数为 0。", ACLSOLVER_STATUS_INTERNAL_ERROR);

    sgeqrf_kernel_do(numBlocks, stream, reinterpret_cast<GM_ADDR>(A), reinterpret_cast<GM_ADDR>(TAU),
                     reinterpret_cast<GM_ADDR>(Workspace), mi, ni, ldai, ws.ldy, QR_COLUMNS, QR_ALIGN_BD, ws.offY,
                     ws.offT, ws.offPanelW, ws.offW, reinterpret_cast<GM_ADDR>(devInfo));

    return ACLSOLVER_STATUS_SUCCESS;
}
