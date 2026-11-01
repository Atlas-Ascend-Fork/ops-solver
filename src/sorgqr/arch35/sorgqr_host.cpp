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
 * \file sorgqr_host.cpp
 * \brief aclsolverSorgqr / aclsolverSorgqr_bufferSize 的 Host 侧实现
 *
 * 语义对齐 cusolverDnSorgqr：由 aclsolverSgeqrf 输出的 Householder 表示生成显式 Q 的前 n 列，
 * 原地覆盖 A。列主序、任意 lda（禁止要求 lda == m）、workspace 以**元素个数**计、
 * devInfo 为 device 侧 int32（0 成功 / -i 表示第 i 个参数非法，不计 handle）。
 *
 * 参数序号（任务书 §2.1 第 3 条，不计 handle）：
 *   aclsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo)
 *                           1  2  3  4   5    6     7     8       9
 *   aclsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
 *                                      1  2  3  4   5    6     7
 */

// 五个 host 都要的那批头（标准库、acl、assert/gm_addr/align_up、platform）收在这里
#include "../../utils/host_common.h"

#include "../kernel/orgqr_launch.h"

namespace
{
// 块宽度。实测 nb=64 优于 128（4096³ 上 128 要 17.9 ms 超预算、64 只要 12.5 ms）。
constexpr int32_t ORGQR_NB = 64;
// kernel 侧 UB：单列要整列驻留（物化 V 时每列一次 DataCopyPad），与 kernel 的 MAX_ROWS 一致
constexpr int32_t ORGQR_MAX_ROWS = 4096;
// workspace 各区段起始按 512B（128 个 float）对齐
constexpr int64_t ORGQR_WS_ALIGN = 128;
// Y 的 leading dimension 按 64 行对齐，匹配 L1TileShape 的 M 方向分形
constexpr int32_t ORGQR_LDY_ALIGN = 64;
// 分 K 的份数上限。必须 >= kernel 里的 KSPLIT_MAX；kernel 只用 capP 做 clamp，
// 给小了只是少分几份（慢但正确），给大了只是多占 workspace。
constexpr int32_t ORGQR_KSPLIT_MAX = 32;


/// workspace 布局（单位：float 元素个数）。host 算偏移、kernel 只接收不重算。
struct WsLayout
{
    int32_t nb = 0;
    int32_t ldy = 0;
    int64_t offY = 0;  // 物化的 V：ldy x nb（列主序，对角 1、上方 0）
    int64_t offT = 0;  // -T：nb x nb（行主序，ld = jb）
    int64_t offG = 0;  // G = V^T V：nb x nb（行主序）
    int64_t offW = 0;  // LARFB 的 W / W2（复用同一块，jb <= 256 时安全）
    int64_t offP = 0;  // 分 K 的 partial 区
    int64_t capP = 0;  // partial 区容量（元素）。**host 是唯一来源**，kernel 只拿它 clamp 份数
    int64_t total = 0;
};

WsLayout PlanWorkspace(int32_t m, int32_t n, int32_t k)
{
    WsLayout ws;
    ws.nb = std::max<int32_t>(1, std::min(ORGQR_NB, std::max(k, 1)));
    ws.ldy = static_cast<int32_t>(AlignUp64(m, ORGQR_LDY_ALIGN));
    int64_t cur = 0;
    auto take = [&cur](int64_t elems) {
        const int64_t off = cur;
        cur += AlignUp64(elems, ORGQR_WS_ALIGN);
        return off;
    };
    ws.offY = take(static_cast<int64_t>(ws.ldy) * ws.nb);
    ws.offT = take(static_cast<int64_t>(ws.nb) * ws.nb);
    ws.offG = take(static_cast<int64_t>(ws.nb) * ws.nb);
    // W = Y^T C 与 W2 = (-T)W 复用同一块：L1 的 N 方向只有一个 tile（jb <= 256）时，
    // 每个输出行块由同一个核读一次写一次，原地安全。若将来把 nb 抬到 > 256 必须拆开。
    ws.offW = take(static_cast<int64_t>(ws.nb) * std::max(m, n));
    // 分 K 的 partial 区：第 s 份是一整个 (cols x jb) 的 W，连续存放。
    // 份数 S = ceil(AIC/ceil(cols/256)) ⇒ S*cols <= ORGQR_KSPLIT_MAX*256 + cols，
    // 所以容量按 nb*(KSPLIT_MAX*256 + max(n, nb)) 给一定够（G = V^T V 的 nb*nb*S 更小）。
    ws.capP = static_cast<int64_t>(ws.nb) * (ORGQR_KSPLIT_MAX * 256 + std::max(n, ws.nb));
    ws.offP = take(ws.capP);
    ws.total = cur;
    return ws;
}

/// 入参校验失败的统一出口（实现在 utils/host_common.h，五个算子共用）
#define ORGQR_PARAM_CHECK(cond, idx, msg) SOLVER_PARAM_CHECK(devInfo, cond, idx, msg)

/// m / n / k / lda 的合法性（§2.4.2：m>=0、0<=n<=m、0<=k<=n、lda>=max(1,m)）
aclsolverStatus_t CheckShape(int32_t m, int32_t n, int32_t k, int32_t lda, int32_t *badIdx)
{
    if (m < 0)
    {
        *badIdx = 1;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (n < 0 || n > m)
    {
        *badIdx = 2;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (k < 0 || k > n)
    {
        *badIdx = 3;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    // LAPACK 约定：lda >= max(1, m)。**不要求** lda == m（任务书 §2.1 第 1 条明文禁止）
    if (lda < std::max<int32_t>(1, m))
    {
        *badIdx = 5;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

} // namespace

aclsolverStatus_t aclsolverSorgqr_bufferSize(aclsolverHandle_t handle, int m, int n, int k, const float *A, int lda,
                                             const float *tau, int *lwork)
{
    (void)handle; // 尺寸查询与 handle/stream 无关
    // A / tau 不参与尺寸推导（既不读内容也不读地址），允许传 nullptr：
    // 调用方常在分配之前先查 workspace 尺寸，强制非空会让这种正常用法失败。
    (void)A;
    (void)tau;
    SOLVER_ECHECK(lwork != nullptr, "aclsolverSorgqr_bufferSize invalid param 7: lwork is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    int32_t badIdx = 0;
    if (CheckShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
                   static_cast<int32_t>(lda), &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        SOLVER_ECHECK(false, "aclsolverSorgqr_bufferSize invalid param: 需满足 m >= 0、0 <= n <= m、0 <= k <= n、lda >= max(1, m)。",
                      ACLSOLVER_STATUS_INVALID_VALUE);
    }
    if (m == 0 || n == 0)
    {
        *lwork = 0;
        return ACLSOLVER_STATUS_SUCCESS;
    }
    const WsLayout ws = PlanWorkspace(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    SOLVER_ECHECK(ws.total <= INT32_MAX, "aclsolverSorgqr_bufferSize: workspace 元素数超出 int32 范围。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    *lwork = static_cast<int>(ws.total);
    return ACLSOLVER_STATUS_SUCCESS;
}

aclsolverStatus_t aclsolverSorgqr(aclsolverHandle_t handle, int m, int n, int k, float *A, int lda, const float *tau,
                                  float *work, int lwork, int *devInfo)
{
    SOLVER_ECHECK(devInfo != nullptr, "aclsolverSorgqr invalid param 9: devInfo is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(handle != nullptr, "aclsolverSorgqr: handle is nullptr.", ACLSOLVER_STATUS_HANDLE_IS_NULLPTR);

    const int32_t mi = static_cast<int32_t>(m);
    const int32_t ni = static_cast<int32_t>(n);
    const int32_t ki = static_cast<int32_t>(k);
    const int32_t ldai = static_cast<int32_t>(lda);

    int32_t badIdx = 0;
    if (CheckShape(mi, ni, ki, ldai, &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        (void)WriteInfoSync(devInfo, -badIdx);
        SOLVER_ECHECK(false, "aclsolverSorgqr invalid param: 需满足 m >= 0、0 <= n <= m、0 <= k <= n、lda >= max(1, m)。", ACLSOLVER_STATUS_INVALID_VALUE);
    }

    // 空问题（m == 0 或 n == 0）按成功返回、不下发 kernel。
    // 注意 **k == 0 不是空问题**：此时 Q = 单位阵的前 n 列，仍须真实写出（对齐 LAPACK dorgqr）。
    if (mi == 0 || ni == 0)
    {
        const aclError rt = WriteInfoSync(devInfo, 0);
        SOLVER_ECHECK(rt == ACL_SUCCESS, "aclsolverSorgqr: 写 devInfo 失败。", ACLSOLVER_STATUS_MAPPING_ERROR);
        return ACLSOLVER_STATUS_SUCCESS;
    }

    ORGQR_PARAM_CHECK(A != nullptr, 4, "aclsolverSorgqr invalid param 4: A is nullptr.");
    ORGQR_PARAM_CHECK(ki == 0 || tau != nullptr, 6, "aclsolverSorgqr invalid param 6: tau is nullptr while k > 0.");

    const WsLayout ws = PlanWorkspace(mi, ni, ki);
    ORGQR_PARAM_CHECK(ws.total <= INT32_MAX, 8, "aclsolverSorgqr: workspace 元素数超出 int32 范围。");
    ORGQR_PARAM_CHECK(work != nullptr, 7, "aclsolverSorgqr invalid param 7: work is nullptr.");
    ORGQR_PARAM_CHECK(static_cast<int64_t>(lwork) >= ws.total, 8,
                      "aclsolverSorgqr invalid param 8: lwork 小于 _bufferSize 返回的尺寸。");

    if (mi > ORGQR_MAX_ROWS)
    {
        (void)WriteInfoSync(devInfo, -1);
        SOLVER_ECHECK(false, "aclsolverSorgqr: m 超过当前 kernel 支持的行数上限 ORGQR_MAX_ROWS。",
                      ACLSOLVER_STATUS_NOT_SUPPORTED);
    }

    aclrtStream stream = nullptr;
    const aclsolverStatus_t stRt = aclsolverGetStream(handle, &stream);
    SOLVER_ECHECK(stRt == ACLSOLVER_STATUS_SUCCESS, "aclsolverSorgqr: aclsolverGetStream 失败。", stRt);

    // 成功路径上 devInfo 置 0 **交给 kernel 自己写**，不在 host 侧下 aclrtMemsetAsync：
    // 那是流上一条独立任务，实测 min 8.9 us，而小形状的整体地板只有 25 us（36% 都是它）。
    const auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    SOLVER_ECHECK(platform != nullptr, "aclsolverSorgqr: 取平台信息失败。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    // MIX kernel 的 blockDim 以 AIC 块数为单位（每块配 2 个 AIV）
    const uint32_t numBlocks = platform->GetCoreNumAic();
    SOLVER_ECHECK(numBlocks > 0, "aclsolverSorgqr: AIC 核数为 0。", ACLSOLVER_STATUS_INTERNAL_ERROR);

    sorgqr_kernel_do(numBlocks, stream, reinterpret_cast<GM_ADDR>(A),
                     reinterpret_cast<GM_ADDR>(const_cast<float *>(tau)), reinterpret_cast<GM_ADDR>(work), mi, ni, ki,
                     ldai, ws.ldy, ws.nb, ws.offY, ws.offT, ws.offG, ws.offW, ws.offP, ws.capP, reinterpret_cast<GM_ADDR>(devInfo));

    return ACLSOLVER_STATUS_SUCCESS;
}
