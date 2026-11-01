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
 * \file sgels_host.cpp
 * \brief aclsolverSSgels / aclsolverSSgels_bufferSize 的 Host 侧实现
 *
 * 语义对齐 cusolverDnSSgels：解超定/方阵最小二乘 min ||A X - B||，A 为 m x n（m >= n），
 * 走 QR 路线三段：
 *   1. aclsolverSgeqrf(A)            —— A 原地变成 R 与反射子（任务书允许破坏 dA）
 *   2. aclsolverSormqr(LEFT, OP_T)   —— 把 Q^T 作用到 B 的副本上
 *   3. 上三角回代 R X = (Q^T B)(1:n, :)，结果写进独立的 dX
 *
 * 几个刻意的实现选择：
 *   - 前两段**直接调本仓的公开接口**，不另写一份 workspace 布局：布局公式写两遍是这类
 *     kernel 上踩过的坑（越界写进相邻缓冲表现为挂死而不是算错）。
 *   - **dB 是只读的**（任务书 §2.4.5 把它列为"输入"），所以先把 B 整块拷进 workspace 再
 *     让 ormqr 原地作用，绝不在用户的 B 上动手。
 *   - `niter` 是 **host 标量**：本实现是直接 QR 解、不做迭代精化，恒置 0。
 *   - 三个 kernel 连续下发**不在中间插同步**（实测连续下发结果与"中间同步"逐位相同，
 *     见 test/sormqr 的 chain 模式）。
 *
 * 参数序号（不计 handle）：
 *   aclsolverSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niter, dinfo)
 *                           1  2  3     4    5     6    7     8    9     10          11          12     13
 */

// 五个 host 都要的那批头（标准库、acl、assert/gm_addr/align_up、platform）收在这里
#include "../../utils/host_common.h"

#include "../kernel/trsm_launch.h"

namespace
{
// 回代的块宽。与 kernel 里的 MAX_NB 一致，且必须是 8 的倍数（UB 里按 nb 做行距）
constexpr int32_t GELS_TRSM_NB = 64;
// 任务书 §2.6：m ∈ [n, 4096]，n ∈ [1, 4096]，nrhs ∈ [1, 128]
constexpr int32_t GELS_MAX_DIM = 4096;
// workspace 各区段起始按 512B（128 个 float）对齐
constexpr int64_t GELS_WS_ALIGN = 128;
// 分 K 的份数上限，必须 >= kernel 里的 KSPLIT_MAX
constexpr int64_t GELS_KSPLIT_MAX = 32;


/// workspace 布局（单位：float 元素个数；对外的查询值换成字节）
struct WsLayout
{
    int64_t offTau = 0;   // min(m,n) = n 个反射子系数
    int64_t offBw = 0;    // B 的副本：lddb x nrhs（dB 只读，不能原地做）
    int64_t offPhase = 0; // geqrf / ormqr / 回代 三段**复用**同一块（它们在流上串行）
    int64_t phase = 0;    // 这块的容量（元素）
    int64_t offW = 0;     // 回代的 W：nb x nrhs（相对 workspace 起点）
    int64_t offP = 0;     // 回代的分 K partial 区
    int64_t capP = 0;
    int64_t total = 0;
};

/// 三段里最大的那个 workspace 需求。前两段直接问本仓的公开查询接口，不另写一份公式。
aclsolverStatus_t PlanWorkspace(aclsolverHandle_t handle, int32_t m, int32_t n, int32_t nrhs, int32_t ldda,
                               int32_t lddb, WsLayout *ws)
{
    int lwG = 0;
    const aclsolverStatus_t s1 = aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, ldda, &lwG);
    if (s1 != ACLSOLVER_STATUS_SUCCESS)
    {
        return s1;
    }
    int lwM = 0;
    const aclsolverStatus_t s2 = aclsolverSormqr_bufferSize(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, m, nrhs, n,
                                                            nullptr, ldda, nullptr, nullptr, lddb, &lwM);
    if (s2 != ACLSOLVER_STATUS_SUCCESS)
    {
        return s2;
    }
    const int32_t nb = std::min(GELS_TRSM_NB, std::max(n, 1));
    const int64_t nbAligned = AlignUp64(nb, 8);
    const int64_t capP = GELS_KSPLIT_MAX * std::max<int64_t>(nrhs, 1) * nbAligned;
    const int64_t trsmNeed = AlignUp64(nbAligned * std::max<int64_t>(nrhs, 1), GELS_WS_ALIGN) + capP;

    int64_t cur = 0;
    auto take = [&cur](int64_t elems) {
        const int64_t off = cur;
        cur += AlignUp64(elems, GELS_WS_ALIGN);
        return off;
    };
    ws->offTau = take(std::max<int32_t>(n, 1));
    ws->offBw = take(static_cast<int64_t>(lddb) * std::max<int32_t>(nrhs, 1));
    ws->phase = std::max<int64_t>(std::max<int64_t>(lwG, lwM), trsmNeed);
    ws->offPhase = take(ws->phase);
    // 回代的两段落在复用区里（它与 geqrf/ormqr 的 workspace 不会同时存活）
    ws->offW = ws->offPhase;
    ws->offP = ws->offPhase + AlignUp64(nbAligned * std::max<int64_t>(nrhs, 1), GELS_WS_ALIGN);
    ws->capP = capP;
    ws->total = cur;
    return ACLSOLVER_STATUS_SUCCESS;
}

/// 入参校验失败的统一出口（实现在 utils/host_common.h，五个算子共用）
#define GELS_PARAM_CHECK(cond, idx, msg) SOLVER_PARAM_CHECK(dinfo, cond, idx, msg)

/// 维数与 leading dimension 的合法性（§2.4.5：m >= n >= 0、nrhs >= 0、
/// ldda >= max(1,m)、lddb >= max(1,m)、lddx >= max(1,n)）
aclsolverStatus_t CheckShape(int32_t m, int32_t n, int32_t nrhs, int32_t ldda, int32_t lddb, int32_t lddx,
                            int32_t *badIdx)
{
    if (m < 0 || m < n)
    {
        *badIdx = 1; // m 这一行同时管 "m >= n"
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (n < 0)
    {
        *badIdx = 2;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (nrhs < 0)
    {
        *badIdx = 3;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (ldda < std::max<int32_t>(1, m))
    {
        *badIdx = 5;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (lddb < std::max<int32_t>(1, m))
    {
        *badIdx = 7;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (lddx < std::max<int32_t>(1, n))
    {
        *badIdx = 9;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

} // namespace

aclsolverStatus_t aclsolverSSgels_bufferSize(aclsolverHandle_t handle, int m, int n, int nrhs, float *dA, int ldda,
                                             float *dB, int lddb, float *dX, int lddx, void *dwork,
                                             size_t *lwork_bytes)
{
    // dA / dB / dX / dwork 不参与尺寸推导（既不读内容也不读地址），允许传 nullptr：
    // 调用方常在分配之前先查 workspace 尺寸。
    (void)dA;
    (void)dB;
    (void)dX;
    (void)dwork;
    SOLVER_ECHECK(lwork_bytes != nullptr, "aclsolverSSgels_bufferSize invalid param 11: lwork_bytes is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(handle != nullptr, "aclsolverSSgels_bufferSize: handle is nullptr.",
                  ACLSOLVER_STATUS_HANDLE_IS_NULLPTR);
    int32_t badIdx = 0;
    if (CheckShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(nrhs),
                   static_cast<int32_t>(ldda), static_cast<int32_t>(lddb), static_cast<int32_t>(lddx),
                   &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        SOLVER_ECHECK(false, "aclsolverSSgels_bufferSize invalid param: 需满足 m >= n >= 0、nrhs >= 0、ldda >= max(1, m)、lddb >= max(1, m)、lddx >= max(1, n)。",
                      ACLSOLVER_STATUS_INVALID_VALUE);
    }
    if (m == 0 || n == 0 || nrhs == 0)
    {
        *lwork_bytes = 0;
        return ACLSOLVER_STATUS_SUCCESS;
    }
    WsLayout ws;
    const aclsolverStatus_t st = PlanWorkspace(handle, static_cast<int32_t>(m), static_cast<int32_t>(n),
                                               static_cast<int32_t>(nrhs), static_cast<int32_t>(ldda),
                                               static_cast<int32_t>(lddb), &ws);
    if (st != ACLSOLVER_STATUS_SUCCESS)
    {
        return st;
    }
    *lwork_bytes = static_cast<size_t>(ws.total) * sizeof(float);
    return ACLSOLVER_STATUS_SUCCESS;
}

namespace {
/// 三段串起来：B 拷副本 -> geqrf -> ormqr(Q^T) -> 回代。前两段直接调公开接口。
aclsolverStatus_t GelsRunPhases(aclsolverHandle_t handle, aclrtStream stream, int32_t mi, int32_t ni, int32_t ri,
                                int32_t lda, int32_t ldb, int32_t ldx, float *dA, float *dB, float *dX,
                                void *dWorkspace, const WsLayout &ws, int *dinfo)
{
    float *const work = reinterpret_cast<float *>(dWorkspace);
    float *const tau = work + ws.offTau;
    float *const Bw = work + ws.offBw;
    float *const phase = work + ws.offPhase;
    const int lwPhase = static_cast<int>(std::min<int64_t>(ws.phase, INT32_MAX));

    // 1. B 整块拷进 workspace（保持 ld = lddb，一次 memcpy 就够；逐列拷在 nrhs 大时
    //    会变成 nrhs 条流上任务，每条约 9 us）
    const aclError cp = aclrtMemcpyAsync(Bw, static_cast<size_t>(ldb) * ri * sizeof(float), dB,
                                         static_cast<size_t>(ldb) * ri * sizeof(float),
                                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    SOLVER_ECHECK(cp == ACL_SUCCESS, "aclsolverSSgels: 拷贝 B 失败。", ACLSOLVER_STATUS_MAPPING_ERROR);

    // 2. A = Q R（原地，任务书允许破坏 dA）
    const aclsolverStatus_t st1 = aclsolverSgeqrf(handle, mi, ni, dA, lda, tau, phase, lwPhase, dinfo);
    SOLVER_ECHECK(st1 == ACLSOLVER_STATUS_SUCCESS, "aclsolverSSgels: 第 1 段 geqrf 失败。", st1);

    // 3. Bw <- Q^T Bw（只动副本）
    const aclsolverStatus_t st2 = aclsolverSormqr(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, mi, ri, ni, dA, lda,
                                                  tau, Bw, ldb, phase, lwPhase, dinfo);
    SOLVER_ECHECK(st2 == ACLSOLVER_STATUS_SUCCESS, "aclsolverSSgels: 第 2 段 ormqr 失败。", st2);

    // 4. 回代 R X = Bw(1:n, :)
    const auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    SOLVER_ECHECK(platform != nullptr, "aclsolverSSgels: 取平台信息失败。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    const uint32_t numBlocks = platform->GetCoreNumAic();
    SOLVER_ECHECK(numBlocks > 0, "aclsolverSSgels: AIC 核数为 0。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    const int32_t nb = std::min(GELS_TRSM_NB, std::max(ni, 1));
    strsm_kernel_do(numBlocks, stream, reinterpret_cast<GM_ADDR>(dA), reinterpret_cast<GM_ADDR>(Bw),
                    reinterpret_cast<GM_ADDR>(dX), reinterpret_cast<GM_ADDR>(dWorkspace), ni, ri, lda, ldb, ldx,
                    static_cast<int32_t>(AlignUp64(nb, 8)), ws.offW, ws.offP, ws.capP);

    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace

aclsolverStatus_t aclsolverSSgels(aclsolverHandle_t handle, int m, int n, int nrhs, float *dA, int ldda, float *dB,
                                  int lddb, float *dX, int lddx, void *dWorkspace, size_t lwork_bytes, int *niter,
                                  int *dinfo)
{
    SOLVER_ECHECK(dinfo != nullptr, "aclsolverSSgels invalid param 13: dinfo is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(handle != nullptr, "aclsolverSSgels: handle is nullptr.", ACLSOLVER_STATUS_HANDLE_IS_NULLPTR);
    GELS_PARAM_CHECK(niter != nullptr, 12, "aclsolverSSgels invalid param 12: niter is nullptr.");
    // 本实现是直接 QR 解，不做迭代精化 ⇒ 迭代次数恒为 0（host 标量，立即可读）
    *niter = 0;

    const int32_t mi = static_cast<int32_t>(m);
    const int32_t ni = static_cast<int32_t>(n);
    const int32_t ri = static_cast<int32_t>(nrhs);
    const int32_t lda = static_cast<int32_t>(ldda);
    const int32_t ldb = static_cast<int32_t>(lddb);
    const int32_t ldx = static_cast<int32_t>(lddx);

    int32_t badIdx = 0;
    if (CheckShape(mi, ni, ri, lda, ldb, ldx, &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        (void)WriteInfoSync(dinfo, -badIdx);
        SOLVER_ECHECK(false, "aclsolverSSgels invalid param: 需满足 m >= n >= 0、nrhs >= 0、ldda >= max(1, m)、lddb >= max(1, m)、lddx >= max(1, n)。", ACLSOLVER_STATUS_INVALID_VALUE);
    }

    // 空问题：没有方程、没有未知量、或没有右端项 ⇒ 没有要写的解
    if (mi == 0 || ni == 0 || ri == 0)
    {
        const aclError rt = WriteInfoSync(dinfo, 0);
        SOLVER_ECHECK(rt == ACL_SUCCESS, "aclsolverSSgels: 写 dinfo 失败。", ACLSOLVER_STATUS_MAPPING_ERROR);
        return ACLSOLVER_STATUS_SUCCESS;
    }

    GELS_PARAM_CHECK(dA != nullptr, 4, "aclsolverSSgels invalid param 4: dA is nullptr.");
    GELS_PARAM_CHECK(dB != nullptr, 6, "aclsolverSSgels invalid param 6: dB is nullptr.");
    GELS_PARAM_CHECK(dX != nullptr, 8, "aclsolverSSgels invalid param 8: dX is nullptr.");
    // dX 必须是独立输出：与 dB 同址时 ormqr 会把用户的 B 写坏，而 §2.4.5 把 dB 列为"输入"
    GELS_PARAM_CHECK(dX != dB, 8, "aclsolverSSgels invalid param 8: dX 不能与 dB 同址。");
    // dX 也不能与 dA 同址：回代第 0 步先把 Q^T B 整块拷进 X，若同址就把 dA 里的 R 覆盖掉，
    // 之后读到的"R"全是解，静默算错。dA 虽被文档化为"内容会被破坏"，但不等于可做输出缓冲。
    GELS_PARAM_CHECK(dX != dA, 8, "aclsolverSSgels invalid param 8: dX 不能与 dA 同址。");
    GELS_PARAM_CHECK(mi <= GELS_MAX_DIM && ni <= GELS_MAX_DIM, 1,
                     "aclsolverSSgels: m / n 超过当前 kernel 上限 4096。");

    WsLayout ws;
    const aclsolverStatus_t stWs = PlanWorkspace(handle, mi, ni, ri, lda, ldb, &ws);
    SOLVER_ECHECK(stWs == ACLSOLVER_STATUS_SUCCESS, "aclsolverSSgels: 推导 workspace 失败。", stWs);
    const size_t need = static_cast<size_t>(ws.total) * sizeof(float);
    GELS_PARAM_CHECK(dWorkspace != nullptr, 10, "aclsolverSSgels invalid param 10: dWorkspace is nullptr.");
    GELS_PARAM_CHECK(lwork_bytes >= need, 11,
                     "aclsolverSSgels invalid param 11: lwork_bytes 小于 _bufferSize 返回的字节数。");

    aclrtStream stream = nullptr;
    const aclsolverStatus_t stRt = aclsolverGetStream(handle, &stream);
    SOLVER_ECHECK(stRt == ACLSOLVER_STATUS_SUCCESS, "aclsolverSSgels: aclsolverGetStream 失败。", stRt);

    return GelsRunPhases(handle, stream, mi, ni, ri, lda, ldb, ldx, dA, dB, dX, dWorkspace, ws, dinfo);
}
