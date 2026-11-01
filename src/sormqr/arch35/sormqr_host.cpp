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
 * \file sormqr_host.cpp
 * \brief aclsolverSormqr / aclsolverSormqr_bufferSize 的 Host 侧实现
 *
 * 语义对齐 cusolverDnSormqr：把 aclsolverSgeqrf 输出的 (A, tau) 所表示的 op(Q)
 * 左乘（side = LEFT）或右乘（side = RIGHT）到 C，C 原地更新。
 * 实数下 trans = T 与 C 等价，两者都走转置路径。
 * 列主序、任意 lda/ldc（禁止要求 lda == nq）、workspace 以**元素个数**计、
 * devInfo 为 device 侧 int32（0 成功 / -i 表示第 i 个参数非法，不计 handle）。
 *
 * 参数序号（与 LAPACK dormqr 的实参顺序一致，不计 handle）：
 *   aclsolverSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
 *                           1     2      3  4  5  6   7    8   9   10    11    12     13
 *   aclsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
 *                                      1     2      3  4  5  6   7    8   9   10    11
 */

// 五个 host 都要的那批头（标准库、acl、assert/gm_addr/align_up、platform）收在这里
#include "../../utils/host_common.h"

#include "../kernel/ormqr_launch.h"

namespace
{
// 块宽度。实测 nb=64 优于 128（128 档 2048³ 反而超预算）。
constexpr int32_t ORMQR_NB = 64;
// kernel 侧 UB：物化 V 时单列整列驻留，与 kernel 的 MAX_ROWS 一致。约束落在 nq 上。
constexpr int32_t ORMQR_MAX_ROWS = 4096;
// workspace 各区段起始按 512B（128 个 float）对齐
constexpr int64_t ORMQR_WS_ALIGN = 128;
// Y 的 leading dimension 按 64 行对齐，匹配 L1TileShape 的 M 方向分形
constexpr int32_t ORMQR_LDY_ALIGN = 64;
// 分 K 的份数上限。必须 >= kernel 里的 KSPLIT_MAX；kernel 只用 capP 做 clamp，
// 给小了只是少分几份（慢但正确），给大了只是多占 workspace。
constexpr int32_t ORMQR_KSPLIT_MAX = 32;


/// workspace 布局（单位：float 元素个数）。host 算偏移、kernel 只接收不重算。
struct WsLayout
{
    int32_t nb = 0;
    int32_t ldy = 0;
    int64_t offY = 0;   // 物化的 V 块：ldy x nb（列主序，对角 1、上方 0）
    int64_t offT = 0;   // -T：nb x nb（行主序，ld = jb）
    int64_t offG = 0;   // G = V^T V：nb x nb（行主序）
    int64_t offW = 0;   // W  = V^T C（LEFT 下 jb x n、RIGHT 下 m x jb）
    int64_t offW2 = 0;  // W2 = (-T) W
    int64_t offP = 0;   // 分 K 的 partial 区
    int64_t capP = 0;   // partial 区容量（元素）。**host 是唯一来源**，kernel 只拿它 clamp 份数
    int64_t total = 0;
};

/// nq = op(Q) 的阶数，也是 V 的行数（LEFT 时 m、RIGHT 时 n）
inline int32_t Nq(bool isLeft, int32_t m, int32_t n)
{
    return isLeft ? m : n;
}

WsLayout PlanWorkspace(bool isLeft, int32_t m, int32_t n, int32_t k)
{
    WsLayout ws;
    const int32_t nq = Nq(isLeft, m, n);
    const int32_t mx = std::max(std::max(m, n), 1);
    ws.nb = std::max<int32_t>(1, std::min(ORMQR_NB, std::max(k, 1)));
    ws.ldy = static_cast<int32_t>(AlignUp64(std::max(nq, 1), ORMQR_LDY_ALIGN));
    int64_t cur = 0;
    auto take = [&cur](int64_t elems) {
        const int64_t off = cur;
        cur += AlignUp64(elems, ORMQR_WS_ALIGN);
        return off;
    };
    ws.offY = take(static_cast<int64_t>(ws.ldy) * ws.nb);
    ws.offT = take(static_cast<int64_t>(ws.nb) * ws.nb);
    ws.offG = take(static_cast<int64_t>(ws.nb) * ws.nb);
    // W 与 W2 必须分开：W2 = (-T)W 的那次 GEMM 在读 W 的同时写 W2，输出 tile 与输入
    // 行块不是一一对应（分 K 时更不是），原地会读到半成品。
    ws.offW = take(static_cast<int64_t>(ws.nb) * mx);
    ws.offW2 = take(static_cast<int64_t>(ws.nb) * mx);
    // 分 K 的 partial 区：第 s 份是一整份 W。份数 S = ceil(AIC/tiles) 的量级，
    // 满足 S*mx <= ORMQR_KSPLIT_MAX*256 + mx ⇒ 容量按下式给一定够
    // （G = V^T V 用的 nb*nb*S 更小）。
    // 实测过把容量抬到 8*nb*mx（S 上限 3→8）：4096³ 只值 -2%、2048³ 反而 +1.5%，
    // 多分的份被归约流量吃回去了 ⇒ 保持这个紧的容量。
    ws.capP = static_cast<int64_t>(ws.nb) * (ORMQR_KSPLIT_MAX * 256 + mx);
    ws.offP = take(ws.capP);
    ws.total = cur;
    return ws;
}

/// 入参校验失败的统一出口（实现在 utils/host_common.h，五个算子共用）
#define ORMQR_PARAM_CHECK(cond, idx, msg) SOLVER_PARAM_CHECK(devInfo, cond, idx, msg)

/// 枚举与维数的合法性，判据照 LAPACK dormqr（含参数序号）
aclsolverStatus_t CheckShape(aclsolverSideMode_t side, aclsolverOperation_t trans, int32_t m, int32_t n, int32_t k,
                             int32_t lda, int32_t ldc, int32_t *badIdx)
{
    if (side != ACLSOLVER_SIDE_LEFT && side != ACLSOLVER_SIDE_RIGHT)
    {
        *badIdx = 1;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    // 实数下 T 与 C 等价，但枚举本身仍要校验：只接受 N / T / C 三个值
    if (trans != ACLSOLVER_OP_N && trans != ACLSOLVER_OP_T && trans != ACLSOLVER_OP_C)
    {
        *badIdx = 2;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (m < 0)
    {
        *badIdx = 3;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (n < 0)
    {
        *badIdx = 4;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    const int32_t nq = Nq(side == ACLSOLVER_SIDE_LEFT, m, n);
    if (k < 0 || k > nq)
    {
        *badIdx = 5;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    // LAPACK 约定：lda >= max(1, nq)、ldc >= max(1, m)。**不要求**等号
    if (lda < std::max<int32_t>(1, nq))
    {
        *badIdx = 7;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max<int32_t>(1, m))
    {
        *badIdx = 10;
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

} // namespace

aclsolverStatus_t aclsolverSormqr_bufferSize(aclsolverHandle_t handle, aclsolverSideMode_t side,
                                             aclsolverOperation_t trans, int m, int n, int k, const float *A, int lda,
                                             const float *tau, const float *C, int ldc, int *lwork)
{
    (void)handle; // 尺寸查询与 handle/stream 无关
    // A / tau / C 不参与尺寸推导（既不读内容也不读地址），允许传 nullptr：
    // 调用方常在分配之前先查 workspace 尺寸，强制非空会让这种正常用法失败。
    (void)A;
    (void)tau;
    (void)C;
    SOLVER_ECHECK(lwork != nullptr, "aclsolverSormqr_bufferSize invalid param 11: lwork is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    int32_t badIdx = 0;
    if (CheckShape(side, trans, static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k),
                   static_cast<int32_t>(lda), static_cast<int32_t>(ldc), &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        SOLVER_ECHECK(false, "aclsolverSormqr_bufferSize invalid param: 需满足 side/trans 为合法枚举、m >= 0、n >= 0、k >= 0、k <= nq、lda >= max(1, nq)、ldc >= max(1, m)。",
                      ACLSOLVER_STATUS_INVALID_VALUE);
    }
    // k == 0 时 op(Q) = I，C 不变 ⇒ 不下发 kernel，也不需要 workspace
    if (m == 0 || n == 0 || k == 0)
    {
        *lwork = 0;
        return ACLSOLVER_STATUS_SUCCESS;
    }
    const WsLayout ws = PlanWorkspace(side == ACLSOLVER_SIDE_LEFT, static_cast<int32_t>(m), static_cast<int32_t>(n),
                                      static_cast<int32_t>(k));
    SOLVER_ECHECK(ws.total <= INT32_MAX, "aclsolverSormqr_bufferSize: workspace 元素数超出 int32 范围。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    *lwork = static_cast<int>(ws.total);
    return ACLSOLVER_STATUS_SUCCESS;
}

namespace {
/// 取流与核数，然后把 kernel 下发出去。
///
/// 成功路径上 devInfo 置 0 **交给 kernel 自己写**，不在 host 侧下 aclrtMemsetAsync：
/// 那是流上一条独立任务，实测 min 8.9 us，而小形状的整体地板只有 25 us（36% 都是它）。
aclsolverStatus_t SormqrLaunch(aclsolverHandle_t handle, bool isLeft, aclsolverOperation_t trans, int32_t mi,
                               int32_t ni, int32_t ki, int32_t ldai, int32_t ldci, const float *A, const float *tau,
                               float *C, float *work, int *devInfo, const WsLayout &ws)
{
    aclrtStream stream = nullptr;
    const aclsolverStatus_t stRt = aclsolverGetStream(handle, &stream);
    SOLVER_ECHECK(stRt == ACLSOLVER_STATUS_SUCCESS, "aclsolverSormqr: aclsolverGetStream 失败。", stRt);

    const auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    SOLVER_ECHECK(platform != nullptr, "aclsolverSormqr: 取平台信息失败。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    // MIX kernel 的 blockDim 以 AIC 块数为单位（每块配 2 个 AIV）
    const uint32_t numBlocks = platform->GetCoreNumAic();
    SOLVER_ECHECK(numBlocks > 0, "aclsolverSormqr: AIC 核数为 0。", ACLSOLVER_STATUS_INTERNAL_ERROR);

    // 实数下 ACLSOLVER_OP_C 与 OP_T 等价（共轭转置 = 转置），kernel 只认 0/1
    const uint32_t transFlag = (trans == ACLSOLVER_OP_N) ? 0u : 1u;
    sormqr_kernel_do(numBlocks, stream, reinterpret_cast<GM_ADDR>(const_cast<float *>(A)),
                     reinterpret_cast<GM_ADDR>(const_cast<float *>(tau)), reinterpret_cast<GM_ADDR>(C),
                     reinterpret_cast<GM_ADDR>(work), reinterpret_cast<GM_ADDR>(devInfo), mi, ni, ki, ldai, ldci,
                     ws.nb, ws.ldy, isLeft ? 0u : 1u,
                     transFlag, ws.offY, ws.offT, ws.offG, ws.offW, ws.offW2, ws.offP, ws.capP);

    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace

aclsolverStatus_t aclsolverSormqr(aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                  int m, int n, int k, const float *A, int lda, const float *tau, float *C, int ldc,
                                  float *work, int lwork, int *devInfo)
{
    SOLVER_ECHECK(devInfo != nullptr, "aclsolverSormqr invalid param 13: devInfo is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(handle != nullptr, "aclsolverSormqr: handle is nullptr.", ACLSOLVER_STATUS_HANDLE_IS_NULLPTR);

    const int32_t mi = static_cast<int32_t>(m);
    const int32_t ni = static_cast<int32_t>(n);
    const int32_t ki = static_cast<int32_t>(k);
    const int32_t ldai = static_cast<int32_t>(lda);
    const int32_t ldci = static_cast<int32_t>(ldc);

    int32_t badIdx = 0;
    if (CheckShape(side, trans, mi, ni, ki, ldai, ldci, &badIdx) != ACLSOLVER_STATUS_SUCCESS)
    {
        (void)WriteInfoSync(devInfo, -badIdx);
        SOLVER_ECHECK(false, "aclsolverSormqr invalid param: 需满足 side/trans 为合法枚举、m >= 0、n >= 0、k >= 0、k <= nq、lda >= max(1, nq)、ldc >= max(1, m)。", ACLSOLVER_STATUS_INVALID_VALUE);
    }

    const bool isLeft = (side == ACLSOLVER_SIDE_LEFT);

    // 空问题：m/n 为 0 没有元素可写；**k == 0 时 op(Q) = I，C 原样就是答案**，
    // 同样不下发 kernel（与 orgqr 不同 —— orgqr 的 k==0 还要写出单位列）。
    if (mi == 0 || ni == 0 || ki == 0)
    {
        const aclError rt = WriteInfoSync(devInfo, 0);
        SOLVER_ECHECK(rt == ACL_SUCCESS, "aclsolverSormqr: 写 devInfo 失败。", ACLSOLVER_STATUS_MAPPING_ERROR);
        return ACLSOLVER_STATUS_SUCCESS;
    }

    ORMQR_PARAM_CHECK(A != nullptr, 6, "aclsolverSormqr invalid param 6: A is nullptr.");
    ORMQR_PARAM_CHECK(tau != nullptr, 8, "aclsolverSormqr invalid param 8: tau is nullptr while k > 0.");
    ORMQR_PARAM_CHECK(C != nullptr, 9, "aclsolverSormqr invalid param 9: C is nullptr.");

    const WsLayout ws = PlanWorkspace(isLeft, mi, ni, ki);
    ORMQR_PARAM_CHECK(ws.total <= INT32_MAX, 12, "aclsolverSormqr: workspace 元素数超出 int32 范围。");
    ORMQR_PARAM_CHECK(work != nullptr, 11, "aclsolverSormqr invalid param 11: work is nullptr.");
    ORMQR_PARAM_CHECK(static_cast<int64_t>(lwork) >= ws.total, 12,
                      "aclsolverSormqr invalid param 12: lwork 小于 _bufferSize 返回的尺寸。");

    const int32_t nq = Nq(isLeft, mi, ni);
    if (nq > ORMQR_MAX_ROWS)
    {
        (void)WriteInfoSync(devInfo, isLeft ? -3 : -4);
        SOLVER_ECHECK(false,
                      "aclsolverSormqr: nq（LEFT 时为 m、RIGHT 时为 n）超过当前 kernel 支持的上限 ORMQR_MAX_ROWS。",
                      ACLSOLVER_STATUS_NOT_SUPPORTED);
    }

    return SormqrLaunch(handle, isLeft, trans, mi, ni, ki, ldai, ldci, A, tau, C, work, devInfo, ws);
}
