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
 * \file xlarft_host.cpp
 * \brief aclsolverXlarft / aclsolverXlarft_bufferSize 的 Host 侧实现
 *
 * 语义对齐 cusolverDnXlarft：由 k 个 Householder 反射子 (V, tau) 算出块反射子的三角因子 T，
 *   direct = FORWARD  : H_1 H_2 ... H_k = I - V T V^T，T 上三角
 *   direct = BACKWARD : H_k ... H_2 H_1 = I - V T V^T，T 下三角
 * 只写 T **被引用的那半个三角**（另一半原样不动，对齐 LAPACK slarft）。
 *
 * 与另外三个接口的几点不同（任务书 §2.1 / §2.4.4）：
 *   - 维数是 **int64_t**（对齐 cuSolver 的 64-bit API）；
 *   - workspace 以**字节**计，而且要 **device 与 host 两个**查询值；
 *   - **没有 devInfo**：所有参数错误只通过返回码报告；
 *   - `params` 允许 nullptr；本实现不提供 aclsolverCreateParams，所以**只接受 nullptr**；
 *   - 四个 dataType 必须全是 ACLSOLVER_R_32F，否则返回 **NOT_SUPPORTED**（不是 INVALID_VALUE）。
 */

// 五个 host 都要的那批头（标准库、acl、assert/gm_addr/align_up、platform）收在这里
#include "../../utils/host_common.h"
#include "../kernel/xlarft_launch.h"

namespace
{
// 任务书 §2.6：n ∈ [1, 4096]，k ∈ [1, min(n, 256)]
constexpr int64_t XLARFT_MAX_N = 4096;
constexpr int64_t XLARFT_MAX_K = 256;
// workspace 各区段起始按 512B（128 个 float）对齐
constexpr int64_t XLARFT_WS_ALIGN = 128;
// Y 的 leading dimension 按 64 行对齐，匹配 L1TileShape 的 M 方向分形
constexpr int64_t XLARFT_LDY_ALIGN = 64;
// 分 K 的份数上限。必须 >= kernel 里的 KSPLIT_MAX；kernel 只用 capP 做 clamp
constexpr int64_t XLARFT_KSPLIT_MAX = 32;

/// workspace 布局（单位：float 元素个数；对外的查询值换成字节）
struct WsLayout
{
    int64_t ldy = 0;
    int64_t offY = 0;  // 物化后的 V：ldy x k
    int64_t offG = 0;  // G = V^T V：k x k 行主序
    int64_t offP = 0;  // 分 K 的 partial 区
    int64_t capP = 0;  // partial 区容量（元素）。**host 是唯一来源**
    int64_t total = 0;
};

WsLayout PlanWorkspace(int64_t n, int64_t k)
{
    WsLayout ws;
    ws.ldy = AlignUp64(std::max<int64_t>(n, 1), XLARFT_LDY_ALIGN);
    int64_t cur = 0;
    auto take = [&cur](int64_t elems)
    {
        const int64_t off = cur;
        cur += AlignUp64(elems, XLARFT_WS_ALIGN);
        return off;
    };
    ws.offY = take(ws.ldy * std::max<int64_t>(k, 1));
    ws.offG = take(std::max<int64_t>(k, 1) * std::max<int64_t>(k, 1));
    // G 的分 K：one = k*k，S <= KSPLIT_MAX ⇒ 容量 k*k*KSPLIT_MAX 足够
    ws.capP = std::max<int64_t>(k, 1) * std::max<int64_t>(k, 1) * XLARFT_KSPLIT_MAX;
    ws.offP = take(ws.capP);
    ws.total = cur;
    return ws;
}

/// 四个 dataType 必须全是 FLOAT32（任务书：其他组合返回 NOT_SUPPORTED）
bool AllFloat32(aclsolverDataType_t dv, aclsolverDataType_t dt, aclsolverDataType_t dT, aclsolverDataType_t dc)
{
    return dv == ACLSOLVER_R_32F && dt == ACLSOLVER_R_32F && dT == ACLSOLVER_R_32F && dc == ACLSOLVER_R_32F;
}

/// 枚举、维数、leading dimension 的合法性。dtype 不合法单独返回 NOT_SUPPORTED。
aclsolverStatus_t CheckArgs(aclsolverParams_t params, aclsolverDirectMode_t direct, aclsolverStorevMode_t storev,
                            int64_t n, int64_t k, aclsolverDataType_t dv, int64_t ldv, aclsolverDataType_t dt,
                            aclsolverDataType_t dT, int64_t ldt, aclsolverDataType_t dc)
{
    // 本实现不提供 params 对象（没有 aclsolverCreateParams/DestroyParams），
    // 所以非 nullptr 一律按非法指针处理 —— 默默忽略会让调用方以为设置生效了。
    SOLVER_ECHECK(params == nullptr, "aclsolverXlarft: params 必须为 nullptr（本实现不提供 params 对象）。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(direct == ACLSOLVER_DIRECT_FORWARD || direct == ACLSOLVER_DIRECT_BACKWARD,
                  "aclsolverXlarft: direct 只能是 FORWARD 或 BACKWARD。", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(storev == ACLSOLVER_STOREV_COLUMNWISE, "aclsolverXlarft: storev 只支持 COLUMNWISE。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(n >= 0, "aclsolverXlarft: n 不能为负。", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(k >= 0, "aclsolverXlarft: k 不能为负。", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(n >= k, "aclsolverXlarft: 要求 n >= k。", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(ldv >= std::max<int64_t>(1, n), "aclsolverXlarft: ldv 小于 max(1, n)。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(ldt >= std::max<int64_t>(1, k), "aclsolverXlarft: ldt 小于 max(1, k)。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(AllFloat32(dv, dt, dT, dc), "aclsolverXlarft: 本任务只支持 FLOAT32 四元组。",
                  ACLSOLVER_STATUS_NOT_SUPPORTED);
    return ACLSOLVER_STATUS_SUCCESS;
}

}  // namespace

aclsolverStatus_t aclsolverXlarft_bufferSize(aclsolverHandle_t handle, aclsolverParams_t params,
                                             aclsolverDirectMode_t direct, aclsolverStorevMode_t storev, int64_t n,
                                             int64_t k, aclsolverDataType_t dataTypeV, const void *V, int64_t ldv,
                                             aclsolverDataType_t dataTypeTau, const void *tau,
                                             aclsolverDataType_t dataTypeT, void *T, int64_t ldt,
                                             aclsolverDataType_t computeType, size_t *workspaceInBytesOnDevice,
                                             size_t *workspaceInBytesOnHost)
{
    (void)handle;  // 尺寸查询与 handle/stream 无关
    // V / tau / T 不参与尺寸推导（既不读内容也不读地址），允许传 nullptr：
    // 调用方常在分配之前先查 workspace 尺寸。
    (void)V;
    (void)tau;
    (void)T;
    SOLVER_ECHECK(workspaceInBytesOnDevice != nullptr,
                  "aclsolverXlarft_bufferSize: workspaceInBytesOnDevice is nullptr.", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(workspaceInBytesOnHost != nullptr, "aclsolverXlarft_bufferSize: workspaceInBytesOnHost is nullptr.",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    const aclsolverStatus_t st =
        CheckArgs(params, direct, storev, n, k, dataTypeV, ldv, dataTypeTau, dataTypeT, ldt, computeType);
    if (st != ACLSOLVER_STATUS_SUCCESS)
    {
        return st;
    }
    // host 侧不需要暂存：全部计算在 device 上
    *workspaceInBytesOnHost = 0;
    if (n == 0 || k == 0)
    {
        *workspaceInBytesOnDevice = 0;
        return ACLSOLVER_STATUS_SUCCESS;
    }
    // 与执行入口同一对上限：超限的入参在尺寸推导（int64 乘法）之前拒绝，
    // 避免 ldy*k / k*k*KSPLIT_MAX 的有符号溢出 UB 误导 workspace 分配。
    SOLVER_ECHECK(n <= XLARFT_MAX_N, "aclsolverXlarft_bufferSize: n 超过当前 kernel 上限 4096。",
                  ACLSOLVER_STATUS_NOT_SUPPORTED);
    SOLVER_ECHECK(k <= XLARFT_MAX_K, "aclsolverXlarft_bufferSize: k 超过当前 kernel 上限 256。",
                  ACLSOLVER_STATUS_NOT_SUPPORTED);
    const WsLayout ws = PlanWorkspace(n, k);
    *workspaceInBytesOnDevice = static_cast<size_t>(ws.total) * sizeof(float);
    return ACLSOLVER_STATUS_SUCCESS;
}

namespace
{
/// 取流、查核数，然后把 kernel 下发出去。
///
/// MIX kernel 的 blockDim 以 AIC 块数为单位（每块配 2 个 AIV）。kernel 里 T 的行按
/// AIV 核号轮转分配、UB 只给 32 行 ⇒ 要求 2*numBlocks*32 >= k。
aclsolverStatus_t XlarftLaunch(aclsolverHandle_t handle, aclsolverDirectMode_t direct, int64_t n, int64_t k,
                               const void *V, int64_t ldv, const void *tau, void *T, int64_t ldt, void *bufferOnDevice,
                               const WsLayout &ws)
{
    aclrtStream stream = nullptr;
    const aclsolverStatus_t stRt = aclsolverGetStream(handle, &stream);
    SOLVER_ECHECK(stRt == ACLSOLVER_STATUS_SUCCESS, "aclsolverXlarft: aclsolverGetStream 失败。", stRt);

    const auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    SOLVER_ECHECK(platform != nullptr, "aclsolverXlarft: 取平台信息失败。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    const uint32_t numBlocks = platform->GetCoreNumAic();
    SOLVER_ECHECK(numBlocks > 0, "aclsolverXlarft: AIC 核数为 0。", ACLSOLVER_STATUS_INTERNAL_ERROR);
    SOLVER_ECHECK(static_cast<int64_t>(numBlocks) * 2 * 32 >= k, "aclsolverXlarft: AIV 核数不足以放下 T 的行。",
                  ACLSOLVER_STATUS_INTERNAL_ERROR);

    xlarft_kernel_do(numBlocks, stream, reinterpret_cast<GM_ADDR>(const_cast<void *>(V)),
                     reinterpret_cast<GM_ADDR>(const_cast<void *>(tau)), reinterpret_cast<GM_ADDR>(T),
                     reinterpret_cast<GM_ADDR>(bufferOnDevice), static_cast<int32_t>(n), static_cast<int32_t>(k),
                     static_cast<int32_t>(ldv), static_cast<int32_t>(ldt), static_cast<int32_t>(ws.ldy),
                     (direct == ACLSOLVER_DIRECT_BACKWARD) ? 1u : 0u, ws.offY, ws.offG, ws.offP, ws.capP);

    // Kernel launches are asynchronous; do not report success when enqueueing
    // failed. Device execution errors are still observed by stream synchronization.
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS ? ACLSOLVER_STATUS_SUCCESS
                                                                    : ACLSOLVER_STATUS_EXECUTION_FAILED;
}
}  // namespace

aclsolverStatus_t aclsolverXlarft(aclsolverHandle_t handle, aclsolverParams_t params, aclsolverDirectMode_t direct,
                                  aclsolverStorevMode_t storev, int64_t n, int64_t k, aclsolverDataType_t dataTypeV,
                                  const void *V, int64_t ldv, aclsolverDataType_t dataTypeTau, const void *tau,
                                  aclsolverDataType_t dataTypeT, void *T, int64_t ldt, aclsolverDataType_t computeType,
                                  void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost,
                                  size_t workspaceInBytesOnHost)
{
    (void)bufferOnHost;  // host workspace 尺寸恒为 0，不读
    SOLVER_ECHECK(handle != nullptr, "aclsolverXlarft: handle is nullptr.", ACLSOLVER_STATUS_HANDLE_IS_NULLPTR);
    const aclsolverStatus_t stChk =
        CheckArgs(params, direct, storev, n, k, dataTypeV, ldv, dataTypeTau, dataTypeT, ldt, computeType);
    if (stChk != ACLSOLVER_STATUS_SUCCESS)
    {
        return stChk;
    }

    // 空问题：没有反射子就没有 T 可写，直接成功返回（不下发 kernel）
    if (n == 0 || k == 0)
    {
        return ACLSOLVER_STATUS_SUCCESS;
    }

    SOLVER_ECHECK(n <= XLARFT_MAX_N, "aclsolverXlarft: n 超过当前 kernel 上限 4096。", ACLSOLVER_STATUS_NOT_SUPPORTED);
    SOLVER_ECHECK(k <= XLARFT_MAX_K, "aclsolverXlarft: k 超过当前 kernel 上限 256。", ACLSOLVER_STATUS_NOT_SUPPORTED);

    SOLVER_ECHECK(V != nullptr, "aclsolverXlarft: V is nullptr.", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(tau != nullptr, "aclsolverXlarft: tau is nullptr.", ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(T != nullptr, "aclsolverXlarft: T is nullptr.", ACLSOLVER_STATUS_INVALID_VALUE);

    const WsLayout ws = PlanWorkspace(n, k);
    const size_t need = static_cast<size_t>(ws.total) * sizeof(float);
    SOLVER_ECHECK(bufferOnDevice != nullptr, "aclsolverXlarft: bufferOnDevice is nullptr。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    SOLVER_ECHECK(workspaceInBytesOnDevice >= need,
                  "aclsolverXlarft: workspaceInBytesOnDevice 小于 _bufferSize 返回的字节数。",
                  ACLSOLVER_STATUS_INVALID_VALUE);
    (void)workspaceInBytesOnHost;

    return XlarftLaunch(handle, direct, n, k, V, ldv, tau, T, ldt, bufferOnDevice, ws);
}
