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
 * \file xlarft_test.cpp
 * \brief aclsolverXlarft 的端到端测试：判据、确定性、参数校验、FORWARD/BACKWARD 覆盖
 *
 * ★ 判据是**自定的**：任务书 §3.2.1 没有给 Xlarft 的精度判据，也没有 GPU 基线。
 *   口径写在这里，报告里要照抄：
 *       err       = 被引用三角里 |T_dev - T_ref64| 的最大值
 *       ratio     = err / ( k * max(1, max|T_ref64|) * eps_f32 )
 *       ratio_cpu = 同一公式，但 T 换成**同算法的 fp32 CPU 实现**
 *       PASS      <=> ratio <= max(5 * ratio_cpu, 1.0)
 *   参考实现是 float64 的 LAPACK slarft 等价实现（qr_core 的 Larft / LarftBackward），
 *   与被测 kernel 是两套独立算法（kernel 走 G = V^T V + 按行分核递推）。
 *   另外两条结构性要求也进判据：
 *     - 被引用三角必须**全部写到**（T 事先投毒 -7）；
 *     - 另一半三角必须**原样不动**（LAPACK slarft 不碰它）。
 *
 * 性能只记录绝对耗时，不做达标判定（没有 GPU 基线可比）。
 *
 * 模式：
 *   xlarft_test            功能 + 参数校验 + 一组形状（FORWARD/BACKWARD 各一遍）
 *   xlarft_test perf       记录几个形状的绝对耗时（含任务书 P-08 / P-09 两条）
 */

// acl/acl.h、cann_ops_solver.h 以及 algorithm / chrono / random / string / vector
// 这批都由 qr_test_utils.h 统一提供，这里只列本文件额外要的。
#include "../utils/qr_test_utils.h"
#include <cmath>
#include "../../src/sgeqrf/kernel/householder.hpp"

namespace
{
struct Buffers
{
    uint8_t *V = nullptr;
    uint8_t *tau = nullptr;
    uint8_t *T = nullptr;
    uint8_t *ws = nullptr;
    void Free()
    {
        if (V) aclrtFree(V);
        if (tau) aclrtFree(tau);
        if (T) aclrtFree(T);
        if (ws) aclrtFree(ws);
        V = tau = T = ws = nullptr;
    }
};

/// 造一组合法输入。FORWARD 直接用 CPU geqrf 的输出；
/// BACKWARD 把同一批反射子**上下左右翻转**（V_b(r,c) = V_f(n-1-r, k-1-c)、tau 反序），
/// 这样隐含 1 自然落在第 n-k+c 行、下方为 0，而且元素量级还是反射子分量的量级。
/// ★ 不要拿 geqrf 结果的上三角（那是 R）去填 BACKWARD 的 V 上方：T 会按 (2n)^k 爆掉，
///   fp32 直接 inf/nan，连 CPU 参考一起 nan，很容易误判成 kernel 算错。
void MakeInput(bool backward, int n, int k, int ldv, int caseIdx, std::vector<float> &V, std::vector<float> &tau)
{
    std::mt19937 rng(0x1a2b0000u + static_cast<uint32_t>(caseIdx));
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    std::vector<float> fac(static_cast<size_t>(ldv) * std::max(k, 1), 0.0f);
    for (auto &x : fac)
    {
        x = d(rng);
    }
    std::vector<float> tauF(static_cast<size_t>(std::max(std::min(n, k), 1)), 0.0f);
    if (n > 0 && k > 0)
    {
        QrCore::Geqr2(n, k, fac.data(), ldv, tauF.data());
    }
    V.assign(static_cast<size_t>(ldv) * std::max(k, 1), 0.0f);
    for (int c = 0; c < k; ++c)
    {
        for (int r = 0; r < n; ++r)
        {
            if (!backward)
            {
                V[static_cast<size_t>(c) * ldv + r] = fac[static_cast<size_t>(c) * ldv + r];
            }
            else
            {
                const int rf = n - 1 - r;
                const int cf = k - 1 - c;
                V[static_cast<size_t>(c) * ldv + r] = (rf > cf) ? fac[static_cast<size_t>(cf) * ldv + rf] : 0.0f;
            }
        }
    }
    tau.assign(tauF.begin(), tauF.begin() + std::max(k, 0));
    if (backward)
    {
        std::reverse(tau.begin(), tau.end());
    }
}

/// 一个被测形状。把 RunShape 那几段之间要来回传的标量收在一起，免得每段都挂一长串形参。
struct XCase
{
    bool backward;
    int64_t n, k, ldv, ldt;
    aclsolverDirectMode_t Direct() const
    {
        return backward ? ACLSOLVER_DIRECT_BACKWARD : ACLSOLVER_DIRECT_FORWARD;
    }
    size_t TBytes() const
    {
        return static_cast<size_t>(ldt) * static_cast<size_t>(k) * sizeof(float);
    }
};

/// T 的投毒值：既能看出"该写的没写"，也能看出"不该写的被写了"
constexpr float kPoison = -7.0f;

/// 查 workspace 大小并把输入搬上设备。hostBytes 预置非 0，接口必须把它改写成 0。
bool XlarftAlloc(aclsolverHandle_t handle, const XCase &c, const std::vector<float> &Vh,
                 const std::vector<float> &tauH, Buffers *buf, size_t *devBytes)
{
    size_t hostBytes = 123;
    if (aclsolverXlarft_bufferSize(handle, nullptr, c.Direct(), ACLSOLVER_STOREV_COLUMNWISE, c.n, c.k,
                                   ACLSOLVER_R_32F, nullptr, c.ldv, ACLSOLVER_R_32F, nullptr, ACLSOLVER_R_32F,
                                   nullptr, c.ldt, ACLSOLVER_R_32F, devBytes, &hostBytes)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  bufferSize 失败\n");
        return false;
    }
    if (hostBytes != 0)
    {
        printf("  host workspace 应为 0，实得 %zu\n", hostBytes);
        return false;
    }
    const size_t vBytes = Vh.size() * sizeof(float);
    const size_t tauBytes = std::max<size_t>(tauH.size(), 1) * sizeof(float);
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->V), vBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->tau), tauBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->T), c.TBytes(), ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->ws), std::max<size_t>(*devBytes, 1),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMemcpy(buf->V, vBytes, Vh.data(), vBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    TEST_CHECK(aclrtMemcpy(buf->tau, tauBytes, tauH.data(), tauH.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    return true;
}

/// 跑 runs 次：每次先投毒 T 再下发，顺带比跨次 bit-wise 一致（§3.5 的确定性要求）
bool XlarftRepeat(aclsolverHandle_t handle, aclrtStream stream, const XCase &c, const Buffers &buf, size_t devBytes,
                  int runs, std::vector<float> *Tdev, double *bestUs)
{
    const std::vector<float> poison(static_cast<size_t>(c.ldt) * c.k, kPoison);
    std::vector<float> first;
    *bestUs = 1e30;
    for (int rep = 0; rep < runs; ++rep)
    {
        TEST_CHECK(aclrtMemcpy(buf.T, c.TBytes(), poison.data(), c.TBytes(), ACL_MEMCPY_HOST_TO_DEVICE));
        TEST_CHECK(aclrtSynchronizeStream(stream));
        const auto t0 = std::chrono::steady_clock::now();
        const aclsolverStatus_t st = aclsolverXlarft(
            handle, nullptr, c.Direct(), ACLSOLVER_STOREV_COLUMNWISE, c.n, c.k, ACLSOLVER_R_32F, buf.V, c.ldv,
            ACLSOLVER_R_32F, buf.tau, ACLSOLVER_R_32F, buf.T, c.ldt, ACLSOLVER_R_32F, buf.ws, devBytes, nullptr, 0);
        const aclError sync = aclrtSynchronizeStream(stream);
        const auto t1 = std::chrono::steady_clock::now();
        if (st != ACLSOLVER_STATUS_SUCCESS || sync != ACL_SUCCESS)
        {
            printf("  status=%d syncRc=%d\n", (int)st, (int)sync);
            return false;
        }
        *bestUs = std::min(*bestUs, std::chrono::duration<double, std::micro>(t1 - t0).count());
        Tdev->assign(poison.size(), 0.0f);
        TEST_CHECK(aclrtMemcpy(Tdev->data(), c.TBytes(), buf.T, c.TBytes(), ACL_MEMCPY_DEVICE_TO_HOST));
        if (rep == 0)
        {
            first = *Tdev;
        }
        else if (first != *Tdev)
        {
            printf("  第 %d 次与第 1 次结果不同（确定性失败）\n", rep + 1);
            return false;
        }
    }
    return true;
}

/// 参考：float64 的 LAPACK slarft 等价实现；另跑一遍同式 fp32 版，当判据阈值的标尺
void XlarftReference(const XCase &c, std::vector<float> &Vh, std::vector<float> &tauH, std::vector<double> *Tref,
                     std::vector<float> *Tf32)
{
    const int ni = static_cast<int>(c.n), ki = static_cast<int>(c.k);
    const int ldvi = static_cast<int>(c.ldv), ldti = static_cast<int>(c.ldt);
    std::vector<double> Vd(Vh.begin(), Vh.end()), taud(tauH.begin(), tauH.end());
    Tref->assign(static_cast<size_t>(c.ldt) * c.k, 0.0);
    Tf32->assign(static_cast<size_t>(c.ldt) * c.k, 0.0f);
    if (c.backward)
    {
        QrCore::LarftBackward<double>(ni, ki, Vd.data(), ldvi, taud.data(), Tref->data(), ldti);
        QrCore::LarftBackward<float>(ni, ki, Vh.data(), ldvi, tauH.data(), Tf32->data(), ldti);
    }
    else
    {
        QrCore::Larft<double>(ni, ki, Vd.data(), ldvi, taud.data(), Tref->data(), ldti);
        QrCore::Larft<float>(ni, ki, Vh.data(), ldvi, tauH.data(), Tf32->data(), ldti);
    }
}

/// 判据：ratio = max|T_dev - T_ref| / (k·max(scale,1)·eps)，阈值 max(5·ratio_cpu, 1)。
/// 另外要求被引用的三角里没有残留投毒值（未写），且三角外一个都没被动过（越界）。
bool XlarftVerify(const XCase &c, const std::vector<float> &Tdev, const std::vector<double> &Tref,
                  const std::vector<float> &Tf32, bool verbose)
{
    const int ki = static_cast<int>(c.k), ldti = static_cast<int>(c.ldt);
    double errDev = 0.0, errCpu = 0.0, scale = 0.0;
    int untouched = 0, dirty = 0;
    for (int col = 0; col < ki; ++col)
    {
        for (int r = 0; r < ki; ++r)
        {
            const size_t idx = static_cast<size_t>(col) * ldti + r;
            const bool referenced = c.backward ? (r >= col) : (r <= col);
            if (!referenced)
            {
                if (Tdev[idx] != kPoison)
                {
                    ++dirty;
                }
                continue;
            }
            scale = std::max(scale, std::fabs(Tref[idx]));
            errDev = std::max(errDev, std::fabs(static_cast<double>(Tdev[idx]) - Tref[idx]));
            errCpu = std::max(errCpu, std::fabs(static_cast<double>(Tf32[idx]) - Tref[idx]));
            if (Tdev[idx] == kPoison && Tref[idx] != static_cast<double>(kPoison))
            {
                ++untouched;
            }
        }
    }
    const double den = static_cast<double>(ki) * std::max(scale, 1.0) * kEpsF32;
    const double ratio = SafeDiv(errDev, den);
    const double ratioCpu = SafeDiv(errCpu, den);
    const double th = std::max(5.0 * ratioCpu, 1.0);
    const bool pass = (ratio <= th) && (untouched == 0) && (dirty == 0);
    if (verbose)
    {
        printf("  n=%-5lld k=%-4lld %s ldv=%-5lld ldt=%-4lld ratio=%.4f ratio_cpu=%.4f th=%.4f 未写=%d 越界=%d %s\n",
               (long long)c.n, (long long)c.k, c.backward ? "B" : "F", (long long)c.ldv, (long long)c.ldt, ratio,
               ratioCpu, th, untouched, dirty, pass ? "PASS" : "**FAIL**");
    }
    return pass;
}

/// 跑一个形状：被测 vs float64 参考，runs 次重复用于 bit-wise 确定性
bool RunShape(aclsolverHandle_t handle, aclrtStream stream, int caseIdx, bool backward, int64_t n, int64_t k,
              int64_t ldv, int64_t ldt, int runs, bool verbose, double *usOut)
{
    const XCase c{backward, n, k, ldv, ldt};
    std::vector<float> Vh, tauH;
    MakeInput(backward, static_cast<int>(n), static_cast<int>(k), static_cast<int>(ldv), caseIdx, Vh, tauH);

    Buffers buf;
    size_t devBytes = 0;
    if (!XlarftAlloc(handle, c, Vh, tauH, &buf, &devBytes))
    {
        buf.Free();
        return false;
    }

    std::vector<float> Tdev;
    double bestUs = 1e30;
    const bool ran = XlarftRepeat(handle, stream, c, buf, devBytes, runs, &Tdev, &bestUs);
    buf.Free();
    if (!ran)
    {
        return false;
    }
    if (usOut != nullptr)
    {
        *usOut = bestUs;
    }

    std::vector<double> Tref;
    std::vector<float> Tf32;
    XlarftReference(c, Vh, tauH, &Tref, &Tf32);
    return XlarftVerify(c, Tdev, Tref, Tf32, verbose);
}

// ---- 参数校验 ----
// 一组固定的合法入参，各组检查只改其中被测的那一个。
constexpr int64_t kBpN = 64;
constexpr int64_t kBpK = 16;
constexpr int64_t kBpLdv = 64;
constexpr int64_t kBpLdt = 16;
constexpr aclsolverDirectMode_t kFwd = ACLSOLVER_DIRECT_FORWARD;
constexpr aclsolverStorevMode_t kColw = ACLSOLVER_STOREV_COLUMNWISE;
constexpr aclsolverDataType_t kR32 = ACLSOLVER_R_32F;

/// 带固定缓冲发一次 Xlarft。三个数据指针恒取 buf 里的，签名只留被测参数。
aclsolverStatus_t CallXlarft(const Buffers &buf, aclsolverHandle_t h, aclsolverParams_t p, aclsolverDirectMode_t d,
                             aclsolverStorevMode_t sv, int64_t nn, int64_t kk, aclsolverDataType_t dv, int64_t lv,
                             aclsolverDataType_t dt, aclsolverDataType_t dT, int64_t lt, aclsolverDataType_t dc,
                             void *wsp, size_t wsb)
{
    return aclsolverXlarft(h, p, d, sv, nn, kk, dv, buf.V, lv, dt, buf.tau, dT, buf.T, lt, dc, wsp, wsb, nullptr, 0);
}

/// handle / params / direct / storev 四个枚举与句柄位
void CheckHandleAndModes(aclsolverHandle_t handle, const Buffers &buf, size_t devBytes, bool *ok)
{
    ExpectStatus("handle=nullptr",
                 CallXlarft(buf, nullptr, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, ok);
    ExpectStatus("params 非 nullptr",
                 CallXlarft(buf, handle, reinterpret_cast<aclsolverParams_t>(0x1), kFwd, kColw, kBpN, kBpK, kR32,
                            kBpLdv, kR32, kR32, kBpLdt, kR32, buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("direct 非法",
                 CallXlarft(buf, handle, nullptr, static_cast<aclsolverDirectMode_t>(7), kColw, kBpN, kBpK, kR32,
                            kBpLdv, kR32, kR32, kBpLdt, kR32, buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("storev 非 COLUMNWISE（ROWWISE）",
                 CallXlarft(buf, handle, nullptr, kFwd, static_cast<aclsolverStorevMode_t>(1), kBpN, kBpK, kR32,
                            kBpLdv, kR32, kR32, kBpLdt, kR32, buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
}

/// 形状与 ld 的取值范围
void CheckShapeRanges(aclsolverHandle_t handle, const Buffers &buf, size_t devBytes, bool *ok)
{
    ExpectStatus("n<0",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, -1, kBpK, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("k<0",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, -1, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("n<k",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, 8, 16, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32, buf.ws,
                            devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("ldv<n",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpN - 1, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("ldt<k",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, kR32, kBpK - 1, kR32,
                            buf.ws, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
}

/// 非 FLOAT32 的四元组、以及超出 kernel 上限的形状，一律 NOT_SUPPORTED（不是 INVALID_VALUE）
void CheckNotSupported(aclsolverHandle_t handle, const Buffers &buf, size_t devBytes, bool *ok)
{
    const auto C32 = ACLSOLVER_C_32F;
    const auto want = ACLSOLVER_STATUS_NOT_SUPPORTED;
    ExpectStatus("dataTypeV 非 FLOAT32 -> NOT_SUPPORTED",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, C32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 want, ok);
    ExpectStatus("dataTypeTau 非 FLOAT32 -> NOT_SUPPORTED",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, C32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 want, ok);
    ExpectStatus("dataTypeT 非 FLOAT32 -> NOT_SUPPORTED",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, C32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 want, ok);
    ExpectStatus("computeType 非 FLOAT32 -> NOT_SUPPORTED",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, kR32, kBpLdt, C32,
                            buf.ws, devBytes),
                 want, ok);
    ExpectStatus("k>256 -> NOT_SUPPORTED",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, 512, 257, kR32, 512, kR32, kR32, 257, kR32, buf.ws,
                            devBytes),
                 want, ok);
    ExpectStatus("n>4096 -> NOT_SUPPORTED",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, 4097, 16, kR32, 4097, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 want, ok);
}

/// workspace 与三个数据指针的空指针/容量
void CheckWorkspaceAndPointers(aclsolverHandle_t handle, const Buffers &buf, size_t devBytes, bool *ok)
{
    ExpectStatus("bufferOnDevice=nullptr",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            nullptr, devBytes),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("device workspace 太小",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes - 1),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("合法调用",
                 CallXlarft(buf, handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, kBpLdv, kR32, kR32, kBpLdt, kR32,
                            buf.ws, devBytes),
                 ACLSOLVER_STATUS_SUCCESS, ok);
    (void)aclrtSynchronizeStream(nullptr);
    ExpectStatus("V=nullptr",
                 aclsolverXlarft(handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, nullptr, kBpLdv, kR32, buf.tau,
                                 kR32, buf.T, kBpLdt, kR32, buf.ws, devBytes, nullptr, 0),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("tau=nullptr",
                 aclsolverXlarft(handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, buf.V, kBpLdv, kR32, nullptr, kR32,
                                 buf.T, kBpLdt, kR32, buf.ws, devBytes, nullptr, 0),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("T=nullptr",
                 aclsolverXlarft(handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, buf.V, kBpLdv, kR32, buf.tau, kR32,
                                 nullptr, kBpLdt, kR32, buf.ws, devBytes, nullptr, 0),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
}

/// 查询接口自己的两个出参不能为空
void CheckQueryNullptrs(aclsolverHandle_t handle, bool *ok)
{
    size_t one = 0;
    ExpectStatus("bufferSize hostBytes=nullptr",
                 aclsolverXlarft_bufferSize(handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, nullptr, kBpLdv, kR32,
                                            nullptr, kR32, nullptr, kBpLdt, kR32, &one, nullptr),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("bufferSize devBytes=nullptr",
                 aclsolverXlarft_bufferSize(handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, nullptr, kBpLdv, kR32,
                                            nullptr, kR32, nullptr, kBpLdt, kR32, nullptr, &one),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
}

/// 参数校验：Xlarft 没有 devInfo，只看返回码
bool RunBadParams(aclsolverHandle_t handle)
{
    size_t devBytes = 0, hostBytes = 0;
    if (aclsolverXlarft_bufferSize(handle, nullptr, kFwd, kColw, kBpN, kBpK, kR32, nullptr, kBpLdv, kR32, nullptr,
                                   kR32, nullptr, kBpLdt, kR32, &devBytes, &hostBytes)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  bufferSize 失败\n");
        return false;
    }
    Buffers buf;
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.V), static_cast<size_t>(kBpLdv) * kBpK * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.tau), static_cast<size_t>(kBpK) * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.T), static_cast<size_t>(kBpLdt) * kBpK * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.ws), std::max<size_t>(devBytes, 1),
                           ACL_MEM_MALLOC_HUGE_FIRST));

    bool ok = true;
    CheckHandleAndModes(handle, buf, devBytes, &ok);
    CheckShapeRanges(handle, buf, devBytes, &ok);
    CheckNotSupported(handle, buf, devBytes, &ok);
    CheckWorkspaceAndPointers(handle, buf, devBytes, &ok);
    CheckQueryNullptrs(handle, &ok);
    buf.Free();
    return ok;
}

/// 空问题：n=0 或 k=0 时成功返回、workspace 查询为 0
bool RunEmpty(aclsolverHandle_t handle, int64_t n, int64_t k)
{
    size_t devBytes = 7, hostBytes = 7;
    const auto F = ACLSOLVER_DIRECT_FORWARD;
    const auto R32 = ACLSOLVER_R_32F;
    if (aclsolverXlarft_bufferSize(handle, nullptr, F, ACLSOLVER_STOREV_COLUMNWISE, n, k, R32, nullptr,
                                   std::max<int64_t>(n, 1), R32, nullptr, R32, nullptr, std::max<int64_t>(k, 1), R32,
                                   &devBytes, &hostBytes) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  空问题 bufferSize 失败\n");
        return false;
    }
    const aclsolverStatus_t st =
        aclsolverXlarft(handle, nullptr, F, ACLSOLVER_STOREV_COLUMNWISE, n, k, R32, nullptr,
                        std::max<int64_t>(n, 1), R32, nullptr, R32, nullptr, std::max<int64_t>(k, 1), R32, nullptr, 0,
                        nullptr, 0);
    const bool ok = (st == ACLSOLVER_STATUS_SUCCESS) && (devBytes == 0) && (hostBytes == 0);
    printf("  空问题 n=%lld k=%lld: status=%d devBytes=%zu hostBytes=%zu  %s\n", (long long)n, (long long)k, (int)st,
           devBytes, hostBytes, ok ? "PASS" : "FAIL");
    return ok;
}

/// 绝对耗时模式：任务包里没有 xlarft 的 GPU 基线，所以只报耗时、不做达标判定
void RunPerfMode(aclsolverHandle_t handle, aclrtStream stream)
{
    printf("== Xlarft 绝对耗时（任务书无 GPU 基线，**不做达标判定**）==\n");
    struct P
    {
        int64_t n, k;
        bool backward;
        const char *tag;
    };
    const P ps[] = {{4096, 128, false, "P-08"}, {2048, 256, false, "P-09"}, {4096, 256, false, ""},
                    {2048, 64, false, ""},      {256, 64, false, ""},       {4096, 128, true, ""},
                    {2048, 256, true, ""},      {1, 1, false, ""}};
    int idx = 0;
    for (const auto &p : ps)
    {
        double us = 0.0;
        ++idx;
        const bool ok = RunShape(handle, stream, idx, p.backward, p.n, p.k, p.n, p.k, 4, false, &us);
        printf("  %-6s n=%-5lld k=%-4lld %s  %9.2f us  %s\n", p.tag, (long long)p.n, (long long)p.k,
               p.backward ? "BACKWARD" : "FORWARD ", us, ok ? "" : "**结果不对**");
    }
}

/// 形状覆盖：每条形状 FORWARD / BACKWARD 各跑一遍，每遍内部再跑 2 次比 bit-wise
void RunShapeSweep(aclsolverHandle_t handle, aclrtStream stream, int *pass, int *total)
{
    printf("== 形状覆盖（FORWARD / BACKWARD 各一遍，每条跑 2 次比 bit-wise）==\n");
    struct Shape
    {
        int64_t n, k, ldvExtra, ldtExtra;
    };
    const Shape shapes[] = {
        {1, 1, 0, 0},      {2, 2, 0, 0},     {8, 4, 0, 0},      {16, 16, 0, 0},   {37, 11, 5, 3},
        {64, 64, 0, 0},    {65, 33, 7, 7},   {100, 37, 5, 3},   {128, 128, 0, 0}, {256, 64, 0, 0},
        {256, 256, 0, 0},  {257, 129, 7, 3}, {1024, 128, 0, 0}, {2048, 256, 0, 0}, {4096, 128, 0, 0},
        {4096, 256, 0, 0},
        // §3.5 入参生成规则：n 覆盖 2 的幂与 2 的幂-1；lda/ldt 再压 +8 / +32 两档 padding
        {3, 3, 0, 0},      {7, 7, 0, 0},      {15, 15, 0, 0},   {31, 31, 0, 0},
        {63, 63, 0, 0},    {127, 127, 0, 0},  {255, 255, 0, 0}, {511, 255, 0, 0},
        {1023, 256, 0, 0}, {2047, 256, 0, 0}, {4095, 256, 0, 0},
        {256, 64, 8, 8},   {256, 64, 32, 32}, {1024, 128, 8, 8}, {1024, 128, 32, 32},
        {4096, 256, 8, 8}, {4096, 256, 32, 32},
    };
    int idx = 0;
    for (const auto &s : shapes)
    {
        for (bool backward : {false, true})
        {
            ++idx;
            ++(*total);
            *pass += RunShape(handle, stream, idx, backward, s.n, s.k, s.n + s.ldvExtra, s.k + s.ldtExtra, 2, true,
                              nullptr)
                         ? 1
                         : 0;
        }
    }
}


} // namespace

int main(int argc, char **argv)
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }

    if (argc >= 2 && std::string(argv[1]) == "perf")
    {
        RunPerfMode(handle, stream);
        TeardownAcl(handle, stream);
        return 0;
    }

    int pass = 0, total = 0;
    printf("== 参数校验（Xlarft 无 devInfo，只看返回码）==\n");
    Tally(RunBadParams(handle), &pass, &total);

    printf("== 空问题 ==\n");
    for (auto t : {std::vector<int64_t>{0, 0}, {8, 0}, {0, 0}})
    {
        Tally(RunEmpty(handle, t[0], t[1]), &pass, &total);
    }

    RunShapeSweep(handle, stream, &pass, &total);

    printf("\n%d / %d PASS\n", pass, total);
    TeardownAcl(handle, stream);
    return (pass == total) ? 0 : 1;
}
