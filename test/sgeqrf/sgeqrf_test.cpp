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
 * \file sgeqrf_test.cpp
 * \brief aclsolverSgeqrf 的端到端测试：判据、确定性、参数校验的 -i 语义
 *
 * 判据取任务书 §3.2.1-1 原文：
 *     ratio = ||R - Q^T·A||_1 / ( m·||A||_1·eps )，  ratio <= max(5·ratio_cpu, 0.1) 为通过
 * ratio_cpu 用同一份 Householder 约定的 CPU 无分块 geqr2（FLOAT32）现场实测，
 * 不写死常数 —— 小形状上 CPU 参考本身的 ratio 就会超过 0.1。
 */

// acl/acl.h、cann_ops_solver.h 以及 algorithm / chrono / random / string / vector
// 这批都由 qr_test_utils.h 统一提供，这里只列本文件额外要的。
#include "../utils/qr_test_utils.h"
#include <cmath>
#include "../../src/sgeqrf/kernel/householder.hpp"

namespace
{
double Norm1Col(const std::vector<double> &a, int ld, int rows, int cols)
{
    double best = 0.0;
    for (int c = 0; c < cols; ++c)
    {
        double s = 0.0;
        for (int r = 0; r < rows; ++r)
        {
            s += std::fabs(a[static_cast<size_t>(c) * ld + r]);
        }
        best = std::max(best, s);
    }
    return best;
}

/// SQRT01 残差：ratio = max_j ( sum_i |R_ij - (Q^T A)_ij| ) / ( m*||A||_1*eps )
///
/// 不显式构造 Q —— 那是 O(k*m^2) 时间加 m^2 的 double 内存（4096 档就是 134 MB）。
/// 改成把反射子**逐列施加**：Q = H_1...H_k  =>  Q^T = H_k...H_1，
/// 对一列从 H_1 依次作用到 H_k，每列 O(k*m)。
///
/// 列数超过 maxCols 时按等步长抽样。**被抽到的列是精确的**，所以得到的是真 1-范数的
/// 下界 => 超阈一定是真失败；未超阈时调用方须标注"抽样下界"。
double Sqrt01(int m, int n, int lda, const std::vector<double> &Aorig, const std::vector<float> &Afac,
              const std::vector<float> &tauFac, int maxCols = (1 << 30), bool *exact = nullptr)
{
    const int k = std::min(m, n);
    const int step = (maxCols > 0 && n > maxCols) ? ((n + maxCols - 1) / maxCols) : 1;
    if (exact != nullptr)
    {
        *exact = (step == 1);
    }
    const double anorm = Norm1Col(Aorig, lda, m, n);
    std::vector<double> y(static_cast<size_t>(m), 0.0);
    double best = 0.0;
    for (int c = 0; c < n; c += step)
    {
        for (int r = 0; r < m; ++r)
        {
            y[r] = Aorig[static_cast<size_t>(c) * lda + r];
        }
        for (int j = 0; j < k; ++j)
        {
            const double tj = tauFac[j];
            if (tj == 0.0)
            {
                continue;
            }
            double dot = y[j];  // v_j[j] 隐含为 1
            for (int r = j + 1; r < m; ++r)
            {
                dot += static_cast<double>(Afac[static_cast<size_t>(j) * lda + r]) * y[r];
            }
            const double w = tj * dot;
            y[j] -= w;
            for (int r = j + 1; r < m; ++r)
            {
                y[r] -= w * static_cast<double>(Afac[static_cast<size_t>(j) * lda + r]);
            }
        }
        double col = 0.0;
        for (int r = 0; r < m; ++r)
        {
            const double rr = (r <= c) ? static_cast<double>(Afac[static_cast<size_t>(c) * lda + r]) : 0.0;
            col += std::fabs(rr - y[r]);
        }
        best = std::max(best, col);
    }
    return best / (static_cast<double>(m) * anorm * kEpsF32);
}

// A / tau / workspace / devInfo 这组缓冲与 sorgqr 完全一样，用共享头里的那份
using Buffers = QrBuffers;

bool GeqrfAlloc(Buffers *buf, size_t aElems, int k, int Lwork)
{
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->A), aElems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->tau), std::max(k, 1) * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    if (Lwork > 0)
    {
        TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->ws), static_cast<size_t>(Lwork) * sizeof(float),
                               ACL_MEM_MALLOC_HUGE_FIRST));
    }
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf->info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    return true;
}

/// 设备与 CPU 参考的最大逐元素差（A 只看被写的 m x n 区域，lda padding 不算）
void GeqrfMaxDiff(int m, int n, int lda, int k, const std::vector<float> &Aout, const std::vector<float> &tauOut,
                  const std::vector<float> &Aref, const std::vector<float> &tauRef, double *dA, double *dT)
{
    *dA = 0.0;
    *dT = 0.0;
    for (int c = 0; c < n; ++c)
    {
        for (int r = 0; r < m; ++r)
        {
            const size_t idx = static_cast<size_t>(c) * lda + r;
            *dA = std::max(*dA, std::fabs(static_cast<double>(Aout[idx]) - Aref[idx]));
        }
    }
    for (int t = 0; t < k; ++t)
    {
        *dT = std::max(*dT, std::fabs(static_cast<double>(tauOut[t]) - tauRef[t]));
    }
}

/// 逐元素把设备值与 CPU 参考值并排打出来
void GeqrfDumpValues(int m, int n, int lda, int k, const std::vector<float> &Aout,
                     const std::vector<float> &tauOut, const std::vector<float> &Aref,
                     const std::vector<float> &tauRef)
{
    for (int c = 0; c < n; ++c)
    {
        for (int r = 0; r < m; ++r)
        {
            const size_t idx = static_cast<size_t>(c) * lda + r;
            printf("         A[%d][%d] dev=%.9e cpu=%.9e\n", r, c, static_cast<double>(Aout[idx]),
                   static_cast<double>(Aref[idx]));
        }
    }
    for (int t = 0; t < k; ++t)
    {
        printf("         tau[%d] dev=%.9e cpu=%.9e\n", t, static_cast<double>(tauOut[t]),
               static_cast<double>(tauRef[t]));
    }
}

/// 小形状上判据非常敏感（分母含 m），超阈值时把设备与 CPU 的逐元素值打出来，
/// 用来区分「舍入路径差一两个 ulp」和「真算错」。
void GeqrfDumpSmallDiff(int m, int n, int lda, int k, const std::vector<float> &Aout,
                        const std::vector<float> &tauOut, const std::vector<float> &Aref,
                        const std::vector<float> &tauRef)
{
    double dA = 0.0, dT = 0.0;
    GeqrfMaxDiff(m, n, lda, k, Aout, tauOut, Aref, tauRef, &dA, &dT);
    printf("         max|dA|=%.3e  max|dtau|=%.3e\n", dA, dT);
    GeqrfDumpValues(m, n, lda, k, Aout, tauOut, Aref, tauRef);
}

/// RunShape 一趟之间要共享的量。first 存第 1 趟的 (A, tau) 拼接，用于比 bit-wise。
struct GeqrfRepState
{
    double ratio0 = 0.0;
    std::vector<float> first;
};

/// 把 A 与 tau 从设备回读
bool GeqrfReadBack(const Buffers &buf, size_t aElems, int k, std::vector<float> *Aout,
                   std::vector<float> *tauOut)
{
    Aout->assign(aElems, 0.0f);
    tauOut->assign(std::max(k, 1), 0.0f);
    TEST_CHECK(aclrtMemcpy(Aout->data(), Aout->size() * sizeof(float), buf.A, Aout->size() * sizeof(float),
                           ACL_MEMCPY_DEVICE_TO_HOST));
    TEST_CHECK(aclrtMemcpy(tauOut->data(), tauOut->size() * sizeof(float), buf.tau,
                           tauOut->size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    return true;
}

/// 输出里不能有 inf/NaN
bool GeqrfAllFinite(const std::vector<float> &v)
{
    for (float x : v)
    {
        if (!std::isfinite(x))
        {
            return false;
        }
    }
    return true;
}

/// 一趟的判定：判据 ratio（SQRT01）+ 与第 1 趟的 bit-wise 一致。
/// 小形状超阈值时额外打逐元素差，用来区分舍入与算错。
void GeqrfJudgeRep(int m, int n, int lda, int k, const std::vector<double> &Aorig,
                   const std::vector<float> &Aout, const std::vector<float> &tauOut,
                   const std::vector<float> &Aref, const std::vector<float> &tauRef, double thresh, int rep,
                   GeqrfRepState *state, bool *ok)
{
    std::vector<float> cur(Aout.begin(), Aout.end());
    cur.insert(cur.end(), tauOut.begin(), tauOut.end());
    const double ratio = Sqrt01(m, n, lda, Aorig, Aout, tauOut);
    if (rep == 0)
    {
        state->ratio0 = ratio;
        state->first = cur;
    }
    else if (cur != state->first)
    {
        printf("  [FAIL] 第 %d 次与第 1 次输出不是 bit-wise 一致\n", rep + 1);
        *ok = false;
    }
    if (ratio > thresh)
    {
        printf("  [FAIL] ratio=%.4f > 阈值 %.4f\n", ratio, thresh);
        *ok = false;
        if (rep == 0 && m <= 8 && n <= 8)
        {
            GeqrfDumpSmallDiff(m, n, lda, k, Aout, tauOut, Aref, tauRef);
        }
    }
}

/// 下发一次并校验：devInfo、非有限值、判据 ratio、与第 1 趟的 bit-wise 一致。
/// 返回 false 表示这一趟没跑成（接口返回错误），判据不过只把 ok 拉低。
bool GeqrfOneRep(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int lda, int k, int Lwork,
                 const Buffers &buf, const std::vector<float> &Ah, const std::vector<double> &Aorig,
                 const std::vector<float> &Aref, const std::vector<float> &tauRef, double thresh, int rep,
                 GeqrfRepState *state, bool *ok)
{
    TEST_CHECK(aclrtMemcpy(buf.A, Ah.size() * sizeof(float), Ah.data(), Ah.size() * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    const aclsolverStatus_t st =
        aclsolverSgeqrf(handle, m, n, reinterpret_cast<float *>(buf.A), lda, reinterpret_cast<float *>(buf.tau),
                        reinterpret_cast<float *>(buf.ws), Lwork, reinterpret_cast<int *>(buf.info));
    if (st != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  [FAIL] aclsolverSgeqrf 返回 %d\n", (int)st);
        return false;
    }
    TEST_CHECK(aclrtSynchronizeStream(stream));

    int32_t info = -12345;
    TEST_CHECK(aclrtMemcpy(&info, sizeof(info), buf.info, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST));
    if (info != 0)
    {
        printf("  [FAIL] devInfo = %d（应为 0）\n", info);
        *ok = false;
    }

    std::vector<float> Aout, tauOut;
    if (!GeqrfReadBack(buf, Ah.size(), k, &Aout, &tauOut))
    {
        return false;
    }
    if (!GeqrfAllFinite(Aout))
    {
        printf("  [FAIL] 输出含非有限值\n");
        *ok = false;
    }
    GeqrfJudgeRep(m, n, lda, k, Aorig, Aout, tauOut, Aref, tauRef, thresh, rep, state, ok);
    return true;
}

/// 正向用例：跑 3 次，验判据 + devInfo + 跨次 bit-wise 一致
/// scale 用来把输入整体放大/缩小到极端量级，触发 Householder 范数的定标兜底
/// （不传就是 1.0f，与原来的用例完全一致）。
bool RunShape(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int pad, float scale = 1.0f)
{
    const int lda = m + pad;
    const int k = std::min(m, n);

    int Lwork = -1;
    const aclsolverStatus_t stQ = aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &Lwork);
    if (stQ != ACLSOLVER_STATUS_SUCCESS || Lwork < 0)
    {
        printf("  [FAIL] bufferSize 返回 %d, Lwork=%d\n", (int)stQ, Lwork);
        return false;
    }

    std::mt19937 rng(20261014u + static_cast<uint32_t>(m) * 131u + static_cast<uint32_t>(n));
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> Ah(static_cast<size_t>(lda) * n);
    for (auto &x : Ah)
    {
        x = dist(rng) * scale;
    }
    const std::vector<double> Aorig(Ah.begin(), Ah.end());

    Buffers buf;
    if (!GeqrfAlloc(&buf, Ah.size(), k, Lwork))
    {
        buf.Free();
        return false;
    }

    // CPU 同精度参考（同一份 Householder 约定）
    std::vector<float> Aref(Ah.begin(), Ah.end());
    std::vector<float> tauRef(std::max(k, 1), 0.0f);
    QrCore::Geqr2(m, n, Aref.data(), lda, tauRef.data());
    const double ratioCpu = Sqrt01(m, n, lda, Aorig, Aref, tauRef);
    const double thresh = std::max(5.0 * ratioCpu, 0.1);

    GeqrfRepState state;
    bool ok = true;
    for (int rep = 0; rep < 3; ++rep)
    {
        if (!GeqrfOneRep(handle, stream, m, n, lda, k, Lwork, buf, Ah, Aorig, Aref, tauRef, thresh, rep, &state,
                         &ok))
        {
            buf.Free();
            return false;
        }
    }
    buf.Free();
    printf("  m=%-5d n=%-5d lda=%-5d Lwork=%-9d ratio=%.4f ratio_cpu=%.4f 阈值=%.4f  %s\n", m, n, lda, Lwork,
           state.ratio0, ratioCpu, thresh, ok ? "PASS" : "FAIL");
    return ok;
}

/// 空问题：m 或 n 为 0 时须成功返回且 info 为 0
bool RunEmpty(aclsolverHandle_t handle, int m, int n)
{
    int Lwork = -1;
    if (aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, std::max(m, 1), &Lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  [FAIL] 空问题 bufferSize 失败\n");
        return false;
    }
    uint8_t *info = nullptr;
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    const int32_t poison = -999;
    TEST_CHECK(aclrtMemcpy(info, sizeof(poison), &poison, sizeof(poison), ACL_MEMCPY_HOST_TO_DEVICE));
    const aclsolverStatus_t st = aclsolverSgeqrf(handle, m, n, nullptr, std::max(m, 1), nullptr, nullptr, Lwork,
                                                 reinterpret_cast<int *>(info));
    int32_t got = -12345;
    (void)aclrtMemcpy(&got, sizeof(got), info, sizeof(got), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(info);
    const bool ok = (st == ACLSOLVER_STATUS_SUCCESS) && (Lwork == 0) && (got == 0);
    printf("  空问题 m=%d n=%d: status=%d Lwork=%d info=%d  %s\n", m, n, (int)st, Lwork, got, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- 参数校验 ----
// 一组固定的合法入参，各条检查只改其中被测的那一个。
constexpr int kBpM = 64;
constexpr int kBpN = 32;
constexpr int kBpLda = 64;

/// 每个参数位置各一条：须返回 INVALID_VALUE，且 devInfo 写入 -i（i 不计 handle）
void CheckArgPositions(aclsolverHandle_t handle, aclrtStream stream, const Buffers &buf, int Lwork, bool *ok)
{
    float *A = reinterpret_cast<float *>(buf.A);
    float *TAU = reinterpret_cast<float *>(buf.tau);
    float *WS = reinterpret_cast<float *>(buf.ws);
    int *INFO = reinterpret_cast<int *>(buf.info);
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    // 这两条接口还没走到写 devInfo 那步（devInfo 本身为空 / handle 为空），只看返回码
    ExpectStatus("devInfo=nullptr", aclsolverSgeqrf(handle, kBpM, kBpN, A, kBpLda, TAU, WS, Lwork, nullptr), bad,
                 ok);
    ExpectStatus("handle=nullptr", aclsolverSgeqrf(nullptr, kBpM, kBpN, A, kBpLda, TAU, WS, Lwork, INFO),
                 ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, ok);
    ExpectInfo(stream, buf.info, "m<0 -> info=-1",
               aclsolverSgeqrf(handle, -1, kBpN, A, kBpLda, TAU, WS, Lwork, INFO), bad, -1, ok);
    ExpectInfo(stream, buf.info, "n<0 -> info=-2",
               aclsolverSgeqrf(handle, kBpM, -1, A, kBpLda, TAU, WS, Lwork, INFO), bad, -2, ok);
    ExpectInfo(stream, buf.info, "A=nullptr -> info=-3",
               aclsolverSgeqrf(handle, kBpM, kBpN, nullptr, kBpLda, TAU, WS, Lwork, INFO), bad, -3, ok);
    ExpectInfo(stream, buf.info, "lda<m -> info=-4",
               aclsolverSgeqrf(handle, kBpM, kBpN, A, kBpM - 1, TAU, WS, Lwork, INFO), bad, -4, ok);
    ExpectInfo(stream, buf.info, "TAU=nullptr -> info=-5",
               aclsolverSgeqrf(handle, kBpM, kBpN, A, kBpLda, nullptr, WS, Lwork, INFO), bad, -5, ok);
    ExpectInfo(stream, buf.info, "Workspace=nullptr -> info=-6",
               aclsolverSgeqrf(handle, kBpM, kBpN, A, kBpLda, TAU, nullptr, Lwork, INFO), bad, -6, ok);
    ExpectInfo(stream, buf.info, "Lwork 太小 -> info=-7",
               aclsolverSgeqrf(handle, kBpM, kBpN, A, kBpLda, TAU, WS, Lwork - 1, INFO), bad, -7, ok);
    // lda > m 必须被接受（任务书明文禁止要求 lda == m）
    ExpectInfo(stream, buf.info, "lda>m 合法",
               aclsolverSgeqrf(handle, kBpM, kBpN, A, kBpLda + 7, TAU, WS, Lwork, INFO),
               ACLSOLVER_STATUS_SUCCESS, 0, ok);
}

/// 查询接口自己的参数校验
void CheckBufferSizeArgs(aclsolverHandle_t handle, bool *ok)
{
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    int Lw = -1;
    ExpectStatus("bufferSize Lwork=nullptr",
                 aclsolverSgeqrf_bufferSize(handle, kBpM, kBpN, nullptr, kBpLda, nullptr), bad, ok);
    ExpectStatus("bufferSize lda<m", aclsolverSgeqrf_bufferSize(handle, kBpM, kBpN, nullptr, kBpM - 1, &Lw), bad,
                 ok);
}

/// 参数校验：须返回 INVALID_VALUE，且 devInfo 写入 -i（i 不计 handle）
bool RunBadParams(aclsolverHandle_t handle, aclrtStream stream)
{
    int Lwork = 0;
    if (aclsolverSgeqrf_bufferSize(handle, kBpM, kBpN, nullptr, kBpLda, &Lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        return false;
    }
    Buffers buf;
    if (!GeqrfAlloc(&buf, static_cast<size_t>(kBpLda) * kBpN, kBpN, Lwork))
    {
        buf.Free();
        return false;
    }
    bool ok = true;
    CheckArgPositions(handle, stream, buf, Lwork, &ok);
    CheckBufferSizeArgs(handle, &ok);
    buf.Free();
    return ok;
}


/// 性能模式：只跑一个形状、重复 iters 次、不做任何 host 校验。
/// 任务书 §3.3 要求"采集时收敛到单条用例，避免多条用例的 kernel 在统计中混叠"，
/// 所以性能采集必须用这个模式，不能直接 profile 上面的功能测试。
/// 墙钟采样：预热 3 次（第 1 次 launch 约贵 10 us，不计入），再取 3 趟 iters 次的最小值。
///
/// **每次 launch 后都同步**：这是计时口径的需要（每趟要重拷 A、把一次 launch 单独圈出来），
/// 不是因为连续下发有问题。早期曾怀疑"连续下发不同步会挂死"，**该现象已不复现**：
/// 见 test/sgels 场景 5（五接口一口气下发、中间零同步，3 形状 × 3 趟逐位一致）
/// 与本仓 sormqr_test 的 chain 模式（不同步 vs 中间同步，结果逐位相同、耗时一致）。
/// 同步版的墙钟只作为优化指南针；验收口径仍是 msprof 的 Task Duration。
void GeqrfPerfMeasure(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int lda, int Lwork,
                      const Buffers &buf, int iters)
{
    auto once = [&]() {
        return aclsolverSgeqrf(handle, m, n, reinterpret_cast<float *>(buf.A), lda,
                               reinterpret_cast<float *>(buf.tau), reinterpret_cast<float *>(buf.ws), Lwork,
                               reinterpret_cast<int *>(buf.info));
    };
    for (int i = 0; i < 3; ++i)
    {
        (void)once();
        (void)aclrtSynchronizeStream(stream);
    }
    double best = 1e30, sum = 0.0;
    for (int rep = 0; rep < 3; ++rep)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            (void)once();
            (void)aclrtSynchronizeStream(stream);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = SafeDiv(std::chrono::duration<double, std::micro>(t1 - t0).count(), iters);
        best = std::min(best, us);
        sum += us;
    }
    printf("PERF m=%d n=%d lda=%d iters=%d  墙钟单次(含同步): min=%.3f us  mean=%.3f us\n", m, n, lda, iters, best,
           sum / 3.0);
}

int RunPerf(int m, int n, int lda, int iters)
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }

    int Lwork = 0;
    if (aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &Lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("bufferSize 失败\n");
        return 1;
    }
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> Ah(static_cast<size_t>(lda) * n);
    for (auto &x : Ah)
    {
        x = dist(rng);
    }
    Buffers buf;
    if (!GeqrfAlloc(&buf, Ah.size(), std::min(m, n), Lwork))
    {
        printf("aclrtMalloc 失败\n");
        buf.Free();
        return 1;
    }
    (void)aclrtMemcpy(buf.A, Ah.size() * sizeof(float), Ah.data(), Ah.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);

    GeqrfPerfMeasure(handle, stream, m, n, lda, Lwork, buf, iters);

    buf.Free();
    TeardownAcl(handle, stream);
    return 0;
}

/// 全量性能扫描：逐 case 与任务包 bench_result.json 的 GPU 平均耗时比，判据 T_NPU <= T_GPU/0.35
///
/// 计时口径（固定下来，别每次换）：
///   - 每次 launch 后同步（为把单次 launch 圈出来；连续下发本身没问题，见场景 5 的回归）
///   - 每趟开始前重拷一次 A，**计时窗口从拷完之后开始**：既避免把 H2D 算进去，
///     也避免一趟里连续几十次对"已分解矩阵"再分解导致数值发散
///   - 取 3 趟的**最小值**当稳态（首趟芯片未提频，实测 mean 能比 min 高近一倍）
/// 查 workspace 并造一份固定种子的随机输入（性能口径不需要换种子）
bool GeqrfPrepareInput(aclsolverHandle_t handle, int m, int n, int lda, int *Lwork, std::vector<float> *Ah)
{
    if (aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, Lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  m=%-5d n=%-5d  bufferSize 失败\n", m, n);
        return false;
    }
    Ah->assign(static_cast<size_t>(lda) * n, 0.0f);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    for (auto &x : *Ah)
    {
        x = d(rng);
    }
    return true;
}

/// 测一个形状的稳态耗时：热机 3 次后取 3 趟的最小值。
/// 每趟重拷 A：连续几十次对"已分解矩阵"再分解会让数值发散。
/// 返回 false 表示这条没测成（原因已打印）。
bool GeqrfTimeOne(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int lda, double *best)
{
    int Lwork = 0;
    std::vector<float> Ah;
    if (!GeqrfPrepareInput(handle, m, n, lda, &Lwork, &Ah))
    {
        return false;
    }
    Buffers buf;
    if (!GeqrfAlloc(&buf, Ah.size(), std::min(m, n), Lwork))
    {
        printf("  m=%-5d n=%-5d  显存分配失败\n", m, n);
        buf.Free();
        return false;
    }
    auto once = [&]() {
        return aclsolverSgeqrf(handle, m, n, reinterpret_cast<float *>(buf.A), lda,
                               reinterpret_cast<float *>(buf.tau), reinterpret_cast<float *>(buf.ws), Lwork,
                               reinterpret_cast<int *>(buf.info));
    };
    auto reload = [&]() {
        (void)aclrtMemcpy(buf.A, Ah.size() * sizeof(float), Ah.data(), Ah.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    };
    const double work = static_cast<double>(m) * n;
    const int iters = (work < 1.0e4) ? 30 : ((work < 1.0e6) ? 10 : 3);
    // 热机 3 次，同时确认这个形状能跑通
    reload();
    bool rcOk = true;
    for (int i = 0; i < 3; ++i)
    {
        if (once() != ACLSOLVER_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        {
            rcOk = false;
        }
    }
    if (!rcOk)
    {
        printf("  m=%-5d n=%-5d  **kernel 失败**\n", m, n);
        buf.Free();
        return false;
    }
    *best = TimeSteadyMin(stream, iters, 0, reload, once);
    buf.Free();
    return true;
}

int RunPerfSweep(aclsolverHandle_t handle, aclrtStream stream, const char *path, int ldaExtra)
{
    std::vector<std::vector<double>> rows;
    if (!LoadCaseRows(path, rows))
    {
        printf("打不开用例清单 %s\n", path);
        return 1;
    }
    int pass = 0;
    int total = 0;
    int noBase = 0;
    double worstRatio = 1e30;
    int worstM = 0;
    int worstN = 0;
    for (const auto &row : rows)
    {
        if (row.size() < 3)
        {
            continue;
        }
        const int m = RowInt(row, 0, 0);
        const int n = RowInt(row, 1, 0);
        const double gpuUs = RowDouble(row, 3, -1.0);
        const int lda = std::max(RowInt(row, 2, 0), m) + ldaExtra;

        double best = 1.0e30;
        if (!GeqrfTimeOne(handle, stream, m, n, lda, &best))
        {
            continue;
        }

        if (gpuUs <= 0.0)
        {
            ++noBase;
            printf("  m=%-5d n=%-5d  NPU=%9.2f us   （无 GPU 基线，不判定）\n", m, n, best);
            continue;
        }
        ++total;
        const std::string label = "m=" + std::to_string(m) + " n=" + std::to_string(n);
        double times = 0.0;
        pass += ExpectPerf(label.c_str(), best, gpuUs, &times) ? 1 : 0;
        if (times < worstRatio)
        {
            worstRatio = times;
            worstM = m;
            worstN = n;
        }
    }
    printf("\n性能扫描：%d / %d PASS（判据 >= 0.35x GPU）；最差 %5.2fx 于 m=%d n=%d；%d 例无基线\n", pass,
           total, worstRatio, worstM, worstN, noBase);
    return (pass == total) ? 0 : 1;
}

/// 连跑两次，两次的 (A, tau) 都带回来用于比 bit-wise。
/// 返回 false 表示这条没跑成（原因已打印）。
bool GeqrfRunTwice(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int lda, int k, int Lwork,
                   const std::vector<float> &Ah, std::vector<float> *outA, std::vector<float> *outTau,
                   int32_t *info)
{
    Buffers buf;
    if (!GeqrfAlloc(&buf, Ah.size(), k, Lwork))
    {
        printf("  m=%-5d n=%-5d lda=%-5d  显存分配失败\n", m, n, lda);
        buf.Free();
        return false;
    }
    aclsolverStatus_t st = ACLSOLVER_STATUS_SUCCESS;
    bool rcOk = true;
    for (int rep = 0; rep < 2; ++rep)
    {
        (void)aclrtMemcpy(buf.A, Ah.size() * sizeof(float), Ah.data(), Ah.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        st = aclsolverSgeqrf(handle, m, n, reinterpret_cast<float *>(buf.A), lda,
                             reinterpret_cast<float *>(buf.tau), reinterpret_cast<float *>(buf.ws), Lwork,
                             reinterpret_cast<int *>(buf.info));
        if (st != ACLSOLVER_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        {
            rcOk = false;
            break;
        }
        outA[rep].assign(Ah.size(), 0.0f);
        outTau[rep].assign(std::max(k, 1), 0.0f);
        (void)aclrtMemcpy(outA[rep].data(), outA[rep].size() * sizeof(float), buf.A,
                          outA[rep].size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        (void)aclrtMemcpy(outTau[rep].data(), outTau[rep].size() * sizeof(float), buf.tau,
                          outTau[rep].size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        (void)aclrtMemcpy(info, sizeof(*info), buf.info, sizeof(*info), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    buf.Free();
    if (!rcOk)
    {
        printf("  m=%-5d n=%-5d lda=%-5d  **kernel 失败** status=%d\n", m, n, lda, static_cast<int>(st));
        return false;
    }
    return true;
}

/// 判据 SQRT01 + devInfo + 非有限值 + 跨次 bit-wise，打一行明细
bool GeqrfSweepVerify(int m, int n, int lda, int k, int maxCols, const std::vector<float> &Ah,
                      const std::vector<double> &Aorig, const std::vector<float> *outA,
                      const std::vector<float> *outTau, int32_t info, bool *exact)
{
    const bool deterministic = (outA[0] == outA[1]) && (outTau[0] == outTau[1]);
    bool nanFree = true;
    for (float x : outA[0])
    {
        if (!std::isfinite(x))
        {
            nanFree = false;
            break;
        }
    }
    std::vector<float> Aref(Ah.begin(), Ah.end());
    std::vector<float> tauRef(std::max(k, 1), 0.0f);
    QrCore::Geqr2(m, n, Aref.data(), lda, tauRef.data());
    const double ratioCpu = Sqrt01(m, n, lda, Aorig, Aref, tauRef, maxCols, exact);
    const double ratio = Sqrt01(m, n, lda, Aorig, outA[0], outTau[0], maxCols, nullptr);
    const double thresh = std::max(5.0 * ratioCpu, 0.1);
    const bool ok = (info == 0) && nanFree && deterministic && (ratio <= thresh);
    printf("  m=%-5d n=%-5d lda=%-5d info=%d det=%d ratio=%.4f ratio_cpu=%.4f th=%.4f %s %s\n", m, n, lda, info,
           deterministic ? 1 : 0, ratio, ratioCpu, thresh, *exact ? "精确" : "抽样下界",
           ok ? "PASS" : "**FAIL**");
    fflush(stdout);
    return ok;
}

/// 全量 case 扫描：精度（§3.2.1-1）+ 跨次 bit-wise 确定性
/// 用例清单每行 "m n lda gpu_us"，取自任务包 cases.json（见 data/sgeqrf_cases.txt）
int RunSweep(aclsolverHandle_t handle, aclrtStream stream, const char *path, int maxCols, int ldaExtra)
{
    int pass = 0;
    int total = 0;
    int sampled = 0;
    const bool loaded = ForEachCaseRow(path, 3, [&](const std::vector<double> &row, int caseIdx) {
        const int m = RowInt(row, 0, 0);
        const int n = RowInt(row, 1, 0);
        const int lda = std::max(RowInt(row, 2, 0), m) + ldaExtra;
        ++total;

        int Lwork = 0;
        if (aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &Lwork) != ACLSOLVER_STATUS_SUCCESS)
        {
            printf("  m=%-5d n=%-5d lda=%-5d  bufferSize 失败\n", m, n, lda);
            return;
        }
        const int k = std::min(m, n);
        std::vector<float> Ah(static_cast<size_t>(lda) * n);
        FillMatrix(Ah, 20261015u + static_cast<uint32_t>(caseIdx), caseIdx);
        const std::vector<double> Aorig(Ah.begin(), Ah.end());

        std::vector<float> outA[2];
        std::vector<float> outTau[2];
        int32_t info = -1;
        if (!GeqrfRunTwice(handle, stream, m, n, lda, k, Lwork, Ah, outA, outTau, &info))
        {
            return;
        }
        bool exact = true;
        const bool ok = GeqrfSweepVerify(m, n, lda, k, maxCols, Ah, Aorig, outA, outTau, info, &exact);
        pass += ok ? 1 : 0;
        sampled += exact ? 0 : 1;
    });
    if (!loaded)
    {
        return 1;
    }
    printf("\n全量扫描：%d / %d PASS（其中 %d 例判据为抽样下界，未抽到的列未验）\n", pass, total, sampled);
    return (pass == total) ? 0 : 1;
}


/// 全量清单模式：sweep 跑精度、perfsweep 跑性能，两者共用同一份用例清单
int RunSweepMode(const std::string &mode, int argc, char **argv)
{
    const char *path = (argc > 2) ? argv[2] : "./data/sgeqrf_cases.txt";
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    int rc = 0;
    if (mode == "perfsweep")
    {
        const int ldaExtra = ArgInt(argc, argv, 3, 0);
        printf("== 全量性能扫描（判据 T_NPU <= T_GPU/0.35）ldaExtra=%d ==\n", ldaExtra);
        rc = RunPerfSweep(handle, stream, path, ldaExtra);
    }
    else
    {
        const int maxCols = ArgInt(argc, argv, 3, 48);  // 判据抽样的最大列数
        const int ldaExtra = ArgInt(argc, argv, 4, 0);  // §3.5 要求覆盖 +8 / +32 padding
        printf("== 全量 case 扫描（精度 §3.2.1-1 + 跨次 bit-wise 确定性）maxCols=%d ldaExtra=%d ==\n", maxCols,
               ldaExtra);
        rc = RunSweep(handle, stream, path, maxCols, ldaExtra);
    }
    TeardownAcl(handle, stream);
    return rc;
}

/// 正向形状覆盖：取自任务包 cases.json，各跑 3 次
void RunShapeCases(aclsolverHandle_t handle, aclrtStream stream, int *pass, int *total)
{
    printf("== 正向用例（取自任务包 cases.json，各跑 3 次）==\n");
    struct Shape
    {
        int m, n, pad;
    };
    const Shape shapes[] = {
        {2, 2, 0},     {4, 4, 0},    {8, 8, 0},    {16, 16, 0},  {18, 8, 0},    {22, 22, 0},
        {32, 4, 0},    {57, 57, 0},  {64, 64, 0},  {88, 88, 0},  {100, 18, 0},  {128, 128, 0},
        {135, 135, 0}, {161, 86, 0}, {205, 100, 0}, {256, 256, 0}, {332, 66, 0}, {512, 512, 0},
        // lda > m：任务包里全是 lda == m，这里额外压一遍非紧凑排布
        {128, 64, 7},  {256, 100, 32},
        // m < n：§3.5 明确要求 geqrf 覆盖（任务包的 cases.json 里一条都没有，
        // 只在 sgeqrf_cases_pow2.txt 里有 6 条）。默认套里也放几个，验收人一跑就能看到。
        {8, 128, 0},   {64, 128, 0},  {37, 100, 3},  {128, 512, 0},
    };
    for (const auto &s : shapes)
    {
        Tally(RunShape(handle, stream, s.m, s.n, s.pad), pass, total);
    }
}

} // namespace

int main(int argc, char **argv)
{
    const std::string mode = (argc >= 2) ? argv[1] : "";
    if (mode == "perfsweep" || mode == "sweep")
    {
        return RunSweepMode(mode, argc, argv);
    }
    if (mode == "perf")
    {
        const int m = ArgInt(argc, argv, 2, 1024);
        const int n = ArgInt(argc, argv, 3, 1024);
        const int lda = ArgInt(argc, argv, 4, m);
        const int iters = ArgInt(argc, argv, 5, 25);
        return RunPerf(m, n, lda, iters);
    }

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    int pass = 0, total = 0;
    if (!BeginTests(&stream, &handle, &pass, &total, RunBadParams))
    {
        return 1;
    }

    printf("== 空问题（m 或 n 为 0）==\n");
    for (auto mn : {std::pair<int, int>{0, 8}, {8, 0}, {0, 0}})
    {
        Tally(RunEmpty(handle, mn.first, mn.second), &pass, &total);
    }

    RunShapeCases(handle, stream, &pass, &total);

    // 极端量级的用例这里**刻意不设**：实测可用窗口是双边的（见 include/cann_ops_solver.h
    // 的 @param A），窗口外属契约外输入，不是本测试要判的东西。两端的实测根因：
    //   · max|A| 约 >= 1.8e19：面板内点积在 ~max|A| 的 v 上做，乘积 ~max|A|^2 超出 fp32；
    //   · max|A| 约 <= 5e-20：tauPrime ~ 1/max|A|^2 超出 fp32。
    // 两者都源于"为性能把归一化推迟到 GeqrfNormalizeStore"，定标求范数救不了，
    // 要支持得按 2 的幂把活动子矩阵搬进窗口（另行实现）。
    // 顺带记一个实测事实：本芯片的 Mul/ReduceSum **保留次正规**，所以 5e-20 这种
    // 量级（平方 2.5e-39 低于 FLT_MIN 但仍合法）累加后回到正常数范围、根本不触发下溢。
    printf("\n%d / %d PASS\n", pass, total);
    TeardownAcl(handle, stream);
    return (pass == total) ? 0 : 1;
}
