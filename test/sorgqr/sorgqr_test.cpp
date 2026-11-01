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
 * \file sorgqr_test.cpp
 * \brief aclsolverSorgqr 的端到端测试：判据、确定性、参数校验的 -i 语义、全量 case 扫描
 *
 * 判据取任务书 §3.2.1-2 原文（SQRT02）：
 *     ratio = ||I - Q^T·Q||_1 / ( m·eps )，   ratio <= max(5·ratio_cpu, 1.5) 为通过
 * 分母固定取**行数 m**，不随 I 的维度 n 变化。ratio_cpu 用同一份 Householder 约定的
 * CPU 参考（QrCore::Org2r，对齐 LAPACK dorg2r）现场实测，不写死常数。
 *
 * 模式：
 *   sorgqr_test                           功能 + 参数校验 + 一组形状
 *   sorgqr_test sweep  <清单> [ldaExtra]  全量 case 精度 + 跨次 bit-wise 确定性
 *   sorgqr_test perfsweep <清单> [ldaExtra]  全量 case 性能（判据 >= 0.35x GPU）
 */

// acl/acl.h、cann_ops_solver.h 以及 algorithm / chrono / random / string / vector
// 这批都由 qr_test_utils.h 统一提供，这里只列本文件额外要的。
#include "../utils/qr_test_utils.h"
#include <cmath>
#include "../../src/sgeqrf/kernel/householder.hpp"

namespace
{
// A / tau / workspace / devInfo 这组缓冲与 sgeqrf 完全一样，用共享头里的那份
using Buffers = QrBuffers;

/// SQRT02：ratio = ||I - Q^T Q||_1 / (m*eps)，Q 为 m x n（ld = lda）
/// 列数超过 maxCols 时等步长抽样：被抽到的列精确 => 得到真 1-范数的下界
double Sqrt02(int m, int n, int lda, const std::vector<float> &Q, int maxCols, bool *exact)
{
    const int step = (maxCols > 0 && n > maxCols) ? ((n + maxCols - 1) / maxCols) : 1;
    if (exact != nullptr)
    {
        *exact = (step == 1);
    }
    double worst = 0.0;
    for (int b = 0; b < n; b += step)
    {
        worst = std::max(worst, OrthColumnSum(m, n, lda, Q, b));
    }
    return worst / (static_cast<double>(m) * kEpsF32);
}

/// 造 orgqr 的合法输入：先用 CPU Geqr2 分解，得到 (A 下梯形里的 v, tau)
/// ★ Geqr2 写 min(m,n) 个 tau，**不是 k 个** —— 按 k 开容量在 k<min(m,n) 时会堆越界
void MakeInput(int m, int n, int k, int lda, int caseIdx, std::vector<float> &fac, std::vector<float> &tau)
{
    fac.assign(static_cast<size_t>(lda) * n, 0.0f);
    FillMatrix(fac, 20261016u + static_cast<uint32_t>(caseIdx), caseIdx);
    std::vector<float> tauFull(static_cast<size_t>(std::min(m, n)), 0.0f);
    QrCore::Geqr2(m, n, fac.data(), lda, tauFull.data());
    tau.assign(tauFull.begin(), tauFull.begin() + k);
}

bool AllocFor(Buffers &buf, size_t aElems, int k, int lwork)
{
    if (aclrtMalloc(reinterpret_cast<void **>(&buf.A), aElems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS)
    {
        return false;
    }
    if (aclrtMalloc(reinterpret_cast<void **>(&buf.tau), std::max(k, 1) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS)
    {
        return false;
    }
    if (aclrtMalloc(reinterpret_cast<void **>(&buf.info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        return false;
    }
    if (lwork > 0 && aclrtMalloc(reinterpret_cast<void **>(&buf.ws), static_cast<size_t>(lwork) * sizeof(float),
                                 ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        return false;
    }
    return true;
}

/// 连跑两次 Sorgqr，两次的输出都带回来用于比 bit-wise。失败时已打印原因。
bool OrgqrRunTwice(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int k, int lda, int lwork,
                   const std::vector<float> &fac, const std::vector<float> &tau, std::vector<float> *out,
                   int32_t *info)
{
    Buffers buf;
    if (!AllocFor(buf, fac.size(), k, lwork))
    {
        printf("  m=%-5d n=%-5d k=%-5d  显存分配失败\n", m, n, k);
        buf.Free();
        return false;
    }
    if (k > 0)
    {
        TEST_CHECK(aclrtMemcpy(buf.tau, tau.size() * sizeof(float), tau.data(), tau.size() * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE));
    }
    for (int rep = 0; rep < 2; ++rep)
    {
        TEST_CHECK(aclrtMemcpy(buf.A, fac.size() * sizeof(float), fac.data(), fac.size() * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE));
        const aclsolverStatus_t st =
            aclsolverSorgqr(handle, m, n, k, reinterpret_cast<float *>(buf.A), lda,
                            reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.ws), lwork,
                            reinterpret_cast<int *>(buf.info));
        if (st != ACLSOLVER_STATUS_SUCCESS)
        {
            printf("  m=%-5d n=%-5d k=%-5d  aclsolverSorgqr 返回 %d\n", m, n, k, static_cast<int>(st));
            buf.Free();
            return false;
        }
        TEST_CHECK(aclrtSynchronizeStream(stream));
        out[rep].assign(fac.size(), 0.0f);
        TEST_CHECK(aclrtMemcpy(out[rep].data(), out[rep].size() * sizeof(float), buf.A,
                               out[rep].size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        TEST_CHECK(aclrtMemcpy(info, sizeof(*info), buf.info, sizeof(*info), ACL_MEMCPY_DEVICE_TO_HOST));
    }
    buf.Free();
    return true;
}

/// 输出里不能有 inf/NaN（只看被写的 m x n 区域，lda padding 不算）
bool OrgqrIsFinite(int m, int n, int lda, const std::vector<float> &q)
{
    for (int c = 0; c < n; ++c)
    {
        for (int r = 0; r < m; ++r)
        {
            if (!std::isfinite(q[static_cast<size_t>(c) * lda + r]))
            {
                return false;
            }
        }
    }
    return true;
}

/// 判据 SQRT02：ratio = ‖I − QᵀQ‖₁ / (m·eps)，阈值 max(5·ratio_cpu, 1.5)。
/// ratio_cpu 用同一份 (v, tau) 的 CPU org2r 现场实测，不写死常数。
bool OrgqrVerify(int m, int n, int k, int lda, int maxCols, const std::vector<float> &fac,
                 const std::vector<float> &tau, const std::vector<float> *out, int32_t info, bool verbose)
{
    const bool deterministic = (out[0] == out[1]);
    const bool nanFree = OrgqrIsFinite(m, n, lda, out[0]);
    std::vector<float> Qcpu(fac.begin(), fac.end());
    QrCore::Org2r(m, n, k, Qcpu.data(), lda, tau.data());
    bool exact = true;
    const double ratioCpu = Sqrt02(m, n, lda, Qcpu, maxCols, &exact);
    const double ratio = Sqrt02(m, n, lda, out[0], maxCols, nullptr);
    const double thresh = std::max(5.0 * ratioCpu, 1.5);
    const bool ok = (info == 0) && nanFree && deterministic && (ratio <= thresh);
    if (verbose || !ok)
    {
        printf("  m=%-5d n=%-5d k=%-5d lda=%-5d info=%d det=%d ratio=%.4f ratio_cpu=%.4f th=%.4f %s %s\n", m, n, k,
               lda, info, deterministic ? 1 : 0, ratio, ratioCpu, thresh, exact ? "精确" : "抽样下界",
               ok ? "PASS" : "**FAIL**");
        fflush(stdout);
    }
    return ok;
}

/// 一个形状：精度（SQRT02）+ 跨次 bit-wise 确定性
bool RunShape(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int k, int lda, int caseIdx, int maxCols,
              bool verbose)
{
    int lwork = 0;
    if (aclsolverSorgqr_bufferSize(handle, m, n, k, nullptr, lda, nullptr, &lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  m=%-5d n=%-5d k=%-5d  bufferSize 失败\n", m, n, k);
        return false;
    }
    std::vector<float> fac;
    std::vector<float> tau;
    MakeInput(m, n, k, lda, caseIdx, fac, tau);

    std::vector<float> out[2];
    int32_t info = -1;
    if (!OrgqrRunTwice(handle, stream, m, n, k, lda, lwork, fac, tau, out, &info))
    {
        return false;
    }
    return OrgqrVerify(m, n, k, lda, maxCols, fac, tau, out, info, verbose);
}

// ---- 参数校验 ----
// 一组固定的合法入参，各条检查只改其中被测的那一个。
constexpr int kBpM = 64;
constexpr int kBpN = 32;
constexpr int kBpK = 16;
constexpr int kBpLda = 64;

/// 每个参数位置各一条：须返回 INVALID_VALUE，且 devInfo 写入 -i（i 不计 handle）
void CheckArgPositions(aclsolverHandle_t handle, aclrtStream stream, const Buffers &buf, int lwork, bool *ok)
{
    float *A = reinterpret_cast<float *>(buf.A);
    const float *tau = reinterpret_cast<const float *>(buf.tau);
    float *WS = reinterpret_cast<float *>(buf.ws);
    int *INFO = reinterpret_cast<int *>(buf.info);
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    // 这两条接口还没走到写 devInfo 那步，只看返回码
    ExpectStatus("devInfo=nullptr", aclsolverSorgqr(handle, kBpM, kBpN, kBpK, A, kBpLda, tau, WS, lwork, nullptr),
                 bad, ok);
    ExpectStatus("handle=nullptr", aclsolverSorgqr(nullptr, kBpM, kBpN, kBpK, A, kBpLda, tau, WS, lwork, INFO),
                 ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, ok);
    ExpectInfo(stream, buf.info, "m<0 -> info=-1",
               aclsolverSorgqr(handle, -1, kBpN, kBpK, A, kBpLda, tau, WS, lwork, INFO), bad, -1, ok);
    ExpectInfo(stream, buf.info, "n>m -> info=-2",
               aclsolverSorgqr(handle, kBpM, kBpM + 1, kBpK, A, kBpLda, tau, WS, lwork, INFO), bad, -2, ok);
    ExpectInfo(stream, buf.info, "k>n -> info=-3",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpN + 1, A, kBpLda, tau, WS, lwork, INFO), bad, -3, ok);
    ExpectInfo(stream, buf.info, "A=nullptr -> info=-4",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpK, nullptr, kBpLda, tau, WS, lwork, INFO), bad, -4, ok);
    ExpectInfo(stream, buf.info, "lda<m -> info=-5",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpK, A, kBpM - 1, tau, WS, lwork, INFO), bad, -5, ok);
    ExpectInfo(stream, buf.info, "tau=nullptr(k>0) -> info=-6",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpK, A, kBpLda, nullptr, WS, lwork, INFO), bad, -6, ok);
    ExpectInfo(stream, buf.info, "work=nullptr -> info=-7",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpK, A, kBpLda, tau, nullptr, lwork, INFO), bad, -7, ok);
    ExpectInfo(stream, buf.info, "lwork 太小 -> info=-8",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpK, A, kBpLda, tau, WS, lwork - 1, INFO), bad, -8, ok);
    ExpectInfo(stream, buf.info, "lda>m 合法",
               aclsolverSorgqr(handle, kBpM, kBpN, kBpK, A, kBpLda + 7, tau, WS, lwork, INFO),
               ACLSOLVER_STATUS_SUCCESS, 0, ok);
}

/// 查询接口自己的参数校验
void CheckBufferSizeArgs(aclsolverHandle_t handle, bool *ok)
{
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    int lw = -1;
    ExpectStatus("bufferSize lwork=nullptr",
                     aclsolverSorgqr_bufferSize(handle, kBpM, kBpN, kBpK, nullptr, kBpLda, nullptr, nullptr), bad,
                     ok);
    ExpectStatus("bufferSize lda<m",
                     aclsolverSorgqr_bufferSize(handle, kBpM, kBpN, kBpK, nullptr, kBpM - 1, nullptr, &lw), bad, ok);
    // m=0 时 n 必须也是 0：LAPACK 的判据是 N<0 || N>M，所以 m=0,n=8 是 info=-2，
    // 不能当成"空问题"放过去。
    ExpectStatus("bufferSize m=0,n=8 (n>m)",
                     aclsolverSorgqr_bufferSize(handle, 0, 8, 0, nullptr, 1, nullptr, &lw), bad, ok);
}

/// 参数校验：须返回 INVALID_VALUE，且 devInfo 写入 -i（i 不计 handle）
bool RunBadParams(aclsolverHandle_t handle, aclrtStream stream)
{
    int lwork = 0;
    if (aclsolverSorgqr_bufferSize(handle, kBpM, kBpN, kBpK, nullptr, kBpLda, nullptr, &lwork)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        return false;
    }
    Buffers buf;
    if (!AllocFor(buf, static_cast<size_t>(kBpLda) * kBpN, kBpK, lwork))
    {
        buf.Free();
        return false;
    }
    bool ok = true;
    CheckArgPositions(handle, stream, buf, lwork, &ok);
    CheckBufferSizeArgs(handle, &ok);
    buf.Free();
    return ok;
}

/// 空问题：m 或 n 为 0 时成功返回且 info=0
bool RunEmpty(aclsolverHandle_t handle, int m, int n, int k)
{
    int lwork = -1;
    if (aclsolverSorgqr_bufferSize(handle, m, n, k, nullptr, std::max(m, 1), nullptr, &lwork) !=
        ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  空问题 bufferSize 失败\n");
        return false;
    }
    uint8_t *info = nullptr;
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    const int32_t poison = -999;
    TEST_CHECK(aclrtMemcpy(info, sizeof(poison), &poison, sizeof(poison), ACL_MEMCPY_HOST_TO_DEVICE));
    const aclsolverStatus_t st = aclsolverSorgqr(handle, m, n, k, nullptr, std::max(m, 1), nullptr, nullptr, lwork,
                                                 reinterpret_cast<int *>(info));
    int32_t got = -12345;
    (void)aclrtMemcpy(&got, sizeof(got), info, sizeof(got), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(info);
    const bool ok = (st == ACLSOLVER_STATUS_SUCCESS) && (lwork == 0) && (got == 0);
    printf("  空问题 m=%d n=%d k=%d: status=%d lwork=%d info=%d  %s\n", m, n, k, (int)st, lwork, got,
           ok ? "PASS" : "FAIL");
    return ok;
}

/// k=0：Q 应为单位阵的前 n 列（LAPACK sorgqr 语义），须真实写出
bool RunKZero(aclsolverHandle_t handle, aclrtStream stream, int m, int n)
{
    int lwork = 0;
    if (aclsolverSorgqr_bufferSize(handle, m, n, 0, nullptr, m, nullptr, &lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        return false;
    }
    std::vector<float> Ah(static_cast<size_t>(m) * n, -7.0f);  // 刻意填脏值
    Buffers buf;
    if (!AllocFor(buf, Ah.size(), 0, lwork))
    {
        buf.Free();
        return false;
    }
    TEST_CHECK(aclrtMemcpy(buf.A, Ah.size() * sizeof(float), Ah.data(), Ah.size() * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    const aclsolverStatus_t st = aclsolverSorgqr(handle, m, n, 0, reinterpret_cast<float *>(buf.A), m, nullptr,
                                                 reinterpret_cast<float *>(buf.ws), lwork,
                                                 reinterpret_cast<int *>(buf.info));
    bool ok = (st == ACLSOLVER_STATUS_SUCCESS);
    if (ok)
    {
        TEST_CHECK(aclrtSynchronizeStream(stream));
        TEST_CHECK(aclrtMemcpy(Ah.data(), Ah.size() * sizeof(float), buf.A, Ah.size() * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST));
        for (int c = 0; c < n && ok; ++c)
        {
            for (int r = 0; r < m; ++r)
            {
                const float want = (r == c) ? 1.0f : 0.0f;
                if (Ah[static_cast<size_t>(c) * m + r] != want)
                {
                    ok = false;
                    break;
                }
            }
        }
    }
    buf.Free();
    printf("  k=0 (m=%d n=%d) 应得单位阵前 n 列: status=%d  %s\n", m, n, (int)st, ok ? "PASS" : "FAIL");
    return ok;
}

/// 全量 case 扫描
int RunSweep(aclsolverHandle_t handle, aclrtStream stream, const char *path, int maxCols, int ldaExtra)
{
    int pass = 0, total = 0;
    const bool loaded = ForEachCaseRow(path, 4, [&](const std::vector<double> &row, int idx) {
        const int m = RowInt(row, 0, 0);
        const int n = RowInt(row, 1, 0);
        const int k = RowInt(row, 2, 0);
        const int lda0 = RowInt(row, 3, 0);
        Tally(RunShape(handle, stream, m, n, k, std::max(lda0, m) + ldaExtra, idx, maxCols, true), &pass, &total);
    });
    if (!loaded)
    {
        return 1;
    }
    printf("\n全量扫描：%d / %d PASS\n", pass, total);
    return (pass == total) ? 0 : 1;
}

/// 测一个形状的稳态耗时：先热机 3 次，再取 3 趟计时的最小值。
/// 首趟芯片未提频，实测 mean 能比 min 高近一倍，所以取 min 当稳态。
/// 返回 false 表示这条用例没测成（原因已打印），调用方应跳过它。
bool OrgqrTimeOne(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int k, int lda, int lwork,
                  const std::vector<float> &fac, const std::vector<float> &tau, double *best)
{
    Buffers buf;
    if (!AllocFor(buf, fac.size(), k, lwork))
    {
        buf.Free();
        printf("  m=%-5d n=%-5d k=%-5d  显存分配失败\n", m, n, k);
        return false;
    }
    if (k > 0)
    {
        (void)aclrtMemcpy(buf.tau, tau.size() * sizeof(float), tau.data(), tau.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    }
    auto once = [&]() {
        return aclsolverSorgqr(handle, m, n, k, reinterpret_cast<float *>(buf.A), lda,
                               reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.ws), lwork,
                               reinterpret_cast<int *>(buf.info));
    };
    auto reload = [&]() {
        (void)aclrtMemcpy(buf.A, fac.size() * sizeof(float), fac.data(), fac.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    };
    const double work = static_cast<double>(m) * n;
    const int iters = (work < 1.0e4) ? 30 : ((work < 1.0e6) ? 10 : 3);
    *best = TimeSteadyMin(stream, iters, 3, reload, once);
    buf.Free();
    return true;
}

/// 一行用例的性能测量：查 lwork、造输入、再测稳态耗时。
/// 返回 false 表示这条跳过（bufferSize 不接受这个形状，或没测成）。
bool OrgqrTimeRow(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int k, int lda, double *best)
{
    int lwork = 0;
    if (aclsolverSorgqr_bufferSize(handle, m, n, k, nullptr, lda, nullptr, &lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        return false;
    }
    std::vector<float> fac;
    std::vector<float> tau;
    MakeInput(m, n, k, lda, 0, fac, tau);
    return OrgqrTimeOne(handle, stream, m, n, k, lda, lwork, fac, tau, best);
}

/// 全量性能扫描：判据 T_NPU <= T_GPU/0.35
int RunPerfSweep(aclsolverHandle_t handle, aclrtStream stream, const char *path, int ldaExtra)
{
    std::vector<std::vector<double>> rows;
    if (!LoadCaseRows(path, rows))
    {
        printf("打不开用例清单 %s\n", path);
        return 1;
    }
    int pass = 0, total = 0;
    double worst = 1e30;
    int worstM = 0, worstN = 0, worstK = 0;
    for (const auto &row : rows)
    {
        const double gpuUs = RowDouble(row, 4, -1.0);
        if (row.size() < 4 || gpuUs <= 0.0)
        {
            continue;
        }
        const int m = RowInt(row, 0, 0);
        const int n = RowInt(row, 1, 0);
        const int k = RowInt(row, 2, 0);
        const int lda = std::max(RowInt(row, 3, 0), m) + ldaExtra;
        double best = 1e30;
        if (!OrgqrTimeRow(handle, stream, m, n, k, lda, &best))
        {
            continue;
        }
        ++total;
        const std::string label =
            "m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        double times = 0.0;
        pass += ExpectPerf(label.c_str(), best, gpuUs, &times) ? 1 : 0;
        if (times < worst)
        {
            worst = times;
            worstM = m;
            worstN = n;
            worstK = k;
        }
    }
    printf("\n性能扫描：%d / %d PASS（判据 >= 0.35x GPU）；最差 %5.2fx 于 m=%d n=%d k=%d\n", pass, total, worst,
           worstM, worstN, worstK);
    return (pass == total) ? 0 : 1;
}

/// 全量清单模式：sweep 跑精度、perfsweep 跑性能，两者共用同一份用例清单
int RunSweepMode(const std::string &mode, int argc, char **argv)
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    const char *path = (argc > 2) ? argv[2] : "./data/sorgqr_cases.txt";
    const int arg3 = ArgInt(argc, argv, 3, (mode == "sweep" ? 48 : 0));
    const int arg4 = ArgInt(argc, argv, 4, 0);
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    int rc = 0;
    if (mode == "sweep")
    {
        printf("== Sorgqr 全量扫描（SQRT02 + 跨次 bit-wise 确定性）maxCols=%d ldaExtra=%d ==\n", arg3, arg4);
        rc = RunSweep(handle, stream, path, arg3, arg4);
    }
    else
    {
        printf("== Sorgqr 全量性能扫描（判据 T_NPU <= T_GPU/0.35）ldaExtra=%d ==\n", arg3);
        rc = RunPerfSweep(handle, stream, path, arg3);
    }
    TeardownAcl(handle, stream);
    return rc;
}

/// 正向形状覆盖：含 k<n 与 lda padding，每条跑 2 次比 bit-wise
void RunShapeCases(aclsolverHandle_t handle, aclrtStream stream, int *pass, int *total)
{
    printf("== 正向用例（含 k<n、lda padding，各跑 2 次比 bit-wise）==\n");
    struct Shape
    {
        int m, n, k, pad;
    };
    const Shape shapes[] = {
        {2, 2, 2, 0},     {4, 4, 4, 0},     {8, 8, 8, 0},      {16, 16, 8, 0},   {18, 8, 8, 0},
        {37, 29, 11, 3},  {57, 57, 57, 0},  {64, 64, 64, 0},   {88, 88, 44, 0},  {128, 64, 64, 7},
        {129, 129, 129, 7}, {256, 128, 128, 0}, {256, 128, 64, 32}, {512, 512, 512, 0},
    };
    for (int i = 0; i < static_cast<int>(sizeof(shapes) / sizeof(shapes[0])); ++i)
    {
        const Shape &s = shapes[i];
        Tally(RunShape(handle, stream, s.m, s.n, s.k, s.m + s.pad, i, 48, true), pass, total);
    }
}

} // namespace

int main(int argc, char **argv)
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    const std::string mode = (argc >= 2) ? argv[1] : "";

    if (mode == "sweep" || mode == "perfsweep")
    {
        return RunSweepMode(mode, argc, argv);
    }

    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    int pass = 0, total = 0;
    printf("== 参数校验（-i 语义）==\n");
    Tally(RunBadParams(handle, stream), &pass, &total);

    printf("== 空问题 ==\n");
    // 注意 m=0,n=8 **不是**空问题而是非法参数（LAPACK sorgqr 的 N>M 走 info=-2），
    // 已在参数校验里单独覆盖，不要放进这个列表。
    for (auto t : {std::vector<int>{0, 0, 0}, {8, 0, 0}, {128, 0, 0}})
    {
        Tally(RunEmpty(handle, t[0], t[1], t[2]), &pass, &total);
    }

    printf("== k=0（应得单位阵前 n 列）==\n");
    for (auto mn : {std::pair<int, int>{16, 8}, {64, 64}})
    {
        Tally(RunKZero(handle, stream, mn.first, mn.second), &pass, &total);
    }

    RunShapeCases(handle, stream, &pass, &total);

    printf("\n%d / %d PASS\n", pass, total);
    TeardownAcl(handle, stream);
    return (pass == total) ? 0 : 1;
}
