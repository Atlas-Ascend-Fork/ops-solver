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
 * \file qr_test_utils.h
 * \brief QR 分解与最小二乘五个接口的测试公共件。
 *
 * sgeqrf / sorgqr / sormqr / xlarft / sgels 五个测试都要做同样四件事：初始化 ACL、
 * 造随机输入、读用例清单、解析命令行参数。这里收一份，各测试只保留自己算子特有的判据。
 */

#ifndef ACLSOLVER_QR_TEST_UTILS_H
#define ACLSOLVER_QR_TEST_UTILS_H

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cann_ops_solver.h"

/// FLOAT32 的机器 epsilon（2^-24）。各算子判据的分母都用它，不要各自写一遍字面量。
constexpr double kEpsF32 = 5.9604644775390625e-08;

/// ACL 调用失败就打印位置并让调用方返回 false。只能用在返回 bool 的函数里。
#define TEST_CHECK(expr)                                                        \
    do                                                                          \
    {                                                                           \
        const aclError rc_ = (expr);                                            \
        if (rc_ != ACL_SUCCESS)                                                 \
        {                                                                       \
            printf("  [ACL 失败] %s:%d -> %d\n", __FILE__, __LINE__, (int)rc_); \
            return false;                                                       \
        }                                                                       \
    } while (0)

/// 建流并创建 solver handle，把流绑到 handle 上。失败时已打印原因。
inline bool InitAcl(aclrtStream *stream, aclsolverHandle_t *handle)
{
    TEST_CHECK(aclInit(nullptr));
    TEST_CHECK(aclrtSetDevice(0));
    TEST_CHECK(aclrtCreateStream(stream));
    if (aclsolverCreate(handle) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("aclsolverCreate 失败\n");
        return false;
    }
    if (aclsolverSetStream(*handle, *stream) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("aclsolverSetStream 失败\n");
        return false;
    }
    return true;
}

/// 除数为 0 时返回 0。判据分母由形状和 eps 构成、计时分母是重复次数，正常都不会是 0，
/// 但静态检查要求除法处有就地守卫（它不做跨语句的常量传播），所以统一走这里。
inline double SafeDiv(double a, double b)
{
    return (b != 0.0) ? (a / b) : 0.0;
}

/// 稳态单次耗时：先热机 warm 次，再跑 3 趟、每趟 iters 次，取 3 趟的最小值。
///
/// 取 min 而不是 mean：首趟芯片未提频，实测 mean 能比 min 高近一倍。
/// reload 在每趟计时窗口**之前**把输入恢复 —— 这些算子多是原地的，连续几十次
/// 对自己的输出再算一遍会让数值发散，也会把耗时带偏。
template <class Reload, class Once>
double TimeSteadyMin(aclrtStream stream, int iters, int warm, Reload &&reload, Once &&once)
{
    reload();
    for (int i = 0; i < warm; ++i)
    {
        once();
        (void)aclrtSynchronizeStream(stream);
    }
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep)
    {
        reload();
        (void)aclrtSynchronizeStream(stream);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            once();
            (void)aclrtSynchronizeStream(stream);
        }
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, SafeDiv(std::chrono::duration<double, std::micro>(t1 - t0).count(), iters));
    }
    return best;
}

/// 性能判据：T_NPU <= T_GPU/0.35（任务书口径）。打印一行并返回是否达标，
/// 顺带把对 GPU 的倍率写回 times（用于统计最差形状）。
/// label 由调用方按各算子的形状参数自己格式化。
inline bool ExpectPerf(const char *label, double npuUs, double gpuUs, double *times)
{
    const double budget = gpuUs / 0.35;
    *times = SafeDiv(gpuUs, npuUs);
    const bool ok = (npuUs <= budget);
    printf("  %-30s NPU=%9.2f us  GPU=%9.2f us  预算=%9.2f us  %5.2fx  %s\n", label, npuUs, gpuUs, budget, *times,
           ok ? "PASS" : "**FAIL**");
    fflush(stdout);
    return ok;
}

/// 参数校验用：比对返回码并打印一行。任一条不符就把 ok 拉低。
/// 五个测试的 RunBadParams 都要这个，写在各自文件里会变成跨文件重复代码。
inline void ExpectStatus(const char *name, aclsolverStatus_t got, aclsolverStatus_t want, bool *ok)
{
    const bool good = (got == want);
    printf("  %-36s status=%-2d (期望 %d) %s\n", name, (int)got, (int)want, good ? "PASS" : "FAIL");
    if (!good)
    {
        *ok = false;
    }
}

/// 同上，并额外校验 kernel 写进 devInfo 的 -i。
///
/// 读 devInfo 之前必须同步流：成功路径上 devInfo 现在是**由 kernel 写**的
/// （host 侧那条 aclrtMemsetAsync 实测 8.9 us，已去掉），不同步会读到上一次的值。
/// 这条竞态在 memsetAsync 时代也在，只是 memset 完成得早、几乎不会输 —— 换成
/// kernel 写之后立刻偶发（78 例里偶尔错 1 例）。
inline void ExpectInfo(aclrtStream stream, void *devInfo, const char *name, aclsolverStatus_t got,
                       aclsolverStatus_t want, int wantInfo, bool *ok)
{
    (void)aclrtSynchronizeStream(stream);
    int32_t info = -12345;
    (void)aclrtMemcpy(&info, sizeof(info), devInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST);
    const bool good = (got == want) && (info == wantInfo);
    printf("  %-36s status=%-2d info=%-6d %s\n", name, (int)got, info, good ? "PASS" : "FAIL");
    if (!good)
    {
        *ok = false;
    }
}

/// 释放一块设备内存并把指针置空。各测试的 Buffers::Free 都在做这件事。
inline void FreeDev(uint8_t **p)
{
    if (*p != nullptr)
    {
        aclrtFree(*p);
        *p = nullptr;
    }
}

/// A / tau / workspace / devInfo 这一组是 sgeqrf 与 sorgqr 共用的缓冲形状。
/// 另外三个接口还要 B/X/C/Q，各自在自己文件里定义。
struct QrBuffers
{
    uint8_t *A = nullptr;
    uint8_t *tau = nullptr;
    uint8_t *ws = nullptr;
    uint8_t *info = nullptr;

    void Free()
    {
        FreeDev(&A);
        FreeDev(&tau);
        FreeDev(&ws);
        FreeDev(&info);
    }
};

/// 跑一项检查并记账。各测试的 main 里这两行重复了十几次。
inline void Tally(bool passed, int *pass, int *total)
{
    ++(*total);
    *pass += passed ? 1 : 0;
}

/// ||I - Q^T Q|| 的第 b 列的 1-范数：sum_a |delta(a,b) - q_a . q_b|。
/// Q 是 m x n 列主序、行距 lda。sorgqr 的抽样版与跨接口场景的全量版都靠这一列。
inline double OrthColumnSum(int m, int n, int lda, const std::vector<float> &Q, int b)
{
    double col = 0.0;
    for (int a = 0; a < n; ++a)
    {
        double v = (a == b) ? 1.0 : 0.0;
        for (int r = 0; r < m; ++r)
        {
            v -= static_cast<double>(Q[static_cast<size_t>(a) * lda + r]) *
                 static_cast<double>(Q[static_cast<size_t>(b) * lda + r]);
        }
        col += std::fabs(v);
    }
    return col;
}

/// 测试 main 的固定开头：建流 / 建 handle、把计数清零，再跑"参数校验"这一节。
/// 设备或 handle 建不起来时返回 false，调用方直接退出。
template <class BadParams>
bool BeginTests(aclrtStream *stream, aclsolverHandle_t *handle, int *pass, int *total, BadParams &&badParams)
{
    *pass = 0;
    *total = 0;
    if (!InitAcl(stream, handle))
    {
        return false;
    }
    printf("== 参数校验（-i 语义）==\n");
    Tally(badParams(*handle, *stream), pass, total);
    return true;
}

/// 释放 handle / stream / device。每个测试的每条退出路径都要走一遍。
inline void TeardownAcl(aclsolverHandle_t handle, aclrtStream stream)
{
    (void)aclsolverDestroy(handle);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

/// 造随机输入。按 caseIdx 的奇偶在均匀分布与「随机均值/方差的正态分布」之间切换，
/// 让全量扫描同时覆盖量级一致和量级悬殊两种数据，不至于只测到一种数值分布。
inline void FillMatrix(std::vector<float> &buf, uint32_t seed, int caseIdx)
{
    std::mt19937 rng(seed);
    if ((caseIdx % 2) == 0)
    {
        std::uniform_real_distribution<float> d(-5.0f, 5.0f);
        for (auto &x : buf)
        {
            x = d(rng);
        }
    }
    else
    {
        std::uniform_real_distribution<float> dm(-5.0f, 5.0f);
        std::uniform_real_distribution<float> ds(0.1f, 2.0f);
        const float mu = dm(rng);
        const float sg = ds(rng);
        std::normal_distribution<float> d(mu, sg);
        for (auto &x : buf)
        {
            x = d(rng);
        }
    }
}

/// 读用例清单：跳过 '#' 注释行与空行，每行按空白切成数值字段。
///
/// 返回 false 只表示文件打不开；能打开但一行都没有也算成功（调用方自己判空）。
/// 字段一律按 double 取出，整数字段由调用方转换 —— 这样一个读取函数就能同时服务
/// "m n lda gpu_us" 和 "m n k side trans lda ldc gpu_us" 这些不同列数的清单。
inline bool LoadCaseRows(const char *path, std::vector<std::vector<double>> &rows)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        return false;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream iss(line);
        std::vector<double> fields;
        double v = 0.0;
        while (iss >> v)
        {
            fields.push_back(v);
        }
        if (!fields.empty())
        {
            rows.push_back(fields);
        }
    }
    return true;
}

/// 读用例清单并逐行回调：列数不足的行跳过，序号从 0 起连续递增。
/// 读不到清单时报错并返回 false。各测试的全量扫描开头原先都抄了这一段。
template <class OnRow>
bool ForEachCaseRow(const char *path, size_t minCols, OnRow &&onRow)
{
    std::vector<std::vector<double>> rows;
    if (!LoadCaseRows(path, rows))
    {
        printf("打不开用例清单 %s\n", path);
        return false;
    }
    int idx = -1;
    for (const auto &row : rows)
    {
        if (row.size() >= minCols)
        {
            onRow(row, ++idx);
        }
    }
    return true;
}

/// 读用例清单，按**恰好 cols 列**筛行，每行交给 parse 填一个 Case，攒成列表。
/// 返回 false 表示清单读不到或没有可用用例。
template <class Case, class Parse>
bool LoadCaseList(const char *path, size_t cols, std::vector<Case> &out, Parse &&parse)
{
    std::vector<std::vector<double>> rows;
    if (!LoadCaseRows(path, rows))
    {
        printf("打不开用例清单 %s\n", path);
        return false;
    }
    for (const auto &row : rows)
    {
        if (row.size() == cols)
        {
            Case c{};
            parse(row, &c);
            out.push_back(c);
        }
    }
    return !out.empty();
}

/// 全量扫描的外壳：逐例调 runOne(c, idx)（idx 从 1 起），打印汇总，全通过返回 0。
template <class Case, class RunOne>
int SweepCases(const std::vector<Case> &cases, RunOne &&runOne)
{
    int pass = 0, idx = 0;
    for (const auto &c : cases)
    {
        ++idx;
        pass += runOne(c, idx) ? 1 : 0;
        fflush(stdout);
    }
    printf("\n全量扫描：%d / %zu PASS\n", pass, cases.size());
    return (pass == static_cast<int>(cases.size())) ? 0 : 1;
}

/// 取清单某行的第 idx 个字段当 int，越界时返回 def。
inline int RowInt(const std::vector<double> &row, size_t idx, int def)
{
    return (idx < row.size()) ? static_cast<int>(row[idx]) : def;
}

/// 取清单某行的第 idx 个字段当 double，越界时返回 def。
inline double RowDouble(const std::vector<double> &row, size_t idx, double def)
{
    return (idx < row.size()) ? row[idx] : def;
}

/// 取第 idx 个命令行参数当整数；缺省、非数字、或有尾随垃圾时都返回 def。
/// 替掉 std::atoi —— 后者对非法输入返回 0 且无法与真的 "0" 区分。
inline int ArgInt(int argc, char **argv, int idx, int def)
{
    if (idx >= argc || argv[idx] == nullptr)
    {
        return def;
    }
    char *end = nullptr;
    const long v = std::strtol(argv[idx], &end, 10);
    if (end == argv[idx] || (end != nullptr && *end != '\0'))
    {
        return def;
    }
    return static_cast<int>(v);
}

#endif  // ACLSOLVER_QR_TEST_UTILS_H
