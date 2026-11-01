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
 * \file sormqr_test.cpp
 * \brief aclsolverSormqr 的端到端测试：判据、确定性、参数校验的 -i 语义、全量 case 扫描
 *
 * 判据取任务书 §3.2.1-3 原文（SQRT03，RESULT(1)~(4)）：
 *     ratio = ||op2 - op1||_1 / ( nq * ||C||_1 * eps )，  ratio <= max(5*ratio_cpu, 0.1) 为通过
 * 其中
 *   op1 = 被测 aclsolverSormqr 的输出；
 *   op2 = **参考路径**：同一份 (v, tau) 先经 aclsolverSorgqr 生成显式 nq x nq 的 Q，
 *         再按同一 side/trans 做矩阵乘法；
 *   nq  = op(Q) 的阶数（side = LEFT 时 m、RIGHT 时 n），**不是** C 的行数；
 *   ratio_cpu = 同一公式在纯 CPU 链路（CPU 施加反射子 vs CPU org2r + 矩阵乘）上的实测值。
 *
 * 任务书也写明验收顺序：**Sorgqr 先过，Sormqr 才能验** —— op2 依赖 orgqr 的输出。
 *
 * 1-范数是"列和的最大值"，所以列数很大时按等步长**抽样列**：被抽到的列是精确的
 * => 得到真 1-范数的**下界** => 超阈一定是真失败；未超阈时标注"抽样下界"。
 *
 * 模式：
 *   sormqr_test                              功能 + 参数校验 + 六种 (side,trans) 组合
 *   sormqr_test sweep  <清单> [maxCols] [ldExtra]  全量 case 精度 + 跨次 bit-wise 确定性
 *   sormqr_test perfsweep <清单>             全量 case 性能（判据 >= 0.35x GPU）
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
    uint8_t *A = nullptr;
    uint8_t *tau = nullptr;
    uint8_t *C = nullptr;
    uint8_t *Q = nullptr;
    uint8_t *ws = nullptr;
    uint8_t *info = nullptr;
    void Free()
    {
        if (A) aclrtFree(A);
        if (tau) aclrtFree(tau);
        if (C) aclrtFree(C);
        if (Q) aclrtFree(Q);
        if (ws) aclrtFree(ws);
        if (info) aclrtFree(info);
        A = tau = C = Q = ws = info = nullptr;
    }
};

/// ||C||_1（列和的最大值）
double Norm1(int m, int n, int ldc, const std::vector<float> &C)
{
    double worst = 0.0;
    for (int c = 0; c < n; ++c)
    {
        double s = 0.0;
        for (int r = 0; r < m; ++r)
        {
            s += std::fabs(static_cast<double>(C[static_cast<size_t>(c) * ldc + r]));
        }
        worst = std::max(worst, s);
    }
    return worst;
}

/// CPU 的 ormqr：逐个施加 H_i = I - tau_i v_i v_i^T（这就是 op1 的参考路径，**不要**
/// 换成"先造显式 Q 再乘"，那是 op2 的路径，两条路径必须不同才量得出 ratio_cpu）
///
/// Q = H_1 H_2 ... H_k，所以：
///   LEFT  & N : Q  C = H_1(H_2(...(H_k C)))  => i 从 k-1 到 0
///   LEFT  & T : Q^T C = H_k(...(H_1 C))      => i 从 0 到 k-1
///   RIGHT & N : C Q  = ((C H_1) H_2)...      => i 从 0 到 k-1
///   RIGHT & T : C Q^T = ((C H_k) ...) H_1    => i 从 k-1 到 0
/// 左乘的单个反射子：w = v^T C（长度 n），再 C -= tau * v w^T
void OrmqrCpuLeftStep(int i, int nq, int n, const float *v, float ti, std::vector<float> &w, std::vector<float> &C,
                      int ldc)
{
    for (int c = 0; c < n; ++c)
    {
        float s = C[static_cast<size_t>(c) * ldc + i];
        for (int r = i + 1; r < nq; ++r)
        {
            s += v[r] * C[static_cast<size_t>(c) * ldc + r];
        }
        w[static_cast<size_t>(c)] = s;
    }
    for (int c = 0; c < n; ++c)
    {
        const float f = ti * w[static_cast<size_t>(c)];
        C[static_cast<size_t>(c) * ldc + i] -= f;
        for (int r = i + 1; r < nq; ++r)
        {
            C[static_cast<size_t>(c) * ldc + r] -= f * v[r];
        }
    }
}

/// 右乘的单个反射子：w = C v（长度 m），再 C -= tau * w v^T
void OrmqrCpuRightStep(int i, int nq, int m, const float *v, float ti, std::vector<float> &w, std::vector<float> &C,
                       int ldc)
{
    for (int r = 0; r < m; ++r)
    {
        w[static_cast<size_t>(r)] = C[static_cast<size_t>(i) * ldc + r];
    }
    for (int c = i + 1; c < nq; ++c)
    {
        const float vc = v[c];
        if (vc == 0.0f)
        {
            continue;
        }
        for (int r = 0; r < m; ++r)
        {
            w[static_cast<size_t>(r)] += vc * C[static_cast<size_t>(c) * ldc + r];
        }
    }
    for (int r = 0; r < m; ++r)
    {
        C[static_cast<size_t>(i) * ldc + r] -= ti * w[static_cast<size_t>(r)];
    }
    for (int c = i + 1; c < nq; ++c)
    {
        const float f = ti * v[c];
        if (f == 0.0f)
        {
            continue;
        }
        for (int r = 0; r < m; ++r)
        {
            C[static_cast<size_t>(c) * ldc + r] -= f * w[static_cast<size_t>(r)];
        }
    }
}

void OrmqrCpu(bool isLeft, bool isTrans, int m, int n, int k, const std::vector<float> &A, int lda,
              const std::vector<float> &tau, std::vector<float> &C, int ldc)
{
    const int nq = isLeft ? m : n;
    const bool forward = isLeft ? isTrans : !isTrans;
    std::vector<float> w(static_cast<size_t>(isLeft ? n : m), 0.0f);
    for (int t = 0; t < k; ++t)
    {
        const int i = forward ? t : (k - 1 - t);
        const float ti = tau[static_cast<size_t>(i)];
        if (ti == 0.0f)
        {
            continue;
        }
        // v_i：第 i 个位置为 1，下方取 A(i+1:nq-1, i)
        const float *v = &A[static_cast<size_t>(i) * lda];
        if (isLeft)
        {
            OrmqrCpuLeftStep(i, nq, n, v, ti, w, C, ldc);
        }
        else
        {
            OrmqrCpuRightStep(i, nq, m, v, ti, w, C, ldc);
        }
    }
}

/// op2 的第 c 列（左乘）：op2(:,c) = op(Q) * C(:,c)，op(Q) = Q 或 Q^T
void Sqrt03ColLeft(bool isTrans, int m, int nq, int ldc, const std::vector<float> &Cin, const std::vector<float> &Q,
                   int ldq, int c, std::vector<float> &col)
{
    for (int r = 0; r < m; ++r)
    {
        float s = 0.0f;
        for (int t = 0; t < nq; ++t)
        {
            const float q = isTrans ? Q[static_cast<size_t>(r) * ldq + t]   // Q^T(r,t) = Q(t,r)
                                    : Q[static_cast<size_t>(t) * ldq + r];  // Q(r,t)
            s += q * Cin[static_cast<size_t>(c) * ldc + t];
        }
        col[static_cast<size_t>(r)] = s;
    }
}

/// op2 的第 c 列（右乘）：op2(:,c) = C * op(Q)(:,c) = sum_t C(:,t) * op(Q)(t,c)
void Sqrt03ColRight(bool isTrans, int m, int nq, int ldc, const std::vector<float> &Cin, const std::vector<float> &Q,
                    int ldq, int c, std::vector<float> &col)
{
    for (int r = 0; r < m; ++r)
    {
        col[static_cast<size_t>(r)] = 0.0f;
    }
    for (int t = 0; t < nq; ++t)
    {
        const float q = isTrans ? Q[static_cast<size_t>(t) * ldq + c]   // Q^T(t,c) = Q(c,t)
                                : Q[static_cast<size_t>(c) * ldq + t];  // Q(t,c)
        if (q == 0.0f)
        {
            continue;
        }
        for (int r = 0; r < m; ++r)
        {
            col[static_cast<size_t>(r)] += q * Cin[static_cast<size_t>(t) * ldc + r];
        }
    }
}

/// SQRT03：ratio = ||op2 - op1||_1 / (nq * ||C||_1 * eps)
///
/// op2 由显式 Q（nq x nq，ld = ldq）在 **fp32** 下乘出来（照 LAPACK sqrt03 用 sgemm 的口径；
/// 用 double 乘会让 op2 过于"干净"，把 ratio 抬高，判据就不是同一个判据了）。
/// 1-范数是列和最大值 => 列数 > maxCols 时等步长抽样，结果是真范数的下界。
double Sqrt03(bool isLeft, bool isTrans, int m, int n, int ldc, const std::vector<float> &Cin,
              const std::vector<float> &op1, const std::vector<float> &Q, int ldq, double normC, int maxCols,
              bool *exact)
{
    const int nq = isLeft ? m : n;
    const int step = (maxCols > 0 && n > maxCols) ? ((n + maxCols - 1) / maxCols) : 1;
    if (exact != nullptr)
    {
        *exact = (step == 1);
    }
    double worst = 0.0;
    std::vector<float> col(static_cast<size_t>(m), 0.0f);
    for (int c = 0; c < n; c += step)
    {
        if (isLeft)
        {
            Sqrt03ColLeft(isTrans, m, nq, ldc, Cin, Q, ldq, c, col);
        }
        else
        {
            Sqrt03ColRight(isTrans, m, nq, ldc, Cin, Q, ldq, c, col);
        }
        double s = 0.0;
        for (int r = 0; r < m; ++r)
        {
            s += std::fabs(static_cast<double>(col[static_cast<size_t>(r)]) -
                           static_cast<double>(op1[static_cast<size_t>(c) * ldc + r]));
        }
        worst = std::max(worst, s);
    }
    const double den = static_cast<double>(nq) * std::max(normC, 1e-30) * kEpsF32;
    return SafeDiv(worst, den);
}

/// 造一组合法输入：A/tau 来自对一个 nq x k 随机矩阵做 CPU geqrf，C 为同分布随机矩阵
void MakeInput(bool isLeft, int m, int n, int k, int lda, int ldc, int caseIdx, std::vector<float> &A,
               std::vector<float> &tau, std::vector<float> &C)
{
    const int nq = isLeft ? m : n;
    A.assign(static_cast<size_t>(lda) * std::max(k, 1), 0.0f);
    FillMatrix(A, 0x5eed0000u + static_cast<uint32_t>(caseIdx), caseIdx);
    // ★ Geqr2 写 min(nq, k) 个 tau。这里 k <= nq 所以正好 k 个；若将来放宽 k 的范围，
    //   按 k 开容量会堆越界（症状会伪装成"某个形状挂住"）。
    tau.assign(static_cast<size_t>(std::max(std::min(nq, k), 1)), 0.0f);
    if (k > 0 && nq > 0)
    {
        QrCore::Geqr2(nq, k, A.data(), lda, tau.data());
    }
    C.assign(static_cast<size_t>(ldc) * std::max(n, 1), 0.0f);
    FillMatrix(C, 0xC0ffee00u + static_cast<uint32_t>(caseIdx), caseIdx + 1);
}

bool AllocFor(Buffers &buf, size_t aElems, size_t tauElems, size_t cElems, size_t qElems, int lwork)
{
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.A), std::max<size_t>(aElems, 1) * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.tau), std::max<size_t>(tauElems, 1) * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.C), std::max<size_t>(cElems, 1) * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    if (qElems > 0)
    {
        TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.Q), qElems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    }
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.ws), std::max<size_t>(lwork, 1) * sizeof(float),
                           ACL_MEM_MALLOC_HUGE_FIRST));
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&buf.info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    return true;
}

/// 一个被测形状。RunShape 分出来的几段之间要来回传的标量都收在这里，
/// 免得每段都挂一长串形参。
struct MCase
{
    aclsolverSideMode_t side;
    aclsolverOperation_t trans;
    int m, n, k, lda, ldc;

    bool IsLeft() const
    {
        return side == ACLSOLVER_SIDE_LEFT;
    }
    bool IsTrans() const
    {
        return trans != ACLSOLVER_OP_N;
    }
    /// op(Q) 的边长：LEFT 时是 m，RIGHT 时是 n
    int Nq() const
    {
        return IsLeft() ? m : n;
    }
    /// k == 0 时 op(Q) = I，不需要造显式 Q
    bool NeedQ() const
    {
        return k > 0 && Nq() > 0;
    }
};

/// 查两个 workspace 大小。op2 需要显式 Q（nq x nq），用 aclsolverSorgqr 生成；
/// 它要求 n <= m 且 k <= n，这里取 m = n = nq、k = k，所以 Q 的存储要 nq 列
/// （不能复用 A 的 k 列）。
bool OrmqrQueryLwork(aclsolverHandle_t handle, const MCase &c, int *lwork, int *lworkQ)
{
    if (aclsolverSormqr_bufferSize(handle, c.side, c.trans, c.m, c.n, c.k, nullptr, c.lda, nullptr, nullptr, c.ldc,
                                   lwork)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  bufferSize 失败\n");
        return false;
    }
    if (c.NeedQ()
        && aclsolverSorgqr_bufferSize(handle, c.Nq(), c.Nq(), c.k, nullptr, c.Nq(), nullptr, lworkQ)
               != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  sorgqr_bufferSize 失败\n");
        return false;
    }
    return true;
}

/// 连跑 runs 次 Sormqr，比跨次 bit-wise 一致，带回最后一次的输出与 devInfo
bool OrmqrRunRepeats(aclsolverHandle_t handle, aclrtStream stream, const MCase &c, const Buffers &buf, int lwork,
                     const std::vector<float> &A, const std::vector<float> &tau, const std::vector<float> &C0,
                     int runs, std::vector<float> *op1, int32_t *info)
{
    const size_t aBytes = A.size() * sizeof(float);
    const size_t cBytes = C0.size() * sizeof(float);
    std::vector<float> first;
    for (int rep = 0; rep < runs; ++rep)
    {
        if (aclrtMemcpy(buf.A, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(buf.tau, tau.size() * sizeof(float), tau.data(), tau.size() * sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(buf.C, cBytes, C0.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        {
            printf("  上传失败\n");
            return false;
        }
        const aclsolverStatus_t st =
            aclsolverSormqr(handle, c.side, c.trans, c.m, c.n, c.k, reinterpret_cast<const float *>(buf.A), c.lda,
                            reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.C), c.ldc,
                            reinterpret_cast<float *>(buf.ws), lwork, reinterpret_cast<int *>(buf.info));
        const aclError sync = aclrtSynchronizeStream(stream);
        if (st != ACLSOLVER_STATUS_SUCCESS || sync != ACL_SUCCESS)
        {
            printf("  status=%d syncRc=%d\n", (int)st, (int)sync);
            return false;
        }
        (void)aclrtMemcpy(info, sizeof(*info), buf.info, sizeof(*info), ACL_MEMCPY_DEVICE_TO_HOST);
        op1->assign(C0.size(), 0.0f);
        (void)aclrtMemcpy(op1->data(), cBytes, buf.C, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (rep == 0)
        {
            first = *op1;
        }
        else if (first != *op1)
        {
            printf("  第 %d 次与第 1 次结果不同（确定性失败）\n", rep + 1);
            return false;
        }
    }
    return true;
}

/// 把 A 的前 k 列铺进 nq x nq 的存储（其余留 0），这是 sorgqr 与 org2r 的共同入参形式
void OrmqrPackQInput(const MCase &c, const std::vector<float> &A, std::vector<float> *Qin)
{
    const int nq = c.Nq();
    Qin->assign(static_cast<size_t>(nq) * nq, 0.0f);
    for (int col = 0; col < c.k; ++col)
    {
        for (int r = 0; r < nq; ++r)
        {
            (*Qin)[static_cast<size_t>(col) * nq + r] = A[static_cast<size_t>(col) * c.lda + r];
        }
    }
}

/// op2 的来源：同一份 (v, tau) 经 NPU sorgqr 生成显式 Q。
/// tau 已经在 OrmqrRunRepeats 里传上设备，这里直接用 buf.tau。
bool OrmqrBuildQ(aclsolverHandle_t handle, aclrtStream stream, const MCase &c, const Buffers &buf,
                 const std::vector<float> &A, int lworkQ, std::vector<float> *Q)
{
    const int nq = c.Nq();
    std::vector<float> Qin;
    OrmqrPackQInput(c, A, &Qin);
    const size_t qBytes = Qin.size() * sizeof(float);
    if (aclrtMemcpy(buf.Q, qBytes, Qin.data(), qBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
    {
        return false;
    }
    const aclsolverStatus_t stQ =
        aclsolverSorgqr(handle, nq, nq, c.k, reinterpret_cast<float *>(buf.Q), nq,
                        reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.ws), lworkQ,
                        reinterpret_cast<int *>(buf.info));
    if (stQ != ACLSOLVER_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
    {
        printf("  sorgqr status=%d\n", (int)stQ);
        return false;
    }
    Q->assign(Qin.size(), 0.0f);
    (void)aclrtMemcpy(Q->data(), qBytes, buf.Q, qBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return true;
}

/// CPU 同式的 ratio_cpu：op1_cpu = 逐个施加反射子；op2_cpu = CPU org2r 造 Q 再乘。
/// 判据阈值 max(5·ratio_cpu, 0.1) 要的就是这个现场实测值，不写死常数。
double OrmqrCpuRatio(const MCase &c, const std::vector<float> &A, const std::vector<float> &tau,
                     const std::vector<float> &C0, double normC, int maxCols)
{
    const int nq = c.Nq();
    std::vector<float> Ccpu = C0;
    OrmqrCpu(c.IsLeft(), c.IsTrans(), c.m, c.n, c.k, A, c.lda, tau, Ccpu, c.ldc);
    std::vector<float> Qcpu;
    OrmqrPackQInput(c, A, &Qcpu);
    QrCore::Org2r(nq, nq, c.k, Qcpu.data(), nq, tau.data());
    return Sqrt03(c.IsLeft(), c.IsTrans(), c.m, c.n, c.ldc, C0, Ccpu, Qcpu, nq, normC, maxCols, nullptr);
}

/// k == 0 时 op(Q) = I，C 必须原样不动
bool OrmqrCheckIdentity(const std::vector<float> &C0, const std::vector<float> &op1)
{
    for (size_t i = 0; i < C0.size(); ++i)
    {
        if (C0[i] != op1[i])
        {
            printf("  k=0 但 C 被改动了\n");
            return false;
        }
    }
    return true;
}

/// 判据 SQRT03：ratio <= max(5·ratio_cpu, 0.1)。verbose 时打一行明细。
bool OrmqrReport(const MCase &c, int32_t info, double ratio, double ratioCpu, bool exact, bool verbose)
{
    const double th = std::max(5.0 * ratioCpu, 0.1);
    const bool pass = (ratio <= th);
    if (verbose)
    {
        const char *sideName = c.IsLeft() ? "L" : "R";
        const char *transName =
            (c.trans == ACLSOLVER_OP_N) ? "N" : ((c.trans == ACLSOLVER_OP_T) ? "T" : "C");
        printf("  m=%-5d n=%-5d k=%-5d %s/%s lda=%-5d ldc=%-5d info=%d ratio=%.4f ratio_cpu=%.4f th=%.4f %s %s\n",
               c.m, c.n, c.k, sideName, transName, c.lda, c.ldc, info, ratio, ratioCpu, th,
               exact ? "精确" : "抽样下界", pass ? "PASS" : "**FAIL**");
    }
    return pass;
}

/// 查两段 workspace 并把设备缓冲分好
bool OrmqrSetup(aclsolverHandle_t handle, const MCase &c, const std::vector<float> &A,
                const std::vector<float> &tau, const std::vector<float> &C0, Buffers *buf, int *lwork,
                int *lworkQ)
{
    if (!OrmqrQueryLwork(handle, c, lwork, lworkQ))
    {
        return false;
    }
    const size_t qElems = c.NeedQ() ? static_cast<size_t>(c.Nq()) * c.Nq() : 0;
    return AllocFor(*buf, A.size(), tau.size(), C0.size(), qElems, std::max(*lwork, *lworkQ));
}

/// 算判据并打一行明细。k == 0 时 op(Q) = I，改为直接核对 C 没被动过。
/// 返回是否通过；ratio 回写给调用方（perf 模式要用）。
bool OrmqrJudge(const MCase &c, const std::vector<float> &A, const std::vector<float> &tau,
                const std::vector<float> &C0, const std::vector<float> &op1, const std::vector<float> &Q,
                int maxCols, int32_t info, bool verbose, double *ratio)
{
    bool exact = true;
    double ratioCpu = 0.0;
    *ratio = 0.0;
    if (c.NeedQ())
    {
        const double normC = Norm1(c.m, c.n, c.ldc, C0);
        *ratio = Sqrt03(c.IsLeft(), c.IsTrans(), c.m, c.n, c.ldc, C0, op1, Q, c.Nq(), normC, maxCols, &exact);
        ratioCpu = OrmqrCpuRatio(c, A, tau, C0, normC, maxCols);
    }
    else if (!OrmqrCheckIdentity(C0, op1))
    {
        return false;
    }
    return OrmqrReport(c, info, *ratio, ratioCpu, exact, verbose);
}

/// 跑一个形状：op1（NPU ormqr）+ op2（NPU orgqr 造 Q 再乘）+ CPU 同式 ratio_cpu
/// runs 次重复用于 bit-wise 确定性
bool RunShape(aclsolverHandle_t handle, aclrtStream stream, int caseIdx, aclsolverSideMode_t side,
              aclsolverOperation_t trans, int m, int n, int k, int lda, int ldc, int maxCols, int runs, bool verbose,
              double *ratioOut)
{
    const MCase c{side, trans, m, n, k, lda, ldc};
    std::vector<float> A, tau, C0;
    MakeInput(c.IsLeft(), m, n, k, lda, ldc, caseIdx, A, tau, C0);

    int lwork = 0;
    int lworkQ = 0;
    Buffers buf;
    if (!OrmqrSetup(handle, c, A, tau, C0, &buf, &lwork, &lworkQ))
    {
        buf.Free();
        return false;
    }

    std::vector<float> op1;
    int32_t info = -12345;
    bool ok = OrmqrRunRepeats(handle, stream, c, buf, lwork, A, tau, C0, runs, &op1, &info);
    if (ok && info != 0)
    {
        printf("  info=%d\n", info);
        ok = false;
    }
    std::vector<float> Q;
    if (ok && c.NeedQ())
    {
        ok = OrmqrBuildQ(handle, stream, c, buf, A, lworkQ, &Q);
    }
    buf.Free();
    if (!ok)
    {
        return false;
    }

    double ratio = 0.0;
    if (!OrmqrJudge(c, A, tau, C0, op1, Q, maxCols, info, verbose, &ratio))
    {
        return false;
    }
    if (ratioOut != nullptr)
    {
        *ratioOut = ratio;
    }
    return true;
}

// ---- 参数校验 ----
// 一组固定的合法入参，各条检查只改其中被测的那一个。
constexpr int kBpM = 64;
constexpr int kBpN = 32;
constexpr int kBpK = 16;
constexpr int kBpLda = 64;
constexpr int kBpLdc = 64;
constexpr aclsolverSideMode_t kLeft = ACLSOLVER_SIDE_LEFT;
constexpr aclsolverOperation_t kOpN = ACLSOLVER_OP_N;

/// 参数校验的设备侧固定实参。各组检查只改签名里被测的那一个。
struct MBadFixture
{
    aclsolverHandle_t handle;
    aclrtStream stream;
    float *A;
    float *tau;
    float *C;
    float *WS;
    int *INFO;
    void *devInfo;
    int lwork;
};

/// devInfo / handle 为空时接口写不了 info，只看返回码
void CheckNullHandleAndInfo(const MBadFixture &fx, bool *ok)
{
    ExpectStatus("devInfo=nullptr",
                 aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, fx.WS,
                                 fx.lwork, nullptr),
                 ACLSOLVER_STATUS_INVALID_VALUE, ok);
    ExpectStatus("handle=nullptr",
                 aclsolverSormqr(nullptr, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, fx.WS,
                                 fx.lwork, fx.INFO),
                 ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, ok);
}

/// 枚举与形状：side / trans / m / n / k（k 的上界随 side 变）
void CheckModeAndShapeArgs(const MBadFixture &fx, bool *ok)
{
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    ExpectInfo(fx.stream, fx.devInfo, "side 非法 -> info=-1",
               aclsolverSormqr(fx.handle, static_cast<aclsolverSideMode_t>(7), kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda,
                               fx.tau, fx.C, kBpLdc, fx.WS, fx.lwork, fx.INFO),
               bad, -1, ok);
    ExpectInfo(fx.stream, fx.devInfo, "trans 非法 -> info=-2",
               aclsolverSormqr(fx.handle, kLeft, static_cast<aclsolverOperation_t>(9), kBpM, kBpN, kBpK, fx.A,
                               kBpLda, fx.tau, fx.C, kBpLdc, fx.WS, fx.lwork, fx.INFO),
               bad, -2, ok);
    ExpectInfo(fx.stream, fx.devInfo, "m<0 -> info=-3",
               aclsolverSormqr(fx.handle, kLeft, kOpN, -1, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, fx.WS,
                               fx.lwork, fx.INFO),
               bad, -3, ok);
    ExpectInfo(fx.stream, fx.devInfo, "n<0 -> info=-4",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, -1, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, fx.WS,
                               fx.lwork, fx.INFO),
               bad, -4, ok);
    ExpectInfo(fx.stream, fx.devInfo, "k>nq -> info=-5",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpM + 1, fx.A, kBpLda, fx.tau, fx.C, kBpLdc,
                               fx.WS, fx.lwork, fx.INFO),
               bad, -5, ok);
    // side=RIGHT 时 nq = n，k 的上界也跟着变（这条曾经容易写成恒用 m）
    ExpectInfo(fx.stream, fx.devInfo, "k>n (RIGHT) -> info=-5",
               aclsolverSormqr(fx.handle, ACLSOLVER_SIDE_RIGHT, kOpN, kBpM, kBpN, kBpN + 1, fx.A, kBpLda, fx.tau,
                               fx.C, kBpLdc, fx.WS, fx.lwork, fx.INFO),
               bad, -5, ok);
}

/// 指针、ld 与 workspace：A / lda / tau / C / ldc / work / lwork，末尾压一条合法调用
void CheckPointerAndLdArgs(const MBadFixture &fx, bool *ok)
{
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    ExpectInfo(fx.stream, fx.devInfo, "A=nullptr -> info=-6",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, nullptr, kBpLda, fx.tau, fx.C, kBpLdc,
                               fx.WS, fx.lwork, fx.INFO),
               bad, -6, ok);
    ExpectInfo(fx.stream, fx.devInfo, "lda<nq -> info=-7",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpM - 1, fx.tau, fx.C, kBpLdc, fx.WS,
                               fx.lwork, fx.INFO),
               bad, -7, ok);
    ExpectInfo(fx.stream, fx.devInfo, "tau=nullptr -> info=-8",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, nullptr, fx.C, kBpLdc, fx.WS,
                               fx.lwork, fx.INFO),
               bad, -8, ok);
    ExpectInfo(fx.stream, fx.devInfo, "C=nullptr -> info=-9",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, nullptr, kBpLdc,
                               fx.WS, fx.lwork, fx.INFO),
               bad, -9, ok);
    ExpectInfo(fx.stream, fx.devInfo, "ldc<m -> info=-10",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpM - 1, fx.WS,
                               fx.lwork, fx.INFO),
               bad, -10, ok);
    ExpectInfo(fx.stream, fx.devInfo, "work=nullptr -> info=-11",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, nullptr,
                               fx.lwork, fx.INFO),
               bad, -11, ok);
    ExpectInfo(fx.stream, fx.devInfo, "lwork 太小 -> info=-12",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, fx.WS,
                               fx.lwork - 1, fx.INFO),
               bad, -12, ok);
    ExpectInfo(fx.stream, fx.devInfo, "lda>nq 合法",
               aclsolverSormqr(fx.handle, kLeft, kOpN, kBpM, kBpN, kBpK, fx.A, kBpLda, fx.tau, fx.C, kBpLdc, fx.WS,
                               fx.lwork, fx.INFO),
               ACLSOLVER_STATUS_SUCCESS, 0, ok);
}

/// 查询接口自己的参数校验
void CheckBufferSizeArgs(aclsolverHandle_t handle, bool *ok)
{
    const auto bad = ACLSOLVER_STATUS_INVALID_VALUE;
    int lw = -1;
    ExpectStatus("bufferSize lwork=nullptr",
                 aclsolverSormqr_bufferSize(handle, kLeft, kOpN, kBpM, kBpN, kBpK, nullptr, kBpLda, nullptr, nullptr,
                                            kBpLdc, nullptr),
                 bad, ok);
    ExpectStatus("bufferSize lda<nq",
                 aclsolverSormqr_bufferSize(handle, kLeft, kOpN, kBpM, kBpN, kBpK, nullptr, kBpM - 1, nullptr,
                                            nullptr, kBpLdc, &lw),
                 bad, ok);
}

/// 参数校验：13 个参数的 -i 语义（不计 handle）
bool RunBadParams(aclsolverHandle_t handle, aclrtStream stream)
{
    int lwork = 0;
    if (aclsolverSormqr_bufferSize(handle, kLeft, kOpN, kBpM, kBpN, kBpK, nullptr, kBpLda, nullptr, nullptr, kBpLdc,
                                   &lwork)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  bufferSize 失败\n");
        return false;
    }
    Buffers buf;
    if (!AllocFor(buf, static_cast<size_t>(kBpLda) * kBpK, static_cast<size_t>(kBpK),
                  static_cast<size_t>(kBpLdc) * kBpN, 0, lwork))
    {
        buf.Free();
        return false;
    }
    const MBadFixture fx{handle,
                         stream,
                         reinterpret_cast<float *>(buf.A),
                         reinterpret_cast<float *>(buf.tau),
                         reinterpret_cast<float *>(buf.C),
                         reinterpret_cast<float *>(buf.ws),
                         reinterpret_cast<int *>(buf.info),
                         buf.info,
                         lwork};
    bool ok = true;
    CheckNullHandleAndInfo(fx, &ok);
    CheckModeAndShapeArgs(fx, &ok);
    CheckPointerAndLdArgs(fx, &ok);
    CheckBufferSizeArgs(handle, &ok);
    buf.Free();
    return ok;
}

/// 空问题与 k=0：成功返回、info=0、lwork=0
bool RunEmpty(aclsolverHandle_t handle, int m, int n, int k)
{
    int lwork = -1;
    const aclsolverSideMode_t L = ACLSOLVER_SIDE_LEFT;
    if (aclsolverSormqr_bufferSize(handle, L, ACLSOLVER_OP_N, m, n, k, nullptr, std::max(m, 1), nullptr, nullptr,
                                   std::max(m, 1), &lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  空问题 bufferSize 失败\n");
        return false;
    }
    uint8_t *info = nullptr;
    TEST_CHECK(aclrtMalloc(reinterpret_cast<void **>(&info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    const int32_t poison = -999;
    TEST_CHECK(aclrtMemcpy(info, sizeof(poison), &poison, sizeof(poison), ACL_MEMCPY_HOST_TO_DEVICE));
    const aclsolverStatus_t st =
        aclsolverSormqr(handle, L, ACLSOLVER_OP_N, m, n, k, nullptr, std::max(m, 1), nullptr, nullptr,
                        std::max(m, 1), nullptr, lwork, reinterpret_cast<int *>(info));
    int32_t got = -12345;
    (void)aclrtMemcpy(&got, sizeof(got), info, sizeof(got), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(info);
    const bool ok = (st == ACLSOLVER_STATUS_SUCCESS) && (lwork == 0) && (got == 0);
    printf("  空问题 m=%d n=%d k=%d: status=%d lwork=%d info=%d  %s\n", m, n, k, (int)st, lwork, got,
           ok ? "PASS" : "FAIL");
    return ok;
}

/// trans = C 必须与 trans = T **逐位相同**（实数下共轭转置等于转置）
bool RunTransCEqualsT(aclsolverHandle_t handle, aclrtStream stream, aclsolverSideMode_t side, int m, int n, int k)
{
    const bool isLeft = (side == ACLSOLVER_SIDE_LEFT);
    const int nq = isLeft ? m : n;
    const int lda = nq, ldc = m;
    std::vector<float> A, tau, C0;
    MakeInput(isLeft, m, n, k, lda, ldc, 4, A, tau, C0);
    int lwork = 0;
    if (aclsolverSormqr_bufferSize(handle, side, ACLSOLVER_OP_T, m, n, k, nullptr, lda, nullptr, nullptr, ldc,
                                   &lwork) != ACLSOLVER_STATUS_SUCCESS)
    {
        return false;
    }
    Buffers buf;
    if (!AllocFor(buf, A.size(), tau.size(), C0.size(), 0, lwork))
    {
        buf.Free();
        return false;
    }
    std::vector<float> out[2];
    const size_t aBytes = A.size() * sizeof(float), cBytes = C0.size() * sizeof(float);
    const aclsolverOperation_t ops[2] = {ACLSOLVER_OP_T, ACLSOLVER_OP_C};
    for (int i = 0; i < 2; ++i)
    {
        TEST_CHECK(aclrtMemcpy(buf.A, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        TEST_CHECK(aclrtMemcpy(buf.tau, tau.size() * sizeof(float), tau.data(), tau.size() * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE));
        TEST_CHECK(aclrtMemcpy(buf.C, cBytes, C0.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        const aclsolverStatus_t st = aclsolverSormqr(
            handle, side, ops[i], m, n, k, reinterpret_cast<const float *>(buf.A), lda,
            reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.C), ldc,
            reinterpret_cast<float *>(buf.ws), lwork, reinterpret_cast<int *>(buf.info));
        if (st != ACLSOLVER_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        {
            buf.Free();
            return false;
        }
        out[i].assign(C0.size(), 0.0f);
        TEST_CHECK(aclrtMemcpy(out[i].data(), cBytes, buf.C, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    buf.Free();
    const bool ok = (out[0] == out[1]);
    printf("  trans=C 与 T 逐位相同 (%s m=%d n=%d k=%d): %s\n", isLeft ? "LEFT" : "RIGHT", m, n, k,
           ok ? "PASS" : "FAIL");
    return ok;
}

struct Case
{
    int m, n, k, side, trans, lda, ldc;
    double gpuUs;
};

bool LoadCases(const char *path, std::vector<Case> &out)
{
    return LoadCaseList(path, 8, out, [](const std::vector<double> &row, Case *c) {
        c->m = RowInt(row, 0, 0);
        c->n = RowInt(row, 1, 0);
        c->k = RowInt(row, 2, 0);
        c->side = RowInt(row, 3, 0);
        c->trans = RowInt(row, 4, 0);
        c->lda = RowInt(row, 5, 0);
        c->ldc = RowInt(row, 6, 0);
        c->gpuUs = row[7];
    });
}

int RunSweep(aclsolverHandle_t handle, aclrtStream stream, const char *path, int maxCols, int ldExtra)
{
    std::vector<Case> cases;
    if (!LoadCases(path, cases))
    {
        return 1;
    }
    return SweepCases(cases, [&](const Case &c, int idx) {
        const aclsolverSideMode_t side = (c.side == 0) ? ACLSOLVER_SIDE_LEFT : ACLSOLVER_SIDE_RIGHT;
        const aclsolverOperation_t trans =
            (c.trans == 0) ? ACLSOLVER_OP_N : ((c.trans == 1) ? ACLSOLVER_OP_T : ACLSOLVER_OP_C);
        // 每例跑两遍比 bit-wise，确定性和精度一起验
        // ldExtra：在清单给的最小合法 lda/ldc 上再加一档 padding（§3.5 要求 +8 / +32 两档）
        return RunShape(handle, stream, idx, side, trans, c.m, c.n, c.k, c.lda + ldExtra, c.ldc + ldExtra, maxCols,
                        2, true, nullptr);
    });
}

/// 测一条用例的稳态耗时：先热一趟（首趟芯片没提频），再取 3 趟的最小值。
/// 每趟前重拷 C（原地算子），**计时窗口放在拷贝之后**。
/// 返回 false 表示这条没测成（原因已打印）。
bool OrmqrTimeCase(aclsolverHandle_t handle, aclrtStream stream, const Case &c, int idx, double *best)
{
    const aclsolverSideMode_t side = (c.side == 0) ? ACLSOLVER_SIDE_LEFT : ACLSOLVER_SIDE_RIGHT;
    const aclsolverOperation_t trans =
        (c.trans == 0) ? ACLSOLVER_OP_N : ((c.trans == 1) ? ACLSOLVER_OP_T : ACLSOLVER_OP_C);
    std::vector<float> A, tau, C0;
    MakeInput(side == ACLSOLVER_SIDE_LEFT, c.m, c.n, c.k, c.lda, c.ldc, idx, A, tau, C0);
    int lwork = 0;
    if (aclsolverSormqr_bufferSize(handle, side, trans, c.m, c.n, c.k, nullptr, c.lda, nullptr, nullptr, c.ldc,
                                   &lwork)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  bufferSize 失败\n");
        return false;
    }
    Buffers buf;
    if (!AllocFor(buf, A.size(), tau.size(), C0.size(), 0, lwork))
    {
        buf.Free();
        return false;
    }
    const size_t aBytes = A.size() * sizeof(float), cBytes = C0.size() * sizeof(float);
    (void)aclrtMemcpy(buf.A, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(buf.tau, tau.size() * sizeof(float), tau.data(), tau.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    auto reload = [&]() {
        (void)aclrtMemcpy(buf.C, cBytes, C0.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    };
    auto once = [&]() {
        return aclsolverSormqr(handle, side, trans, c.m, c.n, c.k, reinterpret_cast<const float *>(buf.A), c.lda,
                               reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.C), c.ldc,
                               reinterpret_cast<float *>(buf.ws), lwork, reinterpret_cast<int *>(buf.info));
    };
    // 热机一次，同时确认这条能跑通
    reload();
    if (once() != ACLSOLVER_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
    {
        printf("  跑挂了\n");
        buf.Free();
        return false;
    }
    *best = TimeSteadyMin(stream, 1, 0, reload, once);
    buf.Free();
    return true;
}

int RunPerfSweep(aclsolverHandle_t handle, aclrtStream stream, const char *path)
{
    std::vector<Case> cases;
    if (!LoadCases(path, cases))
    {
        return 1;
    }
    int pass = 0, idx = 0;
    double worst = 1e30;
    Case worstCase{};
    for (const auto &c : cases)
    {
        ++idx;
        double best = 1e30;
        if (!OrmqrTimeCase(handle, stream, c, idx, &best))
        {
            return 1;
        }
        const std::string label = "m=" + std::to_string(c.m) + " n=" + std::to_string(c.n) +
                                  " k=" + std::to_string(c.k) + " " + ((c.side == 0) ? "L" : "R") + "/" +
                                  ((c.trans == 0) ? "N" : "T");
        double x = 0.0;
        pass += ExpectPerf(label.c_str(), best, c.gpuUs, &x) ? 1 : 0;
        if (x < worst)
        {
            worst = x;
            worstCase = c;
        }
    }
    printf("\n性能扫描：%d / %zu PASS（判据 >= 0.35x GPU）；最差 %5.2fx 于 m=%d n=%d k=%d %s/%s\n", pass,
           cases.size(), worst, worstCase.m, worstCase.n, worstCase.k, (worstCase.side == 0) ? "L" : "R",
           (worstCase.trans == 0) ? "N" : "T");
    return (pass == static_cast<int>(cases.size())) ? 0 : 1;
}

/// 启动地板的分解测量：小形状全部卡在 24~27 us 的平台上（与 m/n/k 无关），
/// 这个模式把"空同步 / 4 字节 memsetAsync+同步 / 一次完整 API 调用"分开量，
/// 用来确认那条 aclrtMemsetAsync(devInfo) 到底值多少。
/// 跑 fn 共 iters 次（外加一次预热），打 min/avg 并返回 min。
/// 取 min 当稳态：首趟芯片未提频，实测 avg 能明显高于 min。
template <class Fn>
double TimeRepeat(const char *name, int iters, Fn &&fn)
{
    fn();  // 预热
    double best = 1e30, sum = 0.0;
    for (int i = 0; i < iters; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        best = std::min(best, us);
        sum += us;
    }
    printf("  %-40s min=%8.3f us  avg=%8.3f us  (n=%d)\n", name, best, SafeDiv(sum, iters), iters);
    return best;
}

/// 启动地板的两个组成：空同步，以及 4 字节 memsetAsync + 同步。
/// 用来确认那条 aclrtMemsetAsync(devInfo) 到底值多少。
int MeasureLaunchFloor(aclrtStream stream)
{
    uint8_t *info = nullptr;
    if (aclrtMalloc(reinterpret_cast<void **>(&info), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        return 1;
    }
    const double tSync = TimeRepeat("空同步", 200, [&]() { (void)aclrtSynchronizeStream(stream); });
    const double tMemset = TimeRepeat("aclrtMemsetAsync(4B) + 同步", 200, [&]() {
        (void)aclrtMemsetAsync(info, sizeof(int32_t), 0, sizeof(int32_t), stream);
        (void)aclrtSynchronizeStream(stream);
    });
    printf("  => memsetAsync 这一条流上任务约 %.3f us\n", tMemset - tSync);
    aclrtFree(info);
    return 0;
}

/// 判据边缘的小形状各测 30 趟（同口径重测，别拿单点样本下结论）
int MeasureTinyCases(aclsolverHandle_t handle, aclrtStream stream)
{
    struct Tiny { int m, n, k; int side; int trans; double gpuUs; };
    const Tiny tiny[] = {{4, 4, 4, 1, 1, 9.3}, {32, 4, 4, 1, 1, 8.3}, {2, 2, 2, 0, 0, 9.0}};
    for (const auto &t : tiny)
    {
        const aclsolverSideMode_t side = (t.side == 0) ? ACLSOLVER_SIDE_LEFT : ACLSOLVER_SIDE_RIGHT;
        const aclsolverOperation_t trans = (t.trans == 0) ? ACLSOLVER_OP_N : ACLSOLVER_OP_T;
        const bool isLeft = (side == ACLSOLVER_SIDE_LEFT);
        const int nq = isLeft ? t.m : t.n;
        std::vector<float> A, tau, C0;
        MakeInput(isLeft, t.m, t.n, t.k, nq, t.m, 2, A, tau, C0);
        int lwork = 0;
        (void)aclsolverSormqr_bufferSize(handle, side, trans, t.m, t.n, t.k, nullptr, nq, nullptr, nullptr, t.m,
                                         &lwork);
        Buffers buf;
        if (!AllocFor(buf, A.size(), tau.size(), C0.size(), 0, lwork))
        {
            buf.Free();
            return 1;
        }
        const size_t aBytes = A.size() * sizeof(float), cBytes = C0.size() * sizeof(float);
        (void)aclrtMemcpy(buf.A, aBytes, A.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        (void)aclrtMemcpy(buf.tau, tau.size() * sizeof(float), tau.data(), tau.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        (void)aclrtMemcpy(buf.C, cBytes, C0.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        const std::string name = "sormqr m=" + std::to_string(t.m) + " n=" + std::to_string(t.n) +
                                 " k=" + std::to_string(t.k) + " " + (isLeft ? "L" : "R") + "/" +
                                 ((t.trans == 0) ? "N" : "T");
        const double best = TimeRepeat(name.c_str(), 30, [&]() {
            (void)aclsolverSormqr(handle, side, trans, t.m, t.n, t.k, reinterpret_cast<const float *>(buf.A), nq,
                                  reinterpret_cast<const float *>(buf.tau), reinterpret_cast<float *>(buf.C), t.m,
                                  reinterpret_cast<float *>(buf.ws), lwork, reinterpret_cast<int *>(buf.info));
            (void)aclrtSynchronizeStream(stream);
        });
        printf("      预算=%.2f us（GPU %.1f / 0.35）=> %s\n", t.gpuUs / 0.35, t.gpuUs,
               (best <= t.gpuUs / 0.35) ? "过" : "不过");
        buf.Free();
    }
    return 0;
}

int RunFloor(aclsolverHandle_t handle, aclrtStream stream)
{
    if (MeasureLaunchFloor(stream) != 0)
    {
        return 1;
    }
    return MeasureTinyCases(handle, stream);
}

/// 连续下发探测：geqrf -> ormqr 中间**不同步**。SSgels 要串三个 kernel，所以专门量这条路径。
/// 早期曾怀疑"连续下发会挂死（跨核 flag 在 kernel 之间残留）"，**实测已不复现**，
/// 本模式现在当回归守卫用：不同步的结果必须与"中间同步"的结果**逐位相同**；真挂住则超时暴露。
/// 连续下发探测用的一组设备缓冲
struct ChainBufs
{
    uint8_t *A = nullptr;
    uint8_t *tau = nullptr;
    uint8_t *B = nullptr;
    uint8_t *ws = nullptr;
    uint8_t *info = nullptr;
    void Free()
    {
        if (A) aclrtFree(A);
        if (tau) aclrtFree(tau);
        if (B) aclrtFree(B);
        if (ws) aclrtFree(ws);
        if (info) aclrtFree(info);
        A = tau = B = ws = info = nullptr;
    }
};

bool ChainAlloc(ChainBufs *d, size_t aBytes, size_t bBytes, int n, int lwMax)
{
    if (aclrtMalloc(reinterpret_cast<void **>(&d->A), aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&d->tau), static_cast<size_t>(n) * 4, ACL_MEM_MALLOC_HUGE_FIRST)
            != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&d->B), bBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&d->ws), static_cast<size_t>(lwMax) * 4, ACL_MEM_MALLOC_HUGE_FIRST)
            != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&d->info), 4, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        printf("  分配失败\n");
        return false;
    }
    return true;
}

/// 一趟 geqrf -> ormqr。mode == 0 时中间同步，mode == 1 时不同步。
/// 判法：不同步的结果必须与"中间同步"的基准逐位相同。
bool ChainOneRep(aclsolverHandle_t handle, aclrtStream stream, const ChainBufs &d, int mode, int rep, int m, int n,
                 int nrhs, int lda, int ldb, int lwG, int lwM, const std::vector<float> &A0,
                 const std::vector<float> &B0, std::vector<float> *ref)
{
    const size_t aB = A0.size() * sizeof(float), bB = B0.size() * sizeof(float);
    (void)aclrtMemcpy(d.A, aB, A0.data(), aB, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(d.B, bB, B0.data(), bB, ACL_MEMCPY_HOST_TO_DEVICE);
    const auto t0 = std::chrono::steady_clock::now();
    const aclsolverStatus_t s1 =
        aclsolverSgeqrf(handle, m, n, reinterpret_cast<float *>(d.A), lda, reinterpret_cast<float *>(d.tau),
                        reinterpret_cast<float *>(d.ws), lwG, reinterpret_cast<int *>(d.info));
    if (mode == 0)
    {
        (void)aclrtSynchronizeStream(stream);
    }
    const aclsolverStatus_t s2 =
        aclsolverSormqr(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, m, nrhs, n,
                        reinterpret_cast<const float *>(d.A), lda, reinterpret_cast<const float *>(d.tau),
                        reinterpret_cast<float *>(d.B), ldb, reinterpret_cast<float *>(d.ws), lwM,
                        reinterpret_cast<int *>(d.info));
    const aclError sync = aclrtSynchronizeStream(stream);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (s1 != ACLSOLVER_STATUS_SUCCESS || s2 != ACLSOLVER_STATUS_SUCCESS || sync != ACL_SUCCESS)
    {
        printf("  %s 第 %d 趟失败: s1=%d s2=%d sync=%d\n", mode ? "不同步" : "中间同步", rep + 1, (int)s1, (int)s2,
               (int)sync);
        return false;
    }
    std::vector<float> got(B0.size(), 0.0f);
    (void)aclrtMemcpy(got.data(), bB, d.B, bB, ACL_MEMCPY_DEVICE_TO_HOST);
    if (mode == 0 && rep == 0)
    {
        *ref = got;
    }
    const bool same = (got == *ref);
    printf("  %s rep%d: %8.1f us  与基准逐位%s\n", mode ? "不同步  " : "中间同步", rep + 1, us,
           same ? "相同" : "**不同**");
    return same;
}

int RunChain(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int nrhs, int reps)
{
    const int lda = m, ldb = m;
    std::vector<float> A0(static_cast<size_t>(lda) * n), B0(static_cast<size_t>(ldb) * nrhs);
    FillMatrix(A0, 0xA11CEu, 0);
    FillMatrix(B0, 0xB22DFu, 1);
    int lwG = 0, lwM = 0;
    (void)aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &lwG);
    (void)aclsolverSormqr_bufferSize(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, m, nrhs, n, nullptr, lda, nullptr,
                                     nullptr, ldb, &lwM);
    ChainBufs d;
    if (!ChainAlloc(&d, A0.size() * sizeof(float), B0.size() * sizeof(float), n, std::max(lwG, lwM)))
    {
        d.Free();
        return 1;
    }
    std::vector<float> ref;
    int rc = 0;
    for (int mode = 0; mode < 2 && rc == 0; ++mode)  // 0 = 中间同步, 1 = 不同步
    {
        for (int rep = 0; rep < reps; ++rep)
        {
            if (!ChainOneRep(handle, stream, d, mode, rep, m, n, nrhs, lda, ldb, lwG, lwM, A0, B0, &ref))
            {
                rc = 1;
                break;
            }
        }
    }
    d.Free();
    return rc;
}


/// chain 模式：geqrf -> ormqr 连续下发探测
int RunChainMode(int argc, char **argv)
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    const int m = ArgInt(argc, argv, 2, 512);
    const int n = ArgInt(argc, argv, 3, 512);
    printf("== 连续下发探测 m=%d n=%d nrhs=1 ==\n", m, n);
    const int rc = RunChain(handle, stream, m, n, 1, 3);
    TeardownAcl(handle, stream);
    return rc;
}

/// floor 模式：启动地板分解
int RunFloorMode()
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    printf("== 启动地板分解 ==\n");
    const int rc = RunFloor(handle, stream);
    TeardownAcl(handle, stream);
    return rc;
}

/// sweep / perfsweep 模式：两者共用同一份用例清单
int RunSweepMode(int argc, char **argv)
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    const std::string mode = argv[1];
    const char *path = argv[2];
    const int arg3 = ArgInt(argc, argv, 3, 32);
    const int arg4 = ArgInt(argc, argv, 4, 0);  // lda/ldc 的额外 padding
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    int rc = 0;
    if (mode == "sweep")
    {
        printf("== Sormqr 全量精度扫描（判据 max(5*ratio_cpu, 0.1)）maxCols=%d ldExtra=%d ==\n", arg3, arg4);
        rc = RunSweep(handle, stream, path, arg3, arg4);
    }
    else
    {
        printf("== Sormqr 全量性能扫描（判据 T_NPU <= T_GPU/0.35）==\n");
        rc = RunPerfSweep(handle, stream, path);
    }
    TeardownAcl(handle, stream);
    return rc;
}

/// 形状 x (side, trans) 的全组合覆盖，每条跑 2 次比 bit-wise
void RunShapeMatrix(aclsolverHandle_t handle, aclrtStream stream, int *pass, int *total)
{
    printf("== 六种 (side,trans) 组合 + 形状覆盖（各跑 2 次比 bit-wise）==\n");
    struct Shape
    {
        int m, n, k, ldaExtra, ldcExtra;
    };
    const Shape shapes[] = {
        {2, 2, 2, 0, 0},     {4, 4, 4, 0, 0},      {8, 8, 8, 0, 0},      {16, 16, 8, 0, 0},
        {37, 29, 11, 3, 5},  {64, 64, 64, 0, 0},   {65, 33, 17, 7, 7},   {128, 64, 32, 0, 0},
        {129, 129, 129, 7, 3}, {256, 128, 64, 0, 0}, {256, 100, 100, 32, 8}, {512, 512, 512, 0, 0},
    };
    int idx = 0;
    for (const auto &s : shapes)
    {
        for (auto side : {ACLSOLVER_SIDE_LEFT, ACLSOLVER_SIDE_RIGHT})
        {
            for (auto trans : {ACLSOLVER_OP_N, ACLSOLVER_OP_T, ACLSOLVER_OP_C})
            {
                const int nq = (side == ACLSOLVER_SIDE_LEFT) ? s.m : s.n;
                if (s.k > nq)
                {
                    continue;  // k <= nq 是接口约束，不是缺陷
                }
                ++idx;
                ++(*total);
                *pass += RunShape(handle, stream, idx, side, trans, s.m, s.n, s.k, nq + s.ldaExtra,
                                  s.m + s.ldcExtra, 1 << 30, 2, true, nullptr)
                             ? 1
                             : 0;
            }
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    const std::string mode = (argc >= 2) ? argv[1] : "";
    if (mode == "chain")
    {
        return RunChainMode(argc, argv);
    }
    if (mode == "floor")
    {
        return RunFloorMode();
    }
    if (argc >= 3 && (mode == "sweep" || mode == "perfsweep"))
    {
        return RunSweepMode(argc, argv);
    }

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (!InitAcl(&stream, &handle))
    {
        return 1;
    }
    int pass = 0, total = 0;
    printf("== 参数校验（-i 语义）==\n");
    Tally(RunBadParams(handle, stream), &pass, &total);

    printf("== 空问题与 k=0 ==\n");
    for (auto t : {std::vector<int>{0, 0, 0}, {8, 0, 0}, {8, 8, 0}})
    {
        Tally(RunEmpty(handle, t[0], t[1], t[2]), &pass, &total);
    }

    printf("== trans=C 与 trans=T 等价 ==\n");
    for (auto side : {ACLSOLVER_SIDE_LEFT, ACLSOLVER_SIDE_RIGHT})
    {
        Tally(RunTransCEqualsT(handle, stream, side, 128, 64, 32), &pass, &total);
    }

    RunShapeMatrix(handle, stream, &pass, &total);

    printf("\n%d / %d PASS\n", pass, total);
    TeardownAcl(handle, stream);
    return (pass == total) ? 0 : 1;
}
