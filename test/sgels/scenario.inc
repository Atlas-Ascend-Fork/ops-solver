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
 * \file scenario.inc
 * \brief 任务书 §3.5 要求的**跨接口必测场景**（被 sgels_test.cpp 直接 include）
 *
 * 放在 sgels 的测试里而不是单开一个 test 目录：SSgels 本身就是
 * "geqrf + ormqr + 回代"的组合算子，它的测试已经链接了全部五个接口；
 * 而仓里 `test/<op>` 与 `src/<op>` 是一一对应的，多一个无主目录会破坏这个约定。
 *
 * 覆盖：
 *   场景 1  geqrf + orgqr 还原 Q（正交性 + Q·R == A）
 *   场景 2  geqrf + ormqr 往返（Qᵀ 再 Q，回到原矩阵）
 *   场景 3  Xlarft 的 T 语义（块公式 vs 逐个施加反射子，FORWARD/BACKWARD）
 *   场景 4  SSgels 端到端（直接复用 RunShape 的两类判据）
 *   场景 5  五个接口连续下发、中间**完全不同步**（曾怀疑会挂死，现为回归守卫）
 *   场景 6  非默认 stream，而且中途换一条
 *   场景 7  确定性：A / tau / info **全部**逐位比对
 *
 * 几个阈值（60）的来由：这些是"两条数学等价路径在 fp32 下的差"，量纲已被
 * `maxdim * ||.|| * eps` 归一化，实测都在 10 以内；取 60 留一倍余量，
 * 同时仍能抓住"算错一个块"这类真缺陷（那会是 1e3 以上）。
 */

namespace
{
constexpr double kScenTol = 60.0;

struct Dev
{
    uint8_t *p = nullptr;
    void Alloc(size_t bytes)
    {
        (void)aclrtMalloc(reinterpret_cast<void **>(&p), std::max<size_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    void Free()
    {
        if (p != nullptr)
        {
            aclrtFree(p);
            p = nullptr;
        }
    }
    float *F() const { return reinterpret_cast<float *>(p); }
    int *I() const { return reinterpret_cast<int *>(p); }
};

/// 两个矩阵之差的 1-范数（列和最大值）。各场景的"两条路径应当一致"都归到这里。
double ColMaxAbsDiff(int rows, int cols, int ld, const std::vector<float> &X, const std::vector<float> &Y)
{
    double worst = 0.0;
    for (int c = 0; c < cols; ++c)
    {
        double col = 0.0;
        for (int r = 0; r < rows; ++r)
        {
            col += std::fabs(static_cast<double>(X[static_cast<size_t>(c) * ld + r]) -
                             static_cast<double>(Y[static_cast<size_t>(c) * ld + r]));
        }
        worst = std::max(worst, col);
    }
    return worst;
}

/// Q 的正交性残差：||I - Q^T Q||_1 / (m * eps)
double OrthResidual(int m, int n, int lda, const std::vector<float> &Q)
{
    double orth = 0.0;
    for (int b = 0; b < n; ++b)
    {
        orth = std::max(orth, OrthColumnSum(m, n, lda, Q, b));
    }
    return SafeDiv(orth, static_cast<double>(m) * kEpsF32);
}

/// 还原残差：||Q R - A||_1 / (m * ||A||_1 * eps)。R 取 fac 的上三角。
double QrBackResidual(int m, int n, int lda, const std::vector<float> &Q, const std::vector<float> &fac,
                      const std::vector<float> &A0)
{
    const double normA = Norm1(m, n, lda, A0);
    double back = 0.0;
    for (int c = 0; c < n; ++c)
    {
        double col = 0.0;
        for (int r = 0; r < m; ++r)
        {
            double v = 0.0;
            for (int t = 0; t <= c; ++t)
            {
                v += static_cast<double>(Q[static_cast<size_t>(t) * lda + r]) *
                     static_cast<double>(fac[static_cast<size_t>(c) * lda + t]);
            }
            col += std::fabs(v - static_cast<double>(A0[static_cast<size_t>(c) * lda + r]));
        }
        back = std::max(back, col);
    }
    return SafeDiv(back, static_cast<double>(m) * normA * kEpsF32);
}

/// 场景 1：geqrf + orgqr 还原 Q —— 同时验 Q 的正交性与 Q*R == A
bool ScenRestoreQ(aclsolverHandle_t handle, aclrtStream stream, int m, int n)
{
    const int lda = m;
    std::vector<float> A0(static_cast<size_t>(lda) * n);
    FillMatrix(A0, 0x51C10000u, 0);
    int lwG = 0;
    int lwQ = 0;
    (void)aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &lwG);
    (void)aclsolverSorgqr_bufferSize(handle, m, n, n, nullptr, lda, nullptr, &lwQ);
    Dev dA;
    Dev dTau;
    Dev dWs;
    Dev dInfo;
    dA.Alloc(A0.size() * sizeof(float));
    dTau.Alloc(static_cast<size_t>(n) * sizeof(float));
    dWs.Alloc(static_cast<size_t>(std::max(lwG, lwQ)) * sizeof(float));
    dInfo.Alloc(sizeof(int32_t));
    (void)aclrtMemcpy(dA.p, A0.size() * sizeof(float), A0.data(), A0.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    const aclsolverStatus_t s1 = aclsolverSgeqrf(handle, m, n, dA.F(), lda, dTau.F(), dWs.F(), lwG, dInfo.I());
    (void)aclrtSynchronizeStream(stream);
    std::vector<float> fac(A0.size(), 0.0f);
    (void)aclrtMemcpy(fac.data(), fac.size() * sizeof(float), dA.p, fac.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    const aclsolverStatus_t s2 = aclsolverSorgqr(handle, m, n, n, dA.F(), lda, dTau.F(), dWs.F(), lwQ, dInfo.I());
    (void)aclrtSynchronizeStream(stream);
    std::vector<float> Q(A0.size(), 0.0f);
    (void)aclrtMemcpy(Q.data(), Q.size() * sizeof(float), dA.p, Q.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    int32_t info = -1;
    (void)aclrtMemcpy(&info, sizeof(info), dInfo.p, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST);
    dA.Free();
    dTau.Free();
    dWs.Free();
    dInfo.Free();

    const double orth = OrthResidual(m, n, lda, Q);
    const double back = QrBackResidual(m, n, lda, Q, fac, A0);

    const bool ok = (s1 == ACLSOLVER_STATUS_SUCCESS) && (s2 == ACLSOLVER_STATUS_SUCCESS) && (info == 0) &&
                    (orth <= kScenTol) && (back <= kScenTol);
    printf("  场景1 geqrf+orgqr 还原 Q (m=%d n=%d): ||I-Q^TQ||/(m eps)=%.2f  ||QR-A||/(m||A||eps)=%.2f  %s\n", m, n,
           orth, back, ok ? "PASS" : "FAIL");
    return ok;
}

/// 场景 2 的设备段：geqrf -> ormqr(Q^T) -> ormqr(Q)，把回来的 C 与 devInfo 带出
void RoundTripOnDevice(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int nrhs, int lda, int ldc,
                       const std::vector<float> &A0, const std::vector<float> &C0, std::vector<float> *C1,
                       int32_t *info)
{
    int lwG = 0;
    int lwM = 0;
    (void)aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &lwG);
    (void)aclsolverSormqr_bufferSize(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, m, nrhs, n, nullptr, lda, nullptr,
                                     nullptr, ldc, &lwM);
    Dev dA;
    Dev dTau;
    Dev dC;
    Dev dWs;
    Dev dInfo;
    dA.Alloc(A0.size() * sizeof(float));
    dTau.Alloc(static_cast<size_t>(n) * sizeof(float));
    dC.Alloc(C0.size() * sizeof(float));
    dWs.Alloc(static_cast<size_t>(std::max(lwG, lwM)) * sizeof(float));
    dInfo.Alloc(sizeof(int32_t));
    (void)aclrtMemcpy(dA.p, A0.size() * sizeof(float), A0.data(), A0.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(dC.p, C0.size() * sizeof(float), C0.data(), C0.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclsolverSgeqrf(handle, m, n, dA.F(), lda, dTau.F(), dWs.F(), lwG, dInfo.I());
    (void)aclsolverSormqr(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, m, nrhs, n, dA.F(), lda, dTau.F(), dC.F(),
                          ldc, dWs.F(), lwM, dInfo.I());
    (void)aclsolverSormqr(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_N, m, nrhs, n, dA.F(), lda, dTau.F(), dC.F(),
                          ldc, dWs.F(), lwM, dInfo.I());
    (void)aclrtSynchronizeStream(stream);
    C1->assign(C0.size(), 0.0f);
    (void)aclrtMemcpy(C1->data(), C1->size() * sizeof(float), dC.p, C1->size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(info, sizeof(*info), dInfo.p, sizeof(*info), ACL_MEMCPY_DEVICE_TO_HOST);
    dA.Free();
    dTau.Free();
    dC.Free();
    dWs.Free();
    dInfo.Free();
}

/// 场景 2：geqrf + ormqr 往返 —— 先 Q^T C 再 Q (Q^T C)，应回到 C
bool ScenOrmqrRoundTrip(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int nrhs)
{
    const int lda = m;
    const int ldc = m;
    std::vector<float> A0(static_cast<size_t>(lda) * n);
    std::vector<float> C0(static_cast<size_t>(ldc) * nrhs);
    FillMatrix(A0, 0x7e510000u, 0);
    FillMatrix(C0, 0x7e510001u, 1);
    std::vector<float> C1;
    int32_t info = -1;
    RoundTripOnDevice(handle, stream, m, n, nrhs, lda, ldc, A0, C0, &C1, &info);

    const double normC = Norm1(m, nrhs, ldc, C0);
    const double diff = SafeDiv(ColMaxAbsDiff(m, nrhs, ldc, C1, C0),
                                static_cast<double>(m) * std::max(normC, 1e-30) * kEpsF32);
    const bool ok = (info == 0) && (diff <= kScenTol);
    printf("  场景2 geqrf+ormqr 往返 (m=%d n=%d nrhs=%d): ||Q Q^T C - C||/(m||C||eps)=%.2f  %s\n", m, n, nrhs, diff,
           ok ? "PASS" : "FAIL");
    return ok;
}

/// 造一组合法的 (V, tau)：对一个 n x k 随机矩阵做 CPU geqr2 取反射子。
/// BACKWARD 时把 V 按行列双向翻转、tau 反序 —— 这是 LAPACK 的 BACKWARD 存储约定。
void XlarftMakeVTau(int n, int k, int ldv, bool backward, std::vector<float> *Vh, std::vector<float> *tauH)
{
    std::vector<float> fac(static_cast<size_t>(ldv) * k);
    FillMatrix(fac, 0x1ced0000u + (backward ? 1u : 0u), 0);
    std::vector<float> t(static_cast<size_t>(std::min(n, k)), 0.0f);
    QrCore::Geqr2(n, k, fac.data(), ldv, t.data());
    Vh->assign(static_cast<size_t>(ldv) * k, 0.0f);
    for (int c = 0; c < k; ++c)
    {
        for (int r = 0; r < n; ++r)
        {
            if (!backward)
            {
                (*Vh)[static_cast<size_t>(c) * ldv + r] = fac[static_cast<size_t>(c) * ldv + r];
            }
            else
            {
                const int rf = n - 1 - r;
                const int cf = k - 1 - c;
                (*Vh)[static_cast<size_t>(c) * ldv + r] = (rf > cf) ? fac[static_cast<size_t>(cf) * ldv + rf] : 0.0f;
            }
        }
    }
    tauH->assign(t.begin(), t.end());
    if (backward)
    {
        std::reverse(tauH->begin(), tauH->end());
    }
}

/// 下发一次 Xlarft，把 T 带回 host。失败时已打印原因。
bool XlarftCallDevice(aclsolverHandle_t handle, aclrtStream stream, int n, int k, int ldv, int ldt, bool backward,
                      const std::vector<float> &Vh, const std::vector<float> &tauH, std::vector<float> *Th)
{
    const aclsolverDirectMode_t direct = backward ? ACLSOLVER_DIRECT_BACKWARD : ACLSOLVER_DIRECT_FORWARD;
    size_t devB = 0;
    size_t hostB = 0;
    if (aclsolverXlarft_bufferSize(handle, nullptr, direct, ACLSOLVER_STOREV_COLUMNWISE, n, k, ACLSOLVER_R_32F,
                                   nullptr, ldv, ACLSOLVER_R_32F, nullptr, ACLSOLVER_R_32F, nullptr, ldt,
                                   ACLSOLVER_R_32F, &devB, &hostB)
        != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  场景3 bufferSize 失败\n");
        return false;
    }
    Dev dV;
    Dev dTau;
    Dev dT;
    Dev dWs;
    dV.Alloc(Vh.size() * sizeof(float));
    dTau.Alloc(tauH.size() * sizeof(float));
    dT.Alloc(static_cast<size_t>(ldt) * k * sizeof(float));
    dWs.Alloc(devB);
    (void)aclrtMemcpy(dV.p, Vh.size() * sizeof(float), Vh.data(), Vh.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(dTau.p, tauH.size() * sizeof(float), tauH.data(), tauH.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    const aclsolverStatus_t st =
        aclsolverXlarft(handle, nullptr, direct, ACLSOLVER_STOREV_COLUMNWISE, n, k, ACLSOLVER_R_32F, dV.p, ldv,
                        ACLSOLVER_R_32F, dTau.p, ACLSOLVER_R_32F, dT.p, ldt, ACLSOLVER_R_32F, dWs.p, devB, nullptr,
                        0);
    (void)aclrtSynchronizeStream(stream);
    Th->assign(static_cast<size_t>(ldt) * k, 0.0f);
    (void)aclrtMemcpy(Th->data(), Th->size() * sizeof(float), dT.p, Th->size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    dV.Free();
    dTau.Free();
    dT.Free();
    dWs.Free();
    if (st != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  场景3 status=%d\n", (int)st);
        return false;
    }
    return true;
}

/// 显式造 V（含隐含的对角 1 与该为 0 的那一半），double 存放、列主序 ld = n
void XlarftExpandV(int n, int k, int ldv, bool backward, const std::vector<float> &Vh, std::vector<double> *V)
{
    V->assign(static_cast<size_t>(n) * k, 0.0);
    for (int c = 0; c < k; ++c)
    {
        const int d = backward ? (n - k + c) : c;
        for (int r = 0; r < n; ++r)
        {
            double v = 0.0;
            if (r == d)
            {
                v = 1.0;
            }
            else if ((!backward && r > d) || (backward && r < d))
            {
                v = static_cast<double>(Vh[static_cast<size_t>(c) * ldv + r]);
            }
            (*V)[static_cast<size_t>(c) * n + r] = v;
        }
    }
}

/// (a) 块公式：y = x - V (T (V^T x))。T 只引用被 direct 决定的那一半三角。
void XlarftApplyBlock(int n, int k, int ldt, bool backward, const std::vector<double> &V,
                      const std::vector<float> &Th, const std::vector<double> &x, std::vector<double> *y)
{
    std::vector<double> w(static_cast<size_t>(k), 0.0);
    std::vector<double> tw(static_cast<size_t>(k), 0.0);
    for (int c = 0; c < k; ++c)
    {
        double s = 0.0;
        for (int r = 0; r < n; ++r)
        {
            s += V[static_cast<size_t>(c) * n + r] * x[static_cast<size_t>(r)];
        }
        w[static_cast<size_t>(c)] = s;
    }
    for (int r = 0; r < k; ++r)
    {
        double s = 0.0;
        for (int c = 0; c < k; ++c)
        {
            const bool referenced = backward ? (r >= c) : (r <= c);
            if (referenced)
            {
                s += static_cast<double>(Th[static_cast<size_t>(c) * ldt + r]) * w[static_cast<size_t>(c)];
            }
        }
        tw[static_cast<size_t>(r)] = s;
    }
    *y = x;
    for (int c = 0; c < k; ++c)
    {
        for (int r = 0; r < n; ++r)
        {
            (*y)[static_cast<size_t>(r)] -= V[static_cast<size_t>(c) * n + r] * tw[static_cast<size_t>(c)];
        }
    }
}

/// (b) 逐个反射子：FORWARD 是 H_1...H_k x（i 从 k-1 到 0），BACKWARD 是 H_k...H_1 x（i 递增）
void XlarftApplyReflectors(int n, int k, bool backward, const std::vector<double> &V,
                           const std::vector<float> &tauH, const std::vector<double> &x, std::vector<double> *y)
{
    *y = x;
    for (int t = 0; t < k; ++t)
    {
        const int i = backward ? t : (k - 1 - t);
        const double ti = static_cast<double>(tauH[static_cast<size_t>(i)]);
        if (ti == 0.0)
        {
            continue;
        }
        double s = 0.0;
        for (int r = 0; r < n; ++r)
        {
            s += V[static_cast<size_t>(i) * n + r] * (*y)[static_cast<size_t>(r)];
        }
        for (int r = 0; r < n; ++r)
        {
            (*y)[static_cast<size_t>(r)] -= ti * V[static_cast<size_t>(i) * n + r] * s;
        }
    }
}

/// 场景 3：Xlarft 的 T 语义 —— 用它算出的 T 按块公式 (I - V T V^T) 作用，
/// 必须与"逐个施加 k 个反射子"一致。两条路径都在 double 下做，比的是 T 本身的语义。
bool ScenXlarftSemantics(aclsolverHandle_t handle, aclrtStream stream, int n, int k, bool backward)
{
    const int ldv = n;
    const int ldt = k;
    std::vector<float> Vh;
    std::vector<float> tauH;
    XlarftMakeVTau(n, k, ldv, backward, &Vh, &tauH);

    std::vector<float> Th;
    if (!XlarftCallDevice(handle, stream, n, k, ldv, ldt, backward, Vh, tauH, &Th))
    {
        return false;
    }

    std::vector<double> V;
    XlarftExpandV(n, k, ldv, backward, Vh, &V);

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    std::mt19937 rng(0xB10Cu);
    std::uniform_real_distribution<double> d01(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
    {
        x[static_cast<size_t>(i)] = d01(rng);
    }
    std::vector<double> yBlock;
    std::vector<double> yRefl;
    XlarftApplyBlock(n, k, ldt, backward, V, Th, x, &yBlock);
    XlarftApplyReflectors(n, k, backward, V, tauH, x, &yRefl);

    double err = 0.0;
    double scale = 0.0;
    for (int i = 0; i < n; ++i)
    {
        err = std::max(err, std::fabs(yBlock[static_cast<size_t>(i)] - yRefl[static_cast<size_t>(i)]));
        scale = std::max(scale, std::fabs(yRefl[static_cast<size_t>(i)]));
    }
    const double ratio = SafeDiv(err, static_cast<double>(k) * std::max(scale, 1e-30) * kEpsF32);
    const bool ok = (ratio <= kScenTol);
    printf("  场景3 Xlarft 的 T 语义 (n=%d k=%d %s): ||块公式-逐个反射子||/(k eps)=%.2f  %s\n", n, k,
           backward ? "BACKWARD" : "FORWARD ", ratio, ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace
