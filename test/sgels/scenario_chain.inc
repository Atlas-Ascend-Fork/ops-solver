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
 * \file scenario_chain.inc
 * \brief 任务书 §3.5 跨接口必测场景的后半：五接口串联、多流、确定性。
 *
 * 和 scenario.inc 一起被 sgels_test.cpp include（拆成两个文件是为了让每个文件
 * 都留在 codecheck 的头文件体量红线内）。同一个翻译单元里的匿名 namespace 是同一个，
 * 所以 Dev / FillMatrix / RunShape / ColMaxAbsDiff 这些在这里都能直接用。
 *
 * 覆盖：
 *   场景 5  五个接口连续下发、中间**完全不同步**（曾怀疑会挂死，现为回归守卫）
 *   场景 6  非默认 stream，而且中途换一条
 *   场景 7  确定性：A / tau / info **全部**逐位比对
 *   以及把场景 1~7 串起来跑的 RunScenario
 */

namespace
{
/// 场景 5 的形状、workspace 大小与设备缓冲。一趟一趟地传这些量太长，收成一个盒子。
struct ChainCtx
{
    int m = 0;
    int n = 0;
    int nrhs = 2;
    int lda = 0;
    int ldb = 0;
    int ldx = 0;
    // Xlarft 的 k 上限是 min(n, 256)（任务书 §2.6），这里串联时要跟着收窄 ——
    // 否则 n>256 的形状会（正确地）拿到 NOT_SUPPORTED，看起来像链路挂了。
    int kT = 0;
    int ldt = 0;
    int lwG = 0;
    int lwQ = 0;
    int lwM = 0;
    size_t xlB = 0;
    size_t glB = 0;
    Dev dA, dA2, dTau, dB, dX, dT, dWs, dWs2, dInfo;

    void Free()
    {
        dA.Free();
        dA2.Free();
        dTau.Free();
        dB.Free();
        dX.Free();
        dT.Free();
        dWs.Free();
        dWs2.Free();
        dInfo.Free();
    }
};

/// 查五个接口的 workspace 并把设备缓冲都分好
void ChainSetup(aclsolverHandle_t handle, int m, int n, size_t aElems, size_t bElems, ChainCtx *cx)
{
    cx->m = m;
    cx->n = n;
    cx->lda = m;
    cx->ldb = m;
    cx->ldx = n;
    cx->kT = std::min(n, 256);
    cx->ldt = cx->kT;
    (void)aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, cx->lda, &cx->lwG);
    (void)aclsolverSorgqr_bufferSize(handle, m, n, n, nullptr, cx->lda, nullptr, &cx->lwQ);
    (void)aclsolverSormqr_bufferSize(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, m, cx->nrhs, n, nullptr, cx->lda,
                                     nullptr, nullptr, cx->ldb, &cx->lwM);
    size_t xlH = 0;
    (void)aclsolverXlarft_bufferSize(handle, nullptr, ACLSOLVER_DIRECT_FORWARD, ACLSOLVER_STOREV_COLUMNWISE, n,
                                     cx->kT, ACLSOLVER_R_32F, nullptr, cx->lda, ACLSOLVER_R_32F, nullptr,
                                     ACLSOLVER_R_32F, nullptr, cx->ldt, ACLSOLVER_R_32F, &cx->xlB, &xlH);
    (void)aclsolverSSgels_bufferSize(handle, m, n, cx->nrhs, nullptr, cx->lda, nullptr, cx->ldb, nullptr, cx->ldx,
                                     nullptr, &cx->glB);
    cx->dA.Alloc(aElems * sizeof(float));
    cx->dA2.Alloc(aElems * sizeof(float));
    cx->dTau.Alloc(static_cast<size_t>(n) * sizeof(float));
    cx->dB.Alloc(bElems * sizeof(float));
    cx->dX.Alloc(static_cast<size_t>(cx->ldx) * cx->nrhs * sizeof(float));
    cx->dT.Alloc(static_cast<size_t>(cx->ldt) * cx->kT * sizeof(float));
    // ★ 连在一起的调用不能共用一块 workspace：它们在流上重叠，后一个会把前一个的
    //   中间结果冲掉。这里给 SSgels 单独一块，其余三个共用另一块（它们之间有数据依赖、
    //   天然串行：geqrf 写 A/tau -> ormqr 读 -> xlarft 读 -> orgqr 读写 A）。
    cx->dWs.Alloc(
        std::max(static_cast<size_t>(std::max(std::max(cx->lwG, cx->lwQ), cx->lwM)) * sizeof(float), cx->xlB));
    cx->dWs2.Alloc(cx->glB);
    cx->dInfo.Alloc(sizeof(int32_t));
}

/// 一趟：五个接口一口气下发、**中间一次同步都没有**，然后把 A / T / X 拼起来带回
bool ChainOneRep(aclsolverHandle_t handle, aclrtStream stream, const ChainCtx &cx, int rep,
                 const std::vector<float> &A0, const std::vector<float> &B0, std::vector<float> *got)
{
    const size_t nQ = A0.size();
    const size_t nT = static_cast<size_t>(cx.ldt) * cx.kT;
    const size_t nX = static_cast<size_t>(cx.ldx) * cx.nrhs;
    (void)aclrtMemcpy(cx.dA.p, nQ * sizeof(float), A0.data(), nQ * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(cx.dA2.p, nQ * sizeof(float), A0.data(), nQ * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(cx.dB.p, B0.size() * sizeof(float), B0.data(), B0.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    int niter = -1;
    const aclsolverStatus_t r1 =
        aclsolverSSgels(handle, cx.m, cx.n, cx.nrhs, cx.dA2.F(), cx.lda, cx.dB.F(), cx.ldb, cx.dX.F(), cx.ldx,
                        cx.dWs2.p, cx.glB, &niter, cx.dInfo.I());
    const aclsolverStatus_t r2 = aclsolverSgeqrf(handle, cx.m, cx.n, cx.dA.F(), cx.lda, cx.dTau.F(), cx.dWs.F(),
                                                 cx.lwG, cx.dInfo.I());
    const aclsolverStatus_t r3 =
        aclsolverSormqr(handle, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_T, cx.m, cx.nrhs, cx.n, cx.dA.F(), cx.lda,
                        cx.dTau.F(), cx.dB.F(), cx.ldb, cx.dWs.F(), cx.lwM, cx.dInfo.I());
    const aclsolverStatus_t r4 =
        aclsolverXlarft(handle, nullptr, ACLSOLVER_DIRECT_FORWARD, ACLSOLVER_STOREV_COLUMNWISE, cx.n, cx.kT,
                        ACLSOLVER_R_32F, cx.dA.p, cx.lda, ACLSOLVER_R_32F, cx.dTau.p, ACLSOLVER_R_32F, cx.dT.p,
                        cx.ldt, ACLSOLVER_R_32F, cx.dWs.p, cx.xlB, nullptr, 0);
    const aclsolverStatus_t r5 =
        aclsolverSorgqr(handle, cx.m, cx.n, cx.n, cx.dA.F(), cx.lda, cx.dTau.F(), cx.dWs.F(), cx.lwQ, cx.dInfo.I());
    const aclError sync = aclrtSynchronizeStream(stream);
    if (r1 != ACLSOLVER_STATUS_SUCCESS || r2 != ACLSOLVER_STATUS_SUCCESS || r3 != ACLSOLVER_STATUS_SUCCESS ||
        r4 != ACLSOLVER_STATUS_SUCCESS || r5 != ACLSOLVER_STATUS_SUCCESS || sync != ACL_SUCCESS)
    {
        printf("  场景5 第 %d 趟失败: %d %d %d %d %d sync=%d\n", rep + 1, (int)r1, (int)r2, (int)r3, (int)r4,
               (int)r5, (int)sync);
        return false;
    }
    got->assign(nQ + nT + nX, 0.0f);
    (void)aclrtMemcpy(got->data(), nQ * sizeof(float), cx.dA.p, nQ * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(got->data() + nQ, nT * sizeof(float), cx.dT.p, nT * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(got->data() + nQ + nT, nX * sizeof(float), cx.dX.p, nX * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    return true;
}

/// 场景 5：五个接口连续下发、中间**完全不同步**
/// 早期怀疑这样会挂死，实测不复现；保留为回归守卫（挂住则本用例超时，不会静默通过）。
bool ScenNoSyncChain(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int reps)
{
    ChainCtx cx;
    std::vector<float> A0(static_cast<size_t>(m) * n);
    std::vector<float> B0(static_cast<size_t>(m) * cx.nrhs);
    FillMatrix(A0, 0xC4A10000u, 0);
    FillMatrix(B0, 0xC4A10001u, 1);
    ChainSetup(handle, m, n, A0.size(), B0.size(), &cx);

    std::vector<float> ref;
    std::vector<float> got;
    bool ok = true;
    for (int rep = 0; rep < reps && ok; ++rep)
    {
        if (!ChainOneRep(handle, stream, cx, rep, A0, B0, &got))
        {
            ok = false;
            break;
        }
        if (rep == 0)
        {
            ref = got;
        }
        else if (ref != got)
        {
            printf("  场景5 第 %d 趟与第 1 趟不同\n", rep + 1);
            ok = false;
        }
    }
    cx.Free();
    printf("  场景5 五接口连续下发不同步 (m=%d n=%d, %d 趟): %s\n", m, n, reps, ok ? "PASS" : "FAIL");
    return ok;
}

/// 场景 6：非默认 stream，而且中途换一条
/// 把 handle 绑到指定的流上跑一次 geqrf，结果带回 out。失败时已打印原因。
bool GeqrfOnStream(aclsolverHandle_t handle, aclrtStream s, int idx, int m, int n, int lda, int lwG, const Dev &dA,
                   const Dev &dTau, const Dev &dWs, const Dev &dInfo, const std::vector<float> &A0,
                   std::vector<float> *out)
{
    if (aclsolverSetStream(handle, s) != ACLSOLVER_STATUS_SUCCESS)
    {
        printf("  场景6 第 %d 条流 SetStream 失败\n", idx + 1);
        return false;
    }
    (void)aclrtMemcpy(dA.p, A0.size() * sizeof(float), A0.data(), A0.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    const aclsolverStatus_t st = aclsolverSgeqrf(handle, m, n, dA.F(), lda, dTau.F(), dWs.F(), lwG, dInfo.I());
    const aclError sync = aclrtSynchronizeStream(s);
    int32_t info = -1;
    (void)aclrtMemcpy(&info, sizeof(info), dInfo.p, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST);
    if (st != ACLSOLVER_STATUS_SUCCESS || sync != ACL_SUCCESS || info != 0)
    {
        printf("  场景6 第 %d 条流失败: status=%d sync=%d info=%d\n", idx + 1, (int)st, (int)sync, info);
        return false;
    }
    out->assign(A0.size(), 0.0f);
    (void)aclrtMemcpy(out->data(), A0.size() * sizeof(float), dA.p, A0.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    return true;
}

/// 问出 geqrf 的 workspace 大小，再把 A / tau / ws / info 四块设备内存一次开好，
/// 返回 workspace 的元素数。场景 6 与场景 7 要的是同一组（只有填充的随机种子不同）。
int AllocGeqrfSet(aclsolverHandle_t handle, int m, int n, int lda, size_t aElems, Dev *dA, Dev *dTau, Dev *dWs,
                  Dev *dInfo)
{
    int lwG = 0;
    (void)aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &lwG);
    dA->Alloc(aElems * sizeof(float));
    dTau->Alloc(static_cast<size_t>(n) * sizeof(float));
    dWs->Alloc(static_cast<size_t>(lwG) * sizeof(float));
    dInfo->Alloc(sizeof(int32_t));
    return lwG;
}

bool ScenStreams(aclsolverHandle_t handle, aclrtStream original, int m, int n)
{
    aclrtStream s1 = nullptr;
    aclrtStream s2 = nullptr;
    if (aclrtCreateStream(&s1) != ACL_SUCCESS || aclrtCreateStream(&s2) != ACL_SUCCESS)
    {
        printf("  场景6 建流失败\n");
        return false;
    }
    const int lda = m;
    std::vector<float> A0(static_cast<size_t>(lda) * n);
    FillMatrix(A0, 0x57E10000u, 0);
    Dev dA;
    Dev dTau;
    Dev dWs;
    Dev dInfo;
    const int lwG = AllocGeqrfSet(handle, m, n, lda, A0.size(), &dA, &dTau, &dWs, &dInfo);
    std::vector<float> out[2];
    bool ok = true;
    aclrtStream streams[2] = {s1, s2};
    for (int i = 0; i < 2; ++i)
    {
        if (!GeqrfOnStream(handle, streams[i], i, m, n, lda, lwG, dA, dTau, dWs, dInfo, A0, &out[i]))
        {
            ok = false;
            break;
        }
    }
    if (ok && out[0] != out[1])
    {
        printf("  场景6 两条流的结果不一致\n");
        ok = false;
    }
    dA.Free();
    dTau.Free();
    dWs.Free();
    dInfo.Free();
    (void)aclsolverSetStream(handle, original); // 还回原来的流，别影响后面的场景
    (void)aclrtDestroyStream(s1);
    (void)aclrtDestroyStream(s2);
    printf("  场景6 非默认 stream + 中途换流 (m=%d n=%d): %s\n", m, n, ok ? "PASS" : "FAIL");
    return ok;
}

/// 场景 7：确定性 —— 同输入重复，**A / tau / info 全部**逐位比对
bool ScenDeterminism(aclsolverHandle_t handle, aclrtStream stream, int m, int n, int reps)
{
    const int lda = m;
    std::vector<float> A0(static_cast<size_t>(lda) * n);
    FillMatrix(A0, 0xDE7E0000u, 0);
    Dev dA;
    Dev dTau;
    Dev dWs;
    Dev dInfo;
    const int lwG = AllocGeqrfSet(handle, m, n, lda, A0.size(), &dA, &dTau, &dWs, &dInfo);
    std::vector<float> ref;
    std::vector<float> got;
    int32_t refInfo = 0;
    bool ok = true;
    for (int rep = 0; rep < reps && ok; ++rep)
    {
        (void)aclrtMemcpy(dA.p, A0.size() * sizeof(float), A0.data(), A0.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        (void)aclsolverSgeqrf(handle, m, n, dA.F(), lda, dTau.F(), dWs.F(), lwG, dInfo.I());
        (void)aclrtSynchronizeStream(stream);
        got.assign(A0.size() + static_cast<size_t>(n), 0.0f);
        (void)aclrtMemcpy(got.data(), A0.size() * sizeof(float), dA.p, A0.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        (void)aclrtMemcpy(got.data() + A0.size(), static_cast<size_t>(n) * sizeof(float), dTau.p,
                          static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        int32_t info = -1;
        (void)aclrtMemcpy(&info, sizeof(info), dInfo.p, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST);
        if (rep == 0)
        {
            ref = got;
            refInfo = info;
        }
        else if (ref != got || info != refInfo)
        {
            printf("  场景7 第 %d 趟与第 1 趟不同（含 tau/info）\n", rep + 1);
            ok = false;
        }
    }
    dA.Free();
    dTau.Free();
    dWs.Free();
    dInfo.Free();
    printf("  场景7 确定性 A/tau/info 逐位 (m=%d n=%d, %d 趟): %s\n", m, n, reps, ok ? "PASS" : "FAIL");
    return ok;
}

int RunScenario(aclsolverHandle_t handle, aclrtStream stream)
{
    int pass = 0;
    int total = 0;
    ++total;
    pass += ScenRestoreQ(handle, stream, 256, 128) ? 1 : 0;
    ++total;
    pass += ScenRestoreQ(handle, stream, 129, 129) ? 1 : 0;
    ++total;
    pass += ScenOrmqrRoundTrip(handle, stream, 256, 128, 4) ? 1 : 0;
    ++total;
    pass += ScenOrmqrRoundTrip(handle, stream, 513, 257, 1) ? 1 : 0;
    ++total;
    pass += ScenXlarftSemantics(handle, stream, 256, 64, false) ? 1 : 0;
    ++total;
    pass += ScenXlarftSemantics(handle, stream, 256, 64, true) ? 1 : 0;
    ++total;
    pass += ScenXlarftSemantics(handle, stream, 129, 37, false) ? 1 : 0;
    printf("  场景4 SSgels 端到端（两类判据，复用 RunShape）\n");
    ++total;
    pass += RunShape(handle, stream, 1, 256, 128, 4, 256, 256, 128, 2, true, true, nullptr) ? 1 : 0;
    ++total;
    pass += RunShape(handle, stream, 2, 512, 512, 1, 512, 512, 512, 2, true, true, nullptr) ? 1 : 0;
    ++total;
    pass += ScenNoSyncChain(handle, stream, 64, 64, 3) ? 1 : 0;
    ++total;
    pass += ScenNoSyncChain(handle, stream, 256, 128, 3) ? 1 : 0;
    ++total;
    pass += ScenNoSyncChain(handle, stream, 1024, 512, 3) ? 1 : 0;
    ++total;
    pass += ScenStreams(handle, stream, 256, 128) ? 1 : 0;
    ++total;
    pass += ScenDeterminism(handle, stream, 256, 128, 3) ? 1 : 0;
    ++total;
    pass += ScenDeterminism(handle, stream, 1024, 1024, 3) ? 1 : 0;
    printf("\n跨接口场景：%d / %d PASS\n", pass, total);
    return (pass == total) ? 0 : 1;
}

} // namespace
