/*
 * PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
 *
 * Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
 * All rights reserved.
 *
 * This software is dual-licensed:
 *   1) Community license: use, modify, and redistribute freely under the
 *      GNU General Public License v3 or later (GPL-3.0-or-later);
 *   2) Commercial license: for closed-source or commercial use / redistribution,
 *      contact the copyright holder for a separate license.
 * You may choose either. Default terms are GPL-3.0-or-later.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you may redistribute it and/or modify it under
 * the terms of the GPL. It is distributed "as is", without warranty.
 * See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
 * Commercial licensing contact: yuantang@fudan.edu.cn
 *
 * -------------------------------------------------------------------------
 * 中文版 (Chinese):
 *
 * PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
 *
 * 本软件采用双重许可（dual licensing）：
 *   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
 *      （GPL-3.0-or-later）自由使用、修改与再分发；
 *   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
 * 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
 *
 * 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
 * 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */

/*!
 * \file sgeqrf_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <securec.h>
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "../utils/assert.h"
#include "../utils/gm_addr.h"

#include "tiling/platform/platform_ascendc.h"

#include "sgeqrf_launchers.hpp"

aclError aclsolverSgeqrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, float *A, const int64_t lda, float *tau, int32_t *info)
{
    SOLVER_ECHECK(m > 0 && n > 0 && lda >= m && A != nullptr && tau != nullptr && info != nullptr,
                  "aclsolverSgeqrf invalid param: m, n <= 0, or lda < m or A, tau, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m <= INT32_MAX && n <= INT32_MAX, "aclsolverSgeqrf invalid param: m or n exceeds int32 range.",
                  ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr)
    {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }
    SOLVER_ECHECK(numBlocks >= 16, "the number of aicores < 16, aclsolverSgeqrf only supports >= 16, check the device.",
                  ACL_ERROR_INVALID_PARAM);
    int64_t M, N, minMN, padM, padN;
    M = m;
    N = n;
    minMN = M > N ? N : M;
    int blockDim = numBlocks;
    padM = ((M + 63) / 64) * 64;
    padN = ((N + 63) / 64) * 64;

    constexpr uint32_t MAX_PANEL_SIZE = 128;
    size_t aSize = M * N * sizeof(float);
    size_t aPadSize = padM * padN * sizeof(float);
    size_t tauSize = minMN * sizeof(float);
    // kernel 在 M >= 64 分支按 padM / padN 迭代，tau 下标最多到 min(padM, padN)：
    // device 侧按该上界分配并清零，host 侧仍只回读 min(m, n) 个，避免对 padding 列的 tau 写/读越界
    size_t tauDeviceSize = static_cast<size_t>(std::max(minMN, std::min(padM, padN))) * sizeof(float);
    size_t tSize = MAX_PANEL_SIZE * MAX_PANEL_SIZE * sizeof(float) * 2;
    size_t wSize = (padM * padN + 2048) * sizeof(float);
    size_t ySize = padM * MAX_PANEL_SIZE * sizeof(float) * 2;
    size_t workSize = tSize + wSize + ySize;

    uint8_t *aHost = reinterpret_cast<uint8_t*>(A);
    uint8_t *aDevice = nullptr;
    uint8_t *aPadDevice = nullptr;
    uint8_t *tauHost = reinterpret_cast<uint8_t*>(tau);
    uint8_t *tauDevice = nullptr;
    uint8_t *workDevice = nullptr;
    uint8_t *sync = nullptr;

    // aDevice 存放按 M 间距打包的输入，aPadDevice 存放 padM * padN 布局的中间结果与最终结果；
    // Pad() 只写有效行列，padding 区必须预先清零，否则未初始化内存中的 NaN/Inf
    // 会经 T 矩阵、Y^T*C 等中间量污染整个面板
    auto cleanup = [&]()
    {
        if (aDevice) aclrtFree(aDevice);
        if (aPadDevice) aclrtFree(aPadDevice);
        if (tauDevice) aclrtFree(tauDevice);
        if (workDevice) aclrtFree(workDevice);
    };

    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void **>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy2dAsync(aDevice, M * sizeof(float), aHost, lda * sizeof(float), M * sizeof(float), N, ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void **>(&aPadDevice), aPadSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemsetAsync(aPadDevice, aPadSize, 0, aPadSize, stream), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void **>(&tauDevice), tauDeviceSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemsetAsync(tauDevice, tauDeviceSize, 0, tauDeviceSize, stream), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void **>(&workDevice), workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    sgeqrf_kernel_do(sync, M, N, padM, padN, aDevice, aPadDevice, tauDevice, workDevice, blockDim, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    // 结果统一写在 aPad（padM 间距，前 M 行 × 前 N 列），按 lda 间距展开回 host
    CHECK_ACLRT(aclrtMemcpy2dAsync(aHost, lda * sizeof(float), aPadDevice, padM * sizeof(float), M * sizeof(float), N, ACL_MEMCPY_DEVICE_TO_HOST, stream), cleanup());
    CHECK_ACLRT(aclrtMemcpy(tauHost, tauSize, tauDevice, tauSize, ACL_MEMCPY_DEVICE_TO_HOST), cleanup());

    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());
    cleanup();
    return ACL_SUCCESS;
}
