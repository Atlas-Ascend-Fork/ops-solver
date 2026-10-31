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
 * \file cgeqrf_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/aicore_num.h"
#include "../utils/assert.h"
#include "../utils/gm_addr.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "cgeqrf_launchers.hpp"

typedef std::complex<float> complex64;

bool GenerateGatherTiling(int M, int N, uint8_t* gatherBuf, uint8_t* eyeBuf)
{
    try
    {
        auto buf = reinterpret_cast<uint32_t*>(gatherBuf);
        int offset = 64, stride = 2 * offset, columns = 32;
        // gatherN
        for (int j = 0; j < columns; ++j) buf[j] = (j * stride) * sizeof(float);
        for (int j = 0; j < columns; ++j) buf[j + columns] = (j * stride + offset) * sizeof(float);
        buf += 128;

        int idx = 0;
        static constexpr int TILE_LENGTH = 4096;
        for (int i = 0; i < TILE_LENGTH; ++i)
        {
            buf[idx++] = (i) * sizeof(float);
            buf[idx++] = (i + TILE_LENGTH) * sizeof(float);
        }

        {
            auto buf = reinterpret_cast<float*>(eyeBuf);
            for (int i = 0; i < 64; ++i)
            {
                for (int j = 0; j < 64; ++j)
                {
                    if (i == j)
                        buf[i * 64 + j] = -1;
                    else
                        buf[i * 64 + j] = 0;
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}

aclError aclsolverCgeqrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, std::complex<float>* A,
                         const int64_t lda, std::complex<float>* tau, int32_t* info)
{
    SOLVER_ECHECK(handle != nullptr, "aclsolverCgeqrf invalid param: handle is nullptr.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m > 0 && n > 0 && lda >= m && A != nullptr && tau != nullptr && info != nullptr,
                  "aclsolverCgeqrf invalid param: m, n <= 0, or lda < m, or A, tau, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m <= INT32_MAX && n <= INT32_MAX, "aclsolverCgeqrf invalid param: m or n exceeds int32 range.",
                  ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    aclsolverGetStream(handle, &stream);
    SOLVER_ECHECK(stream != nullptr, "aclsolverCgeqrf invalid param: handle has no bound stream.",
                  ACL_ERROR_INVALID_PARAM);
    uint32_t numBlocks = GetAiCoreNumOrZero();
    SOLVER_ECHECK(numBlocks > 0, "No available AIcore found on current device.", ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
    int64_t M, N, minMN, padM, padN;
    M = m;
    N = n;
    minMN = M < N ? M : N;
    int blockDim = numBlocks;
    padM = ((M + 127) / 128) * 128;
    padN = ((N + 127) / 128) * 128;

    constexpr int MAX_PANEL_SIZE = 128;
    size_t aSize = M * N * sizeof(complex64);
    size_t aPadSize = padM * padN * sizeof(float) * 2;
    size_t tauSize = minMN * sizeof(complex64);
    // 内核写 tau 时按补齐后的 minMN（64 对齐）寻址，设备侧需按 max(minMN, min(padM, padN)) 分配并清零，
    // 避免越界写；host 侧仍只回拷 minMN 个
    size_t tauDeviceSize = static_cast<size_t>(std::max(minMN, std::min(padM, padN))) * sizeof(complex64);
    size_t tSize = padM * padN * sizeof(float) * 3;
    size_t wSize = (std::max(padM, padN) * 128 + 4096 + 64) * sizeof(float) * 2;
    size_t ySize = padM * padN * sizeof(float) * 3;
    size_t workSize = tSize + wSize + ySize;
    size_t gatherSize = 128 * sizeof(uint32_t) + 8192 * sizeof(uint32_t);
    size_t eyeSize = 64 * 64 * sizeof(float);

    uint8_t* aHost = reinterpret_cast<uint8_t*>(A);
    uint8_t* aDevice = nullptr;
    uint8_t* aPadDevice = nullptr;
    uint8_t* tauHost = reinterpret_cast<uint8_t*>(tau);
    uint8_t* tauDevice = nullptr;
    uint8_t* workDevice = nullptr;
    uint8_t* eyeHost = nullptr;
    uint8_t* eyeDevice = nullptr;
    uint8_t* gatherHost = nullptr;
    uint8_t* gatherDevice = nullptr;
    uint8_t* sync = nullptr;

    auto cleanup = [&]()
    {
        if (aDevice) aclrtFree(aDevice);
        if (aPadDevice) aclrtFree(aPadDevice);
        if (tauDevice) aclrtFree(tauDevice);
        if (workDevice) aclrtFree(workDevice);
        if (eyeHost) aclrtFreeHost(eyeHost);
        if (eyeDevice) aclrtFree(eyeDevice);
        if (gatherHost) aclrtFreeHost(gatherHost);
        if (gatherDevice) aclrtFree(gatherDevice);
    };

    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy2dAsync(aDevice, M * sizeof(complex64), aHost, lda * sizeof(complex64),
                                   M * sizeof(complex64), N, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());

    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&aPadDevice), aPadSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    // Pad() 只写有效行列，padding 区必须预先清零，否则未初始化内存中的 NaN/Inf
    // 会经中间量污染面板，与 sgeqrf 同族防护口径一致
    CHECK_ACLRT(aclrtMemsetAsync(aPadDevice, aPadSize, 0, aPadSize, stream), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&tauDevice), tauDeviceSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemsetAsync(tauDevice, tauDeviceSize, 0, tauDeviceSize, stream), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&workDevice), workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost(reinterpret_cast<void**>(&eyeHost), eyeSize), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&eyeDevice), eyeSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMallocHost(reinterpret_cast<void**>(&gatherHost), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&gatherDevice), gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    if (!GenerateGatherTiling(M, N, gatherHost, eyeHost))
    {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }

    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice, gatherSize, gatherHost, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(eyeDevice, eyeSize, eyeHost, eyeSize, ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void**)&sync), cleanup());

    cgeqrf_kernel_do(sync, M, N, M, padM, aDevice, aPadDevice, tauDevice, workDevice, gatherDevice, eyeDevice, blockDim,
                     stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy2dAsync(aHost, lda * sizeof(complex64), aDevice, M * sizeof(complex64),
                                   M * sizeof(complex64), N, ACL_MEMCPY_DEVICE_TO_HOST, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpy(tauHost, tauSize, tauDevice, tauSize, ACL_MEMCPY_DEVICE_TO_HOST), cleanup());

    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());
    cleanup();
    return ACL_SUCCESS;
}
