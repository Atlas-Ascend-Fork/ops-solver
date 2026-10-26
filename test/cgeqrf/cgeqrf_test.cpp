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
 * \file cgeqrf_test.cpp
 * \brief
 */

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

typedef std::complex<float> complex64;

namespace
{
// 按列主序打印矩阵左上角 (printRows x printCols) 的复元素块，rows 为存储前导维。
void PrintPartOfComplexMatrix(const complex64* data, int rows, int cols, int printRows, int printCols)
{
    int numRows = std::min(rows, printRows);
    int numCols = std::min(cols, printCols);
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            std::cout << data[i + j * rows] << " ";
        }
        std::cout << std::endl;
    }
}

// 打印复向量前 min(count, 8) 个元素。
void PrintPartialVector(const complex64* data, int count)
{
    int numElems = std::min(count, 8);
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < numElems; ++i)
    {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}
}  // namespace

int main(int argc, char** argv)
{
    int deviceId, M, N;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    M = (argc > 2) ? std::atoi(argv[2]) : 1024;
    N = (argc > 3) ? std::atoi(argv[3]) : 256;
    int minMN = std::min(M, N);
    size_t aMatrixFileSize = M * N * sizeof(complex64);
    size_t tauFileSize = minMN * sizeof(complex64);

    if (M <= 0 || N <= 0)
    {
        LOG_PRINT("Invalid M, N. M = %d, N = %d\n", M, N);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::cout << "deviceId = " << deviceId << ", M = " << M << ", N = " << N << std::endl;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    complex64* A = nullptr;
    complex64* tau = nullptr;
    auto cleanup = [&]() -> aclError
    {
        if (A != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(A));
        }
        if (tau != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(tau));
        }
        CHECK_ACL(aclrtDestroyStream(stream));
        if (handle != nullptr)
        {
            CHECK_ACL(aclsolverDestroy(handle));
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return ACL_SUCCESS;
    };

    aclError mallocRet = aclrtMallocHost((void**)(&A), aMatrixFileSize);
    CHECK_RET(mallocRet == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost A failed. ERROR: %d\n", mallocRet); cleanup();
              return mallocRet);
    CHECK_RET(ReadFile("./test/cgeqrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfComplexMatrix(A, M, N, 8, 8);

    mallocRet = aclrtMallocHost((void**)(&tau), tauFileSize);
    CHECK_RET(mallocRet == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost tau failed. ERROR: %d\n", mallocRet); cleanup();
              return mallocRet);

    int32_t infoVal = 0;
    int32_t* info = &infoVal;

    auto ret = aclsolverCgeqrf(handle, M, N, A, M, tau, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCgeqrf failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] A (QR):" << std::endl;
    PrintPartOfComplexMatrix(A, M, N, 8, 8);
    std::cout << "[Output] tau:" << std::endl;
    PrintPartialVector(tau, minMN);

    CHECK_RET(WriteFile("./test/cgeqrf/data/output/R_gm.bin", A, aMatrixFileSize),
              LOG_PRINT("WriteFile R_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);
    CHECK_RET(WriteFile("./test/cgeqrf/data/output/tau.bin", tau, tauFileSize),
              LOG_PRINT("WriteFile tau.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
