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
 * \file sgeqrf_test.cpp
 * \brief
 */

#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "../utils/test_utils.h"

int main(int argc, char *argv[])
{
    int deviceId, M, N, blockDim, padM;
    deviceId =(argc>1) ?  std::atoi(argv[1]) : 0;
    M = (argc>2) ? std::atoi(argv[2]) : 1024;
    N = (argc>3) ? std::atoi(argv[3]) : 256;
    blockDim = (argc>4) ? std::atoi(argv[4]) : 16;

    padM = ((M + 63) / 64) * 64;
    auto outputFile = "./test/sgeqrf/data/output/result.bin";
    auto tauFile = "./test/sgeqrf/data/output/tau.bin";
    std::cout << "deviceId = " << deviceId << ", M = " << M << ", padM = " << padM << ", N = " << N << ", blockDim = " << blockDim << ", output_file = " << outputFile << std::endl;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = M * N * sizeof(float);
    size_t tauFileSize = N * sizeof(float);

    float* A;
    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    ReadFile("./test/sgeqrf/data/input/matrix.bin", aMatrixFileSize, A, aMatrixFileSize);
    
    float* tau;
    CHECK_ACL(aclrtMallocHost((void**)(&tau), tauFileSize));

    std::cout << "[Input] C:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);

    int32_t infoVal = 0;
    int32_t *info = &infoVal;
    
    auto ret = aclsolverSgeqrf(handle, M, N, A, M, tau, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgeqrf failed. ERROR: %d\n", ret); return ret);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::cout << "[OUTPUT] C(Y \\ QR):" << '\n';
    PrintPartOfMatrix<float>((uint8_t *)A, N, M, 8, 8);

    std::cout << "[OUTPUT] TAU:" << '\n';
    PrintPartOfMatrix<float>((uint8_t *)tau, 1, N, 8, 8);

    WriteFile(outputFile, A, aMatrixFileSize);
    WriteFile(tauFile, tau, tauFileSize);

    CHECK_ACL(aclrtFreeHost(A));
    CHECK_ACL(aclrtFreeHost(tau));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclsolverDestroy(handle));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}