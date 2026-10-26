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
 * \file cgetri_batched_test.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

int32_t main(int32_t argc, char* argv[])
{
    int deviceId, batchSize, n;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    batchSize = (argc > 2) ? std::atoi(argv[2]) : 2;
    n = (argc > 3) ? std::atoi(argv[3]) : 48;

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    std::complex<float>* A = nullptr;

    auto cleanup = [&]()
    {
        if (A) aclrtFreeHost(A);
        if (handle) aclsolverDestroy(handle);
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS)
    {
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS)
    {
        cleanup();
        return ret;
    }
    ret = aclsolverCreate(&handle);
    if (ret != ACL_SUCCESS)
    {
        cleanup();
        return ret;
    }
    ret = aclsolverSetStream(handle, stream);
    if (ret != ACL_SUCCESS)
    {
        cleanup();
        return ret;
    }

    size_t aMatrixFileSize = batchSize * n * n * sizeof(std::complex<float>);
    ret = aclrtMallocHost((void**)(&A), aMatrixFileSize);
    if (ret != ACL_SUCCESS)
    {
        cleanup();
        return ret;
    }

    CHECK_RET(ReadFile("./test/cgetri_batched/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);
    std::vector<std::complex<float>> Ainv(batchSize * n * n, {-1.0f, -1.0f});
    std::vector<int32_t> info(batchSize, 0);

    ret = aclsolverCgetriBatched(handle, n, A, n, Ainv.data(), n, info.data(), batchSize);
    if (ret != ACL_SUCCESS)
    {
        LOG_PRINT("aclsolverCgetriBatched failed. ERROR: %d\n", ret);
        cleanup();
        return ret;
    }

    // 写出校验（issue #259，与 :84 ReadFile 校验对称）：写出失败（目录只读、
    // 磁盘满）而输出路径存在旧结果文件时，进程若仍返回 0，后续 verify_result.py
    // 校验的是过期数据，精度验证"假通过"；失败时 cleanup 并返回错误码
    CHECK_RET(WriteFile("./test/cgetri_batched/data/output/Ainv_gm.bin", Ainv.data(), aMatrixFileSize),
              LOG_PRINT("WriteFile Ainv_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    cleanup();
    return 0;
}
