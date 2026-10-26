/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <chrono>
#include <limits>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <queue>
#include <algorithm>

#include "cann_ops_solver.h"

#include "../utils/test_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"


#include "acl/acl.h"


#include <chrono>
using namespace std::chrono;
using namespace std;


int main(int argc, char **argv) {
	int deviceId, M, N;
    auto outputAFile = "../output/A_gm.bin";

    deviceId = (argc>1) ? std::atoi(argv[1]) : 0;
    M = (argc>2) ? std::atoi(argv[2]) : 255;
    N = (argc>3) ? std::atoi(argv[3]) : 255;
    // 输入合法性检查
    if (deviceId < 0 || deviceId > 8192 || M < 1 || M > 8192 || N < 1 || N > 8192 || M != N) {
        std::cerr << "Error: Invalid input. deviceId: 0-8192, M=N: 1-8192" << std::endl;
        return 1;
    }

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = M * N * sizeof(float);

    float* A;
    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    ReadFile("./test/spotrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);


    int32_t infoVal = 0;
 	int32_t *info = &infoVal;

    auto ret = aclsolverSpotrf(handle,'L', M, N, A, N, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSpotrf failed. ERROR: %d\n", ret); return ret);

    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);

    WriteFile("./test/spotrf/data/output/A_gm.bin", A, aMatrixFileSize);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclsolverDestroy(handle));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}
