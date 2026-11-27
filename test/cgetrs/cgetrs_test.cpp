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
 * \file cgetrs_test.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

namespace
{
// trans 档清单（issue #637）：接口文档承诺 0（AX=B）、1（A^T·X=B）、2（A^H·X=B）三档，逐档回归
const int TRANS_LIST[] = {0, 1, 2};
const int TRANS_COUNT = static_cast<int>(sizeof(TRANS_LIST) / sizeof(TRANS_LIST[0]));

}  // namespace

int main(int argc, char **argv)
{
    int deviceId, N, NRHS;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    N = (argc > 2) ? std::atoi(argv[2]) : 32;
    NRHS = (argc > 3) ? std::atoi(argv[3]) : 32;

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(InitGetrsTest(deviceId, &stream, &handle));

    aclFloatComplex *A = nullptr;
    aclFloatComplex *B = nullptr;
    int32_t *ipiv = nullptr;
    // C ABI 契约（issue #354）：aclsolverCgetrs 入参为 C 结构的 aclFloatComplex，
    // 与 std::complex<float> 二进制字节等价，可直接使用各类型的引用。
    auto loadRet = LoadGetrsInputs<aclFloatComplex>("./test/cgetrs/data", N, NRHS, &A, &B, &ipiv);
    CHECK_RET(loadRet == ACL_SUCCESS, DestroyGetrsTest(stream, handle, deviceId, A, B, ipiv); return loadRet);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, N, N * 2, 8, 8);

    std::cout << "[Input] ipiv:" << std::endl;
    PrintPartOfMatrix<int32_t>((uint8_t *)ipiv, 1, N, 8, 8);

    std::cout << "[Input] B:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)B, N, NRHS * 2, 8, 8);

    // B 会被算子原地覆盖为解 X，逐档执行前从备份恢复（issue #637）
    const std::vector<aclFloatComplex> bBackup(B, B + static_cast<size_t>(N) * NRHS);
    auto sweepRet = RunGetrsTransSweep<aclFloatComplex>(
        [&](uint64_t trans, int64_t n, int64_t nrhs, aclFloatComplex *a, int64_t lda, int32_t *pivots,
            aclFloatComplex *b, int64_t ldb, int32_t *info)
        { return aclsolverCgetrs(handle, trans, n, nrhs, a, lda, pivots, b, ldb, info); },
        TRANS_LIST, TRANS_COUNT, N, NRHS, A, N, ipiv, bBackup, B, NRHS, "./test/cgetrs/data/output/X_gm", NRHS * 2);
    CHECK_RET(sweepRet == ACL_SUCCESS, DestroyGetrsTest(stream, handle, deviceId, A, B, ipiv); return sweepRet);

    auto cleanupRet = DestroyGetrsTest(stream, handle, deviceId, A, B, ipiv);
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
