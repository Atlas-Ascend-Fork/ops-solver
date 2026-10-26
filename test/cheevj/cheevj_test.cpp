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
 * \file cheevj_test.cpp
 * \brief
 */

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

namespace
{
constexpr const char *INPUT_A_PATH = "./test/cheevj/data/input/A_gm.bin";
constexpr const char *OUTPUT_W_PATH = "./test/cheevj/data/output/W_gm.bin";
constexpr const char *OUTPUT_V_PATH = "./test/cheevj/data/output/V_gm.bin";

char NormalizeOption(const char *arg, char defaultValue)
{
    if (arg == nullptr || arg[0] == '\0')
    {
        return defaultValue;
    }
    return static_cast<char>(std::toupper(static_cast<unsigned char>(arg[0])));
}

aclsolverEigMode_t ParseJobz(char jobz)
{
    return jobz == 'N' ? ACLSOLVER_EIG_MODE_NOVECTOR : ACLSOLVER_EIG_MODE_VECTOR;
}

aclsolverFillMode_t ParseUplo(char uplo) { return uplo == 'U' ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER; }

void PrintComplexColumnMajor(const std::complex<float> *data, int64_t rows, int64_t cols, int64_t maxRows,
                             int64_t maxCols)
{
    for (int64_t row = 0; row < std::min(rows, maxRows); ++row)
    {
        for (int64_t col = 0; col < std::min(cols, maxCols); ++col)
        {
            const auto value = data[col * rows + row];
            std::cout << "(" << value.real() << "," << value.imag() << ") ";
        }
        std::cout << std::endl;
    }
}

void PrintVector(const float *data, int64_t size, int64_t maxSize)
{
    for (int64_t i = 0; i < std::min(size, maxSize); ++i)
    {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void WriteResults(char jobz, const std::complex<float> *matrix, int64_t n, const std::vector<float> &eigenvalues,
                  size_t matrixBytes)
{
    std::cout << "[Output] W:" << std::endl;
    PrintVector(eigenvalues.data(), n, 16);
    WriteFile(OUTPUT_W_PATH, eigenvalues.data(), static_cast<size_t>(n) * sizeof(float));
    if (jobz == 'V')
    {
        std::cout << "[Output] V:" << std::endl;
        PrintComplexColumnMajor(matrix, n, n, 8, 8);
        WriteFile(OUTPUT_V_PATH, matrix, matrixBytes);
    }
}
}  // namespace

int32_t main(int32_t argc, char *argv[])
{
    int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    int64_t n = (argc > 2) ? std::atoll(argv[2]) : 8;
    char jobzChar = (argc > 3) ? NormalizeOption(argv[3], 'V') : 'V';
    char uploChar = (argc > 4) ? NormalizeOption(argv[4], 'L') : 'L';

    CHECK_RET(n > 0, LOG_PRINT("cheevj_test requires n > 0.\n"); return 1);
    CHECK_RET(jobzChar == 'V' || jobzChar == 'N', LOG_PRINT("jobz must be V or N, got %c.\n", jobzChar); return 1);
    CHECK_RET(uploChar == 'L' || uploChar == 'U', LOG_PRINT("uplo must be L or U, got %c.\n", uploChar); return 1);
    CHECK_RET(static_cast<uint64_t>(n) <=
                  std::numeric_limits<size_t>::max() / static_cast<uint64_t>(n) / sizeof(std::complex<float>),
              LOG_PRINT("Matrix byte size overflows size_t.\n");
              return 1);

    // 资源句柄先行声明（issue #257，对齐 test/sgetri/sgetri_test.cpp 的 cleanup
    // lambda 范式）：此前的错误路径直接 return 1，绕过 A（pinned host 内存）、
    // stream/handle/device/aclFinalize 清理，反复运行/多卡复用环境会放大成资源
    // 泄漏；所有错误路径统一 cleanup() 后返回
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    std::complex<float> *A = nullptr;
    auto cleanup = [&]()
    {
        if (A != nullptr)
        {
            (void)aclrtFreeHost(A);
            A = nullptr;
        }
        if (stream != nullptr)
        {
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (handle != nullptr)
        {
            (void)aclsolverDestroy(handle);
            handle = nullptr;
        }
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
    };

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    const size_t matrixElements = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t aMatrixFileSize = matrixElements * sizeof(std::complex<float>);
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&A), aMatrixFileSize));
    size_t readSize = aMatrixFileSize;
    CHECK_RET(ReadFile(INPUT_A_PATH, readSize, A, aMatrixFileSize),
              LOG_PRINT("Read input matrix failed: %s\n", INPUT_A_PATH);
              cleanup(); return 1);
    const std::vector<std::complex<float>> inputA(A, A + matrixElements);

    std::vector<float> W(n, 0.0f);
    std::vector<int32_t> info(1, 0);

    std::cout << "[Input] A (column-major Hermitian):" << std::endl;
    PrintComplexColumnMajor(A, n, n, 8, 8);

    auto ret = aclsolverCheevj(handle, ParseJobz(jobzChar), ParseUplo(uploChar), n, A, n, W.data(), info.data());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCheevj failed. ERROR: %d\n", ret); cleanup(); return ret);
    CHECK_RET(info[0] == 0, LOG_PRINT("aclsolverCheevj returned info: %d\n", info[0]); cleanup(); return info[0]);
    if (jobzChar == 'N')
    {
        CHECK_RET(std::memcmp(inputA.data(), A, aMatrixFileSize) == 0,
                  LOG_PRINT("aclsolverCheevj modified A for jobz=N.\n");
                  cleanup(); return 1);
    }

    WriteResults(jobzChar, A, n, W, aMatrixFileSize);

    cleanup();
    return 0;
}
