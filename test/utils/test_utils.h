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
 * \file test_utils.h
 * \brief Test utilities for ops-solver
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "cann_ops_solver_common.h"

typedef enum
{
    DT_UNDEFINED = -1,
    FLOAT = 0,
    HALF = 1,
    INT8_T = 2,
    INT32_T = 3,
    UINT8_T = 4,
    INT16_T = 6,
    UINT16_T = 7,
    UINT32_T = 8,
    INT64_T = 9,
    UINT64_T = 10,
    DOUBLE = 11,
    BOOL = 12,
    STRING = 13,
    COMPLEX64 = 16,
    COMPLEX128 = 17,
    BF16 = 27
} printDataType;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define CHECK_ACL(x)                                                                        \
    do                                                                                      \
    {                                                                                       \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_SUCCESS)                                                           \
        {                                                                                   \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
            const char *__recent_msg = aclGetRecentErrMsg();                                \
            if (__recent_msg != nullptr)                                                    \
            {                                                                               \
                std::cerr << "recent error message: " << __recent_msg << std::endl;         \
            }                                                                               \
            return __ret;                                                                   \
        }                                                                                   \
    } while (0)

#define CHECK_SOLVERDN(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        aclsolverStatus_t __ret = x;                                                               \
        if (__ret != ACLSOLVER_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclsolverStatus:" << __ret << std::endl; \
            return -1;                                                                             \
        }                                                                                          \
    } while (0)

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1)
    {
        ERROR_LOG("failed to get file: %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0)
    {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open())
    {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0)
    {
        ERROR_LOG("file size is 0: %s", filePath.c_str());
        return false;
    }
    if (size > bufferSize)
    {
        ERROR_LOG("file size is larger than buffer size: %s", filePath.c_str());
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    fileSize = size;
    buf->sgetn(static_cast<char *>(buffer), fileSize);
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr)
    {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    size_t pos = filePath.find_last_of("/\\");
    if (pos != std::string::npos) mkdir(filePath.substr(0, pos).c_str(), 0755);

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0)
    {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size)
    {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

template <typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i)
    {
        std::cout << std::setw(10) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1)
        {
            std::cout << std::endl;
        }
    }
}

void DoPrintHalfData(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i)
    {
        std::cout << std::setw(10) << std::setprecision(6) << aclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1)
        {
            std::cout << std::endl;
        }
    }
}

void PrintData(const void *data, size_t count, printDataType dataType, size_t elementsPerRow = 16)
{
    if (data == nullptr)
    {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType)
    {
        case BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case INT8_T:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case UINT8_T:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case INT16_T:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case UINT16_T:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case INT32_T:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case UINT32_T:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case INT64_T:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case UINT64_T:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;
        case HALF:
            DoPrintHalfData(reinterpret_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;
        case FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
    std::cout << std::endl;
}

template <typename T>
void PrintPartOfMatrix(uint8_t *data, int rows, int cols, int m, int n)
{
    T *mat = reinterpret_cast<T *>(data);
    for (int i = 0; i < std::min(rows, m); ++i)
    {
        for (int j = 0; j < std::min(cols, n); ++j)
        {
            printf("%f ", static_cast<float>(mat[i * cols + j]));
        }
        printf("\n");
    }
}

inline void printTensor(const std::complex<float> *data, int64_t batch, int64_t rows, int64_t cols)
{
    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t i = 0; i < rows; ++i)
        {
            for (int64_t j = 0; j < cols; ++j)
            {
                std::cout << data[b * rows * cols + i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

inline bool allclose(const std::complex<float> *a, const std::complex<float> *b, int64_t size, float rtol, float atol)
{
    for (int64_t i = 0; i < size; ++i)
    {
        float dr = std::abs(a[i].real() - b[i].real());
        float di = std::abs(a[i].imag() - b[i].imag());
        if (dr > atol + rtol * std::abs(a[i].real()) || di > atol + rtol * std::abs(a[i].imag()))
        {
            std::cout << "[Failed] verification failed!" << std::endl;
            return false;
        }
    }
    std::cout << "[Success] verification passed." << std::endl;
    return true;
}

// getrs 系测试公共输入加载（issue #637）：按 dtype 读取 A/B/W 三份 Host 数据，dataDir 为算子数据目录
template <typename T>
aclError LoadGetrsInputs(const std::string &dataDir, int n, int nrhs, T **a, T **b, int32_t **ipiv)
{
    size_t aMatrixFileSize = static_cast<size_t>(n) * n * sizeof(T);
    size_t ipivFileSize = static_cast<size_t>(n) * sizeof(int32_t);
    size_t bMatrixFileSize = static_cast<size_t>(n) * nrhs * sizeof(T);
    CHECK_ACL(aclrtMallocHost((void **)a, aMatrixFileSize));
    CHECK_RET(ReadFile(dataDir + "/input/A_gm.bin", aMatrixFileSize, *a, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              return ACL_ERROR_INVALID_PARAM);
    CHECK_ACL(aclrtMallocHost((void **)b, bMatrixFileSize));
    CHECK_RET(ReadFile(dataDir + "/input/B_gm.bin", bMatrixFileSize, *b, bMatrixFileSize),
              LOG_PRINT("ReadFile B_gm.bin failed.\n");
              return ACL_ERROR_INVALID_PARAM);
    CHECK_ACL(aclrtMallocHost((void **)ipiv, ipivFileSize));
    CHECK_RET(ReadFile(dataDir + "/input/W_gm.bin", ipivFileSize, *ipiv, ipivFileSize),
              LOG_PRINT("ReadFile W_gm.bin failed.\n");
              return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

// getrs 系测试公共初始化（issue #637）：aclInit、选卡、创建流与算子句柄并绑定流
inline aclError InitGetrsTest(int deviceId, aclrtStream *stream, aclsolverHandle_t *handle)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(stream));
    CHECK_ACL(aclsolverCreate(handle));
    CHECK_ACL(aclsolverSetStream(*handle, *stream));
    return ACL_SUCCESS;
}

// getrs 系测试公共清理（issue #637）：释放 Host 内存、销毁流与句柄、复位设备并逆初始化
template <typename T>
aclError DestroyGetrsTest(aclrtStream stream, aclsolverHandle_t handle, int deviceId, T *a, T *b, int32_t *ipiv)
{
    if (a != nullptr)
    {
        CHECK_ACL(aclrtFreeHost(a));
    }
    if (b != nullptr)
    {
        CHECK_ACL(aclrtFreeHost(b));
    }
    if (ipiv != nullptr)
    {
        CHECK_ACL(aclrtFreeHost(ipiv));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    if (handle != nullptr)
    {
        CHECK_ACL(aclsolverDestroy(handle));
    }
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return ACL_SUCCESS;
}

// getrs 系多 trans 档扫描（issue #637）：逐档从备份恢复 B、调用算子并按档落盘输出。
// trans=0 输出沿用 <outBase>.bin，非零档追加 _trans<t> 后缀；apiCall 封装各算子 API 差异，
// printCols 为 X 打印列数（实数 nrhs、复数交错布局 nrhs*2）。
inline std::string GetrsTransOutputPath(const std::string &outBase, int trans)
{
    return outBase + (trans == 0 ? std::string(".bin") : "_trans" + std::to_string(trans) + ".bin");
}

template <typename T, typename ApiFn>
aclError RunGetrsTransSweep(ApiFn apiCall, const int *transList, int transCount, int64_t n, int64_t nrhs, T *a,
                            int64_t lda, int32_t *ipiv, const std::vector<T> &bBackup, T *b, int64_t ldb,
                            const std::string &outBase, int printCols)
{
    const size_t bBytes = bBackup.size() * sizeof(T);
    // 清理上一次运行可能残留的输出文件（检视采纳）：避免本轮二进制未产出时
    // 校验脚本读到旧文件造成结果来源不可信
    for (int i = 0; i < transCount; ++i)
    {
        const std::string outPath = GetrsTransOutputPath(outBase, transList[i]);
        (void)remove(outPath.c_str());
    }
    for (int i = 0; i < transCount; ++i)
    {
        const int trans = transList[i];
        std::copy(bBackup.begin(), bBackup.end(), b);
        int32_t info = 0;
        std::cout << "[Run] trans=" << trans << std::endl;
        auto ret = apiCall(static_cast<uint64_t>(trans), n, nrhs, a, lda, ipiv, b, ldb, &info);
        CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclsolver getrs call failed at trans=%d, ERROR: %d", trans, ret);
                  return ret);
        std::cout << "[Output] X (trans=" << trans << "):" << std::endl;
        PrintPartOfMatrix<float>((uint8_t *)b, static_cast<int>(n), printCols, 8, 8);
        const std::string outPath = GetrsTransOutputPath(outBase, trans);
        CHECK_RET(WriteFile(outPath, b, bBytes), ERROR_LOG("WriteFile %s failed.", outPath.c_str());
                  return ACL_ERROR_INVALID_PARAM);
    }
    return ACL_SUCCESS;
}
#endif  // TEST_UTILS_H
