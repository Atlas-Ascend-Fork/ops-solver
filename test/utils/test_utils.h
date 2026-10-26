/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
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
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
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
#endif  // TEST_UTILS_H
