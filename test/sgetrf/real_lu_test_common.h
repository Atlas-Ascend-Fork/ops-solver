/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REAL_LU_TEST_COMMON_H
#define REAL_LU_TEST_COMMON_H
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_solver.h"
#include "solver_adapter.h"

namespace
{
namespace real_lu_test
{
void Require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

struct Runtime
{
    bool initialized = false, selected = false;
    int device = 0;
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    void Init(int id)
    {
        device = id;
        Require(aclInit(nullptr) == ACL_SUCCESS, "aclInit failed");
        initialized = true;
        Require(aclrtSetDevice(device) == ACL_SUCCESS, "aclrtSetDevice failed");
        selected = true;
        Require(aclrtCreateStream(&stream) == ACL_SUCCESS, "create stream failed");
        Require(aclsolverCreate(&handle) == ACLSOLVER_STATUS_SUCCESS, "create handle failed");
        Require(aclsolverSetStream(handle, stream) == ACLSOLVER_STATUS_SUCCESS, "set stream failed");
    }
    bool Close()
    {
        bool closed = true;
        if (stream != nullptr)
        {
            closed = aclrtSynchronizeStream(stream) == ACL_SUCCESS && closed;
        }
        if (handle != nullptr)
        {
            closed = aclsolverDestroy(handle) == ACLSOLVER_STATUS_SUCCESS && closed;
        }
        if (stream != nullptr)
        {
            closed = aclrtDestroyStream(stream) == ACL_SUCCESS && closed;
        }
        if (selected)
        {
            closed = aclrtResetDevice(device) == ACL_SUCCESS && closed;
        }
        if (initialized)
        {
            closed = aclFinalize() == ACL_SUCCESS && closed;
        }
        handle = nullptr;
        stream = nullptr;
        selected = false;
        initialized = false;
        return closed;
    }
    ~Runtime() { Close(); }
};

struct DeviceBuffer
{
    void* value = nullptr;
    aclrtStream stream = nullptr;
    DeviceBuffer() = default;
    bool TryAllocate(size_t bytes, aclrtStream current)
    {
        stream = current;
        return aclrtMalloc(&value, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    DeviceBuffer(size_t bytes, aclrtStream current) : stream(current)
    {
        Require(TryAllocate(bytes, current), "Device allocation failed");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (value != nullptr && aclrtSynchronizeStream(stream) == ACL_SUCCESS)
        {
            aclrtFree(value);
        }
    }
    template <typename T>
    T* As()
    {
        return static_cast<T*>(value);
    }
    void Close()
    {
        Require(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "synchronize before free failed");
        Require(aclrtFree(value) == ACL_SUCCESS, "Device free failed");
        value = nullptr;
    }
};

std::vector<float> ReadMatrix(const std::string& path, int m, int n)
{
    const size_t count = static_cast<size_t>(m) * n;
    Require(count <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float),
            "matrix byte count exceeds stream limit");
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Require(file.is_open(), "cannot open input matrix");
    Require(file.tellg() == bytes, "input matrix byte count does not match dimensions");
    std::vector<float> input(count);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(input.data()), bytes);
    Require(static_cast<bool>(file), "input matrix read failed");
    return input;
}

template <typename T>
void WriteBinary(const std::string& path, const std::vector<T>& values)
{
    Require(values.size() <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T),
            "output byte count overflow");
    const size_t bytes = values.size() * sizeof(T);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    Require(file.is_open(), "cannot open output file");
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(bytes));
    file.close();
    Require(static_cast<bool>(file), "output file write failed");
}

int ParseInt(const char* text)
{
    size_t consumed = 0;
    const std::string value(text);
    const int result = std::stoi(value, &consumed);
    Require(consumed == value.size(), "invalid integer argument");
    return result;
}

struct AdapterState
{
    void* value = nullptr;
    ~AdapterState()
    {
        if (value != nullptr)
        {
            solver_adapter_destroy(value);
        }
    }
    void Close()
    {
        Require(solver_adapter_destroy(value) == 0, "adapter destroy failed");
        value = nullptr;
    }
};

template <typename T>
bool SameBits(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

inline std::vector<float> ColumnMajor(const std::vector<float>& input, int m, int n)
{
    std::vector<float> output(input.size());
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            output[static_cast<size_t>(col) * m + row] = input[static_cast<size_t>(row) * n + col];
        }
    }
    return output;
}

inline void RowMajor(const std::vector<float>& input, std::vector<float>& output, int m, int n, int ld)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            output[static_cast<size_t>(row) * n + col] = input[static_cast<size_t>(col) * ld + row];
        }
    }
}

inline double LuProduct(const std::vector<float>& factors, size_t offset, int m, int n, int row, int col)
{
    double actual = 0;
    for (int k = 0; k < std::min(m, n); ++k)
    {
        const double lower = row == k ? 1.0 : row > k ? factors[offset + static_cast<size_t>(row) * n + k] : 0;
        const double upper = k <= col ? factors[offset + static_cast<size_t>(k) * n + col] : 0;
        actual += lower * upper;
    }
    return actual;
}

inline void CheckDescriptor(const char* name)
{
    const auto* desc = solver_adapter_descriptor();
    Require(desc != nullptr && std::strcmp(desc->op, name) == 0 && std::strcmp(desc->dut_abi, "device") == 0 &&
                std::strcmp(desc->dut_layout, "column_major") == 0,
            "unexpected adapter descriptor");
}

}  // namespace real_lu_test
}  // namespace

namespace
{
namespace real_lu_test
{
inline int FileWorkspace(Runtime& runtime, int m, int n, DeviceBuffer& matrix, bool inverse)
{
    int workElements = 0;
    Require(aclsolverSgetrf_bufferSize(runtime.handle, m, n, matrix.As<float>(), m, &workElements) ==
                ACLSOLVER_STATUS_SUCCESS,
            "factor workspace query failed");
    if (inverse)
    {
        int inverseWork = 0;
        Require(aclsolverSgetri_bufferSize(runtime.handle, n, matrix.As<float>(), m, &inverseWork) ==
                    ACLSOLVER_STATUS_SUCCESS,
                "inverse workspace query failed");
        workElements = std::max(workElements, inverseWork);
    }
    Require(workElements >= 0, "negative file workspace size");
    return workElements;
}

inline int FileInfo(Runtime& runtime, DeviceBuffer& info)
{
    Require(aclrtSynchronizeStream(runtime.stream) == ACL_SUCCESS, "file operation synchronization failed");
    int actualInfo = -99;
    Require(aclrtMemcpy(&actualInfo, sizeof(actualInfo), info.value, sizeof(actualInfo), ACL_MEMCPY_DEVICE_TO_HOST) ==
                ACL_SUCCESS,
            "file INFO readback failed");
    Require(actualInfo == 0, "file operation returned nonzero INFO");
    return actualInfo;
}

inline int FileFactor(Runtime& runtime, int m, int n, DeviceBuffer& matrix, DeviceBuffer& workspace,
                      DeviceBuffer& pivots, DeviceBuffer& info)
{
    Require(aclsolverSgetrf(runtime.handle, m, n, matrix.As<float>(), m, workspace.As<float>(), pivots.As<int>(),
                            info.As<int>()) == ACLSOLVER_STATUS_SUCCESS,
            "file-mode Sgetrf failed");
    return FileInfo(runtime, info);
}

inline int FileInverse(Runtime& runtime, int n, int workElements, DeviceBuffer& matrix, DeviceBuffer& workspace,
                       DeviceBuffer& pivots, DeviceBuffer& info)
{
    // The input file contains the original matrix. LU is prepared on Device first.
    Require(aclrtMemset(info.value, sizeof(int), 0xA5, sizeof(int)) == ACL_SUCCESS, "inverse INFO reset failed");
    Require(aclsolverSgetri(runtime.handle, n, matrix.As<float>(), n, pivots.As<int>(), workspace.As<float>(),
                            workElements, info.As<int>()) == ACLSOLVER_STATUS_SUCCESS,
            "file-mode Sgetri failed");
    return FileInfo(runtime, info);
}

inline void SaveFilePivots(const std::string& directory, int m, int n, DeviceBuffer& pivots)
{
    std::vector<int32_t> hostPivots(static_cast<size_t>(std::min(m, n)));
    const size_t bytes = hostPivots.size() * sizeof(int32_t);
    Require(aclrtMemcpy(hostPivots.data(), bytes, pivots.value, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
            "pivot readback failed");
    WriteBinary(directory + "output/W_gm.bin", hostPivots);
}

inline void RunFileCase(Runtime& runtime, int m, int n, bool inverse)
{
    const std::string name = inverse ? "sgetri" : "sgetrf";
    const std::string directory = "./test/" + name + "/data/";
    auto input = ReadMatrix(directory + "input/A_gm.bin", m, n);
    auto columnMajor = ColumnMajor(input, m, n);
    const size_t bytes = input.size() * sizeof(float);
    DeviceBuffer matrix(bytes, runtime.stream), pivots(std::min(m, n) * sizeof(int), runtime.stream),
        info(sizeof(int), runtime.stream);
    Require(aclrtMemcpy(matrix.value, bytes, columnMajor.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
            "input upload failed");
    Require(aclrtMemset(info.value, sizeof(int), 0xA5, sizeof(int)) == ACL_SUCCESS, "INFO initialization failed");
    const int workElements = FileWorkspace(runtime, m, n, matrix, inverse);
    DeviceBuffer workspace(static_cast<size_t>(workElements) * sizeof(float), runtime.stream);
    int actualInfo = FileFactor(runtime, m, n, matrix, workspace, pivots, info);
    if (inverse)
    {
        actualInfo = FileInverse(runtime, n, workElements, matrix, workspace, pivots, info);
    }
    Require(aclrtMemcpy(columnMajor.data(), bytes, matrix.value, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
            "result readback failed");
    RowMajor(columnMajor, input, m, n, m);
    std::filesystem::create_directories(directory + "output");
    WriteBinary(directory + "output/A_gm.bin", input);
    WriteBinary(directory + "output/INFO_gm.bin", std::vector<int>{actualInfo});
    if (!inverse)
    {
        SaveFilePivots(directory, m, n, pivots);
    }
    workspace.Close();
    info.Close();
    pivots.Close();
    matrix.Close();
    std::cout << name << " file sample PASS: actual Device result and INFO=" << actualInfo << '\n';
}

}  // namespace real_lu_test
}  // namespace
#endif
