/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_solver.h"

namespace ComplexSolverTest
{
inline void Require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

inline void Check(int status, const char* operation)
{
    Require(status == 0, std::string(operation) + ": status=" + std::to_string(status));
}

class DeviceBuffer
{
   public:
    explicit DeviceBuffer(size_t bytes) : bytes_(std::max<size_t>(bytes, 1))
    {
        Check(aclrtMalloc(&pointer_, bytes_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
    ~DeviceBuffer()
    {
        if (pointer_ != nullptr) aclrtFree(pointer_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* As() const
    {
        return static_cast<T*>(pointer_);
    }
    void Upload(const void* source, size_t bytes) const
    {
        Check(aclrtMemcpy(pointer_, bytes_, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    }
    void Download(void* destination, size_t bytes) const
    {
        Check(aclrtMemcpy(destination, bytes, pointer_, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "download");
    }

   private:
    void* pointer_ = nullptr;
    size_t bytes_;
};

class Session
{
   public:
    explicit Session(int device) : device_(device)
    {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        Check(aclsolverCreate(&handle), "aclsolverCreate");
        Check(aclsolverSetStream(handle, stream), "aclsolverSetStream");
    }
    ~Session()
    {
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_);
        aclFinalize();
    }
    void Sync() const { Check(aclrtSynchronizeStream(stream), "stream synchronize"); }
    aclsolverHandle_t handle = nullptr;
    aclrtStream stream = nullptr;

   private:
    int device_;
};

inline std::vector<aclFloatComplex> ReadMatrix(const std::string& path, int m, int n, int batch)
{
    std::vector<aclFloatComplex> rowMajor(static_cast<size_t>(m) * n * batch);
    std::ifstream input(path, std::ios::binary);
    Require(bool(input), "open " + path);
    input.read(reinterpret_cast<char*>(rowMajor.data()), rowMajor.size() * sizeof(aclFloatComplex));
    Require(input.gcount() == static_cast<std::streamsize>(rowMajor.size() * sizeof(aclFloatComplex)), "short input");
    std::vector<aclFloatComplex> columnMajor(rowMajor.size());
    for (int b = 0; b < batch; ++b)
        for (int row = 0; row < m; ++row)
            for (int col = 0; col < n; ++col)
                columnMajor[static_cast<size_t>(b) * m * n + col * m + row] =
                    rowMajor[static_cast<size_t>(b) * m * n + row * n + col];
    return columnMajor;
}

inline void WriteMatrix(const std::string& path, const std::vector<aclFloatComplex>& matrix, int m, int n, int batch)
{
    std::vector<aclFloatComplex> rowMajor(matrix.size());
    for (int b = 0; b < batch; ++b)
        for (int row = 0; row < m; ++row)
            for (int col = 0; col < n; ++col)
                rowMajor[static_cast<size_t>(b) * m * n + row * n + col] =
                    matrix[static_cast<size_t>(b) * m * n + col * m + row];
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(rowMajor.data()), rowMajor.size() * sizeof(aclFloatComplex));
    Require(bool(output), "write " + path);
}

inline void Factor(const Session& session, int m, int n, aclFloatComplex* matrix, int lda, int* pivots, int* info)
{
    int elements = -1;
    Check(aclsolverCgetrf_bufferSize(session.handle, m, n, matrix, lda, &elements), "getrf workspace query");
    DeviceBuffer workspace(static_cast<size_t>(elements) * sizeof(aclFloatComplex));
    Check(aclsolverCgetrf(session.handle, m, n, matrix, lda, elements == 0 ? nullptr : workspace.As<aclFloatComplex>(),
                          pivots, info),
          "Cgetrf");
    session.Sync();
}

inline void CheckInfo(const DeviceBuffer& info, int batch)
{
    std::vector<int> values(batch);
    info.Download(values.data(), values.size() * sizeof(int));
    Require(std::all_of(values.begin(), values.end(), [](int value) { return value == 0; }), "nonzero Device info");
}

inline void RunSingle(const Session& session, bool inverse, int m, int n, const std::string& op)
{
    auto matrix = ReadMatrix("./test/" + op + "/data/input/A_gm.bin", m, n, 1);
    size_t bytes = matrix.size() * sizeof(aclFloatComplex);
    DeviceBuffer a(bytes), pivots(std::min(m, n) * sizeof(int)), info(sizeof(int));
    a.Upload(matrix.data(), bytes);
    Factor(session, m, n, a.As<aclFloatComplex>(), m, pivots.As<int>(), info.As<int>());
    CheckInfo(info, 1);
    if (inverse)
    {
        int elements = 0;
        Check(aclsolverCgetri_bufferSize(session.handle, n, a.As<aclFloatComplex>(), m, &elements), "getri query");
        DeviceBuffer workspace(static_cast<size_t>(elements) * sizeof(aclFloatComplex));
        Check(aclsolverCgetri(session.handle, n, a.As<aclFloatComplex>(), m, pivots.As<int>(),
                              workspace.As<aclFloatComplex>(), elements, info.As<int>()),
              "Cgetri");
        session.Sync();
        CheckInfo(info, 1);
    }
    a.Download(matrix.data(), bytes);
    WriteMatrix("./test/" + op + "/data/output/A_gm.bin", matrix, m, n, 1);
    if (!inverse)
    {
        std::vector<int> values(std::min(m, n));
        pivots.Download(values.data(), values.size() * sizeof(int));
        std::ofstream output("./test/cgetrf/data/output/W_gm.bin", std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(int));
        Require(bool(output), "write pivots");
    }
}

inline void RunBatched(const Session& session, bool direct, int n, int batch, const std::string& op)
{
    auto matrix = ReadMatrix("./test/" + op + "/data/input/A_gm.bin", n, n, batch);
    size_t bytes = matrix.size() * sizeof(aclFloatComplex);
    DeviceBuffer a(bytes), c(bytes), pivots(static_cast<size_t>(n) * batch * sizeof(int)), info(batch * sizeof(int));
    DeviceBuffer aArray(batch * sizeof(void*)), cArray(batch * sizeof(void*));
    a.Upload(matrix.data(), bytes);
    std::vector<aclFloatComplex*> inputs(batch), outputs(batch);
    for (int b = 0; b < batch; ++b)
    {
        inputs[b] = a.As<aclFloatComplex>() + static_cast<size_t>(b) * n * n;
        outputs[b] = c.As<aclFloatComplex>() + static_cast<size_t>(b) * n * n;
        if (!direct) Factor(session, n, n, inputs[b], n, pivots.As<int>() + b * n, info.As<int>() + b);
    }
    aArray.Upload(inputs.data(), batch * sizeof(void*));
    cArray.Upload(outputs.data(), batch * sizeof(void*));
    if (direct)
        Check(aclsolverCmatinvBatched(session.handle, n, aArray.As<const aclFloatComplex*>(), n,
                                      cArray.As<aclFloatComplex*>(), n, info.As<int>(), batch),
              "CmatinvBatched");
    else
        Check(aclsolverCgetriBatched(session.handle, n, aArray.As<const aclFloatComplex*>(), n, pivots.As<int>(),
                                     cArray.As<aclFloatComplex*>(), n, info.As<int>(), batch),
              "CgetriBatched");
    session.Sync();
    CheckInfo(info, batch);
    c.Download(matrix.data(), bytes);
    WriteMatrix("./test/" + op + "/data/output/" + "Ainv_gm.bin", matrix, n, n, batch);
}
}  // namespace ComplexSolverTest
