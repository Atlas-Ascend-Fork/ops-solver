/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fixture_runner.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_solver.h"

namespace CholeskyTest
{
namespace
{
constexpr size_t IO_CHUNK_BYTES = 8 * 1024 * 1024;
constexpr int INFO_SENTINEL = -999;
constexpr int DETERMINISM_RUNS = 2;

void Check(int status, const char* action)
{
    if (status != 0)
    {
        throw std::runtime_error(std::string(action) + ": status=" + std::to_string(status));
    }
}

class Session
{
   public:
    explicit Session(int device) : device_(device)
    {
        try
        {
            Check(aclInit(nullptr), "aclInit");
            initialized_ = true;
            Check(aclrtSetDevice(device_), "aclrtSetDevice");
            deviceSet_ = true;
            Check(aclrtCreateStream(&stream), "aclrtCreateStream");
            Check(aclsolverCreate(&handle), "aclsolverCreate");
            Check(aclsolverSetStream(handle, stream), "aclsolverSetStream");
        }
        catch (...)
        {
            Cleanup();
            throw;
        }
    }
    ~Session() { Cleanup(); }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;

   private:
    void Cleanup() noexcept
    {
        if (stream)
        {
            aclrtSynchronizeStream(stream);
        }
        if (handle)
        {
            if (aclsolverDestroy(handle) != ACLSOLVER_STATUS_SUCCESS)
            {
                // Destroy preserves ownership on error; a retry also releases an emptied handle.
                aclsolverDestroy(handle);
            }
        }
        if (stream)
        {
            aclrtDestroyStream(stream);
        }
        if (deviceSet_)
        {
            aclrtResetDevice(device_);
        }
        if (initialized_)
        {
            aclFinalize();
        }
    }
    int device_;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class DeviceBuffer
{
   public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        if (bytes_)
        {
            Check(aclrtMalloc(&pointer_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        }
    }
    ~DeviceBuffer()
    {
        if (pointer_)
        {
            aclrtFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* As() const
    {
        return static_cast<T*>(pointer_);
    }
    void Upload(const void* data) const
    {
        if (bytes_)
        {
            Check(aclrtMemcpy(pointer_, bytes_, data, bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
        }
    }
    void Reset(const void* deviceSource) const
    {
        if (bytes_)
        {
            Check(aclrtMemcpy(pointer_, bytes_, deviceSource, bytes_, ACL_MEMCPY_DEVICE_TO_DEVICE), "device reset");
        }
    }
    void Download(void* data) const
    {
        if (bytes_)
        {
            Check(aclrtMemcpy(data, bytes_, pointer_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "download");
        }
    }
    size_t Bytes() const { return bytes_; }
    void CopyTo(void* destination) const
    {
        if (bytes_)
        {
            Check(aclrtMemcpy(destination, bytes_, pointer_, bytes_, ACL_MEMCPY_DEVICE_TO_DEVICE), "snapshot");
        }
    }
    void DownloadRange(void* destination, size_t offset, size_t count) const
    {
        Check(aclrtMemcpy(destination, count, As<char>() + offset, count, ACL_MEMCPY_DEVICE_TO_HOST), "chunk readback");
    }
    void Load(const std::string& path) const
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("missing input: " + path);
        }
        std::vector<char> chunk(std::min<size_t>(bytes_, IO_CHUNK_BYTES));
        for (size_t offset = 0; offset < bytes_; offset += chunk.size())
        {
            size_t count = std::min(chunk.size(), bytes_ - offset);
            if (!file.read(chunk.data(), count))
            {
                throw std::runtime_error("short input: " + path);
            }
            Check(aclrtMemcpy(As<char>() + offset, count, chunk.data(), count, ACL_MEMCPY_HOST_TO_DEVICE),
                  "chunk upload");
        }
        if (file.peek() != std::ifstream::traits_type::eof())
        {
            throw std::runtime_error("excess input: " + path);
        }
    }

   private:
    size_t bytes_;
    void* pointer_ = nullptr;
};

class MatrixBatch
{
   public:
    MatrixBatch(int count, size_t elements) : elements_(elements)
    {
        for (int j = 0; j < count; ++j)
        {
            buffers_.push_back(std::make_unique<DeviceBuffer>(elements_ * sizeof(aclsolverFloatComplex)));
            pointers.push_back(elements_ ? buffers_[j]->As<aclsolverFloatComplex>() : nullptr);
        }
    }
    void Reset(const DeviceBuffer& input) const
    {
        for (size_t j = 0; j < buffers_.size(); ++j)
        {
            buffers_[j]->Reset(elements_ ? input.As<aclsolverFloatComplex>() + j * elements_ : nullptr);
        }
    }
    void Snapshot(const DeviceBuffer& snapshot) const
    {
        size_t offset = 0;
        for (const auto& buffer : buffers_)
        {
            buffer->CopyTo(buffer->Bytes() ? snapshot.As<char>() + offset : nullptr);
            offset += buffer->Bytes();
        }
    }
    void SaveAndCompare(const std::string& path, const DeviceBuffer& first) const
    {
        std::ofstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("cannot write output: " + path);
        }
        size_t base = 0;
        std::vector<char> actual(IO_CHUNK_BYTES), reference(actual.size());
        for (const auto& buffer : buffers_)
        {
            for (size_t offset = 0; offset < buffer->Bytes(); offset += actual.size())
            {
                size_t count = std::min(actual.size(), buffer->Bytes() - offset);
                buffer->DownloadRange(actual.data(), offset, count);
                first.DownloadRange(reference.data(), base + offset, count);
                if (std::memcmp(actual.data(), reference.data(), count))
                {
                    throw std::runtime_error("bitwise determinism failed: " + path);
                }
                if (!file.write(actual.data(), count))
                {
                    throw std::runtime_error("output write failed: " + path);
                }
            }
            base += buffer->Bytes();
        }
    }
    std::vector<aclsolverFloatComplex*> pointers;

   private:
    size_t elements_;
    std::vector<std::unique_ptr<DeviceBuffer>> buffers_;
};

template <typename T>
void Write(const std::string& path, const std::vector<T>& data)
{
    std::ofstream file(path, std::ios::binary);
    if (!file || !file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T)))
    {
        throw std::runtime_error("cannot write output: " + path);
    }
}

struct Fixture
{
    int index, n, nrhs, batch, upper, lda, ldb;
    int expectedStatus = ACLSOLVER_STATUS_SUCCESS;
    std::vector<int> expectedInfo;
};

int InvokeFixture(Session& session, Operation operation, const Fixture& f, aclsolverFillMode_t uplo,
                  const std::vector<aclsolverFloatComplex*>& pointersA,
                  const std::vector<aclsolverFloatComplex*>& pointersB, DeviceBuffer& arrayA, DeviceBuffer& arrayB,
                  DeviceBuffer& workspace, DeviceBuffer& deviceInfo, int lwork)
{
    int status = 0;
    switch (operation)
    {
        case Operation::Potrf:
            status = aclsolverCpotrf(session.handle, uplo, f.n, pointersA[0], f.lda,
                                     workspace.As<aclsolverFloatComplex>(), lwork, deviceInfo.As<int>());
            break;
        case Operation::Potri:
            status = aclsolverCpotri(session.handle, uplo, f.n, pointersA[0], f.lda,
                                     workspace.As<aclsolverFloatComplex>(), lwork, deviceInfo.As<int>());
            break;
        case Operation::Potrs:
            status = aclsolverCpotrs(session.handle, uplo, f.n, f.nrhs, pointersA[0], f.lda, pointersB[0], f.ldb,
                                     deviceInfo.As<int>());
            break;
        case Operation::PotrfBatched:
            status = aclsolverCpotrfBatched(session.handle, uplo, f.n, arrayA.As<aclsolverFloatComplex*>(), f.lda,
                                            deviceInfo.As<int>(), f.batch);
            break;
        case Operation::PotrsBatched:
            status =
                aclsolverCpotrsBatched(session.handle, uplo, f.n, f.nrhs, arrayA.As<aclsolverFloatComplex*>(), f.lda,
                                       arrayB.As<aclsolverFloatComplex*>(), f.ldb, deviceInfo.As<int>(), f.batch);
            break;
    }
    return status;
}

void RepeatFixture(Session& session, Operation operation, const Fixture& f, aclsolverFillMode_t uplo,
                   const std::string& stem, MatrixBatch& matricesA, MatrixBatch& matricesB, DeviceBuffer& resetA,
                   DeviceBuffer& resetB, DeviceBuffer& arrayA, DeviceBuffer& arrayB, DeviceBuffer& workspace,
                   DeviceBuffer& deviceInfo, int lwork, DeviceBuffer& firstA, DeviceBuffer& firstB,
                   std::vector<int>& info, std::vector<int>& firstInfo, int repeat)
{
    const auto& pointersA = matricesA.pointers;
    const auto& pointersB = matricesB.pointers;

    matricesA.Reset(resetA);
    matricesB.Reset(resetB);
    std::fill(info.begin(), info.end(), INFO_SENTINEL);
    deviceInfo.Upload(info.data());
    int status =
        InvokeFixture(session, operation, f, uplo, pointersA, pointersB, arrayA, arrayB, workspace, deviceInfo, lwork);
    Check(aclrtSynchronizeStream(session.stream), "synchronize");
    if (status != f.expectedStatus)
    {
        throw std::runtime_error("unexpected Cholesky status: " + stem + ": actual=" + std::to_string(status) +
                                 ", expected=" + std::to_string(f.expectedStatus));
    }
    deviceInfo.Download(info.data());
    if (info != f.expectedInfo)
    {
        throw std::runtime_error("unexpected info: " + stem);
    }
    if (repeat == 0)
    {
        matricesA.Snapshot(firstA);
        matricesB.Snapshot(firstB);
        firstInfo = info;
    }
    else if (firstInfo != info)
    {
        throw std::runtime_error("bitwise determinism failed: " + stem);
    }
}

size_t StoredElements(const Fixture& f, bool rhs)
{
    const bool invalid = f.expectedStatus != ACLSOLVER_STATUS_SUCCESS;
    const int order = invalid ? std::max(1, f.n) : f.n;
    const int leading = rhs ? f.ldb : f.lda;
    const int columns = rhs ? f.nrhs : order;
    const int storageLeading = invalid ? std::max(order, leading) : leading;
    const int storageColumns = invalid ? std::max(1, columns) : columns;
    return static_cast<size_t>(storageLeading) * storageColumns;
}

std::vector<int> ReadExpectedInfo(const std::string& root, const std::string& stem, size_t count)
{
    std::vector<int> values(count);
    std::ifstream expected(root + "/input/" + stem + "-info.bin", std::ios::binary);
    if (!expected || !expected.read(reinterpret_cast<char*>(values.data()), count * sizeof(int)) ||
        expected.peek() != std::ifstream::traits_type::eof())
    {
        throw std::runtime_error("invalid expected info: " + stem);
    }
    return values;
}

void RunCase(Session& session, Operation operation, const std::string& root, Fixture& f)
{
    const bool batched = operation == Operation::PotrfBatched || operation == Operation::PotrsBatched;
    const bool solve = operation == Operation::Potrs || operation == Operation::PotrsBatched;
    const auto uplo = f.upper ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER;
    const std::string stem = std::to_string(f.index);
    const bool invalid = f.expectedStatus != ACLSOLVER_STATUS_SUCCESS;
    const int storageBatch = invalid ? std::max(1, f.batch) : f.batch;
    const size_t aElements = StoredElements(f, false);
    const size_t bElements = solve ? StoredElements(f, true) : 0;
    MatrixBatch matricesA(storageBatch, aElements), matricesB(storageBatch, bElements);
    const auto& pointersA = matricesA.pointers;
    const auto& pointersB = matricesB.pointers;
    DeviceBuffer resetA(aElements * storageBatch * sizeof(aclsolverFloatComplex)),
        resetB(bElements * storageBatch * sizeof(aclsolverFloatComplex));
    resetA.Load(root + "/input/" + stem + "-A.bin");
    resetB.Load(root + "/input/" + stem + "-B.bin");
    DeviceBuffer arrayA(batched ? storageBatch * sizeof(aclsolverFloatComplex*) : 0);
    DeviceBuffer arrayB(batched && solve ? storageBatch * sizeof(aclsolverFloatComplex*) : 0);
    arrayA.Upload(pointersA.data());
    arrayB.Upload(pointersB.data());
    std::vector<int> info(operation == Operation::PotrfBatched ? storageBatch : 1, INFO_SENTINEL);
    f.expectedInfo = ReadExpectedInfo(root, stem, info.size());
    DeviceBuffer deviceInfo(info.size() * sizeof(int));
    int lwork = 0;
    if (!invalid && operation == Operation::Potrf)
    {
        Check(aclsolverCpotrf_bufferSize(session.handle, uplo, f.n, pointersA[0], f.lda, &lwork), "potrf bufferSize");
    }
    else if (!invalid && operation == Operation::Potri)
    {
        Check(aclsolverCpotri_bufferSize(session.handle, uplo, f.n, pointersA[0], f.lda, &lwork), "potri bufferSize");
    }
    DeviceBuffer workspace(static_cast<size_t>(lwork) * sizeof(aclsolverFloatComplex));
    DeviceBuffer firstA(resetA.Bytes()), firstB(resetB.Bytes());
    std::vector<int> firstInfo;
    for (int repeat = 0; repeat < DETERMINISM_RUNS; ++repeat)
    {
        RepeatFixture(session, operation, f, uplo, stem, matricesA, matricesB, resetA, resetB, arrayA, arrayB,
                      workspace, deviceInfo, lwork, firstA, firstB, info, firstInfo, repeat);
    }
    matricesA.SaveAndCompare(root + "/output/" + stem + "-A.bin", firstA);
    matricesB.SaveAndCompare(root + "/output/" + stem + "-B.bin", firstB);
    Write(root + "/output/" + stem + "-info.bin", info);
    std::cout << "case " << f.index << ": NPU calls and bitwise determinism PASS" << std::endl;
}

Fixture ParseFixture(const std::string& line)
{
    Fixture fixture{};
    std::istringstream fields(line);
    if (!(fields >> fixture.index >> fixture.n >> fixture.nrhs >> fixture.batch >> fixture.upper >> fixture.lda >>
          fixture.ldb))
    {
        throw std::runtime_error("invalid fixture manifest row");
    }
    fields >> std::ws;
    if (!fields.eof())
    {
        if (!(fields >> fixture.expectedStatus))
        {
            throw std::runtime_error("invalid expected fixture status");
        }
        fields >> std::ws;
        if (!fields.eof())
        {
            throw std::runtime_error("unexpected fixture manifest column");
        }
    }
    return fixture;
}
}  // namespace

int Run(Operation operation, const char* name, int argc, char** argv)
{
    try
    {
        std::string root = "./test/" + std::string(name) + "/data";
        if (argc > 2)
        {
            throw std::runtime_error("expected at most one device ordinal");
        }
        int device = argc == 2 ? std::stoi(argv[1]) : 0;
        std::ifstream manifest(root + "/input/manifest.tsv");
        if (!manifest)
        {
            throw std::runtime_error("missing manifest; run data/gen_data.py first");
        }
        Session session(device);
        std::string line;
        int count = 0;
        while (std::getline(manifest, line))
        {
            Fixture fixture = ParseFixture(line);
            RunCase(session, operation, root, fixture);
            ++count;
        }
        if (!manifest.eof() || count == 0)
        {
            throw std::runtime_error("invalid or empty manifest");
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << name << ": " << error.what() << '\n';
        return 1;
    }
}
}  // namespace CholeskyTest
