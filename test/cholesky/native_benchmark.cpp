/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Native test binary for msprof; data generation and residual checks stay in Python.
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_solver.h"

namespace
{
void Status(int status)
{
    if (status != 0) throw std::runtime_error("native benchmark status " + std::to_string(status));
}
void Demand(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
struct Context
{
    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    Context()
    {
        Status(aclrtCreateStream(&stream));
        Status(aclsolverCreate(&handle));
        Status(aclsolverSetStream(handle, stream));
    }
    ~Context()
    {
        if (handle) aclsolverDestroy(handle);
        if (stream) aclrtDestroyStream(stream);
    }
};
struct Buffer
{
    void *data = nullptr;
    size_t bytes;
    explicit Buffer(size_t size) : bytes(std::max(size, size_t(4)))
    {
        Status(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer()
    {
        if (data) aclrtFree(data);
    }
    void Put(const void *source, size_t size)
    {
        Status(aclrtMemcpy(data, bytes, source, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void Reset(const Buffer &source, aclrtStream stream)
    {
        Status(aclrtMemcpyAsync(data, bytes, source.data, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
};
std::vector<float> Read(const std::string &path, size_t elements)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Demand(file.good() && file.tellg() == std::streamoff(elements * sizeof(float)), "invalid input binary size");
    file.seekg(0);
    std::vector<float> values(elements);
    Demand(bool(file.read(reinterpret_cast<char *>(values.data()), elements * sizeof(float))),
           "input binary read failed");
    return values;
}
void Initialize(Buffer &buffer, const std::vector<float> &pattern, int batch)
{
    std::vector<float> values(pattern.size() * size_t(batch));
    for (int q = 0; q < batch; ++q)
        std::copy(pattern.begin(), pattern.end(), values.begin() + size_t(q) * pattern.size());
    buffer.Put(values.data(), values.size() * sizeof(float));
}
void Pointers(Buffer &pointers, const Buffer &buffer, size_t stride, int batch)
{
    std::vector<uintptr_t> values(batch);
    for (int q = 0; q < batch; ++q) values[q] = reinterpret_cast<uintptr_t>(buffer.data) + size_t(q) * stride;
    pointers.Put(values.data(), values.size() * sizeof(uintptr_t));
}
void Sample(const Buffer &buffer, size_t elements, int slot, const std::string &path,
            const std::vector<float> *unchanged = nullptr)
{
    std::vector<float> values(elements);
    const char *source = static_cast<const char *>(buffer.data) + size_t(slot) * elements * sizeof(float);
    Status(aclrtMemcpy(values.data(), elements * sizeof(float), source, elements * sizeof(float),
                       ACL_MEMCPY_DEVICE_TO_HOST));
    if (unchanged)
        Demand(std::memcmp(values.data(), unchanged->data(), elements * sizeof(float)) == 0,
               "read-only factor modified");
    if (!path.empty())
    {
        std::ofstream output(path, std::ios::binary);
        Demand(bool(output.write(reinterpret_cast<char *>(values.data()), elements * sizeof(float))),
               "output write failed");
    }
}
struct Observer
{
    using Begin = void (*)();
    using End = void (*)(uint64_t *);
    Begin begin = reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT, "choleskyMemoryBegin"));
    End end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT, "choleskyMemoryEnd"));
    std::vector<std::array<uint64_t, 9>> rows;
    void Run(aclrtStream stream, const std::function<int()> &action)
    {
        std::array<uint64_t, 9> values{};
        size_t before = 0, after = 0, total = 0;
        if (begin && end)
        {
            Status(aclrtGetMemInfo(ACL_HBM_MEM, &before, &total));
            begin();
        }
        int status = action();
        Status(aclrtSynchronizeStream(stream));
        if (begin && end)
        {
            end(values.data());
            Status(aclrtGetMemInfo(ACL_HBM_MEM, &after, &total));
            values[7] = before;
            values[8] = after;
            rows.push_back(values);
        }
        Status(status);
    }
    void Write(std::ostream &out) const
    {
        const char *fields[] = {"baseline_bytes",  "peak_requested_bytes", "retained_bytes",
                                "allocated_bytes", "freed_bytes",          "malloc_count",
                                "free_count",      "hbm_free_before",      "hbm_free_after"};
        out << '[';
        for (size_t r = 0; r < rows.size(); ++r)
        {
            if (r) out << ',';
            out << '{';
            for (size_t k = 0; k < rows[r].size(); ++k)
            {
                if (k) out << ',';
                out << '"' << fields[k] << "\":" << rows[r][k];
            }
            out << ",\"hbm_live_delta_bytes\":" << int64_t(rows[r][7]) - int64_t(rows[r][8]) << ",\"phase\":\""
                << (r + 1 == rows.size() ? "destroy_handle" : "compute") << "\"}";
        }
        out << ']';
    }
};
struct BenchmarkConfig
{
    std::string op, inputA, inputB, prefix;
    int n, nrhs, batch, ld, upper, warmups, iterations;
};
BenchmarkConfig ReadConfig(const char *configPath)
{
    BenchmarkConfig result;
    std::ifstream config(configPath);
    auto &[op, inputA, inputB, prefix, n, nrhs, batch, ld, upper, warmups, iterations] = result;
    Demand(
        bool(config >> op >> n >> nrhs >> batch >> ld >> upper >> warmups >> iterations >> inputA >> inputB >> prefix),
        "invalid native benchmark configuration");
    Demand(n > 0 && n <= 4096 && nrhs > 0 && nrhs <= 128 && batch > 0 && ld >= n && (upper == 0 || upper == 1) &&
               warmups >= 0 && iterations > 0,
           "invalid benchmark dimensions");
    bool factor = op == "spotrf" || op == "spotrfbatched";
    bool solve = op == "spotrs" || op == "spotrsbatched";
    bool batched = op == "spotrfbatched" || op == "spotrsbatched";
    Demand(factor || solve || op == "spotri", "invalid benchmark operator");
    Demand(batched || batch == 1, "single interface requires batch=1");
    Demand(op != "spotrsbatched" || nrhs == 1, "batch solve requires nrhs=1");
    return result;
}
void WriteMetadata(const std::string &prefix, int lwork, size_t free, int64_t vectorCores, const Observer &observer)
{
    Dl_info library{};
    void *entry = dlsym(RTLD_DEFAULT, "aclsolverSpotrf");
    Demand(entry && dladdr(entry, &library) != 0, "cannot identify loaded solver library");
    std::ofstream metadata(prefix + ".json");
    metadata << "{\"library_path\":\"" << library.dli_fname << "\",\"info_zero\":true,\"workspace_floats\":" << lwork
             << ",\"launch_free_bytes\":" << free << ",\"vector_cores\":" << vectorCores << ",\"observations\":";
    observer.Write(metadata);
    metadata << '}';
    Demand(metadata.good(), "metadata write failed");
}
struct Invocation
{
    aclsolverHandle_t handle;
    aclsolverFillMode_t uplo;
    int n, nrhs, ld, batch, lwork;
    const Buffer &a, &b, &workspace, &arrayA, &arrayB, &info;
};
int Invoke(const std::string &op, const Invocation &call)
{
    auto af = static_cast<float *>(call.a.data), bf = static_cast<float *>(call.b.data);
    auto w = static_cast<float *>(call.workspace.data);
    auto ai = static_cast<float **>(call.arrayA.data), bi = static_cast<float **>(call.arrayB.data);
    auto status = static_cast<int *>(call.info.data);
    if (op == "spotrf") return aclsolverSpotrf(call.handle, call.uplo, call.n, af, call.ld, w, call.lwork, status);
    if (op == "spotri") return aclsolverSpotri(call.handle, call.uplo, call.n, af, call.ld, w, call.lwork, status);
    if (op == "spotrs")
        return aclsolverSpotrs(call.handle, call.uplo, call.n, call.nrhs, af, call.ld, bf, call.ld, status);
    if (op == "spotrfbatched")
        return aclsolverSpotrfBatched(call.handle, call.uplo, call.n, ai, call.ld, status, call.batch);
    return aclsolverSpotrsBatched(call.handle, call.uplo, call.n, call.nrhs, ai, call.ld, bi, call.ld, status,
                                  call.batch);
}
void VerifyBenchmarkOutput(const Invocation &call, bool solve, const std::string &prefix,
                           const std::vector<float> &patternA)
{
    size_t aElements = size_t(call.n) * call.ld, bElements = size_t(call.nrhs) * call.ld;
    std::vector<int> statuses(call.info.bytes / sizeof(int), -777);
    Status(aclrtMemcpy(statuses.data(), call.info.bytes, call.info.data, call.info.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    Demand(std::all_of(statuses.begin(), statuses.end(), [](int x) { return x == 0; }), "nonzero benchmark info");
    Sample(solve ? call.b : call.a, solve ? bElements : aElements, 0, prefix + "-first.bin");
    Sample(solve ? call.b : call.a, solve ? bElements : aElements, call.batch - 1, prefix + "-last.bin");
    if (solve)
    {
        Sample(call.a, aElements, 0, "", &patternA);
        Sample(call.a, aElements, call.batch - 1, "", &patternA);
    }
}
}  // namespace

int RunNativeBenchmark(const char *configPath)
{
    auto [op, inputA, inputB, prefix, n, nrhs, batch, ld, upper, warmups, iterations] = ReadConfig(configPath);
    bool factor = op == "spotrf" || op == "spotrfbatched";
    bool solve = op == "spotrs" || op == "spotrsbatched";
    bool batched = op == "spotrfbatched" || op == "spotrsbatched";
    Context context;
    auto uplo = upper ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER;
    size_t aElements = size_t(n) * ld, bElements = solve ? size_t(nrhs) * ld : 0;
    auto patternA = Read(inputA, aElements);
    Buffer a0(aElements * batch * sizeof(float)), a(a0.bytes),
        info(size_t(op == "spotrfbatched" ? batch : 1) * sizeof(int));
    Initialize(a0, patternA, batch);
    Buffer b0(bElements * batch * sizeof(float)), b(b0.bytes);
    if (solve) Initialize(b0, Read(inputB, bElements), batch);
    Buffer arrayA(size_t(batch) * sizeof(uintptr_t)), arrayB(arrayA.bytes);
    Pointers(arrayA, a, aElements * sizeof(float), batch);
    if (solve) Pointers(arrayB, b, bElements * sizeof(float), batch);
    int lwork = 0;
    if (!batched && !solve)
    {
        if (factor)
            Status(aclsolverSpotrf_bufferSize(context.handle, uplo, n, static_cast<float *>(a.data), ld, &lwork));
        else
            Status(aclsolverSpotri_bufferSize(context.handle, uplo, n, static_cast<float *>(a.data), ld, &lwork));
    }
    Buffer workspace(size_t(lwork) * sizeof(float));
    Observer observer;
    size_t free = 0, total = 0;
    int64_t vectorCores = 0;
    Status(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
    Status(aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores));
    Invocation call{context.handle, uplo, n, nrhs, ld, batch, lwork, a, b, workspace, arrayA, arrayB, info};
    auto invoke = [&]() { return Invoke(op, call); };
    for (int iteration = 0; iteration < warmups + iterations; ++iteration)
    {
        a.Reset(a0, context.stream);
        if (solve) b.Reset(b0, context.stream);
        observer.Run(context.stream, invoke);
    }
    VerifyBenchmarkOutput(call, solve, prefix, patternA);
    observer.Run(context.stream,
                 [&]() -> int
                 {
                     int status = aclsolverDestroy(context.handle);
                     context.handle = nullptr;
                     return status;
                 });
    WriteMetadata(prefix, lwork, free, vectorCores, observer);
    return 0;
}
