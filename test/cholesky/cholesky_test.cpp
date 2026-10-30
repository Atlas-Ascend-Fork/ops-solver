/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// A standalone public-header/linking check. All five computations run on NPU.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_solver.h"

void Check(int status)
{
    if (status != 0) throw std::runtime_error("C API/runtime status " + std::to_string(status));
}
void Require(bool condition)
{
    if (!condition) throw std::runtime_error("numerical/ABI check failed");
}
int RunNativeBenchmark(const char *configPath);

void ReadInput(const std::string &path, std::vector<float> &values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Require(file.good() && file.tellg() == std::streamoff(values.size() * sizeof(float)));
    file.seekg(0);
    Require(bool(file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float))));
}
void WriteOutput(const std::string &path, const std::vector<float> &values)
{
    if (path.empty()) return;
    std::ofstream file(path, std::ios::binary);
    Require(bool(file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float))));
}

template <class T>
class Device
{
   public:
    T *data = nullptr;
    size_t count;
    explicit Device(size_t size) : count(size)
    {
        Check(aclrtMalloc(reinterpret_cast<void **>(&data), size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST));
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device()
    {
        if (data) aclrtFree(data);
    }
    void Put(const std::vector<T> &source)
    {
        Require(source.size() == count);
        Check(aclrtMemcpy(data, count * sizeof(T), source.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    std::vector<T> Get() const
    {
        std::vector<T> result(count);
        Check(aclrtMemcpy(result.data(), count * sizeof(T), data, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
        return result;
    }
};

constexpr int SAMPLE_N = 17, SAMPLE_LD = 20;
constexpr float SAMPLE_PADDING = -12345.125f;
std::vector<double> ReferenceFactor(const std::vector<float> &input)
{
    constexpr int n = SAMPLE_N, ld = SAMPLE_LD;
    std::vector<double> lower(n * n);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
        {
            double value = input[i + j * ld];
            for (int k = 0; k < j; ++k) value -= lower[i + k * n] * lower[j + k * n];
            lower[i + j * n] = i == j ? std::sqrt(value) : value / lower[j + j * n];
        }
    return lower;
}
void VerifyFactor(const Device<float> &matrix, const std::vector<double> &lower, bool upper)
{
    constexpr int n = SAMPLE_N, ld = SAMPLE_LD;
    auto values = matrix.Get();
    for (int j = 0; j < n; ++j)
    {
        for (int i = n; i < ld; ++i) Require(values[i + j * ld] == SAMPLE_PADDING);
        for (int i = 0; i < n; ++i)
            if (upper ? i <= j : i >= j)
            {
                double gold = upper ? lower[j + i * n] : lower[i + j * n];
                Require(std::abs(values[i + j * ld] - gold) < 1e-5);
            }
    }
}
void VerifySolution(const Device<float> &vector)
{
    auto values = vector.Get();
    for (int i = 0; i < SAMPLE_N; ++i) Require(std::abs(values[i] - 1. / (3. + .1 * SAMPLE_N)) < 1e-5);
    for (int i = SAMPLE_N; i < SAMPLE_LD; ++i) Require(values[i] == SAMPLE_PADDING);
}
void VerifyInverse(const std::vector<float> &inverse, bool upper)
{
    constexpr int n = SAMPLE_N, ld = SAMPLE_LD;
    for (int j = 0; j < n; ++j)
    {
        for (int i = n; i < ld; ++i) Require(inverse[i + j * ld] == SAMPLE_PADDING);
        for (int i = 0; i < n; ++i)
            if (upper ? i <= j : i >= j)
            {
                double gold = (i == j ? 1. / 3. : 0.) - .1 / (3. * (3. + .1 * n));
                Require(std::abs(inverse[i + j * ld] - gold) < 1e-5);
            }
    }
}
void RunBenchmarkList(const char *path)
{
    std::ifstream list(path);
    Require(list.good());
    std::string config;
    while (std::getline(list, config))
    {
        if (config.empty()) continue;
        RunNativeBenchmark(config.c_str());
        std::cout << "native_case_completed " << config << std::endl;
    }
    Require(list.eof());
}

void SampleInput(std::vector<float> &input, std::vector<float> &rhs, bool fileIo)
{
    constexpr int n = SAMPLE_N, ld = SAMPLE_LD;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) input[i + j * ld] = (i == j ? 3.f : 0.f) + .1f;
    std::fill_n(rhs.begin(), n, 1.f);
    if (fileIo)
    {
        ReadInput("test/cholesky/data/input/a.bin", input);
        ReadInput("test/cholesky/data/input/b.bin", rhs);
    }
}

void SaveSample(const std::string &prefix, const std::string &name, const Device<float> &device)
{
    if (!prefix.empty()) WriteOutput(prefix + name + ".bin", device.Get());
}

void RunBatched(aclsolverHandle_t handle, aclrtStream stream, bool upper, const std::vector<float> &input,
                const std::vector<float> &rhs, const std::vector<double> &lower, const std::string &prefix)
{
    constexpr int n = SAMPLE_N, ld = SAMPLE_LD;
    auto uplo = upper ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER;
    Device<float> a(ld * n), other(ld * n), b(ld), otherB(ld);
    Device<int> info(2);
    auto synchronize = [&]() { Check(aclrtSynchronizeStream(stream)); };
    auto verifyFactor = [&](const Device<float> &m) { VerifyFactor(m, lower, upper); };
    a.Put(input);
    other.Put(input);
    b.Put(rhs);
    otherB.Put(rhs);
    Device<float *> arrayA(2), arrayB(2);
    arrayA.Put({a.data, other.data});
    arrayB.Put({b.data, otherB.data});
    Check(aclsolverSpotrfBatched(handle, uplo, n, arrayA.data, ld, info.data, 2));
    synchronize();
    Require(info.Get() == std::vector<int>({0, 0}));
    verifyFactor(a);
    verifyFactor(other);
    SaveSample(prefix, "batch-factor-0", a);
    SaveSample(prefix, "batch-factor-1", other);
    auto otherFactor = other.Get();
    auto factor = a.Get();
    Check(aclsolverSpotrsBatched(handle, uplo, n, 1, arrayA.data, ld, arrayB.data, ld, info.data, 2));
    synchronize();
    Require(info.Get()[0] == 0);
    VerifySolution(b);
    VerifySolution(otherB);
    SaveSample(prefix, "batch-solve-0", b);
    SaveSample(prefix, "batch-solve-1", otherB);
    Require(a.Get() == factor && other.Get() == otherFactor);
}

void Run(aclsolverHandle_t handle, aclrtStream stream, bool upper, bool fileIo)
{
    constexpr int n = 17, ld = 20;
    constexpr float padding = -12345.125f;
    auto uplo = upper ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER;
    std::vector<float> input(ld * n, padding), rhs(ld, padding);
    SampleInput(input, rhs, fileIo);
    std::string prefix = fileIo ? std::string("test/cholesky/data/output/") + (upper ? "upper-" : "lower-") : "";
    auto lower = ReferenceFactor(input);
    Device<float> a(ld * n), b(ld);
    Device<int> info(2);
    a.Put(input);
    int factorSize = 0, inverseSize = 0;
    Check(aclsolverSpotrf_bufferSize(handle, uplo, n, a.data, ld, &factorSize));
    Check(aclsolverSpotri_bufferSize(handle, uplo, n, a.data, ld, &inverseSize));
    Device<float> work(std::max(1, std::max(factorSize, inverseSize)));
    auto synchronize = [&]() { Check(aclrtSynchronizeStream(stream)); };
    auto verifyFactor = [&](const Device<float> &m) { VerifyFactor(m, lower, upper); };
    Check(aclsolverSpotrf(handle, uplo, n, a.data, ld, work.data, factorSize, info.data));
    synchronize();
    Require(info.Get()[0] == 0);
    verifyFactor(a);
    SaveSample(prefix, "factor", a);
    auto factor = a.Get();
    b.Put(rhs);
    Check(aclsolverSpotrs(handle, uplo, n, 1, a.data, ld, b.data, ld, info.data));
    synchronize();
    Require(info.Get()[0] == 0);
    VerifySolution(b);
    Require(a.Get() == factor);
    SaveSample(prefix, "solve", b);
    Check(aclsolverSpotri(handle, uplo, n, a.data, ld, work.data, inverseSize, info.data));
    synchronize();
    Require(info.Get()[0] == 0);
    auto inverse = a.Get();
    SaveSample(prefix, "inverse", a);
    VerifyInverse(inverse, upper);
    RunBatched(handle, stream, upper, input, rhs, lower, prefix);
}

constexpr int MIXED_N = 129, MIXED_LD = MIXED_N + 3, MIXED_NRHS = 3;

void VerifyMixedScratch(const Device<float> &rhs, const Device<aclsolverFloatComplex> &complexRhs,
                        const Device<int> &info)
{
    constexpr int n = MIXED_N, ld = MIXED_LD, nrhs = MIXED_NRHS;
    Require(info.Get() == std::vector<int>({0, 0}));
    auto actual = rhs.Get();
    auto complexActual = complexRhs.Get();
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < ld; ++i)
        {
            int index = i + j * ld;
            Require(std::abs(actual[index] - (i < n ? 1 : SAMPLE_PADDING)) < 1e-5);
            Require(std::abs(complexActual[index].real - (i < n ? 1 : SAMPLE_PADDING)) < 1e-5);
            Require(std::abs(complexActual[index].imag - (i < n ? .5f : SAMPLE_PADDING)) < 1e-5);
        }
}

void RunMixedScratch(bool complexFirst, bool separateStreams, bool upper)
{
    constexpr int n = MIXED_N, ld = MIXED_LD, nrhs = MIXED_NRHS;
    auto uplo = upper ? ACLSOLVER_FILL_MODE_UPPER : ACLSOLVER_FILL_MODE_LOWER;
    std::vector<float> factor(ld * n, 0), rhs(ld * nrhs, SAMPLE_PADDING);
    std::vector<aclsolverFloatComplex> complexFactor(ld * n, {0, 0});
    std::vector<aclsolverFloatComplex> complexRhs(ld * nrhs, {SAMPLE_PADDING, SAMPLE_PADDING});
    for (int i = 0; i < n; ++i)
    {
        factor[i + i * ld] = 2;
        complexFactor[i + i * ld] = {2, 0};
        for (int j = 0; j < nrhs; ++j)
        {
            rhs[i + j * ld] = 4;
            complexRhs[i + j * ld] = {4, 2};
        }
    }
    Device<float> a(factor.size()), b(rhs.size());
    Device<aclsolverFloatComplex> ca(complexFactor.size()), cb(complexRhs.size());
    Device<int> info(2);
    a.Put(factor);
    b.Put(rhs);
    ca.Put(complexFactor);
    cb.Put(complexRhs);
    info.Put({-777, -777});
    aclrtStream first = nullptr, second = nullptr;
    aclsolverHandle_t mixed = nullptr;
    Check(aclrtCreateStream(&first));
    Check(aclrtCreateStream(&second));
    Check(aclsolverCreate(&mixed));
    auto invoke = [&](bool isComplex)
    {
        if (isComplex)
            Check(aclsolverCpotrs(mixed, uplo, n, nrhs, ca.data, ld, cb.data, ld, info.data + 1));
        else
            Check(aclsolverSpotrs(mixed, uplo, n, nrhs, a.data, ld, b.data, ld, info.data));
    };
    Check(aclsolverSetStream(mixed, first));
    invoke(complexFirst);
    Check(aclsolverSetStream(mixed, separateStreams ? second : first));
    invoke(!complexFirst);
    // Destroy must wait for both families without a preceding caller synchronization.
    Check(aclsolverDestroy(mixed));
    VerifyMixedScratch(b, cb, info);
    Check(aclrtDestroyStream(first));
    Check(aclrtDestroyStream(second));
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 3 && (std::string(argv[1]) == "--benchmark" || std::string(argv[1]) == "--benchmark-list"))
        {
            Check(aclInit(nullptr));
            Check(aclrtSetDevice(0));
            if (std::string(argv[1]) == "--benchmark")
                RunNativeBenchmark(argv[2]);
            else
                RunBenchmarkList(argv[2]);
            Check(aclrtResetDevice(0));
            Check(aclFinalize());
            return 0;
        }
        Require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--smoke"));
        Check(aclInit(nullptr));
        Check(aclrtSetDevice(0));
        aclrtStream stream = nullptr, returned = nullptr;
        aclsolverHandle_t handle = nullptr;
        Check(aclrtCreateStream(&stream));
        Check(aclsolverCreate(&handle));
        Check(aclsolverSetStream(handle, stream));
        Check(aclsolverGetStream(handle, &returned));
        Require(stream == returned);
        Run(handle, stream, false, argc == 1);
        Run(handle, stream, true, argc == 1);
        for (bool complexFirst : {false, true})
            for (bool separateStreams : {false, true})
                for (bool upper : {false, true}) RunMixedScratch(complexFirst, separateStreams, upper);
        Check(aclsolverDestroy(handle));
        Check(aclrtDestroyStream(stream));
        Check(aclrtResetDevice(0));
        Check(aclFinalize());
        std::cout << "{\"test\":\"standalone-C-API/all-five/LOWER-and-UPPER/mixed-scratch-eight\",\"pass\":true}\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
