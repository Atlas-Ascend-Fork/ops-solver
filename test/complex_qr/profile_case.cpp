/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cann_ops_solver.h"

extern "C" void cgeqrf_(const int*, const int*, aclsolverFloatComplex*, const int*, aclsolverFloatComplex*,
                        aclsolverFloatComplex*, const int*, int*);

namespace
{
void Check(int status)
{
    if (status)
    {
        const char* recent = aclGetRecentErrMsg();
        if (recent)
        {
            std::cerr << "ACL diagnostic: " << recent << std::endl;
        }
        throw std::runtime_error("API status " + std::to_string(status));
    }
}

int Integer(const std::string& value)
{
    size_t consumed = 0;
    long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() || parsed < 0 || parsed > INT_MAX)
    {
        throw std::runtime_error("Invalid nonnegative int: " + value);
    }
    return static_cast<int>(parsed);
}

struct DeviceBuffers
{
    std::vector<void*> allocations;
    ~DeviceBuffers()
    {
        for (void* p : allocations)
        {
            aclrtFree(p);
        }
    }
    void* Allocate(size_t bytes)
    {
        void* ptr = nullptr;
        Check(aclrtMalloc(&ptr, std::max(bytes, size_t(32)), ACL_MEM_MALLOC_HUGE_FIRST));
        allocations.push_back(ptr);
        return ptr;
    }
};

void Upload(void* target, const std::vector<aclsolverFloatComplex>& source)
{
    if (!source.empty())
    {
        Check(aclrtMemcpy(target, source.size() * sizeof(aclsolverFloatComplex), source.data(),
                          source.size() * sizeof(aclsolverFloatComplex), ACL_MEMCPY_HOST_TO_DEVICE));
    }
}

void SaveOutput(const std::string& path, const void* source, size_t elements)
{
    std::vector<aclsolverFloatComplex> output(elements);
    const size_t bytes = elements * sizeof(aclsolverFloatComplex);
    if (bytes)
    {
        Check(aclrtMemcpy(output.data(), bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    std::ofstream file(path, std::ios::binary);
    if (!file || !file.write(reinterpret_cast<const char*>(output.data()), bytes))
    {
        throw std::runtime_error("Cannot save profiling output: " + path);
    }
}

void Matrix(std::vector<aclsolverFloatComplex>& matrix, int rows, int cols, int ld, std::mt19937& rng)
{
    matrix.assign(size_t(ld) * cols, {97, -83});
    std::uniform_real_distribution<float> uniform(-5, 5), deviation(.1f, 2);
    std::bernoulli_distribution choose(.5);
    for (int part = 0; part < 2; ++part)
    {
        const float mean = uniform(rng), sigma = deviation(rng);
        std::normal_distribution<float> normal(mean, sigma);
        for (int c = 0; c < cols; ++c)
        {
            for (int r = 0; r < rows; ++r)
            {
                float value = choose(rng) ? uniform(rng) : normal(rng);
                auto& z = matrix[size_t(c) * ld + r];
                if (part == 0)
                {
                    z.real = value;
                }
                else
                {
                    z.imag = value;
                }
            }
        }
    }
}

void FactorCpu(std::vector<aclsolverFloatComplex>& a, std::vector<aclsolverFloatComplex>& tau, int rows, int k, int ld)
{
    if (!k)
    {
        return;
    }
    int lwork = -1, info = 0;
    aclsolverFloatComplex query{};
    cgeqrf_(&rows, &k, a.data(), &ld, tau.data(), &query, &lwork, &info);
    Check(info);
    lwork = static_cast<int>(query.real);
    std::vector<aclsolverFloatComplex> work(lwork);
    cgeqrf_(&rows, &k, a.data(), &ld, tau.data(), work.data(), &lwork, &info);
    Check(info);
}

void ReadInput(const std::string& path, std::vector<aclsolverFloatComplex>& values)
{
    std::ifstream file(path, std::ios::binary);
    const auto bytes = values.size() * sizeof(aclsolverFloatComplex);
    if (!file || !file.read(reinterpret_cast<char*>(values.data()), bytes) || file.peek() != EOF)
    {
        throw std::runtime_error("Missing or incorrectly sized official input: " + path);
    }
}

class ProfileCase
{
   public:
    ProfileCase(const std::map<std::string, std::string>& values, aclsolverHandle_t h, aclrtStream streamValue)
        : args(values), handle(h), stream(streamValue)
    {
    }
    void Run()
    {
        Initialize();
        Load();
        Allocate();
        Measure();
        Output();
    }

   private:
    const std::map<std::string, std::string>& args;
    aclsolverHandle_t handle;
    aclrtStream stream;
    std::string op;
    bool factor = false, generate = false, apply = false, solve = false;
    int m = 0, n = 0, k = 0, rhs = 0, lda = 0, ldb = 0, ldx = 0;
    int side = 0, trans = 0, iterations = 0, warmup = 0, order = 0, lwork = 0;
    size_t bytes = 0;
    std::mt19937 rng;
    std::vector<aclsolverFloatComplex> a, b, tau, x;
    DeviceBuffers buffers;
    aclsolverFloatComplex *da = nullptr, *db = nullptr, *dt = nullptr, *dx = nullptr, *workspace = nullptr;
    int* info = nullptr;
    aclsolverSideMode_t s;
    aclsolverOperation_t t;
    void Initialize()
    {
        auto integer = [&](const std::string& key) { return Integer(args.at(key)); };
        op = args.at("operator");
        factor = op == "cgeqrf", generate = op == "cungqr", apply = op == "cunmqr", solve = op == "cgels";
        if (!factor && !generate && !apply && !solve)
        {
            throw std::runtime_error("Unknown operator");
        }
        m = integer("m"), n = integer("n"), k = integer("k"), rhs = integer("nrhs");
        lda = integer("lda"), ldb = integer("ldb"), ldx = integer("ldx");
        side = integer("side"), trans = integer("trans");
        iterations = integer("iterations"), warmup = integer("warmup");
        order = apply && side ? n : m;
        Validate();
        rng.seed(integer("seed"));
        tau.resize(std::max(1, std::min(order, generate || apply ? k : n)));
        x.assign(size_t(ldx) * rhs, {97, -83});
    }
    void Validate()
    {
        if (iterations < 1 || warmup > INT_MAX - iterations || side > 1 || (trans != 0 && trans != 2) ||
            lda < std::max(1, order) || ldb < std::max(1, m) || ldx < std::max(1, n) ||
            (generate && (n > m || k > n)) || (apply && k > order) || (solve && n > m))
        {
            throw std::runtime_error("Invalid profile shape");
        }
    }
    void Load()
    {
        if (args.count("input-dir"))
        {
            a.resize(size_t(lda) * (apply ? k : n));
            b.resize(size_t(ldb) * (solve ? rhs : n));
            const auto directory = args.at("input-dir");
            ReadInput(directory + "/a.bin", a);
            if (apply || solve)
            {
                ReadInput(directory + "/b.bin", b);
            }
            if (generate || apply)
            {
                ReadInput(directory + "/tau.bin", tau);
            }
        }
        else
        {
            Matrix(a, order, apply ? k : n, lda, rng);
            Matrix(b, m, solve ? rhs : n, ldb, rng);
            if (generate || apply)
            {
                FactorCpu(a, tau, order, k, lda);
            }
        }
    }
    void Allocate()
    {
        da = static_cast<aclsolverFloatComplex*>(buffers.Allocate(a.size() * sizeof(aclsolverFloatComplex)));
        db = static_cast<aclsolverFloatComplex*>(buffers.Allocate(b.size() * sizeof(aclsolverFloatComplex)));
        dt = static_cast<aclsolverFloatComplex*>(buffers.Allocate(tau.size() * sizeof(aclsolverFloatComplex)));
        dx = static_cast<aclsolverFloatComplex*>(buffers.Allocate(x.size() * sizeof(aclsolverFloatComplex)));
        info = static_cast<int*>(buffers.Allocate(sizeof(int)));
        Upload(dt, tau);
        s = static_cast<aclsolverSideMode_t>(side);
        t = static_cast<aclsolverOperation_t>(trans);
        if (factor)
        {
            Check(aclsolverCgeqrf_bufferSize(handle, m, n, da, lda, &lwork));
        }
        if (generate)
        {
            Check(aclsolverCungqr_bufferSize(handle, m, n, k, da, lda, dt, &lwork));
        }
        if (apply)
        {
            Check(aclsolverCunmqr_bufferSize(handle, s, t, m, n, k, da, lda, dt, db, ldb, &lwork));
        }
        if (solve)
        {
            Check(aclsolverCCgels_bufferSize(handle, m, n, rhs, da, lda, db, ldb, dx, ldx, nullptr, &bytes));
        }
        else
        {
            bytes = size_t(lwork) * sizeof(aclsolverFloatComplex);
        }
        workspace = static_cast<aclsolverFloatComplex*>(buffers.Allocate(bytes));
    }
    void Measure()
    {
        for (int i = 0; i < warmup + iterations; ++i)
        {
            // Restore all mutable inputs; QR and explicit Q must never time a previous output as a new input.
            Upload(da, a);
            Upload(db, b);
            Upload(dx, x);
            int niter = -1;
            if (factor)
            {
                Check(aclsolverCgeqrf(handle, m, n, da, lda, dt, workspace, lwork, info));
            }
            if (generate)
            {
                Check(aclsolverCungqr(handle, m, n, k, da, lda, dt, workspace, lwork, info));
            }
            if (apply)
            {
                Check(aclsolverCunmqr(handle, s, t, m, n, k, da, lda, dt, db, ldb, workspace, lwork, info));
            }
            if (solve)
            {
                Check(aclsolverCCgels(handle, m, n, rhs, da, lda, db, ldb, dx, ldx, workspace, bytes, &niter, info));
            }
            Check(aclrtSynchronizeStream(stream));
            int result = -999;
            Check(aclrtMemcpy(&result, sizeof(result), info, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST));
            Check(result);
            if (solve && niter < 0)
            {
                throw std::runtime_error("niter was not written");
            }
        }
    }
    void Output()
    {
        // Optional post-run evidence; transfers are outside all operator kernel timing.
        if (args.count("output-prefix"))
        {
            const auto prefix = args.at("output-prefix");
            SaveOutput(prefix + "a.bin", da, a.size());
            if (factor || generate || apply)
            {
                SaveOutput(prefix + "tau.bin", dt, tau.size());
            }
            if (apply || solve)
            {
                SaveOutput(prefix + "b.bin", db, b.size());
            }
            if (solve)
            {
                SaveOutput(prefix + "x.bin", dx, x.size());
            }
        }
        std::cout << "profile_complete operator=" << op << " m=" << m << " n=" << n << " k=" << k << " nrhs=" << rhs
                  << " lda=" << lda << " ldb=" << ldb << " ldx=" << ldx << " side=" << side << " trans=" << trans
                  << " warmup=" << warmup << " iterations=" << iterations << " workspace_bytes=" << bytes << " info=0"
                  << std::endl;
    }
};

void Run(const std::map<std::string, std::string>& args, aclsolverHandle_t handle, aclrtStream stream)
{
    ProfileCase(args, handle, stream).Run();
}

std::map<std::string, std::string> Parse(std::istream& input)
{
    std::map<std::string, std::string> result;
    std::string key, value;
    while (input >> key)
    {
        if (!(input >> value) || key.rfind("--", 0) != 0 || !result.emplace(key.substr(2), value).second)
        {
            throw std::runtime_error("Invalid/duplicate manifest argument");
        }
    }
    return result;
}
void RunManifest(const std::string& path, int device, aclsolverHandle_t handle, aclrtStream stream)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("Cannot open case manifest");
    }
    std::string line;
    size_t count = 0;
    while (std::getline(input, line))
    {
        std::istringstream tokens(line);
        auto item = Parse(tokens);
        if (item.empty() || Integer(item.at("device")) != device)
        {
            throw std::runtime_error("Invalid manifest device");
        }
        Run(item, handle, stream);
        ++count;
    }
    if (!count)
    {
        throw std::runtime_error("Empty case manifest");
    }
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc % 2 == 0)
        {
            throw std::runtime_error("Expected --key value pairs");
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2)
        {
            std::string key = argv[i];
            if (key.rfind("--", 0) != 0 || !args.emplace(key.substr(2), argv[i + 1]).second)
            {
                throw std::runtime_error("Invalid/duplicate argument");
            }
        }
        int device = Integer(args.at("device"));
        Check(aclInit(nullptr));
        Check(aclrtSetDevice(device));
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream));
        aclsolverHandle_t handle = nullptr;
        Check(aclsolverCreate(&handle));
        Check(aclsolverSetStream(handle, stream));
        if (args.count("manifest"))
        {
            RunManifest(args.at("manifest"), device, handle, stream);
        }
        else
        {
            Run(args, handle, stream);
        }
        Check(aclsolverDestroy(handle));
        Check(aclrtDestroyStream(stream));
        Check(aclrtResetDevice(device));
        Check(aclFinalize());
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
