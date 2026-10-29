/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REAL_LU_TEST_IMPL_H
#define REAL_LU_TEST_IMPL_H
#include "real_lu_test_common.h"

namespace
{
namespace real_lu_test
{
struct TestMode
{
    bool factor, batched, inputLu;
};

struct TestData
{
    int m, n, batch;
    bool pivot, singular;
    TestMode mode;
    SolverAdapterCaseV2 c{};
    std::vector<float> input, original, output;
    std::vector<int32_t> pivots, outPivots, info;

    TestData(TestMode current, int rows, int columns, int batches, bool usePivot, bool isSingular)
        : m(rows), n(columns), batch(batches), pivot(usePivot), singular(isSingular), mode(current)
    {
        c.base.version = SOLVER_ADAPTER_VERSION_V2;
        c.base.struct_size = sizeof(c);
        c.base.n = n;
        c.base.batch = batch;
        c.base.lda = std::max(1, m) + 8;
        c.m = m;
        c.ldc = std::max(1, n) + 8;
        c.base.a_bytes = c.base.out_bytes = static_cast<size_t>(m) * n * batch * sizeof(float);
        c.base.info_count = mode.batched ? batch : 1;
        const size_t pivCount = static_cast<size_t>(std::min(m, n)) * batch;
        c.in_ipiv_count = mode.inputLu && pivot ? pivCount : 0;
        c.out_ipiv_count = mode.factor && pivot ? pivCount : 0;
        input.resize(c.base.a_bytes / sizeof(float));
        original.resize(input.size());
        output.resize(input.size());
        pivots.resize(c.in_ipiv_count);
        outPivots.resize(c.out_ipiv_count);
        info.assign(c.base.info_count, -99);
    }
    size_t Offset(int b) const { return static_cast<size_t>(b) * m * n; }
};

inline void FillInput(TestData& t, int b)
{
    const size_t offset = t.Offset(b);
    const bool bad = t.singular && b == 0;
    for (int row = 0; row < t.m; ++row)
    {
        for (int col = 0; col < t.n; ++col)
        {
            t.input[offset + static_cast<size_t>(row) * t.n + col] =
                row == col                    ? (bad && row == 1 ? 0.0f : static_cast<float>(t.n + row + b + 2))
                : bad                         ? 0.0f
                : t.mode.inputLu && row > col ? 0.125f
                                              : 0.25f;
        }
    }
}

inline void SwapRows(std::vector<float>& matrix, size_t offset, int n, int row, int other)
{
    for (int col = 0; col < n; ++col)
    {
        std::swap(matrix[offset + static_cast<size_t>(row) * n + col],
                  matrix[offset + static_cast<size_t>(other) * n + col]);
    }
}

inline float InputLuProduct(const TestData& t, size_t offset, int row, int col)
{
    double value = 0;
    for (int k = 0; k <= std::min(row, col); ++k)
    {
        const float lower = row == k ? 1.0f : t.input[offset + static_cast<size_t>(row) * t.n + k];
        value += lower * t.input[offset + static_cast<size_t>(k) * t.n + col];
    }
    return static_cast<float>(value);
}

inline void PrepareLuOriginal(TestData& t, int b)
{
    const size_t offset = t.Offset(b);
    for (int row = 0; row < t.n; ++row)
    {
        if (t.pivot)
        {
            t.pivots[static_cast<size_t>(b) * t.n + row] = row == 0 && t.n > 1 ? 2 : row + 1;
        }
        for (int col = 0; col < t.n; ++col)
        {
            t.original[offset + static_cast<size_t>(row) * t.n + col] = InputLuProduct(t, offset, row, col);
        }
    }
    if (t.pivot && t.n > 1)
    {
        SwapRows(t.original, offset, t.n, 0, 1);
    }
}

inline void PrepareMatrix(TestData& t, int b)
{
    FillInput(t, b);
    if (t.mode.inputLu)
    {
        PrepareLuOriginal(t, b);
        return;
    }
    const size_t offset = t.Offset(b);
    if (t.pivot && t.m > 1 && !(t.singular && b == 0))
    {
        SwapRows(t.input, offset, t.n, 0, 1);
    }
    if (t.m > 0 && t.n > 0)
    {
        std::copy_n(t.input.data() + offset, static_cast<size_t>(t.m) * t.n, t.original.data() + offset);
    }
}

inline void ExecuteRepeat(Runtime& runtime, AdapterState& state, TestData& t)
{
    Require(solver_adapter_reset_v2(state.value, t.input.data(), t.c.base.a_bytes, nullptr, 0, t.pivots.data(),
                                    t.pivots.size(), nullptr, 0) == 0,
            "adapter reset failed");
    int status = -99;
    Require(solver_adapter_execute(state.value, &status) == 0 && status == ACLSOLVER_STATUS_SUCCESS,
            "operator execution failed");
    Require(aclrtSynchronizeStream(runtime.stream) == ACL_SUCCESS, "stream synchronization failed");
    Require(solver_adapter_readback_v2(state.value, t.output.data(), t.c.base.out_bytes, t.info.data(), t.info.size(),
                                       t.outPivots.data(), t.outPivots.size(), nullptr, 0) == 0,
            "adapter readback failed");
    for (size_t b = 0; b < t.info.size(); ++b)
    {
        Require(t.info[b] == (t.singular && b == 0 ? 2 : 0), "unexpected INFO");
    }
}

inline void CheckRepeats(Runtime& runtime, AdapterState& state, TestData& t)
{
    std::vector<float> first;
    std::vector<int32_t> firstInfo, firstPivots;
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        ExecuteRepeat(runtime, state, t);
        if (repeat == 0)
        {
            first = t.output;
            firstInfo = t.info;
            firstPivots = t.outPivots;
        }
        else
        {
            Require(SameBits(first, t.output) && SameBits(firstInfo, t.info) && SameBits(firstPivots, t.outPivots),
                    "repeat execution differs bit-wise");
        }
    }
}

inline void ApplyOutputPivots(TestData& t, int b)
{
    for (int row = 0; row < std::min(t.m, t.n); ++row)
    {
        const int other = t.outPivots[static_cast<size_t>(b) * std::min(t.m, t.n) + row] - 1;
        Require(other >= row && other < t.m, "invalid one-based pivot");
        SwapRows(t.original, t.Offset(b), t.n, row, other);
    }
}

inline double OutputProduct(const TestData& t, size_t offset, int row, int col)
{
    if (t.mode.factor)
    {
        return LuProduct(t.output, offset, t.m, t.n, row, col);
    }
    double actual = 0;
    for (int k = 0; k < t.n; ++k)
    {
        actual += static_cast<double>(t.output[offset + static_cast<size_t>(row) * t.n + k]) *
                  t.original[offset + static_cast<size_t>(k) * t.n + col];
    }
    return actual;
}

inline void CheckMatrix(const TestData& t, int b)
{
    const size_t offset = t.Offset(b);
    for (int row = 0; row < t.m; ++row)
    {
        for (int col = 0; col < t.n; ++col)
        {
            const double actual = OutputProduct(t, offset, row, col);
            const double expected = t.mode.factor ? t.original[offset + static_cast<size_t>(row) * t.n + col]
                                    : row == col  ? 1
                                                  : 0;
            Require(std::isfinite(actual) && std::abs(actual - expected) <= 1.0e-4 * std::max(1.0, std::abs(expected)),
                    "basic matrix reconstruction failed");
        }
    }
}

inline void RunCase(Runtime& runtime, TestMode mode, int m, int n, int batch, bool pivot, bool singular)
{
    TestData t(mode, m, n, batch, pivot, singular);
    for (int b = 0; b < batch; ++b)
    {
        PrepareMatrix(t, b);
    }
    AdapterState state;
    Require(solver_adapter_create_v2(&t.c, runtime.handle, runtime.stream, &state.value) == 0, "adapter create failed");
    CheckRepeats(runtime, state, t);
    for (int b = 0; b < batch; ++b)
    {
        if (singular && b == 0)
        {
            continue;
        }
        if (mode.factor && pivot)
        {
            ApplyOutputPivots(t, b);
        }
        CheckMatrix(t, b);
    }
    state.Close();
}

inline void CheckInvalid(Runtime& runtime, int (*call)(aclsolverHandle_t, int*))
{
    int* info = nullptr;
    Require(aclrtMalloc(reinterpret_cast<void**>(&info), sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
            "allocate INFO failed");
    const int status = call(runtime.handle, info);
    int actual = 0;
    const bool passed =
        status == ACLSOLVER_STATUS_INVALID_VALUE && aclrtSynchronizeStream(runtime.stream) == ACL_SUCCESS &&
        aclrtMemcpy(&actual, sizeof(actual), info, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
        actual == -1;
    aclrtFree(info);
    Require(passed, "invalid dimension status/INFO mismatch");
}

}  // namespace real_lu_test
}  // namespace

namespace
{
namespace real_lu_test
{
enum class TestOperation
{
    Sgetrf,
    Sgetri,
    SgetriBatched,
    SmatinvBatched
};
constexpr TestOperation TEST_OPERATION = TestOperation::REAL_LU_TEST_OPERATION;
constexpr TestMode MODE{
    TEST_OPERATION == TestOperation::Sgetrf,
    TEST_OPERATION == TestOperation::SgetriBatched || TEST_OPERATION == TestOperation::SmatinvBatched,
    TEST_OPERATION == TestOperation::Sgetri || TEST_OPERATION == TestOperation::SgetriBatched};

inline const char* TestName()
{
    switch (TEST_OPERATION)
    {
        case TestOperation::Sgetrf:
            return "sgetrf";
        case TestOperation::Sgetri:
            return "sgetri";
        case TestOperation::SgetriBatched:
            return "sgetri_batched";
        default:
            return "smatinv_batched";
    }
}

inline int InvalidOperation(aclsolverHandle_t handle, int* info)
{
    switch (TEST_OPERATION)
    {
        case TestOperation::Sgetrf:
            return aclsolverSgetrf(handle, -1, 2, nullptr, 1, nullptr, nullptr, info);
        case TestOperation::Sgetri:
            return aclsolverSgetri(handle, -1, nullptr, 1, nullptr, nullptr, 0, info);
        case TestOperation::SgetriBatched:
            return aclsolverSgetriBatched(handle, -1, nullptr, 1, nullptr, nullptr, 1, info, 1);
        default:
            return aclsolverSmatinvBatched(handle, 33, nullptr, 33, nullptr, 33, info, 1);
    }
}

inline void RunBasicCases(Runtime& runtime)
{
    const int batch = MODE.batched ? 8 : 1;
    RunCase(runtime, MODE, MODE.factor ? 7 : 5, 5, batch, true, false);
    RunCase(runtime, MODE, 3, 3, batch, false, false);
    RunCase(runtime, MODE, 3, 3, batch, true, true);
    RunCase(runtime, MODE, 0, 0, batch, true, false);
    if (MODE.batched)
    {
        RunCase(runtime, MODE, 1, 1, 1, true, false);
        RunCase(runtime, MODE, 32, 32, 8, true, false);
        RunCase(runtime, MODE, 1, 1, 0, true, false);
    }
    CheckInvalid(runtime, InvalidOperation);
}

struct Arguments
{
    bool files = false;
    int m = 0, n = 0, device = 0;
};

inline Arguments ParseArguments(int argc, char** argv)
{
    Arguments args;
    if (MODE.batched)
    {
        const std::string usage = std::string("usage: ") + TestName() + "_test [device_id]";
        Require(argc <= 2, usage.c_str());
        args.device = argc == 2 ? std::stoi(argv[1]) : 0;
        return args;
    }
    args.files = argc > 1 && std::strcmp(argv[1], "--files") == 0;
    const int required = MODE.factor ? 4 : 3;
    const char* usage = MODE.factor ? "usage: sgetrf_test [device_id] | --files m n [device_id]"
                                    : "usage: sgetri_test [device_id] | --files n [device_id]";
    Require(args.files ? (argc == required || argc == required + 1) : argc <= 2, usage);
    args.m = args.files ? ParseInt(argv[2]) : 0;
    args.n = args.files ? (MODE.factor ? ParseInt(argv[3]) : args.m) : 0;
    Require(!args.files || (args.m > 0 && args.m <= (MODE.factor ? 8192 : 4096) && args.n > 0),
            "file-mode dimensions are invalid");
    args.device =
        args.files ? (argc == required + 1 ? ParseInt(argv[required]) : 0) : (argc == 2 ? ParseInt(argv[1]) : 0);
    return args;
}

inline void RunProgram(int argc, char** argv)
{
    const Arguments args = ParseArguments(argc, argv);
    const char* descriptorName = MODE.batched ? (MODE.inputLu ? "sgetribatched" : "smatinvbatched") : TestName();
    CheckDescriptor(descriptorName);
    Runtime runtime;
    runtime.Init(args.device);
    if (args.files)
    {
        RunFileCase(runtime, args.m, args.n, !MODE.factor);
        return;
    }
    RunBasicCases(runtime);
    std::cout << TestName()
              << " basic tests PASS: actual output/INFO, padding input, three repeats, "
                 "empty/invalid boundary\n";
}

}  // namespace real_lu_test
}  // namespace
int main(int argc, char** argv)
{
    try
    {
        real_lu_test::RunProgram(argc, argv);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << real_lu_test::TestName() << " basic tests FAIL: " << error.what() << '\n';
        return 1;
    }
}
#endif
