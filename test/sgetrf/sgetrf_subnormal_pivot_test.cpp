/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "real_lu_test_common.h"

namespace
{
using real_lu_test::DeviceBuffer;
using real_lu_test::Runtime;

struct FactorizationData
{
    int32_t m, n;
    const std::vector<float>& input;
    std::vector<float> columnMajor, factors;
    std::vector<int32_t> pivots;
    size_t bytes;
    aclrtStream stream = nullptr;
    DeviceBuffer matrix, devicePivots, deviceInfo, workspace;

    FactorizationData(int32_t rows, int32_t columns, const std::vector<float>& values)
        : m(rows),
          n(columns),
          input(values),
          columnMajor(real_lu_test::ColumnMajor(values, m, n)),
          factors(values.size()),
          pivots(static_cast<size_t>(std::min(m, n)), 0),
          bytes(values.size() * sizeof(float))
    {
    }
};

bool PrepareDevice(aclsolverHandle_t handle, FactorizationData& data, const char* name)
{
    if (aclsolverGetStream(handle, &data.stream) != ACLSOLVER_STATUS_SUCCESS)
    {
        std::cerr << name << ": get stream failed\n";
        return false;
    }
    int workElements = 0;
    if (!data.matrix.TryAllocate(data.bytes, data.stream) ||
        !data.devicePivots.TryAllocate(data.pivots.size() * sizeof(int32_t), data.stream) ||
        !data.deviceInfo.TryAllocate(sizeof(int32_t), data.stream) ||
        aclsolverSgetrf_bufferSize(handle, data.m, data.n, data.matrix.As<float>(), data.m, &workElements) !=
            ACLSOLVER_STATUS_SUCCESS ||
        workElements <= 0 ||
        !data.workspace.TryAllocate(static_cast<size_t>(workElements) * sizeof(float), data.stream) ||
        aclrtMemcpy(data.matrix.value, data.bytes, data.columnMajor.data(), data.bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
            ACL_SUCCESS ||
        aclrtMemset(data.deviceInfo.value, sizeof(int32_t), 0xA5, sizeof(int32_t)) != ACL_SUCCESS)
    {
        std::cerr << name << ": Device input/workspace preparation failed\n";
        return false;
    }
    return true;
}

bool ExecuteDevice(aclsolverHandle_t handle, FactorizationData& data, const char* name)
{
    const aclsolverStatus_t status =
        aclsolverSgetrf(handle, data.m, data.n, data.matrix.As<float>(), data.m, data.workspace.As<float>(),
                        data.devicePivots.As<int>(), data.deviceInfo.As<int>());
    int32_t info = -99;
    if (status != ACLSOLVER_STATUS_SUCCESS || aclrtSynchronizeStream(data.stream) != ACL_SUCCESS ||
        aclrtMemcpy(&info, sizeof(info), data.deviceInfo.value, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS ||
        info != 0)
    {
        std::cerr << name << ": API failed, status=" << status << " info=" << info << '\n';
        return false;
    }
    return true;
}

bool ReadFactorization(FactorizationData& data, const char* name)
{
    const size_t pivotBytes = data.pivots.size() * sizeof(int32_t);
    if (aclrtMemcpy(data.columnMajor.data(), data.bytes, data.matrix.value, data.bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS ||
        aclrtMemcpy(data.pivots.data(), pivotBytes, data.devicePivots.value, pivotBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS)
    {
        std::cerr << name << ": result readback failed\n";
        return false;
    }
    real_lu_test::RowMajor(data.columnMajor, data.factors, data.m, data.n, data.m);
    if (!std::all_of(data.factors.begin(), data.factors.end(), [](float value) { return std::isfinite(value); }))
    {
        std::cerr << name << ": non-finite factor output\n";
        return false;
    }
    return true;
}

bool PermuteInput(const FactorizationData& data, std::vector<double>& permuted, const char* name)
{
    for (int32_t step = 0; step < std::min(data.m, data.n); ++step)
    {
        const int32_t pivotRow = data.pivots[static_cast<size_t>(step)] - 1;
        if (pivotRow < step || pivotRow >= data.m)
        {
            std::cerr << name << ": invalid pivot " << pivotRow + 1 << '\n';
            return false;
        }
        for (int32_t col = 0; col < data.n; ++col)
        {
            std::swap(permuted[static_cast<size_t>(step * data.n + col)],
                      permuted[static_cast<size_t>(pivotRow * data.n + col)]);
        }
    }
    return true;
}

bool CheckResidual(const FactorizationData& data, const std::vector<double>& permuted, const char* name)
{
    double maxInput = 0.0;
    double maxResidual = 0.0;
    for (int32_t row = 0; row < data.m; ++row)
    {
        for (int32_t col = 0; col < data.n; ++col)
        {
            const double reconstructed = real_lu_test::LuProduct(data.factors, 0, data.m, data.n, row, col);
            const double expected = permuted[static_cast<size_t>(row * data.n + col)];
            if (!std::isfinite(reconstructed) || !std::isfinite(expected))
            {
                std::cerr << name << ": non-finite factorization at (" << row << ", " << col << ")\n";
                return false;
            }
            maxInput = std::max(maxInput, std::abs(expected));
            maxResidual = std::max(maxResidual, std::abs(reconstructed - expected));
        }
    }
    const double relativeResidual = maxInput == 0.0 ? maxResidual : maxResidual / maxInput;
    std::cout << name << ": relative residual=" << relativeResidual << '\n';
    return std::isfinite(relativeResidual) && relativeResidual <= 1.0e-5;
}

bool CheckFactorization(aclsolverHandle_t handle, int32_t m, int32_t n, const std::vector<float>& input,
                        const char* name)
{
    FactorizationData data(m, n, input);
    if (!PrepareDevice(handle, data, name) || !ExecuteDevice(handle, data, name) || !ReadFactorization(data, name))
    {
        return false;
    }
    std::vector<double> permuted(input.begin(), input.end());
    return PermuteInput(data, permuted, name) && CheckResidual(data, permuted, name);
}

bool RunSubnormalCases(aclsolverHandle_t handle)
{
    const float subnormalScale = 1.0e-39f;
    const std::vector<float> subnormal2x2 = {2.0f * subnormalScale, subnormalScale, subnormalScale,
                                             2.0f * subnormalScale};
    const std::vector<float> normal2x2 = {4.0e-38f, 2.0e-38f, 2.0e-38f, 4.0e-38f};
    const std::vector<float> subnormal513x1(513, subnormalScale);
    bool passed = CheckFactorization(handle, 2, 2, subnormal2x2, "2x2 subnormal pivot");
    passed = CheckFactorization(handle, 2, 2, normal2x2, "2x2 normal-scale control") && passed;
    passed = CheckFactorization(handle, 513, 1, subnormal513x1, "513x1 subnormal pivot") && passed;
    return passed;
}
}  // namespace

int main(int argc, char* argv[])
{
    try
    {
        const int32_t deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
        real_lu_test::Require(argc <= 2, "usage: sgetrf_subnormal_pivot_test [device_id]");
        Runtime runtime;
        runtime.Init(deviceId);
        bool passed = RunSubnormalCases(runtime.handle);
        const bool closed = runtime.Close();
        passed = passed && closed;
        if (!passed)
        {
            std::cerr << "Sgetrf subnormal pivot regression failed\n";
            return 1;
        }
        std::cout << "Sgetrf subnormal pivot regression passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Sgetrf subnormal pivot regression failed: " << error.what() << '\n';
        return 1;
    }
}
