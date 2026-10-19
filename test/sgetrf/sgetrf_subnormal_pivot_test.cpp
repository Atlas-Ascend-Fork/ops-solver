/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cann_ops_solver.h"

namespace {

bool CheckFactorization(
    aclsolverHandle_t handle, int32_t m, int32_t n, const std::vector<float>& input, const char* name)
{
    std::vector<float> factors = input;
    std::vector<int32_t> pivots(static_cast<size_t>(std::min(m, n)), 0);
    int32_t info = 0;
    const aclError status = aclsolverSgetrf(handle, m, n, factors.data(), n, pivots.data(), &info);
    if (status != ACL_SUCCESS || info != 0) {
        std::cerr << name << ": API failed, status=" << status << " info=" << info << '\n';
        return false;
    }

    std::vector<double> permuted(input.begin(), input.end());
    for (int32_t step = 0; step < std::min(m, n); ++step) {
        const int32_t pivotRow = pivots[static_cast<size_t>(step)] - 1;
        if (pivotRow < step || pivotRow >= m) {
            std::cerr << name << ": invalid pivot " << pivotRow + 1 << '\n';
            return false;
        }
        for (int32_t col = 0; col < n; ++col) {
            std::swap(permuted[static_cast<size_t>(step * n + col)], permuted[static_cast<size_t>(pivotRow * n + col)]);
        }
    }

    double maxInput = 0.0;
    double maxResidual = 0.0;
    for (int32_t row = 0; row < m; ++row) {
        for (int32_t col = 0; col < n; ++col) {
            double reconstructed = 0.0;
            for (int32_t k = 0; k < std::min(m, n); ++k) {
                const double lower = row == k ? 1.0 : (row > k ? factors[static_cast<size_t>(row * n + k)] : 0.0);
                const double upper = k <= col ? factors[static_cast<size_t>(k * n + col)] : 0.0;
                reconstructed += lower * upper;
            }
            const double expected = permuted[static_cast<size_t>(row * n + col)];
            if (!std::isfinite(reconstructed) || !std::isfinite(expected)) {
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

} // namespace

int main(int argc, char* argv[])
{
    const int32_t deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "ACL initialization failed\n";
        return 1;
    }

    aclrtStream stream = nullptr;
    aclsolverHandle_t handle = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS || aclsolverCreate(&handle) != ACL_SUCCESS ||
        aclsolverSetStream(handle, stream) != ACL_SUCCESS) {
        std::cerr << "Solver setup failed\n";
        if (handle != nullptr) {
            aclsolverDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        aclFinalize();
        return 1;
    }

    const float subnormalScale = 1.0e-39f;
    const std::vector<float> subnormal2x2 = {
        2.0f * subnormalScale, subnormalScale, subnormalScale, 2.0f * subnormalScale};
    const std::vector<float> normal2x2 = {4.0e-38f, 2.0e-38f, 2.0e-38f, 4.0e-38f};
    const std::vector<float> subnormal513x1(513, subnormalScale);

    bool passed = CheckFactorization(handle, 2, 2, subnormal2x2, "2x2 subnormal pivot") &&
                  CheckFactorization(handle, 2, 2, normal2x2, "2x2 normal-scale control") &&
                  CheckFactorization(handle, 513, 1, subnormal513x1, "513x1 subnormal pivot");

    const aclError syncStatus = aclrtSynchronizeStream(stream);
    const aclError destroyStatus = aclsolverDestroy(handle);
    const aclError streamStatus = aclrtDestroyStream(stream);
    const aclError finalizeStatus = aclFinalize();
    passed = passed && syncStatus == ACL_SUCCESS && destroyStatus == ACL_SUCCESS && streamStatus == ACL_SUCCESS &&
             finalizeStatus == ACL_SUCCESS;
    if (!passed) {
        std::cerr << "Sgetrf subnormal pivot regression failed\n";
        return 1;
    }
    std::cout << "Sgetrf subnormal pivot regression passed\n";
    return 0;
}
