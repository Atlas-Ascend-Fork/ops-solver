/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../utils/complex_solver_test.h"

#include <cmath>
#include <string>

using namespace ComplexSolverTest;
using Complex = std::complex<double>;

namespace
{
Complex Value(aclFloatComplex z) { return {z.real, z.imag}; }

std::vector<aclFloatComplex> Matrix(int m, int n, int lda)
{
    std::vector<aclFloatComplex> a(static_cast<size_t>(lda) * n, {7.0f, -11.0f});
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < m; ++row)
        {
            int logical = (row + 1 == m ? 0 : row + 1);
            a[col * lda + row] = {logical == col ? 8.0f + m : float((logical + col) % 5 - 2) / 16,
                                  logical == col ? 2.0f : float((logical * 3 + col) % 7 - 3) / 16};
        }
    return a;
}

void Padding(const std::vector<aclFloatComplex>& a, int m, int n, int lda)
{
    for (int col = 0; col < n; ++col)
        for (int row = m; row < lda; ++row)
            Require(a[col * lda + row].real == 7.0f && a[col * lda + row].imag == -11.0f, "padding changed");
}

double Norm(const std::vector<aclFloatComplex>& a, int m, int n, int lda)
{
    double norm = 0;
    for (int col = 0; col < n; ++col)
    {
        double sum = 0;
        for (int row = 0; row < m; ++row) sum += std::abs(Value(a[col * lda + row]));
        norm = std::max(norm, sum);
    }
    return norm;
}

void LuResidual(const std::vector<aclFloatComplex>& original, const std::vector<aclFloatComplex>& lu,
                const std::vector<int>& pivots, int m, int n, int lda)
{
    auto permuted = original;
    for (int k = 0; k < int(pivots.size()); ++k)
    {
        Require(pivots[k] >= k + 1 && pivots[k] <= m, "invalid pivot range");
        for (int col = 0; col < n; ++col) std::swap(permuted[col * lda + k], permuted[col * lda + pivots[k] - 1]);
    }
    double error = 0;
    for (int col = 0; col < n; ++col)
    {
        double sum = 0;
        for (int row = 0; row < m; ++row)
        {
            Complex value = 0;
            for (int k = 0; k < std::min(m, n); ++k)
            {
                Complex l = row == k ? Complex(1) : row > k ? Value(lu[k * lda + row]) : Complex(0);
                Complex u = k <= col ? Value(lu[col * lda + k]) : Complex(0);
                value += l * u;
            }
            sum += std::abs(value - Value(permuted[col * lda + row]));
        }
        error = std::max(error, sum);
    }
    double residual = error / (n * Norm(original, m, n, lda) * std::ldexp(1.0, -24));
    Require(residual <= 5.0, "LU residual=" + std::to_string(residual));
}

void InverseResidual(const std::vector<aclFloatComplex>& a, const std::vector<aclFloatComplex>& c, int n, int lda,
                     int ldc)
{
    double error = 0;
    for (int col = 0; col < n; ++col)
    {
        double sum = 0;
        for (int row = 0; row < n; ++row)
        {
            Complex value = 0;
            for (int k = 0; k < n; ++k) value += Value(c[k * ldc + row]) * Value(a[col * lda + k]);
            sum += std::abs(value - Complex(row == col ? 1 : 0));
        }
        error = std::max(error, sum);
    }
    double residual = error / (n * Norm(a, n, n, lda) * Norm(c, n, n, ldc) * std::ldexp(1.0, -24));
    Require(residual <= 5.0, "inverse residual=" + std::to_string(residual));
}

void Single(const Session& session, int m, int n, bool pivot)
{
    std::cout << "single m=" << m << " n=" << n << " pivot=" << pivot << std::endl;
    int lda = m + 3;
    auto original = Matrix(m, n, lda);
    if (!pivot)
    {
        auto rotated = original;
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
                original[col * lda + (row + 1 == m ? 0 : row + 1)] = rotated[col * lda + row];
    }
    size_t bytes = original.size() * sizeof(aclFloatComplex);
    DeviceBuffer a(bytes), pivots(std::min(m, n) * sizeof(int)), info(sizeof(int));
    auto output = original;
    a.Upload(original.data(), bytes);
    Factor(session, m, n, a.As<aclFloatComplex>(), lda, pivot ? pivots.As<int>() : nullptr, info.As<int>());
    CheckInfo(info, 1);
    a.Download(output.data(), bytes);
    std::vector<int> ipiv(std::min(m, n));
    if (pivot)
        pivots.Download(ipiv.data(), ipiv.size() * sizeof(int));
    else
        for (int i = 0; i < int(ipiv.size()); ++i) ipiv[i] = i + 1;
    LuResidual(original, output, ipiv, m, n, lda);
    Padding(output, m, n, lda);
    a.Upload(original.data(), bytes);
    Factor(session, m, n, a.As<aclFloatComplex>(), lda, pivot ? pivots.As<int>() : nullptr, info.As<int>());
    auto repeated = original;
    a.Download(repeated.data(), bytes);
    Require(std::memcmp(output.data(), repeated.data(), bytes) == 0, "LU is not bitwise deterministic");
    if (m != n) return;
    int elements = 0;
    Check(aclsolverCgetri_bufferSize(session.handle, n, a.As<aclFloatComplex>(), lda, &elements), "getri query");
    DeviceBuffer workspace(static_cast<size_t>(elements) * sizeof(aclFloatComplex));
    Check(aclsolverCgetri(session.handle, n, a.As<aclFloatComplex>(), lda, pivot ? pivots.As<int>() : nullptr,
                          workspace.As<aclFloatComplex>(), elements, info.As<int>()),
          "getri");
    session.Sync();
    CheckInfo(info, 1);
    a.Download(output.data(), bytes);
    InverseResidual(original, output, n, lda, lda);
    Padding(output, n, n, lda);
}

void Batched(const Session& session, int n, bool direct, bool reverse = false, int batch = 3, bool packed = false)
{
    std::cout << "batched n=" << n << " direct=" << direct << std::endl;
    int lda = n + (packed ? 0 : 3), ldc = n + (packed ? 0 : 5);
    auto original = Matrix(n, n, lda);
    std::vector<aclFloatComplex> input(original.size() * batch), output(static_cast<size_t>(ldc) * n * batch, {7, -11});
    for (int b = 0; b < batch; ++b)
    {
        auto matrix = original;
        for (int col = 0; col < n; ++col) matrix[col * lda + (col == 0 ? n - 1 : col - 1)].real += b + 1;
        std::copy(matrix.begin(), matrix.end(), input.begin() + b * original.size());
    }
    size_t bytes = input.size() * sizeof(aclFloatComplex), outBytes = output.size() * sizeof(aclFloatComplex);
    DeviceBuffer a(bytes), c(outBytes), pivots(n * batch * sizeof(int)), info(batch * sizeof(int));
    DeviceBuffer aa(batch * sizeof(void*)), cc(batch * sizeof(void*));
    a.Upload(input.data(), bytes);
    c.Upload(output.data(), outBytes);
    std::vector<aclFloatComplex*> inputs(batch), outputs(batch);
    for (int b = 0; b < batch; ++b)
    {
        int physical = reverse ? batch - 1 - b : b;
        inputs[b] = a.As<aclFloatComplex>() + static_cast<size_t>(physical) * lda * n;
        outputs[b] = c.As<aclFloatComplex>() + static_cast<size_t>(physical) * ldc * n;
        if (!direct) Factor(session, n, n, inputs[b], lda, pivots.As<int>() + b * n, info.As<int>() + b);
    }
    aa.Upload(inputs.data(), batch * sizeof(void*));
    cc.Upload(outputs.data(), batch * sizeof(void*));
    std::vector<int> untouched(batch, -777);
    info.Upload(untouched.data(), batch * sizeof(int));
    auto status = direct
                      ? aclsolverCmatinvBatched(session.handle, n, aa.As<const aclFloatComplex*>(), lda,
                                                cc.As<aclFloatComplex*>(), ldc, info.As<int>(), batch)
                      : aclsolverCgetriBatched(session.handle, n, aa.As<const aclFloatComplex*>(), lda,
                                               pivots.As<int>(), cc.As<aclFloatComplex*>(), ldc, info.As<int>(), batch);
    Check(status, "batched inverse");
    session.Sync();
    CheckInfo(info, batch);
    c.Download(output.data(), outBytes);
    for (int b = 0; b < batch; ++b)
    {
        std::vector<aclFloatComplex> one(output.begin() + b * ldc * n, output.begin() + (b + 1) * ldc * n);
        std::vector<aclFloatComplex> matrix(input.begin() + b * lda * n, input.begin() + (b + 1) * lda * n);
        InverseResidual(matrix, one, n, lda, ldc);
        Padding(one, n, n, ldc);
    }
}

void ExpectInfo(const Session& session, const DeviceBuffer& info, int expected)
{
    session.Sync();
    int actual = 999;
    info.Download(&actual, sizeof(actual));
    Require(actual == expected, "info=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

void Invalid(const Session& session)
{
    DeviceBuffer info(sizeof(int)), a(16 * sizeof(aclFloatComplex)), workspace(16 * sizeof(aclFloatComplex));
    Require(aclsolverCgetrf(session.handle, -1, 2, nullptr, 1, nullptr, nullptr, info.As<int>()) ==
                ACLSOLVER_STATUS_INVALID_VALUE,
            "negative m status");
    ExpectInfo(session, info, -1);
    Check(aclsolverCgetrf(session.handle, 0, 2, nullptr, 1, nullptr, nullptr, info.As<int>()), "zero m");
    ExpectInfo(session, info, 0);
    Check(aclsolverCgetri(session.handle, 0, nullptr, 1, nullptr, nullptr, 0, info.As<int>()), "zero n");
    ExpectInfo(session, info, 0);
    Require(aclsolverCgetri(session.handle, 4, a.As<aclFloatComplex>(), 4, nullptr, workspace.As<aclFloatComplex>(), 15,
                            info.As<int>()) == ACLSOLVER_STATUS_INVALID_VALUE,
            "short workspace status");
    ExpectInfo(session, info, -6);
    Require(aclsolverCmatinvBatched(session.handle, 33, nullptr, 33, nullptr, 33, info.As<int>(), 1) ==
                ACLSOLVER_STATUS_INVALID_VALUE,
            "n33 status");
    ExpectInfo(session, info, -1);
    Check(aclsolverCgetriBatched(session.handle, 4, nullptr, 4, nullptr, nullptr, 4, nullptr, 0), "zero batch");
    Require(aclsolverCgetri(session.handle, 4, a.As<aclFloatComplex>(), 4, nullptr, a.As<aclFloatComplex>(), 16,
                            info.As<int>()) == ACLSOLVER_STATUS_INVALID_VALUE,
            "overlap status");
    ExpectInfo(session, info, -5);
}

void InvalidPivots(const Session& session, int n, bool batched)
{
    int batch = batched ? 3 : 1;
    std::vector<aclFloatComplex> input(static_cast<size_t>(n) * n * batch, {0, 0});
    for (int b = 0; b < batch; ++b)
        for (int k = 0; k < n; ++k) input[static_cast<size_t>(b) * n * n + k * n + k] = {1, 0};
    size_t bytes = input.size() * sizeof(aclFloatComplex);
    DeviceBuffer a(bytes), c(bytes), workspace(static_cast<size_t>(n) * n * sizeof(aclFloatComplex));
    DeviceBuffer pivots(static_cast<size_t>(n) * batch * sizeof(int)), info(batch * sizeof(int));
    DeviceBuffer aa(batch * sizeof(void*)), cc(batch * sizeof(void*));
    std::vector<aclFloatComplex*> inputs(batch), outputs(batch);
    std::vector<int> valid(static_cast<size_t>(n) * batch), actual(batch);
    for (int b = 0; b < batch; ++b)
    {
        inputs[b] = a.As<aclFloatComplex>() + static_cast<size_t>(b) * n * n;
        outputs[b] = c.As<aclFloatComplex>() + static_cast<size_t>(b) * n * n;
        for (int k = 0; k < n; ++k) valid[b * n + k] = k + 1;
    }
    aa.Upload(inputs.data(), batch * sizeof(void*));
    cc.Upload(outputs.data(), batch * sizeof(void*));
    auto invoke = [&]()
    {
        return batched ? aclsolverCgetriBatched(session.handle, n, aa.As<const aclFloatComplex*>(), n, pivots.As<int>(),
                                                cc.As<aclFloatComplex*>(), n, info.As<int>(), batch)
                       : aclsolverCgetri(session.handle, n, a.As<aclFloatComplex>(), n, pivots.As<int>(),
                                         workspace.As<aclFloatComplex>(), n * n, info.As<int>());
    };
    for (int b = 0; b < batch; ++b)
        for (int value : {-1, 0, n + 1, n - 1})
        {
            a.Upload(input.data(), bytes);
            auto invalid = valid;
            invalid[b * n + n - 1] = value;
            pivots.Upload(invalid.data(), invalid.size() * sizeof(int));
            Require(invoke() == ACLSOLVER_STATUS_INVALID_VALUE, "invalid pivot value API status");
            session.Sync();
            info.Download(actual.data(), actual.size() * sizeof(int));
            for (int j = 0; j < batch; ++j) Require(actual[j] == (j == b ? -4 : 0), "invalid pivot batch info");
            pivots.Upload(valid.data(), valid.size() * sizeof(int));
            a.Upload(input.data(), bytes);
            Check(invoke(), "valid pivots after Device value update");
            session.Sync();
            CheckInfo(info, batch);
        }
}

void Capacity(const Session& session)
{
    DeviceBuffer storage(512), info(sizeof(int)), workspace(16 * sizeof(aclFloatComplex));
    void* base = nullptr;
    size_t bytes = 0;
    Check(aclrtMemGetAddressRange(storage.As<void>(), &base, &bytes), "allocation range");
    auto* tail = static_cast<unsigned char*>(base) + bytes - sizeof(uint64_t);
    auto* shortMatrix = reinterpret_cast<aclFloatComplex*>(tail);
    Require(aclsolverCgetrf(session.handle, 4, 4, shortMatrix, 4, nullptr, nullptr, info.As<int>()) ==
                ACLSOLVER_STATUS_INVALID_VALUE,
            "short matrix capacity");
    ExpectInfo(session, info, -3);
    Require(aclsolverCgetriBatched(session.handle, 4, reinterpret_cast<const aclFloatComplex**>(tail), 4, nullptr,
                                   reinterpret_cast<aclFloatComplex**>(tail), 4, info.As<int>(),
                                   2) == ACLSOLVER_STATUS_INVALID_VALUE,
            "short pointer array capacity");
    ExpectInfo(session, info, -2);
    auto* shortPivot = reinterpret_cast<int*>(tail + sizeof(int));
    Require(aclsolverCgetri(session.handle, 4, storage.As<aclFloatComplex>(), 4, shortPivot,
                            workspace.As<aclFloatComplex>(), 16, info.As<int>()) == ACLSOLVER_STATUS_INVALID_VALUE,
            "short pivot capacity");
    ExpectInfo(session, info, -4);
    Require(aclsolverCgetri(session.handle, 4, storage.As<aclFloatComplex>(), 4, nullptr, shortMatrix, 16,
                            info.As<int>()) == ACLSOLVER_STATUS_INVALID_VALUE,
            "short workspace allocation");
    ExpectInfo(session, info, -5);
    auto* shortInfo = shortPivot;
    Require(aclsolverCmatinvBatched(session.handle, 0, nullptr, 1, nullptr, 1, shortInfo, 2) ==
                ACLSOLVER_STATUS_INVALID_VALUE,
            "short info capacity");
    Require(aclsolverCgetriBatched(session.handle, 4, nullptr, 4, nullptr, nullptr, 4, info.As<int>(), -1) ==
                ACLSOLVER_STATUS_INVALID_VALUE,
            "negative batch");
    ExpectInfo(session, info, -8);
    Require(aclsolverCmatinvBatched(session.handle, 4, nullptr, 4, nullptr, 4, info.As<int>(), 10000001) ==
                ACLSOLVER_STATUS_INVALID_VALUE,
            "oversized batch");
    ExpectInfo(session, info, -7);
}

void MixedSingular(const Session& session, int n, bool direct, int batch = 3)
{
    auto valid = Matrix(n, n, n);
    std::vector<aclFloatComplex> input(n * n * batch, {0, 0}), output(n * n * batch, {7, -11});
    for (int b = 0; b < batch; ++b)
        if (b != 1) std::copy(valid.begin(), valid.end(), input.begin() + static_cast<size_t>(b) * n * n);
    DeviceBuffer a(input.size() * sizeof(aclFloatComplex)), c(output.size() * sizeof(aclFloatComplex));
    DeviceBuffer aa(batch * sizeof(void*)), cc(batch * sizeof(void*)), info(batch * sizeof(int));
    DeviceBuffer pivots(n * batch * sizeof(int));
    std::vector<aclFloatComplex*> inputs(batch), outputs(batch);
    for (int b = 0; b < batch; ++b)
    {
        inputs[b] = a.As<aclFloatComplex>() + b * n * n;
        outputs[b] = c.As<aclFloatComplex>() + b * n * n;
    }
    a.Upload(input.data(), input.size() * sizeof(aclFloatComplex));
    c.Upload(output.data(), output.size() * sizeof(aclFloatComplex));
    if (!direct)
        for (int b = 0; b < batch; ++b)
            Factor(session, n, n, inputs[b], n, pivots.As<int>() + b * n, info.As<int>() + b);
    aa.Upload(inputs.data(), batch * sizeof(void*));
    cc.Upload(outputs.data(), batch * sizeof(void*));
    std::vector<int> untouched(batch, -777);
    info.Upload(untouched.data(), batch * sizeof(int));
    auto result = direct
                      ? aclsolverCmatinvBatched(session.handle, n, aa.As<const aclFloatComplex*>(), n,
                                                cc.As<aclFloatComplex*>(), n, info.As<int>(), batch)
                      : aclsolverCgetriBatched(session.handle, n, aa.As<const aclFloatComplex*>(), n, pivots.As<int>(),
                                               cc.As<aclFloatComplex*>(), n, info.As<int>(), batch);
    Check(result, "mixed singular batch");
    session.Sync();
    std::vector<int> status(batch);
    info.Download(status.data(), batch * sizeof(int));
    for (int b = 0; b < batch; ++b) Require(status[b] == (b == 1 ? 1 : 0), "mixed singular info");
    c.Download(output.data(), output.size() * sizeof(aclFloatComplex));
    for (int b = 0; b < batch; ++b)
    {
        if (b == 1) continue;
        std::vector<aclFloatComplex> one(output.begin() + b * n * n, output.begin() + (b + 1) * n * n);
        InverseResidual(valid, one, n, n, n);
    }
    for (int i = n * n; i < 2 * n * n; ++i)
        Require(output[i].real == 7 && output[i].imag == -11, "singular batch changed output");
}

void Singular(const Session& session)
{
    std::vector<aclFloatComplex> matrix(16, {0, 0});
    matrix[0] = {1, 0};
    matrix[10] = {3, 0};
    matrix[15] = {4, 0};
    DeviceBuffer a(matrix.size() * sizeof(aclFloatComplex)), pivots(4 * sizeof(int)), info(sizeof(int));
    a.Upload(matrix.data(), matrix.size() * sizeof(aclFloatComplex));
    Factor(session, 4, 4, a.As<aclFloatComplex>(), 4, pivots.As<int>(), info.As<int>());
    ExpectInfo(session, info, 2);
    auto lu = matrix;
    a.Download(lu.data(), lu.size() * sizeof(aclFloatComplex));
    DeviceBuffer workspace(16 * sizeof(aclFloatComplex));
    Check(aclsolverCgetri(session.handle, 4, a.As<aclFloatComplex>(), 4, pivots.As<int>(),
                          workspace.As<aclFloatComplex>(), 16, info.As<int>()),
          "singular getri");
    ExpectInfo(session, info, 2);
    a.Download(matrix.data(), matrix.size() * sizeof(aclFloatComplex));
    Require(std::memcmp(matrix.data(), lu.data(), matrix.size() * sizeof(aclFloatComplex)) == 0,
            "singular overwrote LU");
}
void ShiftedBatched(const Session& session, int n)
{
    std::cout << "shifted batched n=" << n << std::endl;
    const int batch = 3;
    auto matrix = Matrix(n, n, n);
    for (int column = 0; column < n; ++column)
        for (int row = 0; row < n; ++row)
            matrix[column * n + row] = row == column  ? aclFloatComplex{float(n + 2), 1.0f}
                                       : row < column ? aclFloatComplex{0.125f, -0.25f}
                                                      : aclFloatComplex{0, 0};
    std::vector<aclFloatComplex> input, output(batch * n * n);
    for (int b = 0; b < batch; ++b) input.insert(input.end(), matrix.begin(), matrix.end());
    size_t bytes = input.size() * sizeof(aclFloatComplex);
    DeviceBuffer a(bytes + 8), c(bytes + 8), aa(batch * sizeof(void*)), cc(batch * sizeof(void*)),
        info(batch * sizeof(int));
    auto* shiftedA = reinterpret_cast<aclFloatComplex*>(a.As<float>() + 1);
    auto* shiftedC = reinterpret_cast<aclFloatComplex*>(c.As<float>() + 1);
    Check(aclrtMemcpy(shiftedA, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "shifted upload");
    std::vector<aclFloatComplex*> inputs(batch), outputs(batch);
    for (int b = 0; b < batch; ++b)
    {
        inputs[b] = shiftedA + b * n * n;
        outputs[b] = shiftedC + b * n * n;
    }
    aa.Upload(inputs.data(), batch * sizeof(void*));
    cc.Upload(outputs.data(), batch * sizeof(void*));
    Check(aclsolverCgetriBatched(session.handle, n, aa.As<const aclFloatComplex*>(), n, nullptr,
                                 cc.As<aclFloatComplex*>(), n, info.As<int>(), batch),
          "shifted inverse");
    session.Sync();
    CheckInfo(info, batch);
    Check(aclrtMemcpy(output.data(), bytes, shiftedC, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "shifted download");
    for (int b = 0; b < batch; ++b)
    {
        std::vector<aclFloatComplex> one(output.begin() + b * n * n, output.begin() + (b + 1) * n * n);
        InverseResidual(matrix, one, n, n, n);
    }
}
void VectorBatched(const Session& session)
{
    int64_t cores = 0;
    Check(aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores), "vector core count");
    Require(cores > 0 && cores < 10000, "invalid vector core count");
    // A full core wave plus three matrices enters the vector path and exercises its tail.
    int batch = static_cast<int>(cores) * 64 + 3;
    for (int n : {13, 17, 32}) Batched(session, n, true, false, batch, true);
    Batched(session, 31, true, true, batch, true);
    MixedSingular(session, 32, true, batch);
}
void PanelRegression(const Session& session)
{
    // 1000x100: consecutive distributed panels and cube/trailing updates.
    // 800x64: the second panel has 768 rows and switches to the local panel.
    // 1000x40: the last eight columns use the reduced local panel width.
    for (auto shape : {std::pair<int, int>{1000, 100}, {800, 64}, {1000, 40}})
        for (bool pivot : {false, true}) Single(session, shape.first, shape.second, pivot);
}

void MemoryRegression(const Session& session)
{
    PanelRegression(session);
    Single(session, 129, 129, true);
    // Independent pointer-table order, padding, and a non-full tiny-inverse wave.
    for (bool packed : {false, true})
        for (bool reverse : {false, true}) Batched(session, 2, false, reverse, 129, packed);
    Batched(session, 32, false, true);
    Batched(session, 32, true, true);
}

void RunRegression(const Session& session)
{
    PanelRegression(session);
    VectorBatched(session);
    for (int n : {3, 4, 5, 8}) ShiftedBatched(session, n);
    Invalid(session);
    Capacity(session);
    for (int n : {1, 2, 4, 32, 33, 90, 129, 257})
        for (bool batched : {false, true}) InvalidPivots(session, n, batched);
    Singular(session);
    MixedSingular(session, 32, true);
    for (int n : {32, 64, 90, 128, 129, 132, 152, 153, 257}) MixedSingular(session, n, false);
    for (int n : {1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257}) Single(session, n, n, true);
    for (auto shape : {std::pair<int, int>{35, 67},
                       {67, 35},
                       {4, 70},
                       {70, 4},
                       {8, 1792},
                       {8, 1793},
                       {32, 6632},
                       {680, 9},
                       {3574, 15}})
        Single(session, shape.first, shape.second, true);
    Single(session, 8, 8, false);
    for (int n : {1, 2, 3, 31, 32}) Batched(session, n, true);
    for (int n : {2, 4, 8, 31, 32, 33, 64, 65, 96, 97, 104, 105, 128, 129, 131, 132, 152, 153, 256, 257})
        Batched(session, n, false);
    Batched(session, 90, false, false, 129, true);
    Batched(session, 95, false, true, 67, true);
    Batched(session, 32, true, true);
    Batched(session, 129, false, true);
    Single(session, 8, 1793, false);
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Require(argc == 1 || (argc == 2 && (std::string(argv[1]) == "--panel-regression" ||
                                            std::string(argv[1]) == "--memory-regression")),
                "usage: complex_solver_test [--panel-regression|--memory-regression]");
        Session session(0);
        if (argc == 2)
        {
            if (std::string(argv[1]) == "--panel-regression")
                PanelRegression(session);
            else
                MemoryRegression(session);
            std::cout << "complex solver focused regression passed" << std::endl;
            return 0;
        }
        RunRegression(session);
        std::cout << "complex solver regression passed" << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
