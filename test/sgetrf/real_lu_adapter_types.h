/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REAL_LU_ADAPTER_TYPES_H
#define REAL_LU_ADAPTER_TYPES_H
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "cann_ops_solver.h"
#include "solver_adapter.h"

namespace
{
namespace real_lu_adapter
{
static_assert(sizeof(int) == sizeof(int32_t), "The public pivot/info ABI must use int32");
static_assert(sizeof(float) == 4, "The transport and public ABI must use float32");
enum class Operation
{
    Sgetrf,
    Sgetri,
    SgetriBatched,
    SmatinvBatched
};
constexpr Operation OP = Operation::REAL_LU_ADAPTER_OPERATION;
constexpr bool BATCHED = OP == Operation::SgetriBatched || OP == Operation::SmatinvBatched;
constexpr bool FACTOR = OP == Operation::Sgetrf;
constexpr bool INPUT_LU = OP == Operation::Sgetri || OP == Operation::SgetriBatched;
constexpr bool IN_PLACE = !BATCHED;
constexpr int PADDING_SENTINEL = -773;
constexpr size_t STAGING_BYTES = 16 * 1024 * 1024;

// Adapter failures are independent of aclsolverStatus_t, even where values overlap.
enum Status
{
    SUCCESS = 0,
    INVALID_CASE = 1,
    ALLOC_FAILED = 2,
    RUNTIME_FAILED = 3,
    QUERY_FAILED = 4
};

struct DeviceBuffer
{
    void* data = nullptr;
    size_t bytes = 0;

    bool Allocate(size_t capacity)
    {
        bytes = capacity;
        return capacity == 0 || aclrtMalloc(&data, capacity, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }

    bool Clear(int value) { return bytes == 0 || aclrtMemset(data, bytes, value, bytes) == ACL_SUCCESS; }

    bool Free()
    {
        if (data != nullptr && aclrtFree(data) != ACL_SUCCESS)
        {
            return false;
        }
        data = nullptr;
        bytes = 0;
        return true;
    }
};

struct State
{
    aclsolverHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    int m = 0, n = 0, lda = 0, ldc = 0, batch = 0, lwork = 0;
    size_t matrixCells = 0, transportBytes = 0, aStrideCells = 0, cStrideCells = 0;
    size_t inPivots = 0, outPivots = 0, infoCount = 0;
    bool ready = false, reset = false, pending = false;
    DeviceBuffer a, c, workspace, pivots, info, aArray, cArray;
    std::vector<float> staging;
};

inline bool Multiply(size_t a, size_t b, size_t& result)
{
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    {
        return false;
    }
    result = a * b;
    return true;
}

inline bool FitsInt(int64_t value) { return value >= 0 && value <= INT_MAX; }

inline bool ValidateBase(const SolverAdapterCaseV2* c)
{
    return c != nullptr && c->base.version == SOLVER_ADAPTER_VERSION_V2 && c->base.struct_size == sizeof(*c) &&
           FitsInt(c->base.n) && FitsInt(c->base.batch) && FitsInt(c->base.lda) && c->base.b_bytes == 0 &&
           c->in_tau_count == 0 && c->out_tau_count == 0 && (BATCHED || c->base.batch == 1);
}

inline bool ValidateDimensions(const SolverAdapterCaseV2& c, State& s)
{
    s.n = static_cast<int>(c.base.n);
    s.batch = static_cast<int>(c.base.batch);
    s.lda = static_cast<int>(c.base.lda);
    if (FACTOR)
    {
        if (!FitsInt(c.m) || c.m > 8192)
        {
            return false;
        }
        s.m = static_cast<int>(c.m);
    }
    else
    {
        if ((c.m != 0 && c.m != s.n) || s.n > (OP == Operation::SmatinvBatched ? 32 : 4096))
        {
            return false;
        }
        s.m = s.n;
    }
    if (s.lda < std::max(1, s.m))
    {
        return false;
    }
    s.ldc = s.lda;
    if (BATCHED)
    {
        if (!FitsInt(c.ldc) || c.ldc < std::max(1, s.n))
        {
            return false;
        }
        s.ldc = static_cast<int>(c.ldc);
    }
    return true;
}

inline bool ValidatePivots(const SolverAdapterCaseV2& c, size_t pivotCount)
{
    // Zero counts select the public null-pivot path; no identity pivots are fabricated.
    if (FACTOR)
    {
        return c.in_ipiv_count == 0 && (c.out_ipiv_count == 0 || c.out_ipiv_count == pivotCount);
    }
    if (INPUT_LU)
    {
        return c.out_ipiv_count == 0 && (c.in_ipiv_count == 0 || c.in_ipiv_count == pivotCount);
    }
    return c.in_ipiv_count == 0 && c.out_ipiv_count == 0;
}

inline bool ValidateTransport(const SolverAdapterCaseV2& c, State& s)
{
    size_t totalCells = 0, pivotCount = 0, metadataBytes = 0;
    if (!Multiply(static_cast<size_t>(s.m), s.n, s.matrixCells) || !Multiply(s.matrixCells, s.batch, totalCells) ||
        !Multiply(totalCells, sizeof(float), s.transportBytes) ||
        !Multiply(static_cast<size_t>(std::min(s.m, s.n)), s.batch, pivotCount) ||
        !Multiply(pivotCount, sizeof(int32_t), metadataBytes) ||
        !Multiply(static_cast<size_t>(s.batch), sizeof(void*), metadataBytes))
    {
        return false;
    }
    s.infoCount = BATCHED ? static_cast<size_t>(s.batch) : 1;
    if (!Multiply(s.infoCount, sizeof(int32_t), metadataBytes) || c.base.info_count != s.infoCount ||
        c.base.a_bytes != s.transportBytes || c.base.out_bytes != s.transportBytes || !ValidatePivots(c, pivotCount))
    {
        return false;
    }
    s.inPivots = c.in_ipiv_count;
    s.outPivots = c.out_ipiv_count;
    return true;
}

inline bool ValidateStrides(State& s)
{
    size_t aCells = 0, cCells = 0;
    return Multiply(static_cast<size_t>(s.lda), s.n, s.aStrideCells) &&
           Multiply(static_cast<size_t>(s.ldc), s.n, s.cStrideCells) && Multiply(s.aStrideCells, s.batch, aCells) &&
           Multiply(s.cStrideCells, s.batch, cCells) && Multiply(aCells, sizeof(float), s.a.bytes) &&
           Multiply(cCells, sizeof(float), s.c.bytes);
}

inline bool Validate(const SolverAdapterCaseV2* c, State& s)
{
    return ValidateBase(c) && ValidateDimensions(*c, s) && ValidateTransport(*c, s) && ValidateStrides(s);
}

}  // namespace real_lu_adapter
}  // namespace

namespace
{
namespace real_lu_adapter
{
inline aclsolverStatus_t QueryWorkspace(State& s)
{
    if constexpr (FACTOR)
    {
        return aclsolverSgetrf_bufferSize(s.handle, s.m, s.n, nullptr, s.lda, &s.lwork);
    }
    if constexpr (OP == Operation::Sgetri)
    {
        return aclsolverSgetri_bufferSize(s.handle, s.n, nullptr, s.lda, &s.lwork);
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline aclsolverStatus_t ExecuteOperation(State& s)
{
    if constexpr (FACTOR)
    {
        return aclsolverSgetrf(s.handle, s.m, s.n, static_cast<float*>(s.a.data), s.lda,
                               static_cast<float*>(s.workspace.data), static_cast<int*>(s.pivots.data),
                               static_cast<int*>(s.info.data));
    }
    if constexpr (OP == Operation::Sgetri)
    {
        return aclsolverSgetri(s.handle, s.n, static_cast<float*>(s.a.data), s.lda, static_cast<int*>(s.pivots.data),
                               static_cast<float*>(s.workspace.data), s.lwork, static_cast<int*>(s.info.data));
    }
    if constexpr (OP == Operation::SgetriBatched)
    {
        return aclsolverSgetriBatched(s.handle, s.n, static_cast<const float* const*>(s.aArray.data), s.lda,
                                      static_cast<const int*>(s.pivots.data), static_cast<float* const*>(s.cArray.data),
                                      s.ldc, static_cast<int*>(s.info.data), s.batch);
    }
    return aclsolverSmatinvBatched(s.handle, s.n, static_cast<const float* const*>(s.aArray.data), s.lda,
                                   static_cast<float* const*>(s.cArray.data), s.ldc, static_cast<int*>(s.info.data),
                                   s.batch);
}

inline const SolverAdapterDesc* Descriptor()
{
    const char* name = FACTOR                           ? "sgetrf"
                       : OP == Operation::Sgetri        ? "sgetri"
                       : OP == Operation::SgetriBatched ? "sgetribatched"
                                                        : "smatinvbatched";
    const void* symbol = FACTOR                           ? reinterpret_cast<const void*>(&aclsolverSgetrf)
                         : OP == Operation::Sgetri        ? reinterpret_cast<const void*>(&aclsolverSgetri)
                         : OP == Operation::SgetriBatched ? reinterpret_cast<const void*>(&aclsolverSgetriBatched)
                                                          : reinterpret_cast<const void*>(&aclsolverSmatinvBatched);
    static const SolverAdapterDesc descriptor{SOLVER_ADAPTER_VERSION_V2,
                                              sizeof(SolverAdapterDesc),
                                              name,
                                              "float32",
                                              "device",
                                              "column_major",
                                              BATCHED ? "array" : "scalar",
                                              symbol};
    return &descriptor;
}

}  // namespace real_lu_adapter
}  // namespace
#endif
