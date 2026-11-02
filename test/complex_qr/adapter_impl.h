/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Shared transport-only implementation; each operator builds its own adapter DSO.
#ifndef COMPLEX_QR_ADAPTER_IMPL_H
#define COMPLEX_QR_ADAPTER_IMPL_H
#include <acl/acl.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "../../tests/third_party/complex_qr/solver_adapter.h"
#include "cann_ops_solver.h"

namespace
{
constexpr int Op = QR_ADAPTER_OP;  // 0=geqrf, 1=ungqr, 2=unmqr, 3=gels
void Require(bool value)
{
    if (!value)
    {
        throw std::runtime_error("adapter argument");
    }
}
void Check(int status)
{
    if (status)
    {
        throw std::runtime_error("adapter runtime");
    }
}
size_t Bytes(int64_t rows, int64_t cols)
{
    Require(rows >= 0 && rows <= INT_MAX && cols >= 0 && cols <= INT_MAX);
    Require(!cols || size_t(rows) <= std::numeric_limits<size_t>::max() / sizeof(aclsolverFloatComplex) / size_t(cols));
    return size_t(rows) * size_t(cols) * sizeof(aclsolverFloatComplex);
}
struct State
{
    SolverAdapterCaseV2 c{};
    aclsolverHandle_t handle{};
    int m{}, n{}, k{}, rhs{}, lda{}, ldb{}, ldx{}, ar{}, ac{}, bc{}, outRows{}, outCols{};
    bool bad{};
    aclsolverSideMode_t side{};
    aclsolverOperation_t trans{};
    aclsolverFloatComplex *a{}, *b{}, *x{}, *tau{}, *work{};
    int* info{};
    int lwork{}, niter{-999};
    size_t workBytes{};
    std::vector<void*> allocations;
    void* Allocate(size_t bytes)
    {
        void* p = nullptr;
        Check(aclrtMalloc(&p, std::max(bytes, size_t(32)), ACL_MEM_MALLOC_HUGE_FIRST));
        try
        {
            allocations.push_back(p);
        }
        catch (...)
        {
            aclrtFree(p);
            throw;
        }
        return p;
    }
};
void Upload(void* dst, const void* src, size_t bytes)
{
    if (bytes)
    {
        Require(src != nullptr);
        Check(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
}
void Pack(void* dst, const void* src, int rows, int cols, int ld)
{
    Require(src || !rows || !cols);
    std::vector<aclsolverFloatComplex> packed(size_t(ld) * cols, {97, -83});
    auto* in = static_cast<const aclsolverFloatComplex*>(src);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            packed[size_t(c) * ld + r] = in[size_t(r) * cols + c];
        }
    }
    Upload(dst, packed.data(), packed.size() * sizeof(aclsolverFloatComplex));
}
void Unpack(void* dst, void* src, int rows, int cols, int ld)
{
    Require(dst || !rows || !cols);
    std::vector<aclsolverFloatComplex> packed(size_t(ld) * cols);
    const size_t bytes = packed.size() * sizeof(aclsolverFloatComplex);
    if (bytes)
    {
        Check(aclrtMemcpy(packed.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    auto* out = static_cast<aclsolverFloatComplex*>(dst);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            out[size_t(r) * cols + c] = packed[size_t(c) * ld + r];
        }
    }
}
void InitializeShape(State* s, const SolverAdapterCaseV2* c)
{
    s->c = *c;
    s->m = c->m;
    s->n = c->base.n;
    s->k = c->k;
    s->rhs = c->base.nrhs;
    s->lda = c->base.lda;
    s->bad = c->base.info_probe && std::strcmp(c->base.info_probe, "bad_param") == 0;
    Require(!c->base.info_probe || !*c->base.info_probe || s->bad);
    s->side = c->side == 'R' ? ACLSOLVER_SIDE_RIGHT : ACLSOLVER_SIDE_LEFT;
    s->trans = c->trans == 'C' ? ACLSOLVER_OP_C : ACLSOLVER_OP_N;
    if (Op == 2)
    {
        Require((c->side == 'L' || c->side == 'R') && (c->trans == 'N' || c->trans == 'C'));
    }
    s->ar = Op == 2 && c->side == 'R' ? s->n : s->m;
    s->ac = Op == 2 ? s->k : s->n;
    s->bc = Op == 3 ? s->rhs : (Op == 2 ? s->n : 0);
    Require(s->lda >= std::max(1, s->ar));
    if (Op == 1)
    {
        Require(s->n <= s->m && s->k <= s->n);
    }
    if (Op == 2)
    {
        Require(s->k <= s->ar);
    }
    if (Op == 3)
    {
        Require(s->n <= s->m);
    }
}

size_t ValidateLayout(State* s, const SolverAdapterCaseV2* c)
{
    const int64_t ldb = Op == 2 ? c->ldc : (Op == 3 ? c->base.ldb : std::max(1, s->m));
    const int64_t ldx = Op == 3 ? c->ldx : std::max(1, s->n);
    Require(ldb >= std::max(1, s->m) && ldb <= INT_MAX && ldx >= std::max(1, s->n) && ldx <= INT_MAX);
    s->ldb = ldb;
    s->ldx = ldx;
    s->outRows = Op == 3 ? s->n : s->m;
    s->outCols = Op == 3 ? s->rhs : s->n;
    Require(c->base.a_bytes == Bytes(s->ar, s->ac) && c->base.b_bytes == Bytes(s->m, s->bc) &&
            c->base.out_bytes == Bytes(s->outRows, s->outCols));
    const size_t nt = Op == 0 ? std::min(s->m, s->n) : ((Op == 1 || Op == 2) ? s->k : 0);
    Require(c->in_tau_count == ((Op == 1 || Op == 2) ? nt : 0) && c->out_tau_count == (Op == 0 ? nt : 0));
    return nt;
}

void AllocateState(State* s, size_t nt)
{
    s->a = static_cast<aclsolverFloatComplex*>(s->Allocate(Bytes(s->lda, s->ac)));
    s->b = static_cast<aclsolverFloatComplex*>(s->Allocate(Bytes(s->ldb, s->bc)));
    s->x = static_cast<aclsolverFloatComplex*>(s->Allocate(Op == 3 ? Bytes(s->ldx, s->rhs) : 0));
    s->tau = static_cast<aclsolverFloatComplex*>(s->Allocate(nt * sizeof(aclsolverFloatComplex)));
    s->info = static_cast<int*>(s->Allocate(sizeof(int)));
    if (Op == 0)
    {
        Check(aclsolverCgeqrf_bufferSize(s->handle, s->m, s->n, s->a, s->lda, &s->lwork));
    }
    if (Op == 1)
    {
        Check(aclsolverCungqr_bufferSize(s->handle, s->m, s->n, s->k, s->a, s->lda, s->tau, &s->lwork));
    }
    if (Op == 2)
    {
        Check(aclsolverCunmqr_bufferSize(s->handle, s->side, s->trans, s->m, s->n, s->k, s->a, s->lda, s->tau, s->b,
                                         s->ldb, &s->lwork));
    }
    if (Op == 3)
    {
        Check(aclsolverCCgels_bufferSize(s->handle, s->m, s->n, s->rhs, s->a, s->lda, s->b, s->ldb, s->x, s->ldx,
                                         nullptr, &s->workBytes));
    }
    else
    {
        s->workBytes = size_t(s->lwork) * sizeof(aclsolverFloatComplex);
    }
    s->work = static_cast<aclsolverFloatComplex*>(s->Allocate(s->workBytes));
}

}  // namespace

extern "C" const SolverAdapterDesc* solver_adapter_descriptor()
{
#if QR_ADAPTER_OP == 0
    // The A2/A3 C++ overload has seven arguments; bind the 950 C ABI explicitly.
    using FactorFunction = aclsolverStatus_t (*)(aclsolverHandle_t, int, int, aclsolverFloatComplex*, int,
                                                 aclsolverFloatComplex*, aclsolverFloatComplex*, int, int*);
    const auto symbol = static_cast<FactorFunction>(&QR_ADAPTER_SYMBOL);
#else
    const auto symbol = &QR_ADAPTER_SYMBOL;
#endif
    static const SolverAdapterDesc d{SOLVER_ADAPTER_VERSION_V2,
                                     sizeof(SolverAdapterDesc),
                                     QR_ADAPTER_NAME,
                                     "complex64",
                                     "device",
                                     "column_major",
                                     "scalar",
                                     reinterpret_cast<const void*>(symbol)};
    return &d;
}

extern "C" int solver_adapter_create_v2(const SolverAdapterCaseV2* c, void* handle, void* stream, void** state)
{
    if (!state)
    {
        return 1;
    }
    *state = nullptr;
    try
    {
        Require(c && handle && c->base.version == SOLVER_ADAPTER_VERSION_V2 && c->base.struct_size == sizeof(*c) &&
                c->base.batch == 1 && c->base.info_count == 1);
        Require(!c->in_ipiv_count && !c->out_ipiv_count);
        Bytes(c->m, c->base.n);
        Bytes(c->k, c->base.nrhs);
        Require(c->base.lda >= 1 && c->base.lda <= INT_MAX);
        auto* s = new State;
        *state = s;
        s->handle = static_cast<aclsolverHandle_t>(handle);
        InitializeShape(s, c);
        const size_t nt = ValidateLayout(s, c);
        Check(aclsolverSetStream(s->handle, static_cast<aclrtStream>(stream)));
        AllocateState(s, nt);
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" int solver_adapter_reset_v2(void* state, const void* a, size_t aBytes, const void* b, size_t bBytes,
                                       const int32_t*, size_t pivCount, const void* tau, size_t tauBytes)
{
    try
    {
        auto* s = static_cast<State*>(state);
        Require(s && aBytes == s->c.base.a_bytes && bBytes == s->c.base.b_bytes && !pivCount &&
                tauBytes == s->c.in_tau_count * sizeof(aclsolverFloatComplex));
        Pack(s->a, a, s->ar, s->ac, s->lda);
        Pack(s->b, b, s->m, s->bc, s->ldb);
        if (Op == 3)
        {
            std::vector<aclsolverFloatComplex> initial(size_t(s->ldx) * s->rhs, {97, -83});
            Upload(s->x, initial.data(), initial.size() * sizeof(aclsolverFloatComplex));
        }
        if (Op == 0)
        {
            std::vector<aclsolverFloatComplex> initial(s->c.out_tau_count, {97, -83});
            Upload(s->tau, initial.data(), initial.size() * sizeof(aclsolverFloatComplex));
        }
        else
        {
            Upload(s->tau, tau, tauBytes);
        }
        int sentinel = -999;
        Upload(s->info, &sentinel, sizeof(sentinel));
        s->niter = -999;
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" int solver_adapter_execute(void* state, int* status)
{
    auto* s = static_cast<State*>(state);
    if (!s || !status)
    {
        return 1;
    }
    const int m = s->bad ? -1 : s->m;
    if (Op == 0)
    {
        *status = aclsolverCgeqrf(s->handle, m, s->n, s->a, s->lda, s->tau, s->work, s->lwork, s->info);
    }
    if (Op == 1)
    {
        *status = aclsolverCungqr(s->handle, m, s->n, s->k, s->a, s->lda, s->tau, s->work, s->lwork, s->info);
    }
    if (Op == 2)
    {
        *status = aclsolverCunmqr(s->handle, s->bad ? static_cast<aclsolverSideMode_t>(-1) : s->side, s->trans, s->m,
                                  s->n, s->k, s->a, s->lda, s->tau, s->b, s->ldb, s->work, s->lwork, s->info);
    }
    if (Op == 3)
    {
        *status = aclsolverCCgels(s->handle, m, s->n, s->rhs, s->a, s->lda, s->b, s->ldb, s->x, s->ldx, s->work,
                                  s->workBytes, &s->niter, s->info);
    }
    return 0;
}

extern "C" int solver_adapter_readback_v2(void* state, void* out, size_t bytes, int32_t* info, size_t infoCount,
                                          int32_t*, size_t pivCount, void* tau, size_t tauBytes)
{
    try
    {
        auto* s = static_cast<State*>(state);
        Require(s && info && infoCount == 1 && bytes == s->c.base.out_bytes && !pivCount &&
                tauBytes == s->c.out_tau_count * sizeof(aclsolverFloatComplex));
        void* src = Op == 3 ? s->x : (Op == 2 ? s->b : s->a);
        const int ld = Op == 3 ? s->ldx : (Op == 2 ? s->ldb : s->lda);
        Unpack(out, src, s->outRows, s->outCols, ld);
        Check(aclrtMemcpy(info, sizeof(*info), s->info, sizeof(*info), ACL_MEMCPY_DEVICE_TO_HOST));
        if (tauBytes)
        {
            Require(tau != nullptr);
            Check(aclrtMemcpy(tau, tauBytes, s->tau, tauBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" int solver_adapter_destroy(void* state)
{
    auto* s = static_cast<State*>(state);
    if (!s)
    {
        return 0;
    }
    int result = 0;
    for (auto i = s->allocations.rbegin(); i != s->allocations.rend(); ++i)
    {
        if (aclrtFree(*i))
        {
            result = 1;
        }
    }
    delete s;
    return result;
}
#endif
