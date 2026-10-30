/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <memory>
#include <new>
#include <vector>

#include "../../utils/aclsolver_handle_internal.h"
#include "cann_ops_solver.h"
#include "cholesky_internal.h"
#include "tiling/platform/platform_ascendc.h"

namespace Cholesky
{
unsigned CubeCores() { return platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic(); }
unsigned VectorCores() { return platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv(); }
namespace
{
using Handle = _aclsolver_handle;
int Pitch(int n) { return (n + 15) / 16 * 16; }
int Required(int n, bool inverse)
{
    return n == 0 || (!inverse && n <= PANEL) ? 0 : 127 + (inverse ? 3 : 2) * Pitch(n) * Pitch(n);
}
bool ValidUplo(int uplo) { return uplo == 0 || uplo == 1; }
int MatrixError(int u, int n, const void *a, int ld, bool used)
{
    if (!ValidUplo(u)) return 1;
    if (n < 0 || n > 4096) return 2;
    if (used && !a) return 3;
    if (ld < std::max(1, n)) return 4;
    return 0;
}
// Runtime retains a thread error until the caller clears it. Do not hide it or
// synchronize here: callers still check device execution with stream synchronization.
aclsolverStatus_t LaunchStatus()
{
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS ? ACLSOLVER_STATUS_SUCCESS
                                                                    : ACLSOLVER_STATUS_EXECUTION_FAILED;
}
aclsolverStatus_t Error(Handle *h, int *info, int index, int count = 1)
{
    if (info && count > 0)
    {
        SetInfo(info, -index, count, h->stream);
        if (LaunchStatus() != ACLSOLVER_STATUS_SUCCESS) return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    return ACLSOLVER_STATUS_INVALID_VALUE;
}
float *Align(float *w) { return reinterpret_cast<float *>((reinterpret_cast<uintptr_t>(w) + 511) & ~uintptr_t(511)); }
struct Buffer
{
    float *allocation = nullptr;
    float *ptr = nullptr;
    size_t bytes = 0;
    aclrtEvent event = nullptr;
    bool leased = false;
    bool recorded = false;
    bool quarantined = false;
    aclrtStream stream = nullptr;
    ~Buffer()
    {
        if (quarantined && aclrtSynchronizeStream(stream) != ACL_SUCCESS) return;
        if (recorded && aclrtSynchronizeEvent(event) != ACL_SUCCESS) return;
        if (event) aclrtDestroyEvent(event);
        if (allocation) aclrtFree(allocation);
    }
};
struct Pool
{
    std::vector<std::unique_ptr<Buffer>> buffers;
};
void ReleasePool(void *p) { delete static_cast<Pool *>(p); }
bool Completed(const Buffer &b)
{
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    return !b.recorded ||
           (aclrtQueryEventStatus(b.event, &status) == ACL_SUCCESS && status == ACL_EVENT_RECORDED_STATUS_COMPLETE);
}
bool Reusable(const Buffer &b, size_t bytes, aclrtStream stream)
{
    if (b.leased || b.quarantined || b.bytes < bytes) return false;
    // Pending work on the same stream is ordered before the next use.
    return b.stream == stream || Completed(b);
}
Buffer *Acquire(Handle *h, size_t bytes)
{
    if (!h->solverScratch)
    {
        h->solverScratch = new (std::nothrow) Pool;
        if (!h->solverScratch) return nullptr;
        h->releaseScratch = ReleasePool;
    }
    auto *pool = static_cast<Pool *>(h->solverScratch);
    for (auto &b : pool->buffers)
    {
        if (!Reusable(*b, bytes, h->stream)) continue;
        b->leased = true;
        return b.get();
    }
    auto b = std::unique_ptr<Buffer>(new (std::nothrow) Buffer);
    if (!b) return nullptr;
    if (bytes > SIZE_MAX - 511 ||
        aclrtMalloc((void **)&b->allocation, bytes + 511, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtCreateEvent(&b->event) != ACL_SUCCESS)
        return nullptr;
    b->ptr = Align(b->allocation);
    b->bytes = bytes;
    b->leased = true;
    auto *result = b.get();
    try
    {
        pool->buffers.push_back(std::move(b));
    }
    catch (const std::bad_alloc &)
    {
        return nullptr;
    }
    // Commit the growth before retiring old capacity. A failed malloc/event/vector
    // allocation leaves every reusable cached block available for the next call.
    for (auto it = pool->buffers.begin(); it != pool->buffers.end();)
    {
        auto &old = *it;
        if (!old->leased && !old->quarantined && old->bytes < bytes && Completed(*old))
            it = pool->buffers.erase(it);
        else
            ++it;
    }
    return result;
}
aclsolverStatus_t Finish(Handle *h, Buffer *b)
{
    const auto launch = LaunchStatus();
    aclError err = aclrtRecordEvent(b->event, h->stream);
    bool quarantined = err != ACL_SUCCESS && aclrtSynchronizeStream(h->stream) != ACL_SUCCESS;
    b->stream = h->stream;
    b->quarantined = quarantined;
    b->recorded = err == ACL_SUCCESS;
    b->leased = false;
    return err == ACL_SUCCESS ? launch : ACLSOLVER_STATUS_EXECUTION_FAILED;
}

void Factor(MatrixBatch a, float *w, int *info, aclrtStream stream)
{
    if (a.n <= PANEL)
    {
        SmallFactor(a, info, stream);
        return;
    }
    if (a.n <= 128)
    {
        int productPitch = Pitch(a.n - PANEL);
        MediumFactorStage(a, w, productPitch, false, info, stream);
        MediumGram(a, w, productPitch, info, stream);
        MediumFactorStage(a, w, productPitch, true, info, stream);
        return;
    }
    if (a.array)
    {
        int productRows = Pitch(a.n);
        for (int start = 0; start < a.n; start += PANEL)
        {
            if (start) LeftGram(a, w, productRows, start, info, stream);
            LeftFactorStage(a, w, productRows, start, info, stream);
        }
        return;
    }
    int pitch = Pitch(a.n);
    float *product = w + int64_t(a.count) * pitch * pitch;
    Pack(a, w, pitch, info, stream);
    for (int k = 0; k < a.n; k += PANEL)
    {
        int width = std::min(PANEL, a.n - k);
        Panel(w, pitch, a.n, k, a.count, info, stream);
        if (k + width == a.n) break;
        PanelSolve(w, pitch, a.n, k, a.count, info, stream);
        Gram(w, product, pitch, a.n, k, width, a.count, info,
             platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic(), stream);
        Update(w, product, pitch, a.n, k + width, a.count, info, stream);
    }
    Unpack(a, w, pitch, info, stream);
}
aclsolverStatus_t Query(aclsolverHandle_t handle, int u, int n, float *a, int ld, int *lw, bool inverse)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if (MatrixError(u, n, a, ld, n > 0) || !lw) return ACLSOLVER_STATUS_INVALID_VALUE;
    *lw = Required(n, inverse);
    return ACLSOLVER_STATUS_SUCCESS;
}
aclsolverStatus_t Compute(aclsolverHandle_t handle, int u, int n, float *a, int ld, float *w, int lw, int *info,
                          bool inverse)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    auto *h = static_cast<Handle *>(handle);
    int error = MatrixError(u, n, a, ld, n > 0);
    if (!error && n > 0 && n <= 4096 && (Required(n, inverse) || lw > 0) && !w) error = 5;
    if (!error && (lw < 0 || lw < Required(n, inverse))) error = 6;
    if (!error && !info) error = 7;
    if (error) return Error(h, info, error);
    if (!n)
    {
        SetInfo(info, 0, 1, h->stream);
        return LaunchStatus();
    }
    MatrixBatch batch{a, nullptr, n, ld, u, 0, 1};
    if (inverse)
        Invert(batch, Align(w), Pitch(n), info, h->stream);
    else
        Factor(batch, w ? Align(w) : nullptr, info, h->stream);
    return LaunchStatus();
}
}  // namespace
}  // namespace Cholesky

extern "C"
{
    aclsolverStatus_t aclsolverSpotrf_bufferSize(aclsolverHandle_t h, aclsolverFillMode_t u, int n, float *a, int ld,
                                                 int *lw)
    {
        return Cholesky::Query(h, u, n, a, ld, lw, false);
    }
    aclsolverStatus_t aclsolverSpotri_bufferSize(aclsolverHandle_t h, aclsolverFillMode_t u, int n, float *a, int ld,
                                                 int *lw)
    {
        return Cholesky::Query(h, u, n, a, ld, lw, true);
    }
    aclsolverStatus_t aclsolverSpotrf(aclsolverHandle_t h, aclsolverFillMode_t u, int n, float *a, int ld, float *w,
                                      int lw, int *info)
    {
        return Cholesky::Compute(h, u, n, a, ld, w, lw, info, false);
    }
    aclsolverStatus_t aclsolverSpotri(aclsolverHandle_t h, aclsolverFillMode_t u, int n, float *a, int ld, float *w,
                                      int lw, int *info)
    {
        return Cholesky::Compute(h, u, n, a, ld, w, lw, info, true);
    }

    aclsolverStatus_t aclsolverSpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t u, int n, float *a[], int ld,
                                             int *info, int count)
    {
        using namespace Cholesky;
        if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        auto *h = static_cast<_aclsolver_handle *>(handle);
        int error = MatrixError(u, n, a, ld, n > 0 && count > 0);
        if (!error && count > 0 && !info) error = 5;
        if (!error && count < 0) error = 6;
        if (error) return Error(h, info, error, count > 0 ? count : count < 0 ? 1 : 0);
        if (!n || !count)
        {
            if (count) SetInfo(info, 0, count, h->stream);
            return count ? LaunchStatus() : ACLSOLVER_STATUS_SUCCESS;
        }
        MatrixBatch batch{nullptr, a, n, ld, int(u), 0, count};
        if (n <= PANEL)
        {
            SmallFactor(batch, info, h->stream);
            return LaunchStatus();
        }
        size_t one = n <= 128 ? size_t(Pitch(n - PANEL)) * Pitch(n - PANEL) * sizeof(float)
                              : size_t(Pitch(n)) * PANEL * sizeof(float);
        size_t freeBytes = 0, totalBytes = 0;
        if (aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes) != ACL_SUCCESS)
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        size_t budget = std::min<size_t>(size_t(8) << 30, freeBytes / 8);
        int wave = std::min<size_t>(std::min(count, 4096), std::max<size_t>(1, budget / one));
        Buffer *buffer = Acquire(h, one * wave);
        if (!buffer) return ACLSOLVER_STATUS_ALLOC_FAILED;
        for (int64_t first = 0; first < count; first += wave)
        {
            batch.first = int(first);
            batch.count = std::min(wave, count - int(first));
            Factor(batch, buffer->ptr, info + first, h->stream);
        }
        return Finish(h, buffer);
    }

    aclsolverStatus_t aclsolverSpotrs(aclsolverHandle_t handle, aclsolverFillMode_t u, int n, int nrhs, const float *a,
                                      int lda, float *b, int ldb, int *info)
    {
        using namespace Cholesky;
        if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        auto *h = static_cast<_aclsolver_handle *>(handle);
        int error = !ValidUplo(u)            ? 1
                    : n < 0 || n > 4096      ? 2
                    : nrhs < 0 || nrhs > 128 ? 3
                    : n && nrhs && !a        ? 4
                    : lda < std::max(1, n)   ? 5
                    : n && nrhs && !b        ? 6
                    : ldb < std::max(1, n)   ? 7
                    : !info                  ? 8
                                             : 0;
        if (error) return Error(h, info, error);
        SetInfo(info, 0, 1, h->stream);
        if (n > 128 && nrhs > 1)
        {
            int pitch = Pitch(n);
            Buffer *buffer = Acquire(h, size_t(pitch) * nrhs * sizeof(float));
            if (!buffer) return ACLSOLVER_STATUS_ALLOC_FAILED;
            Solve({const_cast<float *>(a), nullptr, n, lda, int(u), 0, 1}, {b, nullptr, n, ldb, 0, 0, 1}, nrhs, info,
                  h->stream, buffer->ptr, pitch);
            return Finish(h, buffer);
        }
        if (n && nrhs)
            Solve({const_cast<float *>(a), nullptr, n, lda, int(u), 0, 1}, {b, nullptr, n, ldb, 0, 0, 1}, nrhs, info,
                  h->stream);
        return LaunchStatus();
    }

    aclsolverStatus_t aclsolverSpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t u, int n, int nrhs,
                                             float *a[], int lda, float *b[], int ldb, int *info, int count)
    {
        using namespace Cholesky;
        if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        auto *h = static_cast<_aclsolver_handle *>(handle);
        bool used = n > 0 && count > 0;
        int error = !ValidUplo(u)                     ? 1
                    : n < 0 || n > 4096               ? 2
                    : nrhs < 0 || (used && nrhs != 1) ? 3
                    : used && !a                      ? 4
                    : lda < std::max(1, n)            ? 5
                    : used && !b                      ? 6
                    : ldb < std::max(1, n)            ? 7
                    : !info                           ? 8
                    : count < 0                       ? 9
                                                      : 0;
        if (error) return Error(h, info, error);
        if (used)
            SetSolveInfo(info, a, b, count, h->stream);
        else
            SetInfo(info, 0, 1, h->stream);
        if (used) Solve({nullptr, a, n, lda, int(u), 0, count}, {nullptr, b, n, ldb, 0, 0, count}, 1, info, h->stream);
        return LaunchStatus();
    }
}
