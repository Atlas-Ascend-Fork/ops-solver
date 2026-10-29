/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REAL_LU_ADAPTER_IMPL_H
#define REAL_LU_ADAPTER_IMPL_H
#include "real_lu_adapter_types.h"

namespace
{
namespace real_lu_adapter
{
inline int SyncPending(State& s)
{
    if (s.pending)
    {
        if (aclrtSynchronizeStream(s.stream) != ACL_SUCCESS)
        {
            return RUNTIME_FAILED;
        }
        s.pending = false;
    }
    return SUCCESS;
}

inline bool CopyToDevice(DeviceBuffer& buffer, const void* source, size_t bytes)
{
    return bytes == 0 ||
           aclrtMemcpy(buffer.data, buffer.bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

inline bool BuildPointerArray(DeviceBuffer& array, const DeviceBuffer& matrices, size_t strideCells, int batch)
{
    if (array.bytes == 0)
    {
        return true;
    }
    // Only pointer metadata is staged for the whole batch; matrices use bounded chunks.
    std::vector<const float*> pointers(static_cast<size_t>(batch));
    for (int b = 0; b < batch; ++b)
    {
        pointers[b] = matrices.data == nullptr
                          ? nullptr
                          : static_cast<const float*>(matrices.data) + static_cast<size_t>(b) * strideCells;
    }
    return CopyToDevice(array, pointers.data(), array.bytes);
}

inline int AllocateStaging(State& s)
{
    if (s.batch == 0 || s.matrixCells == 0)
    {
        return SUCCESS;
    }
    const size_t stride = std::max(s.aStrideCells, s.cStrideCells);
    if (stride == 0)
    {
        return INVALID_CASE;
    }
    const size_t chunk =
        std::min(static_cast<size_t>(s.batch), std::max(size_t{1}, STAGING_BYTES / sizeof(float) / stride));
    size_t stagingCells = 0;
    if (!Multiply(stride, chunk, stagingCells))
    {
        return ALLOC_FAILED;
    }
    s.staging.resize(stagingCells);
    return SUCCESS;
}

inline int Allocate(State& s)
{
    if (QueryWorkspace(s) != ACLSOLVER_STATUS_SUCCESS)
    {
        return QUERY_FAILED;
    }
    size_t workBytes = 0;
    if (s.lwork < 0 || !Multiply(static_cast<size_t>(s.lwork), sizeof(float), workBytes))
    {
        return QUERY_FAILED;
    }
    const size_t aBytes = s.a.bytes;
    const size_t cBytes = IN_PLACE ? 0 : s.c.bytes;
    if (!s.a.Allocate(aBytes) || !s.c.Allocate(cBytes) || !s.workspace.Allocate(workBytes) ||
        !s.pivots.Allocate(std::max(s.inPivots, s.outPivots) * sizeof(int32_t)) ||
        !s.info.Allocate(s.infoCount * sizeof(int32_t)))
    {
        return ALLOC_FAILED;
    }
    if (BATCHED)
    {
        const size_t bytes = static_cast<size_t>(s.batch) * sizeof(void*);
        if (!s.aArray.Allocate(bytes) || !s.cArray.Allocate(bytes))
        {
            return ALLOC_FAILED;
        }
    }
    const int status = AllocateStaging(s);
    s.ready = status == SUCCESS;
    return status;
}

inline bool CopyBytes(void* destination, size_t destinationCapacity, const void* source, size_t sourceCapacity,
                      size_t bytes)
{
    if (bytes > destinationCapacity || bytes > sourceCapacity)
    {
        return false;
    }
    if (bytes == 0)
    {
        return true;
    }
    if (destination == nullptr || source == nullptr)
    {
        return false;
    }
    std::copy_n(static_cast<const unsigned char*>(source), bytes, static_cast<unsigned char*>(destination));
    return true;
}

struct MatrixTransfer
{
    unsigned char* destination;
    const unsigned char* source;
    size_t destinationCapacity, sourceCapacity;
    int m, n, ld;
    bool toColumn;

    bool Element(int row, int col) const
    {
        const size_t rowOffset = (static_cast<size_t>(row) * n + col) * sizeof(float);
        const size_t columnOffset = (static_cast<size_t>(col) * ld + row) * sizeof(float);
        const size_t dstOffset = toColumn ? columnOffset : rowOffset;
        const size_t srcOffset = toColumn ? rowOffset : columnOffset;
        if (dstOffset > destinationCapacity || srcOffset > sourceCapacity)
        {
            return false;
        }
        return CopyBytes(destination + dstOffset, destinationCapacity - dstOffset, source + srcOffset,
                         sourceCapacity - srcOffset, sizeof(float));
    }

    bool Tile(int rowBlock, int colBlock) const
    {
        bool copied = true;
        for (int row = rowBlock; row < std::min(rowBlock + 32, m); ++row)
        {
            for (int col = colBlock; col < std::min(colBlock + 32, n); ++col)
            {
                copied = Element(row, col) && copied;
            }
        }
        return copied;
    }

    bool Copy() const
    {
        bool copied = true;
        for (int rowBlock = 0; rowBlock < m; rowBlock += 32)
        {
            for (int colBlock = 0; colBlock < n; colBlock += 32)
            {
                copied = Tile(rowBlock, colBlock) && copied;
            }
        }
        return copied;
    }
};

inline bool UploadChunk(State& s, const unsigned char* transport, size_t first, size_t count)
{
    const size_t cells = count * s.aStrideCells;
    if (s.lda != s.m)
    {
        std::fill_n(s.staging.data(), cells, static_cast<float>(PADDING_SENTINEL));
    }
    for (size_t b = 0; b < count; ++b)
    {
        auto* matrix = reinterpret_cast<unsigned char*>(s.staging.data() + b * s.aStrideCells);
        const unsigned char* source = transport + (first + b) * s.matrixCells * sizeof(float);
        MatrixTransfer copy{matrix, source, s.aStrideCells * sizeof(float), s.matrixCells * sizeof(float), s.m, s.n,
                            s.lda,  true};
        if (!copy.Copy())
        {
            return false;
        }
    }
    const size_t bytes = cells * sizeof(float);
    auto* destination = static_cast<float*>(s.a.data) + first * s.aStrideCells;
    return aclrtMemcpy(destination, bytes, s.staging.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

inline int UploadMatrices(State& s, const void* input)
{
    if (s.transportBytes == 0)
    {
        return SUCCESS;
    }
    if (s.aStrideCells == 0)
    {
        return INVALID_CASE;
    }
    const size_t chunk = s.staging.size() / s.aStrideCells;
    if (chunk == 0)
    {
        return INVALID_CASE;
    }
    const auto* transport = static_cast<const unsigned char*>(input);
    for (size_t first = 0; first < static_cast<size_t>(s.batch);)
    {
        const size_t count = std::min(chunk, static_cast<size_t>(s.batch) - first);
        if (!UploadChunk(s, transport, first, count))
        {
            return RUNTIME_FAILED;
        }
        first += count;
    }
    return SUCCESS;
}

inline bool ResetBuffers(State& s, const void* input, const int32_t* ipiv, size_t ipivCount)
{
    // Sentinels expose unwritten INFO/output; no expected INFO is inserted.
    if (UploadMatrices(s, input) != SUCCESS || !s.c.Clear(0xA5) || !s.workspace.Clear(0) || !s.info.Clear(0xA5) ||
        !s.pivots.Clear(0xA5) || !CopyToDevice(s.pivots, ipiv, ipivCount * sizeof(int32_t)))
    {
        return false;
    }
    return !BATCHED || (BuildPointerArray(s.aArray, s.a, s.aStrideCells, s.batch) &&
                        BuildPointerArray(s.cArray, s.c, s.cStrideCells, s.batch));
}

inline bool DownloadChunk(State& s, unsigned char* transport, size_t first, size_t count, size_t stride)
{
    const DeviceBuffer& buffer = IN_PLACE ? s.a : s.c;
    const int ld = IN_PLACE ? s.lda : s.ldc;
    const size_t bytes = count * stride * sizeof(float);
    const auto* source = static_cast<const float*>(buffer.data) + first * stride;
    if (aclrtMemcpy(s.staging.data(), s.staging.size() * sizeof(float), source, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS)
    {
        return false;
    }
    for (size_t b = 0; b < count; ++b)
    {
        const auto* matrix = reinterpret_cast<const unsigned char*>(s.staging.data() + b * stride);
        unsigned char* destination = transport + (first + b) * s.matrixCells * sizeof(float);
        MatrixTransfer copy{destination, matrix, s.matrixCells * sizeof(float), stride * sizeof(float), s.m, s.n,
                            ld,          false};
        if (!copy.Copy())
        {
            return false;
        }
    }
    return true;
}

inline int DownloadMatrices(State& s, void* output)
{
    if (s.transportBytes == 0)
    {
        return SUCCESS;
    }
    const size_t stride = IN_PLACE ? s.aStrideCells : s.cStrideCells;
    if (stride == 0)
    {
        return INVALID_CASE;
    }
    const size_t chunk = s.staging.size() / stride;
    if (chunk == 0)
    {
        return INVALID_CASE;
    }
    auto* transport = static_cast<unsigned char*>(output);
    for (size_t first = 0; first < static_cast<size_t>(s.batch);)
    {
        const size_t count = std::min(chunk, static_cast<size_t>(s.batch) - first);
        if (!DownloadChunk(s, transport, first, count, stride))
        {
            return RUNTIME_FAILED;
        }
        first += count;
    }
    return SUCCESS;
}

inline bool DownloadMetadata(State& s, const DeviceBuffer& buffer, int32_t* output, size_t count)
{
    // File-backed output mappings cannot be pinned by every supported driver.
    // Stage DMA into ordinary host storage, as for the matrix readback above.
    int32_t emptyMatrixStaging = 0;
    void* staging = s.staging.empty() ? static_cast<void*>(&emptyMatrixStaging) : s.staging.data();
    const size_t capacity = s.staging.empty() ? sizeof(emptyMatrixStaging) : s.staging.size() * sizeof(float);
    const auto* source = static_cast<const unsigned char*>(buffer.data);
    auto* destination = reinterpret_cast<unsigned char*>(output);
    const size_t bytes = count * sizeof(int32_t);
    for (size_t offset = 0; offset < bytes;)
    {
        const size_t chunk = std::min(capacity, bytes - offset);
        if (aclrtMemcpy(staging, capacity, source + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        {
            return false;
        }
        if (!CopyBytes(destination + offset, bytes - offset, staging, capacity, chunk))
        {
            return false;
        }
        offset += chunk;
    }
    return true;
}
}  // namespace real_lu_adapter
}  // namespace

extern "C" const SolverAdapterDesc* solver_adapter_descriptor(void) { return real_lu_adapter::Descriptor(); }

extern "C" int solver_adapter_create_v2(const SolverAdapterCaseV2* c, void* handle, void* stream, void** state)
{
    using namespace real_lu_adapter;
    if (state == nullptr)
    {
        return INVALID_CASE;
    }
    *state = nullptr;
    if (handle == nullptr)
    {
        return INVALID_CASE;
    }
    auto* s = new (std::nothrow) State;
    if (s == nullptr)
    {
        return ALLOC_FAILED;
    }
    // Publish before any operation that can fail so destroy can release partial state.
    *state = s;
    s->handle = static_cast<aclsolverHandle_t>(handle);
    s->stream = static_cast<aclrtStream>(stream);
    if (!Validate(c, *s))
    {
        return INVALID_CASE;
    }
    aclrtStream actualStream = nullptr;
    if (aclsolverGetStream(s->handle, &actualStream) != ACLSOLVER_STATUS_SUCCESS || actualStream != s->stream)
    {
        return INVALID_CASE;
    }
    try
    {
        return Allocate(*s);
    }
    catch (const std::bad_alloc&)
    {
        return ALLOC_FAILED;
    }
    catch (const std::length_error&)
    {
        return ALLOC_FAILED;
    }
}

extern "C" int solver_adapter_reset_v2(void* state, const void* a, size_t a_bytes, const void*, size_t b_bytes,
                                       const int32_t* ipiv, size_t ipiv_count, const void*, size_t tau_bytes)
{
    using namespace real_lu_adapter;
    auto* s = static_cast<State*>(state);
    if (s == nullptr || !s->ready || a_bytes != s->transportBytes || (a_bytes != 0 && a == nullptr) || b_bytes != 0 ||
        ipiv_count != s->inPivots || (ipiv_count != 0 && ipiv == nullptr) || tau_bytes != 0)
    {
        return INVALID_CASE;
    }
    s->reset = false;
    if (SyncPending(*s) != SUCCESS)
    {
        return RUNTIME_FAILED;
    }
    try
    {
        if (!ResetBuffers(*s, a, ipiv, ipiv_count))
        {
            return RUNTIME_FAILED;
        }
    }
    catch (const std::bad_alloc&)
    {
        return ALLOC_FAILED;
    }
    catch (const std::length_error&)
    {
        return ALLOC_FAILED;
    }
    s->reset = true;
    return SUCCESS;
}

extern "C" int solver_adapter_execute(void* state, int* dut_status)
{
    using namespace real_lu_adapter;
    auto* s = static_cast<State*>(state);
    if (s == nullptr || !s->ready || !s->reset || dut_status == nullptr)
    {
        return INVALID_CASE;
    }
    // The public call can enqueue work even if it returns an error (for example INFO).
    s->pending = true;
    *dut_status = static_cast<int>(ExecuteOperation(*s));
    return SUCCESS;
}

extern "C" int solver_adapter_readback_v2(void* state, void* out, size_t out_bytes, int32_t* info, size_t info_count,
                                          int32_t* ipiv, size_t ipiv_count, void*, size_t tau_bytes)
{
    using namespace real_lu_adapter;
    auto* s = static_cast<State*>(state);
    if (s == nullptr || !s->ready || !s->reset || out_bytes != s->transportBytes ||
        (out_bytes != 0 && out == nullptr) || info_count != s->infoCount || (info_count != 0 && info == nullptr) ||
        ipiv_count != s->outPivots || (ipiv_count != 0 && ipiv == nullptr) || tau_bytes != 0)
    {
        return INVALID_CASE;
    }
    // The harness must synchronize the supplied stream before this callback.
    if (DownloadMatrices(*s, out) != SUCCESS || !DownloadMetadata(*s, s->info, info, info_count) ||
        !DownloadMetadata(*s, s->pivots, ipiv, ipiv_count))
    {
        return RUNTIME_FAILED;
    }
    return SUCCESS;
}

extern "C" int solver_adapter_destroy(void* state)
{
    using namespace real_lu_adapter;
    auto* s = static_cast<State*>(state);
    if (s == nullptr)
    {
        return SUCCESS;
    }
    if (SyncPending(*s) != SUCCESS)
    {
        // Keep state and buffers alive for retry when completion was not established.
        return RUNTIME_FAILED;
    }
    bool released = true;
    DeviceBuffer* buffers[] = {&s->cArray, &s->aArray, &s->info, &s->pivots, &s->workspace, &s->c, &s->a};
    for (DeviceBuffer* buffer : buffers)
    {
        if (!buffer->Free())
        {
            released = false;
        }
    }
    if (!released)
    {
        s->ready = false;
        return RUNTIME_FAILED;
    }
    delete s;
    return SUCCESS;
}

#endif
