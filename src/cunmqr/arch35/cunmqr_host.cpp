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
#include <array>
#include <limits>

#include "../../cungqr/arch35/cungqr_internal.h"
#include "../../cunmqr/arch35/complex_qr_algorithms.h"
#include "../../utils/complex_qr/arch35/complex_qr_transfer.h"
#include "../../utils/complex_qr/complex_qr_host.h"
#include "acl/acl.h"
#include "complex_qr_algorithms.h"
#include "complex_qr_matmul.h"
#include "cunmqr_internal.h"
#include "cunmqr_launch.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace
{
bool Empty(const ComplexQrCall& c) { return c.m == 0 || c.n == 0 || c.k == 0; }
bool WorkspaceSize(const ComplexQrCall& c, size_t& bytes)
{
    uint64_t elements = 0, scratch = 0;
    if (!Empty(c))
    {
        const uint32_t order = c.side == ACLSOLVER_SIDE_RIGHT ? c.n : c.m;
        const uint32_t columns = c.side == ACLSOLVER_SIDE_RIGHT ? c.m : c.n;
        if (ComplexQrBlockedApply(order, columns, c.k) && !ComplexQrBlockWorkspace(order, columns, scratch))
        {
            return false;
        }
    }
    return ComplexQrWorkspaceBytes(elements, scratch, bytes);
}
int CheckExecution(const ComplexQrCall&, int*, bool) { return 0; }
void Execute(float* a, float* tau, float* b, float*, float* work, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrApply(a, tau, b, work, s, stream);
    ComplexQrInfo(info, 0, stream);
}
const ComplexQrPolicy POLICY = {
    ComplexQrCheckApply, Empty, WorkspaceSize, CheckExecution, CunmqrValidate, Execute, 11, 13};
}  // namespace

extern "C" aclsolverStatus_t aclsolverCunmqr_bufferSize(aclsolverHandle_t h, aclsolverSideMode_t side,
                                                        aclsolverOperation_t trans, int m, int n, int k,
                                                        const aclsolverFloatComplex* a, int lda,
                                                        const aclsolverFloatComplex* tau,
                                                        const aclsolverFloatComplex* c, int ldc, int* lwork)
{
    return ComplexQrQuery(
        POLICY, h, {ComplexQrOperation::Apply, m, n, k, 0, int(side), int(trans), lda, ldc, 0, a, tau, c, nullptr},
        lwork);
}
extern "C" aclsolverStatus_t aclsolverCunmqr(aclsolverHandle_t h, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                             int m, int n, int k, const aclsolverFloatComplex* a, int lda,
                                             const aclsolverFloatComplex* tau, aclsolverFloatComplex* c, int ldc,
                                             aclsolverFloatComplex* work, int lwork, int* info)
{
    return ComplexQrRun(POLICY, h,
                        {ComplexQrOperation::Apply, m, n, k, 0, int(side), int(trans), lda, ldc, 0, a, tau, c, nullptr},
                        work, lwork < 0 ? SIZE_MAX : size_t(lwork) * sizeof(aclsolverFloatComplex), info);
}

int ComplexQrCheckApply(const ComplexQrCall& c)
{
    if (c.side != ACLSOLVER_SIDE_LEFT && c.side != ACLSOLVER_SIDE_RIGHT)
    {
        return 1;
    }
    if (c.trans != ACLSOLVER_OP_N && c.trans != ACLSOLVER_OP_C)
    {
        return 2;
    }
    if (c.m < 0)
    {
        return 3;
    }
    if (c.n < 0)
    {
        return 4;
    }
    const int order = c.side == ACLSOLVER_SIDE_LEFT ? c.m : c.n;
    if (c.k < 0 || c.k > order)
    {
        return 5;
    }
    const bool active = c.m && c.n && c.k;
    if (active && !c.a)
    {
        return 6;
    }
    if (c.lda < std::max(1, order))
    {
        return 7;
    }
    if (active && !c.tau)
    {
        return 8;
    }
    if (active && !c.b)
    {
        return 9;
    }
    if (c.ldb < std::max(1, c.m))
    {
        return 10;
    }
    return 0;
}

static void ApplyPanels(float* a, float* tau, float* c, float* work, ComplexQrShape s, void* stream, bool right,
                        bool adjoint, uint32_t count, uint32_t order)
{
    float* target = c;
    float* scratch = work;
    uint64_t ldc = s.ldb;
    if (right)
    {
        adjoint = !adjoint;
    }
    const uint32_t panels = (s.k + COMPLEX_QR_PANEL - 1) / COMPLEX_QR_PANEL;
    for (uint32_t p = 0; p < panels; ++p)
    {
        const uint32_t first = (adjoint ? p : panels - 1 - p) * COMPLEX_QR_PANEL;
        const uint32_t end = first + COMPLEX_QR_PANEL < s.k ? first + COMPLEX_QR_PANEL : s.k;
        // Explicit Q starts at identity. While applying panels in reverse,
        // earlier columns have zeros on the active rows and remain unchanged.
        const uint32_t columnStart = s.operation == ComplexQrOperation::Generate ? first : 0;
        const uint32_t activeColumns = count - columnStart;
        const bool identity = s.operation == ComplexQrOperation::Generate && p == 0;
        if (!ComplexQrBlockUpdate(a, tau, target, scratch, order, activeColumns, s.lda, ldc, first, end, columnStart,
                                  adjoint, stream, identity, right))
        {
            for (uint32_t j = 0; j < end - first; ++j)
            {
                uint32_t k = adjoint ? first + j : end - 1 - j;
                LaunchApplyKernel(ComplexQrBlocks(activeColumns), stream, a, tau, target, order, s.lda, ldc, k,
                                  columnStart, activeColumns, right, right ? !adjoint : adjoint);
            }
        }
    }
}

void ComplexQrApply(float* a, float* tau, float* c, float* work, ComplexQrShape s, void* stream)
{
    bool right = s.side != 0, adjoint = s.trans == 2;
    uint32_t count = right ? s.m : s.n, order = right ? s.n : s.m;
    if (s.operation == ComplexQrOperation::Generate && (count <= 32 || (count <= 64 && order <= 1024)) &&
        order <= 8192 && s.k)
    {
        LaunchCachedSequenceKernel(ComplexQrBlocks(count), stream, a, tau, c, order, s.lda, s.ldb, s.k, count, false,
                                   true);
        return;
    }
    if (ComplexQrBlockedApply(order, count, s.k))
    {
        ApplyPanels(a, tau, c, work, s, stream, right, adjoint, count, order);
        return;
    }
    if (s.k)
    {
        if (order <= 8192)
        {
            LaunchCachedSequenceKernel(ComplexQrBlocks(count), stream, a, tau, c, order, s.lda, s.ldb, s.k, count,
                                       adjoint, false, right);
        }
        else
        {
            LaunchApplySequenceKernel(ComplexQrBlocks(count), stream, a, tau, c, order, s.lda, s.ldb, s.k, count, right,
                                      adjoint);
        }
    }
}

bool ComplexQrBlockWorkspace(uint32_t order, uint32_t columns, uint64_t& elements)
{
    elements = 0;
    const uint64_t paddedOrder = (uint64_t(order) + 127) / 128 * 128;
    const uint64_t paddedColumns =
        columns >= 512 ? (uint64_t(columns) + 255) / 256 * 256 : (uint64_t(columns) + 127) / 128 * 128;
    const uint64_t matrix = paddedOrder * std::max(paddedColumns, uint64_t(2 * COMPLEX_QR_PANEL));
    return ComplexQrAddWorkspace(elements, 5, matrix) &&
           ComplexQrAddWorkspace(elements, 4 * COMPLEX_QR_PANEL, paddedOrder) &&
           ComplexQrAddWorkspace(elements, 5, uint64_t(COMPLEX_QR_PANEL) * COMPLEX_QR_PANEL) &&
           ComplexQrAddWorkspace(elements, 1, COMPLEX_QR_SPLIT_FLOATS / 2);
}

uint32_t ComplexQrBlocks(uint32_t n)
{
    uint32_t available = ComplexQrVectorCores();
    return n < available ? n : available;
}
namespace
{
struct BlockWorkspace
{
    float *ah, *al, *bh, *bl, *hh, *hl, *lh, *gram, *th, *tl, *splitScratch;
};

BlockWorkspace MakeBlockWorkspace(float* work, uint32_t order, uint32_t packedCols)
{
    uint32_t paddedOrder = (order + 127) / 128 * 128;
    uint64_t leftSize = 4 * uint64_t(paddedOrder) * COMPLEX_QR_PANEL;
    uint64_t rightSize =
        2 * uint64_t(paddedOrder) * (packedCols > 2 * COMPLEX_QR_PANEL ? packedCols : 2 * COMPLEX_QR_PANEL);
    float* ah = work;
    float* al = ah + leftSize;
    float* bh = al + leftSize;
    float* bl = bh + rightSize;
    float* hh = bl + rightSize;
    float* hl = hh + rightSize;
    float* lh = hl + rightSize;
    float* gram = lh + rightSize;
    float* th = gram + 2 * COMPLEX_QR_PANEL * COMPLEX_QR_PANEL;
    float* tl = th + 4 * COMPLEX_QR_PANEL * COMPLEX_QR_PANEL;
    float* splitScratch = tl + 4 * COMPLEX_QR_PANEL * COMPLEX_QR_PANEL;
    return {ah, al, bh, bl, hh, hl, lh, gram, th, tl, splitScratch};
}

bool TransformBlockWeights(const BlockWorkspace& w, uint32_t panel, uint32_t rows, uint32_t packedCols,
                           uint32_t vectorCores, void* stream)
{
    if (!ComplexQrCompensatedMatmul(w.th, w.tl, w.bh, w.bl, w.hh, 2 * panel, packedCols, 2 * panel, stream))
    {
        return false;
    }
    LaunchSplitWeightsKernel(vectorCores, stream, w.hh, w.bh, w.bl, 2 * uint64_t(panel) * packedCols);
    return ComplexQrCompensatedMatmul(w.hl, w.lh, w.bh, w.bl, w.hh, 2 * rows, packedCols, 2 * panel, stream);
}

bool ProjectBlockWeights(const BlockWorkspace& w, float* c, uint64_t ldc, uint32_t first, uint32_t columnStart,
                         uint32_t actualRows, uint32_t cols, uint32_t rows, uint32_t panel, uint32_t packedCols,
                         void* stream, bool rightLayout)
{
    const bool transposeB = !rightLayout && rows >= 512 && cols >= 512;
    if (transposeB)
    {
        ComplexQrPackTransposed(c, w.bh, w.bl, ldc, first, columnStart, actualRows, cols, rows, packedCols, stream);
    }
    else
    {
        ComplexQrPackTrailing(c, w.bh, w.bl, ldc, first, columnStart, actualRows, cols, rows, packedCols, stream,
                              rightLayout);
    }
    if (!ComplexQrCompensatedMatmul(w.ah, w.al, w.bh, w.bl, w.hh, 2 * panel, packedCols, 2 * rows, stream,
                                    w.splitScratch, transposeB))
    {
        return false;
    }
    LaunchSplitWeightsKernel(ComplexQrVectorCores(), stream, w.hh, w.bh, w.bl, 2 * uint64_t(panel) * packedCols);
    return true;
}

}  // namespace

bool ComplexQrBlockUpdate(float* a, float* tau, float* c, float* work, uint32_t order, uint32_t cols, uint64_t lda,
                          uint64_t ldc, uint32_t first, uint32_t end, uint32_t columnStart, bool adjoint, void* stream,
                          bool identity, bool rightLayout)
{
    uint32_t actualRows = order - first, actualPanel = end - first;
    if (actualPanel <= 16 && actualRows <= 8192 && cols <= 64 && !rightLayout)
    {
        LaunchCachedSequenceKernel(ComplexQrBlocks(cols), stream, a + 2 * (first + uint64_t(first) * lda),
                                   tau + 2 * first, c + 2 * (first + uint64_t(columnStart) * ldc), actualRows, lda, ldc,
                                   actualPanel, cols, adjoint, identity);
        return true;
    }
    uint32_t rows = (actualRows + 127) / 128 * 128, panel = (actualPanel + 63) / 64 * 64;
    uint32_t packedCols = cols >= 512 ? (cols + 255) / 256 * 256 : (cols + 127) / 128 * 128;
    uint32_t vectorCores = ComplexQrVectorCores();
    const auto workspace = MakeBlockWorkspace(work, order, packedCols);
    auto [ah, al, bh, bl, hh, hl, lh, gram, th, tl, splitScratch] = workspace;
    // DMA reads A once; tiled expansion writes both V and V^H contiguously.
    ComplexQrPackReflectors(a, ah, al, hl, lh, lda, first, rows, panel, actualRows, actualPanel, stream);
    if (!ComplexQrCompensatedMatmul(ah, al, hl, lh, hh, 2 * panel, 2 * panel, 2 * rows, stream, splitScratch))
    {
        return false;
    }
    LaunchUnpackGramKernel(vectorCores, stream, hh, nullptr, nullptr, gram, panel);
    LaunchFormTransformKernel(ComplexQrBlocks(panel), stream, tau, gram, th, tl, first, panel, actualPanel, adjoint);
    if (identity)
    {
        LaunchIdentityWeightsKernel(vectorCores, stream, ah, al, bh, bl, rows, panel, cols, packedCols);
    }
    else if (!ProjectBlockWeights(workspace, c, ldc, first, columnStart, actualRows, cols, rows, panel, packedCols,
                                  stream, rightLayout))
    {
        return false;
    }
    if (!TransformBlockWeights(workspace, panel, rows, packedCols, vectorCores, stream))
    {
        return false;
    }
    ComplexQrUpdateTrailing(c, hh, ldc, first, columnStart, actualRows, cols, rows, packedCols, stream, rightLayout);
    return true;
}

namespace
{
struct MatmulPlan
{
    uint32_t m, n, k, cores;
    uint32_t tileM = 128, tileN = 128, splitParts = 1;
    uint32_t splitK;
};

struct MatmulBuffers
{
    const float *ah, *al, *bh, *bl;
    float *output, *scratch;
    void* stream;
    bool transposeB;
};

void SelectSplit(MatmulPlan& plan, const float* splitScratch)
{
    auto& [m, n, k, cores, tileM, tileN, splitParts, splitK] = plan;
    if (splitScratch && k >= 1024 && m % 128 == 0 && n % 128 == 0 && uint64_t(m) * n <= COMPLEX_QR_SPLIT_OUTPUT_LIMIT)
    {
        uint32_t outputTiles = m / 128 * (n / 128);
        splitParts = std::min(COMPLEX_QR_SPLIT_MAX_PARTS, std::max(1U, cores / outputTiles));
        splitK = (k + 128 * splitParts - 1) / (128 * splitParts) * 128;
        splitParts = (k + splitK - 1) / splitK;
    }
}

uint64_t TileCost(const MatmulPlan& plan, uint32_t tm, uint32_t tn)
{
    if (tm == 0 || tn == 0 || plan.cores == 0)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t count = uint64_t((plan.m + tm - 1) / tm) * ((plan.n + tn - 1) / tn);
    return (count + plan.cores - 1) / plan.cores * tm * tn;
}

void SelectWaveTiles(MatmulPlan& plan)
{
    uint64_t best = TileCost(plan, plan.tileM, plan.tileN);
    for (uint32_t tm : {128U, 64U})
    {
        for (uint32_t tn : {128U, 64U})
        {
            if (tm == 64 && tn == 64)
            {
                continue;
            }
            const uint64_t candidate = TileCost(plan, tm, tn);
            if (candidate * 5 <= best * 4)
            {
                plan.tileM = tm;
                plan.tileN = tn;
                best = candidate;
            }
        }
    }
}

void SelectTiles(MatmulPlan& plan)
{
    auto& [m, n, k, cores, tileM, tileN, splitParts, splitK] = plan;
    auto tiles = [&plan]()
    { return uint64_t((plan.m + plan.tileM - 1) / plan.tileM) * ((plan.n + plan.tileN - 1) / plan.tileN); };
    if (splitParts == 1 && tiles() < (cores + 1) / 2)
    {
        if (m >= n)
        {
            tileM = 64;
        }
        else
        {
            tileN = 64;
        }
    }
    if (splitParts == 1 && tiles() < (cores + 1) / 2)
    {
        tileM = 64;
        tileN = 64;
    }
    // For a long reduction, 29-42 large tiles would leave most cores idle in
    // the second wave. Choose a smaller tile only when it reduces total waves
    // weighted by tile area; merely increasing the tile count is not enough.
    if (splitParts == 1 && k >= 1024)
    {
        SelectWaveTiles(plan);
    }
    if (splitParts == 1 && k <= 256 && m >= 512 && n >= 512 && m % 256 == 0 && n % 256 == 0)
    {
        tileM = 256;
        tileN = 256;
    }
}

void ConfigureReduction(optiling::TCubeTiling& cube, const MatmulPlan& plan,
                        platform_ascendc::PlatformAscendC* platform, bool interior)
{
    uint64_t l0a = 0, l0b = 0, l1 = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    const uint64_t aBytes = uint64_t(cube.get_baseM()) * 128 * sizeof(float);
    const uint64_t bBytes = uint64_t(cube.get_baseN()) * 128 * sizeof(float);
    if (interior && plan.splitParts == 1 && plan.k >= 1024 && plan.tileM <= 128 && plan.tileN <= 128 && aBytes <= l0a &&
        bBytes <= l0b && 2 * (aBytes + bBytes) <= l1)
    {
        // One 128-deep tile occupies at most 64 KiB per L0 operand;
        // two L1 tiles for each operand occupy at most 256 KiB total.
        cube.set_baseK(128);
        cube.set_dbL0A(1);
        cube.set_dbL0B(1);
        cube.set_stepKa(1);
        cube.set_stepKb(1);
        cube.set_depthA1(2);
        cube.set_depthB1(2);
    }
}

bool LaunchPart(matmul_tiling::MultiCoreMatmulTiling& api, const MatmulPlan& plan, const MatmulBuffers& buffers,
                platform_ascendc::PlatformAscendC* platform, bool interior)
{
    api.SetCType(interior ? matmul_tiling::TPosition::CO1 : matmul_tiling::TPosition::GM,
                 interior ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT);
    optiling::TCubeTiling cube{};
    if (api.GetTiling(cube) == -1)
    {
        return false;
    }
    ConfigureReduction(cube, plan, platform, interior);
    alignas(8) std::array<uint8_t, 1024> data{};
    if (cube.GetDataSize() > data.size())
    {
        return false;
    }
    cube.SaveToBuffer(data.data(), data.size());
    ComplexQrCompensatedMatmulLaunch(
        buffers.ah, buffers.al, buffers.bh, buffers.bl, plan.splitParts > 1 ? buffers.scratch : buffers.output, plan.m,
        plan.n, plan.k, data.data(), interior, plan.splitParts, plan.splitK, buffers.stream, buffers.transposeB);
    return true;
}
}  // namespace

bool ComplexQrCompensatedMatmul(const float* ah, const float* al, const float* bh, const float* bl, float* c,
                                uint32_t m, uint32_t n, uint32_t k, void* stream, float* splitScratch, bool transposeB)
{
    if (!m || !n || !k)
    {
        return false;
    }
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
    if (!platform)
    {
        return false;
    }
    MatmulPlan plan{m, n, k, ComplexQrCubeCores(), 128, 128, 1, k};
    SelectSplit(plan, splitScratch);
    SelectTiles(plan);
    MatmulBuffers buffers{ah, al, bh, bl, c, splitScratch, stream, transposeB};
    matmul_tiling::MultiCoreMatmulTiling api(*platform);
    api.SetDim(1);
    api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                 transposeB);
    api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    api.SetOrgShape(m, n, k);
    api.SetShape(plan.tileM, plan.tileN, plan.splitK);
    api.SetFixSplit(plan.tileM, plan.tileN, -1);
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);
    // Keep complete tiles in CO1; use the separate GM path only for tails.
    if (m >= plan.tileM && n >= plan.tileN && !LaunchPart(api, plan, buffers, platform, true))
    {
        return false;
    }
    if ((m % plan.tileM || n % plan.tileN) && !LaunchPart(api, plan, buffers, platform, false))
    {
        return false;
    }
    if (plan.splitParts > 1)
    {
        ComplexQrReduceSplit(splitScratch, c, m * n, plan.splitParts, stream);
    }
    return true;
}
