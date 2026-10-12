/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

bool ProbesReconstruct(const std::vector<Complex>& probeValues, const std::vector<Complex>& reconstructed, int64_t n)
{
    constexpr double tolerance = 5.0e-3;
    for (int probe = 0; probe < ORTHOGONALITY_PROBE_COUNT; ++probe)
    {
        double errorSquared = 0.0;
        for (int64_t col = 0; col < n; ++col)
        {
            const size_t offset = static_cast<size_t>(probe * n + col);
            errorSquared += std::norm(reconstructed[offset] - probeValues[offset]);
        }
        if (std::sqrt(errorSquared) > tolerance)
        {
            return false;
        }
    }
    return true;
}

bool PassesOrthogonalityProbe(const std::vector<Complex>& vectors, int64_t n)
{
    const float inverseRootTwoN = 1.0f / std::sqrt(2.0f * static_cast<float>(n));
    const size_t probeElements = static_cast<size_t>(ORTHOGONALITY_PROBE_COUNT * n);
    std::vector<Complex> probeValues(probeElements);
    std::vector<Complex> transformed(probeElements);
    std::vector<Complex> reconstructed(probeElements);

    BuildOrthogonalityProbes(&probeValues, n, inverseRootTwoN);
    ApplyVectorsToProbes(vectors, probeValues, n, &transformed);
    ApplyAdjointToProbes(vectors, transformed, n, &reconstructed);
    return ProbesReconstruct(probeValues, reconstructed, n);
}

bool HasTightEigenvalueCluster(const float* eigenvalues, int64_t n)
{
    constexpr double clusterTolerance = 1.0e-5;
    double spectralScale = 1.0;
    for (int64_t index = 0; index < n; ++index)
    {
        spectralScale = std::max(spectralScale, std::abs(static_cast<double>(eigenvalues[index])));
    }
    for (int64_t index = 1; index < n; ++index)
    {
        if (std::abs(static_cast<double>(eigenvalues[index]) - eigenvalues[index - 1]) <=
            clusterTolerance * spectralScale)
        {
            return true;
        }
    }
    return false;
}

bool OrthonormalizeCluster(std::vector<Complex>* vectors, int64_t n, int64_t clusterBegin, int64_t clusterEnd)
{
    constexpr double minimumNorm = 1.0e-10;
    for (int64_t column = clusterBegin; column < clusterEnd; ++column)
    {
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int64_t previous = clusterBegin; previous < column; ++previous)
            {
                std::complex<double> dot(0.0, 0.0);
                for (int64_t row = 0; row < n; ++row)
                {
                    dot += std::conj(
                               static_cast<std::complex<double>>((*vectors)[static_cast<size_t>(row + previous * n)])) *
                           static_cast<std::complex<double>>((*vectors)[static_cast<size_t>(row + column * n)]);
                }
                const Complex coefficient(static_cast<float>(dot.real()), static_cast<float>(dot.imag()));
                for (int64_t row = 0; row < n; ++row)
                {
                    (*vectors)[static_cast<size_t>(row + column * n)] -=
                        coefficient * (*vectors)[static_cast<size_t>(row + previous * n)];
                }
            }
        }

        double normSquared = 0.0;
        for (int64_t row = 0; row < n; ++row)
        {
            normSquared +=
                std::norm(static_cast<std::complex<double>>((*vectors)[static_cast<size_t>(row + column * n)]));
        }
        if (normSquared <= minimumNorm)
        {
            return false;
        }
        const float inverseNorm = static_cast<float>(1.0 / std::sqrt(normSquared));
        for (int64_t row = 0; row < n; ++row)
        {
            (*vectors)[static_cast<size_t>(row + column * n)] *= inverseNorm;
        }
    }
    return true;
}

bool OrthonormalizeEigenvalueClusters(std::vector<Complex>* vectors, int64_t n, const float* eigenvalues)
{
    constexpr double clusterTolerance = 1.0e-5;
    double spectralScale = 1.0;
    for (int64_t index = 0; index < n; ++index)
    {
        spectralScale = std::max(spectralScale, std::abs(static_cast<double>(eigenvalues[index])));
    }
    int64_t clusterBegin = 0;
    while (clusterBegin < n)
    {
        int64_t clusterEnd = clusterBegin + 1;
        while (clusterEnd < n &&
               std::abs(static_cast<double>(eigenvalues[clusterEnd]) - eigenvalues[clusterEnd - 1]) <=
                   clusterTolerance * spectralScale)
        {
            ++clusterEnd;
        }
        if (clusterEnd - clusterBegin > 1 && !OrthonormalizeCluster(vectors, n, clusterBegin, clusterEnd))
        {
            return false;
        }
        clusterBegin = clusterEnd;
    }
    return true;
}

uint8_t* TakeWorkspaceSlice(uint8_t* base, size_t* offset, size_t bytes)
{
    uint8_t* slice = base + *offset;
    *offset += bytes;
    return slice;
}

struct FixedWorkspaceSizes
{
    int64_t matrixElems;
    size_t matrixBytes;
    size_t panelBytes;
    size_t vectorBytes;
    size_t panelWorkspaceBytes;
    size_t barrierBytes;
    size_t boundsBytes;
    size_t sixPlaneBytes;
};

// panel 工作区布局为 host/kernel 双编译域共用契约（issue #160）：
//   kernel 侧 cheevj_e2e_fixed.hpp 以 4*panelSize*MatrixN + 2*MatrixN 消费，
//   panelSize 按形状取 512->4 / 1024->8 / 2048->16（cheevj_e2e_fixed.hpp:104）。
// CHEEVJ_FIXED_PANEL_SIZE=16 是 panelSize 的全形状上界，二者经下方 static_assert
// 绑定；kernel 侧 panelSize 或布局公式上调时必须同步提高本常量，否则编译期即失败，
// 杜绝 host 欠分配导致的设备越界写静默发生。
static_assert(CHEEVJ_FIXED_PANEL_SIZE == 16,
              "panel workspace size must stay >= kernel-side panelSize max (2048 shape) "
              "and match the 4*SIZE*n+2*n layout consumed by cheevj_e2e_fixed.hpp (issue #160)");

FixedWorkspaceSizes GetFixedWorkspaceSizes(int64_t n)
{
    constexpr int64_t barrierElems = CHEEVJ_FIXED_PANEL_PARTICIPANTS * 16;
    constexpr int64_t boundsElems = 8;
    const int64_t matrixElems = n * n;
    const int64_t panelElems = n * CHEEVJ_FIXED_PANEL_WIDTH;
    // 布局公式与 kernel 侧 cheevj_e2e_fixed.hpp:130 逐项对应（issue #160）
    const int64_t panelWorkspaceElems = 4 * CHEEVJ_FIXED_PANEL_SIZE * n + 2 * n;
    const size_t matrixBytes = static_cast<size_t>(matrixElems) * sizeof(float);
    return {matrixElems,
            matrixBytes,
            static_cast<size_t>(panelElems) * sizeof(float),
            static_cast<size_t>(n) * sizeof(float),
            static_cast<size_t>(panelWorkspaceElems) * sizeof(float),
            static_cast<size_t>(barrierElems) * sizeof(int32_t),
            static_cast<size_t>(boundsElems) * sizeof(float),
            6 * matrixBytes};
}

struct FixedMatrixSlices
{
    uint8_t* real;
    uint8_t* imag;
    uint8_t* packedReal;
    uint8_t* packedImag;
    uint8_t* updateReal;
    uint8_t* updateImag;
};

FixedMatrixSlices TakeFixedMatrixSlices(uint8_t* workspace, size_t matrixBytes)
{
    size_t offset = 0;
    FixedMatrixSlices slices{};
    slices.real = TakeWorkspaceSlice(workspace, &offset, matrixBytes);
    slices.imag = TakeWorkspaceSlice(workspace, &offset, matrixBytes);
    slices.packedReal = TakeWorkspaceSlice(workspace, &offset, matrixBytes);
    slices.packedImag = TakeWorkspaceSlice(workspace, &offset, matrixBytes);
    slices.updateReal = TakeWorkspaceSlice(workspace, &offset, matrixBytes);
    slices.updateImag = TakeWorkspaceSlice(workspace, &offset, matrixBytes);
    return slices;
}

struct FixedPanelSlices
{
    uint8_t* panelVReal;
    uint8_t* panelVImag;
    uint8_t* panelWReal;
    uint8_t* panelWImag;
    uint8_t* wHReal;
    uint8_t* wHImag;
    uint8_t* wHImagNeg;
    uint8_t* diagonal;
    uint8_t* offDiagonal;
    uint8_t* tauReal;
    uint8_t* tauImag;
    uint8_t* panelWorkspace;
    uint8_t* barrierWorkspace;
    size_t nextOffset;
};

FixedPanelSlices TakeFixedPanelSlices(uint8_t* workspace, const FixedWorkspaceSizes& sizes)
{
    size_t offset = 0;
    FixedPanelSlices slices{};
    slices.panelVReal = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.panelVImag = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.panelWReal = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.panelWImag = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.wHReal = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.wHImag = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.wHImagNeg = TakeWorkspaceSlice(workspace, &offset, sizes.panelBytes);
    slices.diagonal = TakeWorkspaceSlice(workspace, &offset, sizes.vectorBytes);
    slices.offDiagonal = TakeWorkspaceSlice(workspace, &offset, sizes.vectorBytes);
    slices.tauReal = TakeWorkspaceSlice(workspace, &offset, sizes.vectorBytes);
    slices.tauImag = TakeWorkspaceSlice(workspace, &offset, sizes.vectorBytes);
    slices.panelWorkspace = TakeWorkspaceSlice(workspace, &offset, sizes.panelWorkspaceBytes);
    slices.barrierWorkspace = TakeWorkspaceSlice(workspace, &offset, sizes.barrierBytes);
    slices.nextOffset = offset;
    return slices;
}

void PackComplexPlanes(const std::vector<Complex>& packed, int64_t n, std::vector<float>* real,
                       std::vector<float>* imag)
{
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            const Complex value = packed[static_cast<size_t>(CompactColumnMajorIndex(row, col, n))];
            const size_t rowMajor = static_cast<size_t>(row * n + col);
            (*real)[rowMajor] = value.real();
            (*imag)[rowMajor] = value.imag();
        }
    }
}

void UnpackComplexPlanes(const std::vector<float>& real, const std::vector<float>& imag,
                         std::vector<Complex>* packed, int64_t n)
{
#pragma omp parallel for if (n >= HOST_OMP_PARALLEL_MIN_N * 2) num_threads(HOST_OMP_THREAD_NUM) schedule(static)
    for (int64_t col = 0; col < n; ++col)
    {
        for (int64_t row = 0; row < n; ++row)
        {
            const size_t columnMajor = static_cast<size_t>(col * n + row);
            (*packed)[columnMajor] = Complex(real[columnMajor], imag[columnMajor]);
        }
    }
}

aclError CopyFixedEigenvalues(uint8_t* deviceInfo, uint8_t* eigenvalues, size_t vectorBytes, float* w, int32_t* info)
{
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(info, sizeof(int32_t), deviceInfo, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
                       ((void)0));
    if (*info == 0)
    {
        CHEEVJ_CHECK_ACLRT(aclrtMemcpy(w, vectorBytes, eigenvalues, vectorBytes, ACL_MEMCPY_DEVICE_TO_HOST), ((void)0));
    }
    return ACL_SUCCESS;
}

aclError CopyFixedEigenvectors(const FixedMatrixSlices& matrix, const FixedWorkspaceSizes& sizes,
                               std::vector<Complex>* packedA, int64_t n)
{
    std::vector<float> outputReal(static_cast<size_t>(sizes.matrixElems));
    std::vector<float> outputImag(static_cast<size_t>(sizes.matrixElems));
    CHEEVJ_CHECK_ACLRT(
        aclrtMemcpy(outputReal.data(), sizes.matrixBytes, matrix.imag, sizes.matrixBytes, ACL_MEMCPY_DEVICE_TO_HOST),
        ((void)0));
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(outputImag.data(), sizes.matrixBytes, matrix.packedReal, sizes.matrixBytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST),
                       ((void)0));
    UnpackComplexPlanes(outputReal, outputImag, packedA, n);
    return ACL_SUCCESS;
}

aclError LaunchCompleteVector(const FixedMatrixSlices& matrix, const FixedPanelSlices& panel, uint8_t* reflectorReal,
                              uint8_t* reflectorImag, uint8_t* bounds, uint8_t* eigenvalues,
                              uint8_t* commandWorkspace, uint8_t* deviceInfo, int64_t n, aclrtStream stream)
{
    cheevj_fixed_kernel_do(matrix.real, matrix.imag, matrix.packedReal, matrix.packedImag, panel.panelVReal,
                           panel.panelVImag, panel.panelWReal, panel.panelWImag, panel.wHReal, panel.wHImag,
                           panel.wHImagNeg, matrix.updateReal, matrix.updateImag, panel.diagonal, panel.offDiagonal,
                           panel.tauReal, panel.tauImag, panel.panelWorkspace, panel.barrierWorkspace, reflectorReal,
                           reflectorImag, bounds, eigenvalues, matrix.packedImag, matrix.updateReal, matrix.real,
                           commandWorkspace, matrix.imag, matrix.packedReal, deviceInfo, static_cast<int>(n), true,
                           stream);
    CHEEVJ_CHECK_ACLRT(aclrtSynchronizeStream(stream), ((void)0));
    return ACL_SUCCESS;
}

aclError RunCompleteNoVector(const std::vector<Complex>& packedA, int64_t n, float* w, int32_t* info,
                             aclrtStream stream)
{
    const FixedWorkspaceSizes sizes = GetFixedWorkspaceSizes(n);

    // The fixed-panel correctness chain keeps the original, packed trailing
    // matrix, and HER2K update as planar real/imag pairs.  These are exactly
    // the six matrix planes already budgeted by the public jobz=N path.
    const size_t auxiliaryBytes = 7 * sizes.panelBytes + 6 * sizes.vectorBytes + sizes.panelWorkspaceBytes +
                                  sizes.barrierBytes + sizes.boundsBytes;

    std::vector<float> matrixReal(static_cast<size_t>(sizes.matrixElems));
    std::vector<float> matrixImag(static_cast<size_t>(sizes.matrixElems));
    PackComplexPlanes(packedA, n, &matrixReal, &matrixImag);

    uint8_t* matrixWorkspace = nullptr;
    uint8_t* auxiliaryWorkspace = nullptr;
    uint8_t* eigenvalues = nullptr;
    uint8_t* deviceInfo = nullptr;
    // Release all four device buffers on any allocation, memset, memcpy, or
    // synchronize failure instead of returning with them still held (#121).
    auto releaseFixedBuffers = [&]() {
        if (matrixWorkspace != nullptr)
        {
            (void)aclrtFree(matrixWorkspace);
            matrixWorkspace = nullptr;
        }
        if (auxiliaryWorkspace != nullptr)
        {
            (void)aclrtFree(auxiliaryWorkspace);
            auxiliaryWorkspace = nullptr;
        }
        if (eigenvalues != nullptr)
        {
            (void)aclrtFree(eigenvalues);
            eigenvalues = nullptr;
        }
        if (deviceInfo != nullptr)
        {
            (void)aclrtFree(deviceInfo);
            deviceInfo = nullptr;
        }
    };
    CHEEVJ_CHECK_ACLRT(
        aclrtMalloc(reinterpret_cast<void**>(&matrixWorkspace), sizes.sixPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(
        aclrtMalloc(reinterpret_cast<void**>(&auxiliaryWorkspace), auxiliaryBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(
        aclrtMalloc(reinterpret_cast<void**>(&eigenvalues), sizes.vectorBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMalloc(reinterpret_cast<void**>(&deviceInfo), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                       releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemset(matrixWorkspace, sizes.sixPlaneBytes, 0, sizes.sixPlaneBytes),
                       releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemset(auxiliaryWorkspace, auxiliaryBytes, 0, auxiliaryBytes), releaseFixedBuffers());

    const FixedMatrixSlices matrix = TakeFixedMatrixSlices(matrixWorkspace, sizes.matrixBytes);
    const FixedPanelSlices panel = TakeFixedPanelSlices(auxiliaryWorkspace, sizes);
    size_t auxiliaryOffset = panel.nextOffset;
    uint8_t* bounds = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.boundsBytes);
    uint8_t* lowWorkspace = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.vectorBytes);
    uint8_t* highWorkspace = TakeWorkspaceSlice(auxiliaryWorkspace, &auxiliaryOffset, sizes.vectorBytes);

    CHEEVJ_CHECK_ACLRT(
        aclrtMemcpy(matrix.real, sizes.matrixBytes, matrixReal.data(), sizes.matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(
        aclrtMemcpy(matrix.imag, sizes.matrixBytes, matrixImag.data(), sizes.matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        releaseFixedBuffers());
    CHEEVJ_CHECK_ACLRT(aclrtMemcpy(deviceInfo, sizeof(int32_t), info, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE),
                       releaseFixedBuffers());

    // All numeric stages are enqueued through one fixed-shape dispatcher.
    // The stage launches remain separate because AIV, AIC, and global
    // synchronization boundaries have different execution contracts.
    cheevj_fixed_kernel_do(matrix.real, matrix.imag, matrix.packedReal, matrix.packedImag, panel.panelVReal,
                           panel.panelVImag, panel.panelWReal, panel.panelWImag, panel.wHReal, panel.wHImag,
                           panel.wHImagNeg, matrix.updateReal, matrix.updateImag, panel.diagonal, panel.offDiagonal,
                           panel.tauReal, panel.tauImag, panel.panelWorkspace, panel.barrierWorkspace, nullptr, nullptr,
                           bounds, eigenvalues, lowWorkspace, highWorkspace, nullptr, nullptr, nullptr, nullptr,
                           deviceInfo, static_cast<int>(n), false, stream);
    CHEEVJ_CHECK_ACLRT(aclrtSynchronizeStream(stream), releaseFixedBuffers());

    const aclError copyStatus = CopyFixedEigenvalues(deviceInfo, eigenvalues, sizes.vectorBytes, w, info);
    if (copyStatus != ACL_SUCCESS)
    {
        releaseFixedBuffers();
        return copyStatus;
    }

    // 成功释放后立即置空（issue #186）：后续 aclrtFree 失败进入 releaseFixedBuffers
    // 清理时，已成功释放的地址因置空不会再次进入 aclrtFree，避免重复释放
    CHEEVJ_CHECK_ACLRT(aclrtFree(matrixWorkspace), releaseFixedBuffers());
    matrixWorkspace = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(auxiliaryWorkspace), releaseFixedBuffers());
    auxiliaryWorkspace = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(eigenvalues), releaseFixedBuffers());
    eigenvalues = nullptr;
    CHEEVJ_CHECK_ACLRT(aclrtFree(deviceInfo), releaseFixedBuffers());
    deviceInfo = nullptr;
    return ACL_SUCCESS;
}
