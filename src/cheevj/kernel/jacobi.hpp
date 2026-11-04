/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file jacobi.hpp
 * \brief Device-side complex Hermitian Jacobi eigensolver component.
 */

#ifndef CHEEVJ_C64_JACOBI_HPP
#define CHEEVJ_C64_JACOBI_HPP

#include <cstdint>

#include "cheevj_ascendc_symbols.hpp"
#include "cheevj_block_jacobi.hpp"
#include "cheevj_diagonal.hpp"
#include "cheevj_output.hpp"
#include "cheevj_pair_update.hpp"
#include "cheevj_stage_block_jacobi16.hpp"
#include "cheevj_workspace.hpp"
#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

namespace Cheevj
{

constexpr int CHEEVJ_DEVICE_MAX_N = 32;
constexpr int CHEEVJ_MATRIX_ELEMS = CHEEVJ_DEVICE_MAX_N * CHEEVJ_DEVICE_MAX_N;
constexpr int CHEEVJ_COMPLEX_FLOATS = CHEEVJ_MATRIX_ELEMS * 2;
constexpr int CHEEVJ_W_FLOATS = CHEEVJ_DEVICE_MAX_N;
// RunJacobi 本地迭代循环的 sweep 上限（jacobi_local_iteration.inc）；与 host 经 tiling
// 下发的 CHEEVJ_HOST_TILING_SWEEPS=32 相互独立、用途不同（issue #316，两处曾重名同值不同）。
constexpr int CHEEVJ_LOCAL_ITERATION_MAX_SWEEPS = 120;
constexpr int CHEEVJ_LARGE_SCRATCH_PLANES_N = 3;
constexpr int CHEEVJ_LARGE_SCRATCH_PLANES_V = 5;
constexpr int CHEEVJ_LARGE_NAIVE_MAX_N = 32;
constexpr int CHEEVJ_LARGE_NAIVE_MAX_SWEEPS = 96;
constexpr int CHEEVJ_LARGE_BLOCK_OUTER_SWEEPS = 8;
constexpr int CHEEVJ_STAGE_BLOCK16_OUTER_SWEEPS = 32;
constexpr int CHEEVJ_OFFDIAG_REDUCE_TILE = 8192;
constexpr float CHEEVJ_REL_TOL = 1.0e-6f;
constexpr float CHEEVJ_MIN_PIVOT = 1.0e-20f;
constexpr float CHEEVJ_CLUSTER_TOL = 1.0e-3f;
constexpr float CHEEVJ_LARGE_CONV_TOL = 1.0e-3f;

// GM workspace layout for the tiled path:
//   [0, n*n)         A_real
//   [n*n, 2*n*n)     A_imag
//   [2*n*n, 3*n*n)   V_real, only when jobz=V
//   [3*n*n, 4*n*n)   V_imag, only when jobz=V

struct Complex32
{
    float real;
    float imag;
};

__aicore__ inline Complex32 MakeComplex(float real, float imag)
{
    Complex32 value{real, imag};
    return value;
}

__aicore__ inline float AbsFloat(float value) { return value >= 0.0f ? value : -value; }

__aicore__ inline float MaxFloat(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }

__aicore__ inline float SqrtApprox(float value)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    float root = value > 1.0f ? value : 1.0f;
    for (int iter = 0; iter < 24; ++iter)
    {
        root = 0.5f * (root + value / root);
    }
    return root;
}

__aicore__ inline Complex32 ComplexConj(Complex32 value) { return MakeComplex(value.real, -value.imag); }

__aicore__ inline float ComplexAbs(Complex32 value)
{
    return SqrtApprox(value.real * value.real + value.imag * value.imag);
}

__aicore__ inline bool IsVectorMode(int jobz)
{
    return jobz == 1 || jobz == static_cast<int>('V') || jobz == static_cast<int>('v');
}

__aicore__ inline bool IsLowerMode(int uplo)
{
    // 与 host 侧 IsLowerMode 口径一致：仅接受枚举与 'L'/'l'，不收 121/122 魔数
    // （122=Lower 的历史映射与 PLASMA/MAGMA 相反，issue #155）
    return uplo == 0 || uplo == static_cast<int>('L') || uplo == static_cast<int>('l');
}

class CheevjDeviceKernel
{
   public:
    __aicore__ inline CheevjDeviceKernel() {}

    __aicore__ inline void Process(GM_ADDR a, GM_ADDR w, GM_ADDR info, GM_ADDR workspace, int n, int lda, int jobz,
                                   int uplo, bool knownDiagonal, int maxSweeps)
    {
        (void)lda;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));
        infoGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(info));

        if (n <= 0)
        {
            if (GetBlockIdx() == 0)
            {
                infoGlobal.SetValue(0, -4);
            }
            return;
        }
        if (n > CHEEVJ_DEVICE_MAX_N)
        {
            ProcessLargeInput(a, w, workspace, n, lda, jobz, uplo, knownDiagonal, maxSweeps);
            return;
        }

        if (GetBlockIdx() != 0)
        {
            return;
        }

        ProcessLocalInput(n, jobz, uplo);
    }

   private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf;
    TBuf<QuePosition::VECCALC> outputBuf;
    TBuf<QuePosition::VECCALC> matrixRealBuf;
    TBuf<QuePosition::VECCALC> matrixImagBuf;
    TBuf<QuePosition::VECCALC> vectorRealBuf;
    TBuf<QuePosition::VECCALC> vectorImagBuf;
    TBuf<QuePosition::VECCALC> wBuf;
    TBuf<QuePosition::VECCALC> temp0Buf;
    TBuf<QuePosition::VECCALC> temp1Buf;
    TBuf<QuePosition::VECCALC> temp2Buf;
    TBuf<QuePosition::VECCALC> temp3Buf;
    TBuf<QuePosition::VECCALC> temp4Buf;
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> wGlobal;
    GlobalTensor<int32_t> infoGlobal;

#include "jacobi_large_impl_part1.inc"
#include "jacobi_large_impl_part2.inc"
#include "jacobi_local_helpers.inc"
#include "jacobi_local_iteration.inc"
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_JACOBI_HPP
