/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GETRS_TRSM_HELPERS_HPP
#define GETRS_TRSM_HELPERS_HPP

template<typename TMatmul>
__aicore__ inline void getrs_trsm_cube_l2_gemm(TMatmul &opmm, int rows, int curBlockM, int n, int L2M, int L2N,
    int L2K, int strideM, int strideN, int offsetA, int offsetB, int offsetC, int blockNum, int blockIdx, int trans) {
    for (int l2mIdx = 0; l2mIdx < rows; l2mIdx += L2M) {
        int curM = min(rows - l2mIdx, L2M);
        for (int l2nIdx = 0; l2nIdx < n; l2nIdx += L2N) {
            for (int l2kIdx = 0; l2kIdx < curBlockM; l2kIdx += L2K) {
                int gemmOffsetA = trans ? offsetA + l2kIdx * strideM + l2mIdx :
                    offsetA + l2mIdx * strideM + l2kIdx;
                opmm.Gemm(gemmOffsetA, offsetB + l2kIdx * strideN + l2nIdx,
                    offsetC + l2mIdx * strideN + l2nIdx, curM, min(n - l2nIdx, L2N), L2K,
                    blockNum, blockIdx, trans);
            }
        }
    }
}

template<typename TPipePool, typename TTrsm>
__aicore__ inline void getrs_trsm_vec_loop(TPipePool &pipe, TTrsm &optrsm, int totalBlocks, int blockM) {
    optrsm.Process(0);
    if (totalBlocks > 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(VEC_CUBE_FLAGID);
    for (int i = 1, offsetM = blockM; i < totalBlocks; ++i, offsetM += blockM) {
        CrossCoreWaitFlag(CUBE_VEC_FLAGID);
        optrsm.Process(offsetM);
        if (i != totalBlocks - 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(VEC_CUBE_FLAGID);
    }
    pipe.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
}

template<typename TPipePool, typename TTrsm>
__aicore__ inline void getrs_trsm_upper_vec_loop(TPipePool &pipe, TTrsm &optrsm, int totalBlocks, int blockM, int m,
    int orgM) {
    int maskLen = m - orgM;
    int mask = min(maskLen, blockM);
    optrsm.Process(m - blockM, blockM - mask);
    maskLen -= mask;
    if (totalBlocks > 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(VEC_CUBE_FLAGID);
    for (int i = 1, offsetM = blockM; i < totalBlocks; ++i, offsetM += blockM) {
        CrossCoreWaitFlag(CUBE_VEC_FLAGID);
        mask = min(maskLen, blockM);
        optrsm.Process(m - offsetM - blockM, blockM - mask);
        maskLen -= mask;
        if (i != totalBlocks - 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(VEC_CUBE_FLAGID);
    }
    pipe.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(ALL_VEC_FLAGID);
    CrossCoreWaitFlag(ALL_VEC_FLAGID);
}

template<typename TMatmul>
__aicore__ inline void getrs_trsm_cube_large(TMatmul &opmm, int offsetM, int curBlockM, int m, int n,
    int L2M, int L2N, int L2K, int strideM, int strideN, int blockNum, int blockIdx, int trans, int baseM,
    int &totalGemmBlock, int &lastGemmBoundary, int &nextGemmBoundary) {
    int offsetA = trans ? (offsetM - curBlockM) * strideM + offsetM : offsetM * strideM + offsetM - curBlockM;
    int offsetB = (offsetM - curBlockM) * strideN;
    int offsetC = offsetM * strideN;
    CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
    CrossCoreWaitFlag(ALL_CUBE_FLAGID);
    getrs_trsm_cube_l2_gemm(opmm, min(m - offsetM, curBlockM), curBlockM, n, L2M, L2N, L2K, strideM,
        strideN, offsetA, offsetB, offsetC, blockNum, blockIdx, trans);
    CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
    CrossCoreWaitFlag(ALL_CUBE_FLAGID);
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_VEC_FLAGID);
    totalGemmBlock = min(m - offsetM, L2M) / baseM;
    lastGemmBoundary = nextGemmBoundary;
    nextGemmBoundary += totalGemmBlock * baseM;
}

template<typename TMatmul>
__aicore__ inline void getrs_trsm_cube_prefetch(TMatmul &opmm, int offsetM, int baseM, int n, int strideM,
    int strideN, int blockNum, int blockIdx, int trans, int totalGemmBlock, int lastGemmBoundary,
    int nextGemmBoundary) {
    if (nextGemmBoundary - offsetM - baseM > 0) {
        int gemmOffsetA = trans ? (offsetM - baseM) * strideM + (offsetM + baseM) :
            (offsetM + baseM) * strideM + offsetM - baseM;
        int offsetB = (offsetM - baseM) * strideN;
        int offsetC = (offsetM + baseM) * strideN;
        opmm.GemmWorkGenerate(gemmOffsetA, offsetB, offsetC,
            min(nextGemmBoundary - offsetM - baseM, totalGemmBlock / 2 * baseM), n, baseM, blockNum, blockIdx, trans);
        if (nextGemmBoundary - offsetM - baseM < totalGemmBlock / 2 * baseM) {
            int gemmOffsetA2 = trans ? lastGemmBoundary * strideM + (offsetM + baseM) :
                (offsetM + baseM) * strideM + lastGemmBoundary;
            int offsetB = lastGemmBoundary * strideN;
            int offsetC = (offsetM + baseM) * strideN;
            if (offsetM - totalGemmBlock / 2 * baseM > 0) {
                opmm.GemmWorkGenerate(gemmOffsetA2, offsetB, offsetC, baseM, n,
                    offsetM - lastGemmBoundary - totalGemmBlock / 2 * baseM, blockNum, blockIdx, trans);
            }
        }
    }
}

template<typename TMatmul>
__aicore__ inline void getrs_trsm_cube(TMatmul &opmm, int m, int n, int blockM, int L2M, int L2N, int L2K,
    int strideM, int strideN, int blockNum, int blockIdx, int totalBlocks, int baseM, int trsmOffsetN,
    int realBlockN, int trans, int totalGemmBlock, int lastGemmBoundary, int nextGemmBoundary) {
    int remainingSteps = 0;
    for (int i = 1, offsetM = blockM; i < totalBlocks; ++i, offsetM += blockM) {
        int curBlockM = (i & -i) * blockM;
        CrossCoreWaitFlag(VEC_CUBE_FLAGID);
        if (curBlockM >= L2M) {
            getrs_trsm_cube_large(opmm, offsetM, curBlockM, m, n, L2M, L2N, L2K, strideM, strideN,
                blockNum, blockIdx, trans, baseM, totalGemmBlock, lastGemmBoundary, nextGemmBoundary);
        } else if (curBlockM < baseM) {
            int offsetA = trans ? (offsetM - curBlockM) * strideM + offsetM :
                offsetM * strideM + offsetM - curBlockM;
            int offsetB = (offsetM - curBlockM) * strideN;
            int offsetC = offsetM * strideN;
            opmm.Process(offsetA, offsetB + trsmOffsetN, offsetC + trsmOffsetN,
                min(m - offsetM, curBlockM), realBlockN, curBlockM, trans);
            CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_VEC_FLAGID);
            opmm.GemmWorkStep(remainingSteps--, trans);
        } else {
            int offsetA = trans ? (offsetM - baseM) * strideM + offsetM :
                offsetM * strideM + offsetM - baseM;
            int offsetB = (offsetM - baseM) * strideN;
            int offsetC = offsetM * strideN;
            CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
            CrossCoreWaitFlag(ALL_CUBE_FLAGID);
            opmm.Gemm(offsetA, offsetB, offsetC, baseM, n, baseM, blockNum, blockIdx, trans);
            CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
            CrossCoreWaitFlag(ALL_CUBE_FLAGID);
            CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_VEC_FLAGID);
            opmm.GemmWorkClear();
            getrs_trsm_cube_prefetch(opmm, offsetM, baseM, n, strideM, strideN, blockNum, blockIdx, trans,
                totalGemmBlock, lastGemmBoundary, nextGemmBoundary);
            remainingSteps = baseM / blockM;
            opmm.GemmWorkStep(remainingSteps--, trans);
        }
    }
}

template<typename TMatmul>
__aicore__ inline void getrs_trsm_upper_cube_large(TMatmul &opmm, int offsetM, int curBlockM, int m, int n,
    int L2M, int L2N, int L2K, int strideM, int strideN, int blockNum, int blockIdx, int trans, int baseM,
    int &totalGemmBlock, int &lastGemmBoundary, int &nextGemmBoundary) {
    int curM = min(m - offsetM, curBlockM);
    int flipRow = m - offsetM - curM;
    int flipCol = m - offsetM;
    int offsetA = trans ? flipCol * strideM + flipRow : flipRow * strideM + flipCol;
    int offsetB = flipCol * strideN;
    int offsetC = flipRow * strideN;
    CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
    CrossCoreWaitFlag(ALL_CUBE_FLAGID);
    getrs_trsm_cube_l2_gemm(opmm, curM, curBlockM, n, L2M, L2N, L2K, strideM, strideN, offsetA, offsetB,
        offsetC, blockNum, blockIdx, trans);
    CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
    CrossCoreWaitFlag(ALL_CUBE_FLAGID);
    CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_VEC_FLAGID);
    totalGemmBlock = min(m - offsetM, L2M) / baseM;
    lastGemmBoundary = nextGemmBoundary;
    nextGemmBoundary += totalGemmBlock * baseM;
}

template<typename TMatmul>
__aicore__ inline void getrs_trsm_upper_cube_prefetch(TMatmul &opmm, int m, int offsetM, int baseM, int n,
    int strideM, int strideN, int blockNum, int blockIdx, int trans, int totalGemmBlock, int lastGemmBoundary,
    int nextGemmBoundary) {
    if (nextGemmBoundary - offsetM - baseM > 0) {
        int curM_pf1 = min(nextGemmBoundary - offsetM - baseM, totalGemmBlock / 2 * baseM);
        int flipRow1 = m - (offsetM + baseM) - curM_pf1;
        int flipCol1 = m - offsetM;
        int offsetA1 = trans ? flipCol1 * strideM + flipRow1 : flipRow1 * strideM + flipCol1;
        int offsetB1 = flipCol1 * strideN;
        int offsetC1 = flipRow1 * strideN;
        opmm.GemmWorkGenerate(offsetA1, offsetB1, offsetC1, curM_pf1, n, baseM, blockNum, blockIdx, trans);
        if (nextGemmBoundary - offsetM - baseM < totalGemmBlock / 2 * baseM &&
            offsetM - totalGemmBlock / 2 * baseM > 0) {
            int curK_pf2 = offsetM - lastGemmBoundary - totalGemmBlock / 2 * baseM;
            int flipRow2 = m - (offsetM + baseM) - baseM;
            int flipCol2 = m - lastGemmBoundary - curK_pf2;
            int offsetA2 = trans ? flipCol2 * strideM + flipRow2 : flipRow2 * strideM + flipCol2;
            int offsetB2 = flipCol2 * strideN;
            int offsetC2 = flipRow2 * strideN;
            opmm.GemmWorkGenerate(offsetA2, offsetB2, offsetC2, baseM, n, curK_pf2, blockNum, blockIdx, trans);
        }
    }
}

template<typename TMatmul>
__aicore__ inline void getrs_trsm_upper_cube(TMatmul &opmm, int m, int n, int blockM, int L2M, int L2N, int L2K,
    int strideM, int strideN, int blockNum, int blockIdx, int totalBlocks, int baseM, int trsmOffsetN,
    int realBlockN, int trans, int totalGemmBlock, int lastGemmBoundary, int nextGemmBoundary) {
    int remainingSteps = 0;
    for (int i = 1, offsetM = blockM; i < totalBlocks; ++i, offsetM += blockM) {
        int curBlockM = (i & -i) * blockM;
        CrossCoreWaitFlag(VEC_CUBE_FLAGID);
        if (curBlockM >= L2M) {
            getrs_trsm_upper_cube_large(opmm, offsetM, curBlockM, m, n, L2M, L2N, L2K, strideM, strideN,
                blockNum, blockIdx, trans, baseM, totalGemmBlock, lastGemmBoundary, nextGemmBoundary);
        } else if (curBlockM < baseM) {
            int curM = min(m - offsetM, curBlockM);
            int flipRow = m - offsetM - curM;
            int flipCol = m - offsetM;
            int offsetA = trans ? flipCol * strideM + flipRow : flipRow * strideM + flipCol;
            int offsetB = flipCol * strideN;
            int offsetC = flipRow * strideN;
            opmm.Process(offsetA, offsetB + trsmOffsetN, offsetC + trsmOffsetN, curM, realBlockN, curBlockM, trans);
            CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_VEC_FLAGID);
            opmm.GemmWorkStep(remainingSteps--, trans);
        } else {
            int flipRow = m - offsetM - baseM;
            int flipCol = m - offsetM;
            int offsetA = trans ? flipCol * strideM + flipRow : flipRow * strideM + flipCol;
            int offsetB = flipCol * strideN;
            int offsetC = flipRow * strideN;
            CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
            CrossCoreWaitFlag(ALL_CUBE_FLAGID);
            opmm.Gemm(offsetA, offsetB, offsetC, baseM, n, baseM, blockNum, blockIdx, trans);
            CrossCoreSetFlag<0x0, PIPE_FIX>(ALL_CUBE_FLAGID);
            CrossCoreWaitFlag(ALL_CUBE_FLAGID);
            CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_VEC_FLAGID);
            opmm.GemmWorkClear();
            getrs_trsm_upper_cube_prefetch(opmm, m, offsetM, baseM, n, strideM, strideN, blockNum,
                blockIdx, trans, totalGemmBlock, lastGemmBoundary, nextGemmBoundary);
            remainingSteps = baseM / blockM;
            opmm.GemmWorkStep(remainingSteps--, trans);
        }
    }
}

#endif  // GETRS_TRSM_HELPERS_HPP
