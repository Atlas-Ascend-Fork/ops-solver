/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <acl/acl.h>

#include <cstdint>

#if defined(__CCE__) && !defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#define CHO_GM __gm__
#else
#define CHO_GM
#endif
namespace Cholesky
{
constexpr int PANEL = 64;
unsigned VectorCores();
unsigned CubeCores();
struct MatrixBatch
{
    CHO_GM float *single;
    CHO_GM float *CHO_GM *array;
    int n;
    int ld;
    int upper;
    int first;
    int count;
};
void SetInfo(int *info, int value, int count, aclrtStream stream);
void SetSolveInfo(int *info, float *a[], float *b[], int count, aclrtStream stream);
void SmallFactor(const MatrixBatch &a, int *info, aclrtStream stream);
void SmallFactorVector(const MatrixBatch &a, int *info, aclrtStream stream);
void MediumFactorStage(const MatrixBatch &a, float *product, int productPitch, bool finalPhase, int *info,
                       aclrtStream stream);
void MediumGram(const MatrixBatch &a, float *product, int productPitch, int *info, aclrtStream stream);
void LeftFactorStage(const MatrixBatch &a, float *product, int productRows, int start, int *info, aclrtStream stream);
void LeftGram(const MatrixBatch &a, float *product, int productRows, int start, int *info, aclrtStream stream);
void Pack(const MatrixBatch &a, float *w, int pitch, int *info, aclrtStream stream);
void Panel(float *w, int pitch, int n, int start, int count, int *info, aclrtStream stream);
void PanelSolve(float *w, int pitch, int n, int start, int count, int *info, aclrtStream stream);
void Gram(float *w, float *product, int pitch, int n, int start, int width, int count, int *info, unsigned cores,
          aclrtStream stream);
void Update(float *w, float *product, int pitch, int n, int end, int count, int *info, aclrtStream stream);
void Unpack(const MatrixBatch &a, const float *w, int pitch, const int *info, aclrtStream stream);
void Solve(const MatrixBatch &a, const MatrixBatch &b, int nrhs, int *info, aclrtStream stream,
           float *product = nullptr, int productPitch = 0);
void SmallSolve(const MatrixBatch &a, const MatrixBatch &b, int nrhs, int *info, aclrtStream stream);
void SolvePanel(const MatrixBatch &a, const MatrixBatch &b, int nrhs, int start, int width, bool reverse, int *info,
                aclrtStream stream);
void SolveUpdateVector(const MatrixBatch &a, const MatrixBatch &b, int start, int width, bool reverse, int *info,
                       aclrtStream stream);
void Invert(const MatrixBatch &a, float *w, int pitch, int *info, aclrtStream stream);
void RhsUpdate(const MatrixBatch &a, const MatrixBatch &b, int nrhs, float *product, int productPitch, int start,
               int width, bool reverse, int *info, unsigned cores, aclrtStream stream);
void TriUpdate(float *g, float *b, float *product, int pitch, int n, int start, int width, bool reverse, int *info,
               unsigned cores, aclrtStream stream);
}  // namespace Cholesky
