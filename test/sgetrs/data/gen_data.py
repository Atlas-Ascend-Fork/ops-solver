#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys, os
import logging
import numpy as np
import scipy.linalg

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

np.random.seed(149)

# 从命令行读取矩阵维度 n 和右侧向量数量 nrhs
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
nrhs = int(sys.argv[2]) if len(sys.argv) > 2 else 32

# 创建输出目录
os.makedirs("./test/sgetrs/data/input", exist_ok=True)
os.makedirs("./test/sgetrs/data/output", exist_ok=True)

# 1. 生成随机可逆的 n×n 矩阵 A (float32)
np.random.seed(149)  # 与 sgetrf 使用相同种子以保证一致性
A = np.random.randint(1, 11, size=(n, n)).astype(np.float32)

# 确保矩阵可逆（通过对角占优）
for i in range(n):
    row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    A[i, i] = row_sum + np.random.rand() * 10

# 可选：随机打乱矩阵以增加测试覆盖
np.random.shuffle(A)

# 2. 生成随机右侧向量 B (float32) - 维度: n × nrhs
B = np.random.randn(n, nrhs).astype(np.float32)

# 3. 计算 LU 分解 (sgetrf 的等效操作)
# lu_factor 返回组合的 L 和 U 矩阵，以及主元索引 (pivots)
lu, piv = scipy.linalg.lu_factor(A)
LU = lu.astype(np.float32)

# 使用 0-based 索引（与 SciPy 返回的一致）
IPIV = piv.astype(np.int32)

# 4. 计算参考解 X = A^(-1) * B (sgetrs 的等效操作)
X_ref = scipy.linalg.lu_solve((lu, piv), B).astype(np.float32)

# --- 保存输入数据 ---
# 保存 LU 分解后的矩阵 (sgetrs 的输入 A)
with open("./test/sgetrs/data/input/A_gm.bin", "wb") as f:
    f.write(LU.tobytes())

# 保存主元索引数组 (sgetrs 的输入 IPIV) - 0-based
with open("./test/sgetrs/data/input/W_gm.bin", "wb") as f:
    f.write(IPIV.tobytes())

# 保存右侧向量 B (sgetrs 的输入 B)
with open("./test/sgetrs/data/input/B_gm.bin", "wb") as f:
    f.write(B.tobytes())

# --- 保存输出数据 (golden) ---
# 保存参考解向量 X (sgetrs 的输出，通常会覆盖 B)
with open("./test/sgetrs/data/output/golden.bin", "wb") as f:
    f.write(X_ref.tobytes())

# 打印信息用于验证
logging.info(f"Generated SGETRS test data for {n}x{n} matrix with {nrhs} RHS vectors")
logging.info(f"Input LU matrix (first 3x3):\n{LU[:3, :3]}")
logging.info(f"Input IPIV array (first 10 elements, 0-based):\n{IPIV[:10] if len(IPIV) > 10 else IPIV}")
logging.info(f"Input vector B (first 3 rows, first 3 columns):\n{B[:3, :3] if nrhs >= 3 else B[:3, :]}")
logging.info(f"Reference solution X (first 3 rows, first 3 columns):\n{X_ref[:3, :3] if nrhs >= 3 else X_ref[:3, :]}")

# 验证: 计算 A * X 应该等于 B
AX = np.matmul(A, X_ref)
abs_error = np.abs(AX - B)
max_error = np.max(abs_error)
mean_error = np.mean(abs_error)

logging.info("\nVerification results:")
logging.info(f"Max absolute error: {max_error:.3e}")
logging.info(f"Mean absolute error: {mean_error:.3e}")

# 验证 LU 分解的正确性
# 重建原始矩阵 A 从 LU 和 IPIV (0-based)
# 注意: scipy.linalg.lu_factor 返回的 LU 矩阵是包含 L 和 U 的组合形式
# 这里简化验证，仅检查条件数
logging.info(f"Condition number: {np.linalg.cond(A):.3e}")
