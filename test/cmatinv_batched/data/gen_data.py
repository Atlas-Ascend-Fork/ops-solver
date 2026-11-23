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
import numpy as np

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

np.random.seed(42)

# 从命令行读取矩阵维度 n 和 batchSize
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
if n <= 0:
    raise ValueError(f"n 必须是正整数, got {n}")
batchSize = int(sys.argv[2]) if len(sys.argv) > 2 else 2
# 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
if batchSize <= 0:
    raise ValueError(f"batchSize 必须是正整数, got {batchSize}")

# 生成批量随机复数矩阵 (complex64)
a_real = np.random.randn(batchSize, n, n).astype(np.float32)
a_imag = np.random.randn(batchSize, n, n).astype(np.float32)
a = a_real + 1j * a_imag

# 确保每个矩阵对角占优（复数情况）
for b in range(batchSize):
    for i in range(n):
        # 计算非对角线元素的绝对值之和（复数取模）
        row_sum = np.sum(np.abs(a[b, i, :])) - np.abs(a[b, i, i])
        # 调整对角线元素，使其模大于行其他元素模之和
        a[b, i, i] = (row_sum + np.random.rand() * 10) * (1 + 1j)

# 计算每个batch的逆矩阵
inv_a = np.zeros((batchSize, n, n), dtype=np.complex64)
for b in range(batchSize):
    inv_a[b] = np.linalg.inv(a[b]).astype(np.complex64)

os.makedirs(os.path.dirname(f"{_D}input/A_gm.bin"), exist_ok=True)
os.makedirs(os.path.dirname(f"{_D}golden/Ainv_gm.bin"), exist_ok=True)

# 写入输入数据 (原始矩阵)
with open(f"{_D}input/A_gm.bin", "wb") as f:
    f.write(a.tobytes())

# 写入输出数据 (逆矩阵)
with open(f"{_D}golden/Ainv_gm.bin", "wb") as f:
    f.write(inv_a.tobytes())

print(f"Generated batch {batchSize} of {n}x{n} complex matrices and their inverses")


# # 可选：打印矩阵形状和部分数据用于验证
# print("Input matrix batch 0 (first 3x3):\n", a[0, :3, :3])
# print("Inverse matrix batch 0 (first 3x3):\n", inv_a[0, :3, :3])

# # 验证第一个batch
# product = np.matmul(a[0], inv_a[0])
# identity = np.eye(n, dtype=np.complex64)

# # 计算误差
# abs_error = np.abs(product - identity)
# max_error = np.max(abs_error)
# mean_error = np.mean(abs_error)
# print("Verification results for batch 0:")
# print(f"Max absolute error: {max_error:.3e}")
# print(f"Mean absolute error: {mean_error:.3e}")
# print("Product of A and inv(A) (first 3x3):\n", product[:3, :3])