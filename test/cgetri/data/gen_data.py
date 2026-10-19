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

np.random.seed(42)

# 从命令行读取矩阵维度 n
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32 # 使用方式: python script.py <n>

# 生成随机复数矩阵 (complex64)
a_real = np.random.randn(n, n).astype(np.float32)  # 实部
a_imag = np.random.randn(n, n).astype(np.float32)  # 虚部
a = a_real + 1j * a_imag  # 组合成复数矩阵 (complex64)

# 先打乱行顺序，增加测试场景多样性
np.random.shuffle(a)

# 确保对角占优（复数情况）
for i in range(n):
    # 计算非对角线元素的绝对值之和（复数取模）
    row_sum = np.sum(np.abs(a[i, :])) - np.abs(a[i, i])
    # 调整对角线元素，使其模大于行其他元素模之和
    a[i, i] = (row_sum + np.random.rand() * 10) * (1 + 1j)  # 复数对角增强


# 计算逆矩阵 (complex64)
inv_a = np.linalg.inv(a).astype(np.complex64)

data_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(data_dir, "input", "A_gm.bin")
golden_path = os.path.join(data_dir, "output", "golden.bin")

os.makedirs(os.path.dirname(input_path), exist_ok=True)
os.makedirs(os.path.dirname(golden_path), exist_ok=True)

# 写入输入数据 (原始矩阵)
with open(input_path, "wb") as f:
    f.write(a.tobytes())

# 写入输出数据 (逆矩阵)
with open(golden_path, "wb") as f:
    f.write(inv_a.tobytes())

print(f"Generated {n}x{n} complex matrix and its inverse")

# # 可选：打印矩阵形状和部分数据用于验证
# print("Input matrix (first 3x3):\n", a[:3, :3])
# print("Inverse matrix (first 3x3):\n", inv_a[:3, :3])

# product = np.matmul(a, inv_a)
# identity = np.eye(n, dtype=np.complex64)

# # 计算误差
# abs_error = np.abs(product - identity)
# max_error = np.max(abs_error)
# mean_error = np.mean(abs_error)
# print("Verification results:")
# print(f"Max absolute error: {max_error:.3e}")
# print(f"Mean absolute error: {mean_error:.3e}")
# print("Product of A and inv(A) (first 3x3):\n", product[:3, :3])