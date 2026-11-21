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

import sys
import os
import numpy as np

np.random.seed(42)

# 从命令行读取矩阵维度 n
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32  # 使用方式: python script.py <n>
# 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
if n <= 0:
    raise ValueError(f"n 必须是正整数, got {n}")

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


data_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(data_dir, "input", "A_gm.bin")
# 测试二进制经 std::ofstream 写 output/A_gm.bin（ofstream 不自建目录），须预建 output 目录
output_dir = os.path.join(data_dir, "output")

os.makedirs(os.path.dirname(input_path), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 写入输入数据 (原始矩阵)。校验走 verify_result.py 的 A@Ainv≈I 残差式自洽校验，
# 不产出固定读取 NumPy 结果、无任何消费者的 golden.bin（对齐 sgetri 清理口径）。
with open(input_path, "wb") as f:
    f.write(a.tobytes())

print(f"Generated {n}x{n} complex matrix")
