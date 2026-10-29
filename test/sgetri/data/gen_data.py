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

# 生成随机实数矩阵 (float32)
a = np.random.randn(n, n).astype(np.float32)

# 先打乱行顺序，增加测试场景多样性
np.random.shuffle(a)

# 确保对角占优（实数情况）
for i in range(n):
    # 计算非对角线元素的绝对值之和
    row_sum = np.sum(np.abs(a[i, :])) - np.abs(a[i, i])
    # 调整对角线元素，使其绝对值大于行其他元素绝对值之和
    a[i, i] = row_sum + np.random.rand() * 10


os.makedirs(os.path.dirname("./test/sgetri/data/input/A_gm.bin"), exist_ok=True)

# 写入输入数据 (原始矩阵)
with open("./test/sgetri/data/input/A_gm.bin", "wb") as f:
    f.write(a.tobytes())

print(f"Generated {n}x{n} float matrix")
