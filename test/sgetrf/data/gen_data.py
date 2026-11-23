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

np.random.seed(149)

# 从命令行读取矩阵维度 m 和 n
m = int(sys.argv[1]) if len(sys.argv) > 1 else 32
# 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
if m <= 0:
    raise ValueError(f"m 必须是正整数, got {m}")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
# 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
if n <= 0:
    raise ValueError(f"n 必须是正整数, got {n}")

# 生成随机实数矩阵 (float32)
a = np.random.randint(1, 11, size=(m, n)).astype(np.float32)

# 输出路径相对脚本所在目录解析，避免受调用工作目录影响（issue #571，
# 对齐 sgetrs/cgetrs 已合入的 5810f1e/ea7ead4 修复口径）。
data_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(data_dir, "input"), exist_ok=True)

# 写入输入数据
with open(os.path.join(data_dir, "input/A_gm.bin"), "wb") as f:
    f.write(a.tobytes())

print(f"Generated {m}x{n} float matrix")

# # 可选：打印矩阵形状和部分数据用于验证
# print("Input matrix (first 3x3):\n", a[:3, :3])