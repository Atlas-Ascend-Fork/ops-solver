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
import numpy as np
import sys
import os

# 仅使用 numpy 的随机种子（与 C++ mt19937 rng(149); 一致）
np.random.seed(149)

def generate_data(m: int, n: int) -> np.ndarray:
    """
    生成 m × (2n) 的 float32 矩阵，每元素为 [1, 10] 的整数
    """
    return np.random.randint(1, 11, size=(m, n * 2)).astype(np.float32)

def main():
    # 支持命令行参数
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    # 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
    if m <= 0:
        raise ValueError(f"m 必须是正整数, got {m}")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    # 参数域校验（issue #466）：负数/零此前在 numpy 深处报 'negative dimensions'
    if n <= 0:
        raise ValueError(f"n 必须是正整数, got {n}")

    # 生成数据
    a = generate_data(m, n)

    # 确保输出目录存在
    output_dir = "./test/cgetrf/data/input"
    os.makedirs(output_dir, exist_ok=True)

    # 写入二进制文件
    filename = os.path.join(output_dir, "A_gm.bin")
    try:
        with open(filename, "wb") as f:
            f.write(a.tobytes())
        print(f"Generated {m} × {n} matrix")
    except Exception as e:
        print(f"ERROR: 无法写入文件 {filename}，错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()