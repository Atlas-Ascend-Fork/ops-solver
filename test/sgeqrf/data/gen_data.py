#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
#
# Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
# All rights reserved.
#
# This software is dual-licensed:
#   1) Community license: use, modify, and redistribute freely under the
#      GNU General Public License v3 or later (GPL-3.0-or-later);
#   2) Commercial license: for closed-source or commercial use / redistribution,
#      contact the copyright holder for a separate license.
# You may choose either. Default terms are GPL-3.0-or-later.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you may redistribute it and/or modify it under
# the terms of the GPL. It is distributed "as is", without warranty.
# See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
# Commercial licensing contact: yuantang@fudan.edu.cn
#
# -------------------------------------------------------------------------
# 中文版 (Chinese):
#
# PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
#
# 本软件采用双重许可（dual licensing）：
#   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
#      （GPL-3.0-or-later）自由使用、修改与再分发；
#   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
# 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
#
# 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
# 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
# 商业授权联系：yuantang@fudan.edu.cn
import logging
import os
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def main():
    # 解析命令行参数
    rows = 1024 if len(sys.argv) < 2 else int(sys.argv[1])
    cols = 256 if len(sys.argv) < 3 else int(sys.argv[2])
    
    # 创建目录（如果不存在）
    output_dir = "./test/sgeqrf/data/input"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "matrix.bin")
    
    # 生成随机矩阵，元素服从标准正态分布（float32）
    np.random.seed(42)          # 固定随机种子保证可复现
    matrix = np.random.randn(rows, cols).astype(np.float32)
    
    # 保存为二进制文件（按行优先，C 连续）
    with open(output_file, "wb") as f:
        f.write(matrix.tobytes())
    
    logging.info(f"Generated {rows}x{cols} matrix, saved to {output_file}")
    logging.info(f"File size: {os.path.getsize(output_file)} bytes (expected {rows*cols*4})")


if __name__ == "__main__":
    main()