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
import logging
import sys
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

TOLERANCE = 1e-4  # 绝对误差阈值
ERROR_TOL = 1e-4


def load_complex_matrix(filepath, n):
    """从二进制文件加载复数矩阵 (实虚 float32 交错)"""
    expected_size = n * n * 2 * 4  # 两个 float32 每个 4 字节
    if os.path.getsize(filepath) != expected_size:
        logging.warning(f"警告: 文件 {filepath} 大小不符预期 (预期 {expected_size} 字节)")
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size != n * n * 2:
        raise ValueError(f"文件 {filepath} 数据量不足: 需要 {n*n*2} 个 float32，实际 {data.size}")
    # 交错读取：每两个 float 组成一个复数
    real = data[0::2].reshape(n, n)
    imag = data[1::2].reshape(n, n)
    return real + 1j * imag


def check_complex_matrix(uplo, n):
    # 选择文件名
    if uplo == 'U':
        file_gm = "test/cpotrf/data/output/U_gm.bin"
        file_std = "test/cpotrf/data/output/U_std.bin"
    else:
        file_gm = "test/cpotrf/data/output/L_gm.bin"
        file_std = "test/cpotrf/data/output/L_std.bin"

    # 检查文件是否存在
    for f in (file_gm, file_std):
        if not os.path.exists(f):
            raise FileNotFoundError(f"错误: 文件 {f} 不存在")

    # 加载矩阵
    logging.info(f"正在加载矩阵: {file_gm} ...")
    A = load_complex_matrix(file_gm, n)
    logging.info(f"正在加载矩阵: {file_std} ...")
    B = load_complex_matrix(file_std, n)

    # 定义三角掩码
    if uplo == 'U':
        mask = np.triu(np.ones((n, n), dtype=bool))
    else:
        mask = np.tril(np.ones((n, n), dtype=bool))

    # 只考虑三角部分的元素
    indices = np.argwhere(mask)  # (行, 列) 数组
    total = len(indices)

    # 计算误差（复数模）
    diff = A - B
    abs_diff = np.abs(diff)

    # 判断合格
    qualified = abs_diff[mask] <= TOLERANCE
    num_qualified = np.sum(qualified)
    num_failed = total - num_qualified

    logging.info(f"\n比较结果 (uplo={uplo}):")
    logging.info(f"三角部分元素总数: {total}")
    logging.info(f"合格元素数: {num_qualified} ({num_qualified/total*100:.2f}%)")
    logging.info(f"不合格元素数: {num_failed} ({num_failed/total*100:.2f}%)")

    # 打印不合格示例（最多10个）
    if num_failed > 0:
        failed_indices = indices[~qualified]
        logging.info(f"\n不合格元素示例 (最多显示10个):")
        logging.info("  行  列      实部误差        虚部误差        误差模")
        count = 0
        for i, j in failed_indices:
            if count >= 10:
                break
            re_diff = diff[i, j].real
            im_diff = diff[i, j].imag
            mod = abs_diff[i, j]
            # 使用 1-indexed 位置
            logging.info(f"  {i+1:3d} {j+1:3d}   {re_diff:12.6e}   {im_diff:12.6e}   {mod:12.6e}")
            count += 1
        if num_failed > 10:
            logging.info(f"  ... 还有 {num_failed - 10} 个不合格元素未显示")
    else:
        logging.info("\n所有三角元素误差均在容忍范围内！")
    return num_failed / total <= ERROR_TOL


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    u_success = check_complex_matrix('U', n)
    l_success = check_complex_matrix('L', n)
    return u_success and l_success


if __name__ == "__main__":
    try:
        res = main()
        if not res:
            logging.info("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            logging.info("[Success] Case accuracy is verification passed.")
            sys.exit(0)
    except Exception as e:
        logging.info(e)
        sys.exit(1)
