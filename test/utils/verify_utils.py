# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared accuracy verification helpers for solver matrix-inverse ST scripts."""

import logging
import sys

import os

import numpy as np

# 输出格式与既有验证脚本保持一致（纯消息文本，无日志级别前缀）
logging.basicConfig(level=logging.INFO, format="%(message)s")

RELATIVE_TOL = 5e-3
ABSOLUTE_TOL = 5e-3
ERROR_TOL = 1e-4
MAX_PRINTED_MISMATCHES = 32
DEFAULT_MATRIX_ORDER = 32
DTYPE_FLOAT32 = np.float32
DTYPE_COMPLEX64 = np.complex64


def read_binary_strict(path, dtype):
    """按 dtype 读取二进制文件，尾部不足一个完整元素时直接报错（issue #202）。

    numpy.fromfile 会静默丢弃末尾不组成完整元素的字节，损坏文件（截断或
    追加脏字节）会被后续 reshape/数量校验误判为成功。这里显式校验文件
    字节数必须是元素大小的整数倍。
    """
    file_size = os.path.getsize(path)
    element_size = np.dtype(dtype).itemsize
    if file_size % element_size != 0:
        raise ValueError(
            f"corrupted binary file {path}: size {file_size} is not a multiple of "
            f"element size {element_size} for dtype {np.dtype(dtype).name}"
        )
    return np.fromfile(path, dtype=dtype)


def verify_inverse(input_path, inv_path, order, dtype, format_mismatch):
    """校验 inv_path 矩阵是否为 input_path 矩阵的逆（A @ Ainv ≈ I）。

    差异元素由 format_mismatch(index, expected, actual) 负责格式化记录，
    最多记录 MAX_PRINTED_MISMATCHES 条；返回是否满足误差容限。
    """
    a = read_binary_strict(input_path, dtype).reshape(order, order)
    inv_a = read_binary_strict(inv_path, dtype).reshape(order, order)
    if not np.isfinite(a).all() or not np.isfinite(inv_a).all():
        logging.info(
            "Non-finite input or inverse output cannot pass accuracy verification."
        )
        return False
    output = np.dot(a, inv_a).reshape(-1)
    if not np.isfinite(output).all():
        logging.info("Non-finite inverse product cannot pass accuracy verification.")
        return False
    golden = np.eye(order, dtype=dtype).reshape(-1)
    close = np.isclose(
        output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL, equal_nan=True
    )
    mismatch_indexes = np.flatnonzero(~close)
    for printed, index in enumerate(mismatch_indexes):
        if printed >= MAX_PRINTED_MISMATCHES:
            break
        logging.info("%s", format_mismatch(index, golden[index], output[index]))
    error_ratio = mismatch_indexes.size / golden.size
    logging.info("error ratio: %.4f, tolerance: %.4f", error_ratio, ERROR_TOL)
    return error_ratio <= ERROR_TOL


def run_verification(input_path, inv_path, dtype, format_mismatch):
    """ST 验证入口：可选命令行参数指定矩阵阶数（默认 32），返回进程退出码。"""
    try:
        order = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MATRIX_ORDER
        passed = verify_inverse(input_path, inv_path, order, dtype, format_mismatch)
    except Exception as err:
        logging.error("%s", err)
        return 1
    if passed:
        logging.info("[Success] Case accuracy verification passed.")
        return 0
    logging.info("[Failed] Case accuracy verification failed.")
    return 1
