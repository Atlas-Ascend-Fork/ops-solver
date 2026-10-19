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

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
)
from verify_utils import read_binary_strict  # noqa: E402

# for float32
relative_tol = 5e-3
absolute_tol = 5e-3
error_tol = 1e-4


def verify_result(A, invA, batchSize, N):
    A = read_binary_strict(A, np.complex64).reshape(batchSize, N, N)
    invA = read_binary_strict(invA, np.complex64).reshape(batchSize, N, N)

    # 计算每个batch的误差
    total_elements = 0
    total_different = 0

    for b in range(batchSize):
        output = np.dot(A[b], invA[b]).reshape(-1)
        golden = np.eye(N).astype(np.complex64).reshape(-1)
        different_element_results = np.isclose(
            output, golden, rtol=relative_tol, atol=absolute_tol, equal_nan=True
        )
        different_element_indexes = np.flatnonzero(~different_element_results)

        # 打印跨 batch 合计前 32 个不匹配的元素（issue #243）：原实现 total_different
        # 在 batch 循环后才累加，首个坏 batch 打印全部、后续 batch 一条不打，
        # 32 条上限形同虚设；改为明细循环内以局部计数器统一限流
        printed_this_run = 0
        for index in range(len(different_element_indexes)):
            if total_different + printed_this_run >= 32:
                break
            real_index = different_element_indexes[index]
            golden_data = golden[real_index]
            output_data = output[real_index]
            # 除零防护（issue #243）：golden 为单位阵，对角外位置为 0，相对差
            # 分母为 0 时改打印绝对差，避免 ZeroDivisionError 吞掉全部明细
            denom = abs(golden_data)
            if denom != 0:
                diff_desc = "rdiff: %.6f" % (abs(output_data - golden_data) / denom)
            else:
                diff_desc = "absdiff: %.6f" % abs(output_data - golden_data)
            print(
                "batch: %d, data index: %08d, expected: %.9f%+.9fj, actual: %.9f%+.9fj, %s"
                % (
                    b,
                    real_index,
                    golden_data.real,
                    golden_data.imag,
                    output_data.real,
                    output_data.imag,
                    diff_desc,
                )
            )
            printed_this_run += 1

        total_elements += golden.size
        total_different += different_element_indexes.size

    error_ratio = float(total_different) / total_elements
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        batchSize = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        res = verify_result(
            "./test/cmatinv_batched/data/input/A_gm.bin",
            "./test/cmatinv_batched/data/output/Ainv_gm.bin",
            batchSize,
            n,
        )

        if not res:
            print("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            print("[Success] Case accuracy verification passed.")
            sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
