#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys

import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
)
from lu_verify_utils import (  # noqa: E402
    compare_and_report,
    contract_warning,
    load_pivots,
    resolve_data_dir,
    run_cli,
    solve_lu,
)
from verify_utils import read_binary_strict  # noqa: E402

_D = resolve_data_dir(os.path.join("test", "cgetrs", "data"), __file__)

_DEFAULT_DIM = 32
# 接口契约 n、nrhs 上限（docs/zh/cgetrs.md，host 侧已强制）；
# 校验脚本自身同样提示，超限输入不做无意义的全量读入与求解
_CONTRACT_LIMIT = 8192


def load_data(n: int, nrhs: int) -> tuple:
    """读取 lu/rhs/piv/solution 二进制数据，piv 由 1-based 主元契约还原并校验"""
    lu = read_binary_strict(f"{_D}input/A_gm.bin", np.complex64).reshape(n, n)
    rhs = read_binary_strict(f"{_D}input/B_gm.bin", np.complex64).reshape(n, nrhs)
    piv = load_pivots(f"{_D}input/W_gm.bin", n)
    solution = read_binary_strict(f"{_D}output/X_gm.bin", np.complex64).reshape(n, nrhs)
    return lu, rhs, piv, solution


def main() -> bool:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DIM
    nrhs = int(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_DIM

    contract_warning(n, nrhs, _CONTRACT_LIMIT, "docs/zh/cgetrs.md")
    lu, rhs, piv, solution = load_data(n, nrhs)
    if not all(np.isfinite(array).all() for array in (lu, rhs, solution)):
        logging.info("Non-finite input or solution cannot pass accuracy verification.")
        return False

    # 计算 expected 解（整矩阵一次性求解，替代原逐列 Python 循环）
    reference = solve_lu(lu, rhs, piv)
    if not np.isfinite(reference).all():
        logging.info("Non-finite reference solution cannot pass accuracy verification.")
        return False

    return compare_and_report(reference, solution, n, nrhs)


if __name__ == "__main__":
    sys.exit(run_cli(main))
