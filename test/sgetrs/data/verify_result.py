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
import os
import sys

import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
)
from lu_verify_utils import (  # noqa: E402
    contract_warning,
    load_pivots,
    resolve_data_dir,
    run_cli,
    verify_trans_cases,
)
from verify_utils import read_binary_strict  # noqa: E402

_D = resolve_data_dir(os.path.join("test", "sgetrs", "data"), __file__)

_DEFAULT_DIM = 32
# 接口契约 n、nrhs 上限（docs/zh/sgetrs.md，host 侧已强制）；
# 校验脚本自身同样提示，超限输入不做无意义的全量读入与求解
_CONTRACT_LIMIT = 8192
# trans 档清单（issue #637）：实数接口文档承诺 0、1 两档，与测试二进制逐档落盘对齐
_TRANS_LIST = (0, 1)


def load_inputs(n: int, nrhs: int) -> tuple:
    """读取 lu/b/piv 二进制数据，piv 由 1-based 主元契约还原并校验"""
    lu = read_binary_strict(f"{_D}input/A_gm.bin", np.float32).reshape(n, n)
    b = read_binary_strict(f"{_D}input/B_gm.bin", np.float32).reshape(n, nrhs)
    piv = load_pivots(f"{_D}input/W_gm.bin", n)
    return lu, b, piv


def main() -> bool:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DIM
    nrhs = int(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_DIM

    contract_warning(n, nrhs, _CONTRACT_LIMIT, "docs/zh/sgetrs.md")
    lu, b, piv = load_inputs(n, nrhs)
    if not all(np.isfinite(array).all() for array in (lu, b)):
        logging.info("Non-finite input cannot pass accuracy verification.")
        return False

    # 逐 trans 档读算子输出并与参考解比对（issue #637）
    return verify_trans_cases(
        lu, b, piv, n, nrhs, f"{_D}output", np.float32, _TRANS_LIST
    )


if __name__ == "__main__":
    sys.exit(run_cli(main))
