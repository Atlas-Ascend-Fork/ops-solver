# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""getrs 系 verify 共享工具：行分块三角求解与向量化精度比对（issue #612）"""

import logging
import os
from collections.abc import Callable

import numpy as np

from verify_utils import read_binary_strict

EPS = 1e-3
ERROR_TOL = 1e-4

# 行分块尺寸：块间用整块矩阵乘、块内行推进只读块内数据，
# 避免 8192 规格下每行重读整个 (n, nrhs) 累积矩阵造成的 DRAM 带宽墙
_TRI_BLOCK = 256
# 相对误差分母兜底阈值与失败样本打印条数（与既有口径一致）
_DENOM_FLOOR = 1e-12
_MAX_BAD_PRINT = 32


def resolve_data_dir(sub_dir: str, caller_file: str) -> str:
    """数据目录解析：仓根 cwd 布局优先（make test 入口），否则回退脚本所在 data 目录"""
    cwd_dir = os.path.join(sub_dir, "")
    if os.path.isdir(cwd_dir):
        return cwd_dir
    return os.path.join(os.path.dirname(os.path.abspath(caller_file)), "")


def load_pivots(path: str, n: int) -> np.ndarray:
    """读取 W_gm.bin：1-based 主元契约还原 0-based 供行交换使用，并做越界内容校验。

    内容校验对齐同族 getrf verify 形态（issue #460）：越界值此前抛裸 IndexError，
    0 值经 -1 后为合法负索引静默回绕交换末行、以错误基准误判结果（issue #482）。
    """
    raw_piv = read_binary_strict(path, np.int32).reshape(n)
    for i, pivot in enumerate(raw_piv):
        if not (i + 1 <= pivot <= n):
            raise ValueError(
                f"invalid pivot at index {i}: {pivot}, expected [{i + 1}, {n}]"
            )
    return raw_piv - 1


def contract_warning(n: int, nrhs: int, limit: int, doc: str) -> None:
    """接口契约 n、nrhs 上限提示（docs 已文档化且 host 侧已强制）"""
    if n > limit or nrhs > limit:
        logging.warning(
            "warning: n=%d, nrhs=%d exceed the interface contract limit %d (%s).",
            n,
            nrhs,
            limit,
            doc,
        )


def forward_substitution(lu: np.ndarray, b: np.ndarray, piv: np.ndarray) -> np.ndarray:
    """Ly = Pb (前向替换，右端列整体向量化 + 行分块)"""
    n = lu.shape[0]
    y = b.copy()
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # 应用 pivot（按 LAPACK 顺序逐对行交换，行交换与右端列无关）
    for i, p in enumerate(piv):
        y[[i, p], :] = y[[p, i], :]

    # 解 Ly = Pb：行间数据依赖按块推进，块间整块矩阵乘、块内逐行更新
    # （原双重循环在 n=8192 规格下为小时级纯 Python 耗时，issue #612）
    for i0 in range(0, n, _TRI_BLOCK):
        i1 = min(i0 + _TRI_BLOCK, n)
        if i0:
            y[i0:i1, :] -= lu[i0:i1, :i0] @ y[:i0, :]
        for i in range(i0 + 1, i1):
            y[i, :] -= lu[i, i0:i] @ y[i0:i, :]

    return y


def backward_substitution(lu: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ux = y (后向替换，右端列整体向量化 + 行分块)"""
    n = lu.shape[0]
    x = y.copy()

    for i1 in range(n, 0, -_TRI_BLOCK):
        i0 = max(i1 - _TRI_BLOCK, 0)
        if i1 < n:
            x[i0:i1, :] -= lu[i0:i1, i1:] @ x[i1:, :]
        for i in range(i1 - 1, i0 - 1, -1):
            j0 = i + 1
            x[i, :] -= lu[i, j0:i1] @ x[j0:i1, :]
            x[i, :] /= lu[i, i]

    return x


def solve_lu(lu: np.ndarray, b: np.ndarray, piv: np.ndarray) -> np.ndarray:
    """通过 lu 分解求解 Ax=b，b 支持 (n,) 或 (n, nrhs)"""
    y = forward_substitution(lu, b, piv)
    x = backward_substitution(lu, y)
    return x


def l1norm(a: np.ndarray) -> float:
    """计算矩阵 L1 范数"""
    return np.max(np.sum(np.abs(a), axis=0))


def _fmt(value) -> str:
    """复数按既有 complex :.4f 形态、实数按 %.4f 输出（% 格式化，规避 W1203）"""
    if np.iscomplexobj(value):
        return "(%.4f%+.4fj)" % (value.real, value.imag)
    return "%.4f" % value


def compare_and_report(x_ref: np.ndarray, x: np.ndarray, n: int, nrhs: int) -> bool:
    """整体向量化比对（保持与双重循环版本一致的 eps 双条件口径，issue #612）"""
    logging.info("-------------------------------------------------------")
    # 相对误差分母用参考值绝对值，参考值过小时取 1 避免除零
    abs_err = np.abs(x_ref - x)
    denom = np.where(np.abs(x_ref) > _DENOM_FLOOR, np.abs(x_ref), np.float32(1.0))
    rel_err = abs_err / denom
    bad = (abs_err > EPS) & (rel_err > EPS)
    cnt = int(bad.sum())

    # 失败样本保持既有日志格式，输出前 _MAX_BAD_PRINT 条（原双重循环按 j 外层、i 内层顺序）
    if cnt:
        bad_i, bad_j = np.nonzero(bad)
        order = np.lexsort((bad_i, bad_j))
        for k in order[:_MAX_BAD_PRINT]:
            i, j = int(bad_i[k]), int(bad_j[k])
            logging.info(
                "index: %d, %d, golden: %s, actual: %s, abs_err: %.4f, rel_err: %.4f",
                i,
                j,
                _fmt(x_ref[i, j]),
                _fmt(x[i, j]),
                abs_err[i, j],
                rel_err[i, j],
            )
    logging.info("Wrong Indices: %d, ratio: %.4f", cnt, cnt / (n * nrhs))
    logging.info("l1norm diff: %.8f", l1norm(x_ref - x) / l1norm(x_ref))

    return cnt / (n * nrhs) <= ERROR_TOL


def run_cli(main_fn: Callable[[], bool]) -> int:
    """verify 脚本命令行入口装配：判定未通过或异常返回退出码 1，通过返回 0（文案保持既有）。

    退出码由调用方在主进程入口（__main__ 块）统一 sys.exit 上抛，函数内不直接
    退出进程（G.ERR.11）。
    """
    try:
        res = main_fn()
    except Exception as e:
        logging.info(e)
        return 1
    if not res:
        logging.info("[Failed] Case accuracy verification failed.")
        return 1
    logging.info("[Success] Case accuracy verification passed.")
    return 0
