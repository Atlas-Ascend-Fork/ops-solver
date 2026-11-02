# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Supplementary late-element validation checks, separate from official cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import argparse
import ctypes as ct
import json
from itertools import product

import numpy as np
from runtime import Runtime, check
from test_support import report, require


def prepare_validation(rt, case):
    op, m, n, k, side, operand, row, column, value = case
    order = n if op == "Cunmqr" and side else m
    a_cols = k if op == "Cunmqr" else n
    b_cols = 128 if op == "CCgels" else n
    a = np.zeros((order + 7, a_cols), dtype=np.complex64, order="F")
    b = np.zeros((m + 11, b_cols), dtype=np.complex64, order="F")
    # Poison padding in every check: it must not override the intended error.
    a[order:] = complex(float("nan"), float("inf"))
    b[m:] = complex(float("nan"), float("inf"))
    (a if operand == "A" else b)[row, column] = value
    da, db = rt.upload(a), rt.upload(b)
    dt = rt.upload(np.zeros(max(k, min(m, n)), dtype=np.complex64))
    if op == "Cgeqrf":
        query = [m, n, da, a.shape[0]]
        call = query + [dt]
        expected = 3
    elif op == "Cungqr":
        query = [m, n, k, da, a.shape[0], dt]
        call = query
        expected = 4
    elif op == "Cunmqr":
        query = [side, 0, m, n, k, da, a.shape[0], dt, db, b.shape[0]]
        call = query
        expected = 6 if operand == "A" else 9
    else:
        dx = rt.alloc((n + 3) * 128 * 8)
        call = [m, n, 128, da, a.shape[0], db, b.shape[0], dx, n + 3]
        query = call + [None]
        expected = 4 if operand == "A" else 6
    return query, call, expected


def verify(rt, case):
    op, m, n, k, side, operand, row, column, value = case
    query, call, expected = prepare_validation(rt, case)
    length = ct.c_size_t() if op == "CCgels" else ct.c_int()
    check(
        getattr(rt.lib, "aclsolver" + op + "_bufferSize")(
            rt.handle, *query, ct.byref(length)
        )
    )
    workspace = rt.alloc(length.value * (1 if op == "CCgels" else 8))
    info = rt.upload(np.array([-999], dtype=np.int32))
    niter = ct.c_int(-999)
    extra = [ct.byref(niter)] if op == "CCgels" else []
    status = getattr(rt.lib, "aclsolver" + op)(
        rt.handle, *call, workspace, length.value, *extra, info
    )
    got = int(rt.download(info, (1,), np.int32)[0])
    result = dict(
        operator=op,
        m=m,
        n=n,
        k=k,
        side=side,
        operand=operand,
        row=row,
        column=column,
        value=str(value),
        status=status,
        info=got,
        expected_info=-expected,
        passed=status == 3 and got == -expected,
    )
    report(json.dumps(result))
    require(result["passed"], result)
    rt.free_all()


def validation_cases():
    shapes = [
        ("Cgeqrf", 257, 131, 0, 0),
        ("Cgeqrf", 31, 601, 0, 0),
        ("Cungqr", 257, 131, 129, 0),
        ("Cunmqr", 257, 131, 129, 0),
        ("Cunmqr", 131, 257, 129, 1),
        ("Cunmqr", 4096, 270, 270, 0),
        ("CCgels", 257, 131, 0, 0),
    ]
    for shape in shapes:
        yield from shape_cases(shape)
    for rows in (511, 512, 513, 1025):
        for shape in (
            ("Cgeqrf", rows, 131, 0, 0),
            ("Cungqr", rows, 131, 129, 0),
            ("Cunmqr", rows, 131, 129, 0),
            ("Cunmqr", 131, rows, 129, 1),
            ("CCgels", rows, 131, 0, 0),
        ):
            yield from shape_cases(shape)


def shape_cases(shape):
    op, m, n, k, side = shape
    order = n if op == "Cunmqr" and side else m
    columns = k if op in ("Cungqr", "Cunmqr") else n
    operands = ["A", "B"] if op in ("Cunmqr", "CCgels") else ["A"]
    for operand in operands:
        positions = (
            [(1, 0), (order - 1, columns - 1)]
            if operand == "A"
            else [(1, 0), (m - 1, (128 if op == "CCgels" else n) - 1)]
        )
        active_rows = order if operand == "A" else m
        positions += [(row, 0) for row in (511, 512, 513) if row < active_rows]
        for (row, column), value in product(
            positions, (complex(float("nan"), 0), complex(0, float("inf")))
        ):
            yield (*shape, operand, row, column, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    rt = Runtime(args.library, args.device)
    count = 0
    try:
        for case in validation_cases():
            verify(rt, case)
            count += 1
        report(json.dumps(dict(total=count, passed=count)))
    finally:
        rt.close()


if __name__ == "__main__":
    main()
