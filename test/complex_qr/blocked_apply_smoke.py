# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Supplementary blocked Q tests; official inputs and acceptance gates are unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import argparse
import json

import numpy as np
from numerical_checks import evaluate
from runtime import Runtime
from test_support import report


def repeat_case(rt, op, case, index, args):
    hashes, values = [], []
    for _ in range(args.repeat):
        rt.observed.clear()
        value, cpu, multiplier, floor = evaluate(
            rt, op, case, np.random.default_rng(9502026 + index)
        )
        values.append(value)
        hashes.append(tuple(rt.observed))
        rt.free_all()
    limit = max(multiplier * cpu, floor)
    passed = bool(
        np.isfinite(cpu)
        and all(np.isfinite(v) and v <= limit for v in values)
        and all(value == hashes[0] for value in hashes)
    )
    result = dict(
        operator=op,
        case=case,
        ratios=values,
        ratio_cpu=cpu,
        limit=limit,
        repeat=args.repeat,
        deterministic=all(value == hashes[0] for value in hashes),
        output_sha256=hashes[0],
        passed=passed,
    )
    return result


def append_padded_cases(cases):
    for padding in (8, 32):
        for m, n in ((31, 63), (128, 255), (255, 128)):
            cases.append(("cgeqrf", dict(m=m, n=n, lda=m + padding)))
        for m, n, k in ((127, 127, 127), (255, 127, 63)):
            cases.append(("cungqr", dict(m=m, n=n, k=k, lda=m + padding)))
        for side in (0, 1):
            for trans in (0, 2):
                cases.append(
                    (
                        "cunmqr",
                        dict(
                            m=127,
                            n=255,
                            k=63,
                            side=side,
                            trans=trans,
                            lda=(127 if side == 0 else 255) + padding,
                            ldc=127 + padding,
                        ),
                    )
                )
        for m, n, nrhs in (
            (127, 127, 1),
            (255, 127, 8),
            (256, 128, 32),
            (128, 64, 128),
        ):
            cases.append(
                (
                    "cgels",
                    dict(
                        m=m,
                        n=n,
                        nrhs=nrhs,
                        ldda=m + padding,
                        lddb=m + padding,
                        lddx=n + padding,
                    ),
                )
            )


def default_cases():
    cases = []
    for m, n, k in [(129, 97, 65), (96, 160, 33), (257, 17, 17), (17, 257, 17)]:
        for side in (0, 1):
            for trans in (0, 2):
                cases.append(
                    (
                        "cunmqr",
                        dict(
                            m=m,
                            n=n,
                            k=k,
                            side=side,
                            trans=trans,
                            lda=(m if side == 0 else n) + 3,
                            ldc=m + 5,
                        ),
                    )
                )
    for m, n, nrhs in [
        (97, 65, 64),
        (128, 128, 64),
        (65, 17, 48),
        (129, 97, 8),
        (257, 129, 32),
        (257, 129, 128),
        (127, 127, 128),
    ]:
        cases.append(
            ("cgels", dict(m=m, n=n, nrhs=nrhs, ldda=m + 3, lddb=m + 5, lddx=n + 7))
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--repeat",
        type=int,
        choices=(2, 10),
        default=2,
        help="Ten repetitions implement the design's representative determinism check",
    )
    parser.add_argument(
        "--task-boundaries",
        action="store_true",
        help="Also run the task's padding, powers-minus-one and large multi-RHS cases",
    )
    args = parser.parse_args()
    cases = default_cases()
    if args.task_boundaries:
        append_padded_cases(cases)
        for m, n, nrhs in ((4096, 1024, 8), (4096, 4096, 32)):
            cases.append(("cgels", dict(m=m, n=n, nrhs=nrhs, ldda=m, lddb=m, lddx=n)))
    rt = Runtime(args.library, args.device)
    failed = 0
    try:
        for index, (op, case) in enumerate(cases):
            try:
                result = repeat_case(rt, op, case, index, args)
            except Exception as exc:
                result = dict(operator=op, case=case, passed=False, error=str(exc))
            failed += not result["passed"]
            report(json.dumps(result))
            rt.free_all()
    finally:
        rt.close()
    report(json.dumps(dict(supplementary=True, total=len(cases), failed=failed)))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
