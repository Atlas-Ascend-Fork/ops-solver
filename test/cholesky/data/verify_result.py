# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent LAPACK residual verification of the solver test binary's files."""

import json
import logging
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lapack_residual import paired

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
root = Path(__file__).resolve().parent
n, ld = 17, 20


def read(name, columns):
    path = root / name
    if path.stat().st_size != columns * ld * 4:
        raise ValueError(f"Corrupted binary size: {path}")
    values = np.fromfile(path, dtype=np.float32).reshape(columns, ld)
    if not np.all(values[:, n:] == -12345.125):
        raise ValueError(f"Padding changed: {path}")
    return values[:, :n].T


def main():
    a, b = read("input/a.bin", n), read("input/b.bin", 1)
    rows = []
    for upper in (False, True):
        prefix = "upper-" if upper else "lower-"
        for op, suffix, columns in [
            ("spotrf", "factor", n),
            ("spotrs", "solve", 1),
            ("spotri", "inverse", n),
            ("spotrfbatched", "batch-factor-0", n),
            ("spotrsbatched", "batch-solve-0", 1),
        ]:
            output = read("output/" + prefix + suffix + ".bin", columns)
            row = paired(op, a, output, upper, b if "spotrs" in op else None)
            if op.endswith("batched"):
                other = read("output/" + prefix + suffix[:-1] + "1.bin", columns)
                if not np.array_equal(output.view(np.uint32), other.view(np.uint32)):
                    raise ValueError("Physical batch outputs differ bitwise")
            rows.append(row | {"op": op, "uplo": "U" if upper else "L"})
    for row in rows:
        mean = sum(r["ratio_cpu"] for r in rows if r["op"] == row["op"]) / 2
        if row["op"].endswith("batched"):
            mean = row["ratio_cpu"]
        row["threshold"] = max(
            5 * row["ratio_cpu"], 0.1 if row["op"] == "spotri" else 3 * mean
        )
        row["pass"] = row["ratio"] <= row["threshold"]
        logging.info(json.dumps(row))
    if not all(r["pass"] for r in rows):
        return 1
    logging.info("[Success] Case LAPACK residual verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error("[Failed] %s: %s", type(error).__name__, error)
        raise SystemExit(1) from error
