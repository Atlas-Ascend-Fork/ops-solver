# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Task section 3.2.2: FP64 residual arithmetic, paired FP32 LAPACK reference.

All finite numerical cases take this path, independently of elementwise errors.
For batches, all physical slots are checked against their content representative
bit for bit; that representative's residual therefore applies to every slot.
"""

import numpy as np
from scipy.linalg.lapack import spotrf, spotrs, spotri

EPS = 2.0**-23


def symmetric(matrix, upper):
    matrix = np.asarray(matrix, dtype=np.float64)
    half = np.triu(matrix) if upper else np.tril(matrix)
    return half + half.T - np.diag(np.diag(half))


def residual(op, a32, output, upper=False, b32=None):
    """Use the actual FP32 input, lifted to FP64; upper factors use U.T @ U."""
    a = symmetric(a32, upper)
    out = np.asarray(output, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(out).all():
        raise ValueError("Nonfinite input/output in LAPACK numerical residual")
    n = len(a)
    anorm = np.linalg.norm(a, 1)
    if n < 1 or anorm <= 0:
        raise ValueError("Invalid matrix order or zero matrix norm")
    if op.startswith("spotrf"):
        f = np.triu(out) if upper else np.tril(out)
        error = f.T @ f - a if upper else f @ f.T - a
        return float(np.linalg.norm(error, 1) / (n * anorm * EPS))
    if op.startswith("spotrs"):
        b = np.asarray(b32, dtype=np.float64)
        if b.shape != out.shape or not np.isfinite(b).all():
            raise ValueError("Invalid right-hand side")
        numerator = np.abs(b - a @ out).sum(axis=0)
        denominator = anorm * np.abs(out).sum(axis=0) * EPS
        ratios = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.inf),
            where=denominator > 0,
        )
        ratios[(numerator == 0) & (denominator == 0)] = 0
        return float(np.max(ratios))
    c = symmetric(out, upper)
    return float(
        np.linalg.norm(np.eye(n) - a @ c, 1) / (n * anorm * np.linalg.norm(c, 1) * EPS)
    )


def paired(op, a32, output, upper=False, b32=None):
    factor, info = spotrf(np.array(a32, dtype=np.float32, order="F"), lower=not upper)
    if info != 0:
        raise ValueError(f"FP32 LAPACK potrf preparation failed: {info}")
    if op.startswith("spotrf"):
        cpu = factor
    elif op.startswith("spotrs"):
        cpu, info = spotrs(
            factor, np.array(b32, dtype=np.float32, order="F"), lower=not upper
        )
    else:
        cpu, info = spotri(factor, lower=not upper)
    if info != 0:
        raise ValueError(f"FP32 LAPACK reference failed: {info}")
    cpu_ratio = residual(op, a32, cpu, upper, b32)
    if not np.isfinite(cpu_ratio) or cpu_ratio < 0:
        raise ValueError("Nonfinite or negative FP32 LAPACK reference residual")
    return {
        "ratio": residual(op, a32, output, upper, b32),
        "ratio_cpu": cpu_ratio,
        "eps": EPS,
    }


def verify_slot_copies(output, indices, expected, content):
    chunk = max(1, (32 << 20) // max(1, expected.nbytes))
    for start in range(0, len(indices), chunk):
        end = start + chunk
        selected = indices[start:end]
        if not np.all(output[selected].view(np.uint32) == expected):
            raise ValueError(f"Batch content {content}: physical slots differ bitwise")


def evaluate(entry, arrays, dut):
    if entry.get("case_purpose") == "info":
        return []
    if not np.all(np.asarray(dut["info"]) == 0):
        raise ValueError("Unexpected nonzero info for a numerical case")
    op = entry["op"].lower()
    upper = entry["uplo"] == "U"
    mapping = arrays.get("sample_map")
    if not mapping:
        return [
            paired(op, arrays["A32"], dut["out32"], upper, arrays.get("B32"))
            | {"physical_slots_checked": 1}
        ]
    results = []
    output = dut["out32"]
    checked = 0
    coverage = np.zeros(entry["batch"], dtype=bool)
    for item in mapping:
        content, representative = item["content_idx"], item["rep_slot"]
        expected = output[representative].view(np.uint32)
        indices = np.asarray(item["slots"], dtype=np.intp)
        if (
            np.any(indices < 0)
            or np.any(indices >= entry["batch"])
            or not np.any(indices == representative)
        ):
            raise ValueError("Invalid batch slot mapping")
        verify_slot_copies(output, indices, expected, content)
        coverage[indices] = True
        slots = len(indices)
        checked += slots
        b = arrays["B32"][content] if "B32" in arrays else None
        results.append(
            paired(op, arrays["A32"][content], output[representative], upper, b)
            | {
                "content_idx": content,
                "representative_slot": representative,
                "physical_slots_checked": slots,
            }
        )
    if checked != entry["batch"] or not np.all(coverage):
        raise ValueError("Incomplete batch slot coverage")
    mean = sum(r["ratio_cpu"] * r["physical_slots_checked"] for r in results) / checked
    for row in results:
        row["ratio_cpu_mean"] = mean
    return results


def finalize(rows, op, complete_set):
    numerical = [r for row in rows for r in row.get("lapack_residual", [])]
    mean = (
        float(np.mean([r["ratio_cpu"] for r in numerical]))
        if numerical and complete_set
        else None
    )
    for row in rows:
        if row.get("purpose") == "info":
            row["final_precision_pass"] = row.get("attachment_numeric_pass", False)
            row["final_precision_method"] = "info contract (residual not applicable)"
            continue
        checks = row.get("lapack_residual", [])
        for value in checks:
            cpu_mean = value.get("ratio_cpu_mean") if op.endswith("batched") else mean
            threshold = (
                max(5 * value["ratio_cpu"], 0.1)
                if op == "spotri"
                else (
                    max(5 * value["ratio_cpu"], 3 * cpu_mean)
                    if cpu_mean is not None
                    else 5 * value["ratio_cpu"]
                )
            )
            value.update(
                ratio_cpu_mean=cpu_mean,
                threshold=threshold,
                normalized_ratio=value["ratio"] / threshold
                if threshold
                else (0 if value["ratio"] == 0 else None),
                pass_=bool(np.isfinite(value["ratio"]) and value["ratio"] <= threshold),
            )
        row["final_precision_pass"] = (
            "error" not in row and bool(checks) and all(r["pass_"] for r in checks)
        )
        row["final_precision_method"] = "task section 3.2.2 LAPACK residual"
