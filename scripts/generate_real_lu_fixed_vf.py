# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Emit fixed-order VF with named registers; CANN 9.0 cannot unroll register arrays."""

from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RowEmitter:
    op: Callable[[str], None]
    load: Callable[[str, int, int], None]
    diagonal: Callable[[str, int], None]
    term: Callable[[str, str], None]


def emit_upper_rows(order, rows, emitter):
    op, load, diagonal, term = emitter.op, emitter.load, emitter.diagonal, emitter.term
    for row in rows:
        for col in range(order):
            if row == col:
                diagonal(f"x{row}_{col}", col)
            else:
                op(f"MicroAPI::Duplicate<float>(x{row}_{col}, 0.0f, valid);")
        for col in range(row + 1, order):
            diagonal("diagonal", col)
            op("MicroAPI::Neg<float>(negative, diagonal, valid);")
            load("coefficient", row, col)
            op(f"MicroAPI::Mul<float>(x{row}_{col}, coefficient, x{row}_{row}, valid);")
            for k in range(row + 1, col):
                load("coefficient", k, col)
                term(f"x{row}_{col}", f"x{row}_{k}")
            op(f"MicroAPI::Mul<float>(x{row}_{col}, x{row}_{col}, negative, valid);")


def emit_lower_rows(order, rows, emitter):
    op, load, term = emitter.op, emitter.load, emitter.term
    for col in range(order - 2, -1, -1):
        for row in rows:
            op(f"MicroAPI::Duplicate<float>(dot{row}, 0.0f, valid);")
        for k in range(col + 1, order):
            load("coefficient", k, col)
            for row in rows:
                term(f"dot{row}", f"x{row}_{k}")
        for row in rows:
            op(f"MicroAPI::Sub<float>(x{row}_{col}, x{row}_{col}, dot{row}, valid);")


def emit_block(order, begin, width):
    """Keep a few whole inverse rows in registers, including the lower solve."""
    rows = range(begin, begin + width)
    # Use fused ascending dot products for every fixed order.
    fused = True
    name = f"SmallLapack{order}Rows{begin}VF"
    lines = [
        f"__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void {name}(",
        "        __ubuf__ float *lu, __ubuf__ float *x, __ubuf__ float *r, uint16_t matrices, uint32_t count)",
        "{",
        "    uint32_t remaining = count;",
        "    MicroAPI::RegTensor<float> coefficient, product, negative, diagonal;",
    ]
    names = [f"x{row}_{col}" for row in rows for col in range(order)]
    names += [f"dot{row}" for row in rows]
    lines += [
        "    MicroAPI::RegTensor<float> " + ", ".join(names) + ";",
        "    const uint16_t lane = 0;",
        "    {",
        "        auto valid = MicroAPI::UpdateMask<float>(remaining);",
    ]

    def op(value):
        lines.append("        " + value)

    def load(target, row, col):
        op(
            f"MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({target}, "
            f"lu + {row * order + col} * matrices + lane, 1u, valid);"
        )

    def diagonal(target, col):
        op(
            f"MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({target}, "
            f"r + {col} * matrices + lane, 1u, valid);"
        )

    def term(target, entry):
        if fused:
            op(f"MicroAPI::MulAddDst<float>({target}, coefficient, {entry}, valid);")
        else:
            op(f"MicroAPI::Mul<float>(product, coefficient, {entry}, valid);")
            op(f"MicroAPI::Add<float>({target}, {target}, product, valid);")

    emitter = RowEmitter(op, load, diagonal, term)
    emit_upper_rows(order, rows, emitter)
    emit_lower_rows(order, rows, emitter)
    for row in rows:
        for col in range(order):
            op(
                f"MicroAPI::StoreAlign<float>(x + {row * order + col} * matrices + lane, x{row}_{col}, valid);"
            )
    lines += [
        "    }",
        "    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();",
        "}",
    ]
    return "\n".join(lines)


CHUNK_BLOCK = 8
CHUNK_ATTRIBUTE = "__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline))"
CHUNK_BARRIER = "MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();"


def chunk_name(order, phase, first):
    return f"SmallLapack{order}{phase}Block{CHUNK_BLOCK}_{first}VF"


def chunk_load(register, address, indent="    "):
    return (
        indent
        + f"MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({register}, {address}, 1u, valid);"
    )


def chunk_upper(order, first):
    lines = [
        f"{CHUNK_ATTRIBUTE} void {chunk_name(order, 'Upper', first)}(",
        "        __ubuf__ float* lu, __ubuf__ float* x, __ubuf__ float* r,",
        "        uint16_t row, uint16_t pitch, uint32_t count)",
        "{",
        "    // One wave only. row is a caller CPU parameter, never a VF induction variable.",
        "    MicroAPI::RegTensor<float> coefficient, other, target, diagonal, negative;",
        "    auto valid = MicroAPI::UpdateMask<float>(count);",
    ]
    for col in range(max(1, first), min(order, first + CHUNK_BLOCK)):
        lines += [
            f"    // Column {col}: first rounded Mul, then ascending FMA, then rounded scaling.",
            f"    if (row < {col}) {{",
            chunk_load("diagonal", "r + row * pitch", "        "),
            chunk_load(
                "coefficient", f"lu + (row * {order} + {col}) * pitch", "        "
            ),
            "        MicroAPI::Mul<float>(target, coefficient, diagonal, valid);",
            f"        for (uint16_t k = static_cast<uint16_t>(row + 1); k < {col}; ++k) {{",
            chunk_load(
                "coefficient", f"lu + (k * {order} + {col}) * pitch", "            "
            ),
            chunk_load("other", "x + k * pitch", "            "),
            "            MicroAPI::MulAddDst<float>(target, coefficient, other, valid);",
            "        }",
            chunk_load("diagonal", f"r + {col} * pitch", "        "),
            "        MicroAPI::Neg<float>(negative, diagonal, valid);",
            "        MicroAPI::Mul<float>(target, target, negative, valid);",
            f"        MicroAPI::StoreAlign<float>(x + {col} * pitch, target, valid);",
            "        " + CHUNK_BARRIER,
            "    }",
        ]
    return "\n".join(lines + ["}"])


def chunk_lower_starts(order):
    # Established fixed kernels retain their original last-column no-op.
    stop = order if order in (16, 32) else order - 1
    return range(0, stop, CHUNK_BLOCK)


def chunk_lower_column(order, col, rows):
    lines = []
    lines.append(
        f"    // Column {col}: independent positive dot, ascending k, one final Sub."
    )
    lines += [f"    MicroAPI::Duplicate<float>(dot{row}, 0.0f, valid);" for row in rows]
    lines += [
        f"    for (uint16_t k = {col + 1}; k < {order}; ++k) {{",
        chunk_load("coefficient", f"lu + (k * {order} + {col}) * pitch", "        "),
    ]
    lines += [
        chunk_load(f"other{row}", f"x{row} + k * pitch", "        ") for row in rows
    ]
    lines += [
        f"        MicroAPI::MulAddDst<float>(dot{row}, coefficient, other{row}, valid);"
        for row in rows
    ]
    lines.append("    }")
    lines += [chunk_load(f"target{row}", f"x{row} + {col} * pitch") for row in rows]
    lines += [
        f"    MicroAPI::Sub<float>(target{row}, target{row}, dot{row}, valid);"
        for row in rows
    ]
    lines += [
        f"    MicroAPI::StoreAlign<float>(x{row} + {col} * pitch, target{row}, valid);"
        for row in rows
    ]
    lines.append("    " + CHUNK_BARRIER)
    return lines


def chunk_lower(order, first, single=False):
    # Preserve the established n16/n32 kernels; arbitrary orders leave the
    # final diagonal column untouched, matching SmallLowerColumnVF exactly.
    column_limit = order if order in (16, 32) else order - 1
    columns = range(first, min(column_limit, first + CHUNK_BLOCK))
    rows = range(1 if single else 2)
    phase = "LowerSingle" if single else "Lower"
    row_arguments = ", ".join(f"__ubuf__ float* x{row}" for row in rows)
    registers = ["coefficient"]
    registers += [f"other{row}" for row in rows]
    registers += [f"dot{row}" for row in rows]
    registers += [f"target{row}" for row in rows]
    lines = [
        f"{CHUNK_ATTRIBUTE} void {chunk_name(order, phase, first)}(",
        f"        __ubuf__ float* lu, {row_arguments}, uint16_t pitch, uint32_t count)",
        "{",
        "    // Runtime k addresses prevent hoisting the whole triangular input into registers.",
        (
            "    // One final row has no companion row, so no out-of-range x1 is formed."
            if single
            else "    // Two independent rows share each coefficient load, retaining their own dot order."
        ),
        "    MicroAPI::RegTensor<float> " + ", ".join(registers) + ";",
        "    auto valid = MicroAPI::UpdateMask<float>(count);",
    ]
    for col in reversed(columns):
        lines += chunk_lower_column(order, col, rows)

    return "\n".join(lines + ["}"])


def chunk_wrapper(order):
    lines = [
        f"__aicore__ inline void SmallLapack{order}Chunked(",
        "        LocalTensor<float> factors, LocalTensor<float> inverse,",
        "        LocalTensor<float> reciprocals, int matrices)",
        "{",
        "    // DiagonalReciprocals already initialized inverse to zero plus original reciprocals.",
        "    // Column permutation and output are handled by the caller.",
        "    auto lu = reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr());",
        "    auto x = reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr());",
        "    auto r = reinterpret_cast<__ubuf__ float*>(reciprocals.GetPhyAddr());",
        "    const uint16_t pitch = static_cast<uint16_t>(matrices);",
        "    for (int wave = 0; wave < matrices; wave += 64) {",
        "        uint32_t count = static_cast<uint32_t>(matrices - wave < 64 ? matrices - wave : 64);",
        f"        for (uint16_t row = 0; row < {order - order % 2}; row += 2) {{",
        f"            auto rowX = x + static_cast<uint32_t>(row) * {order} * matrices + wave;",
        f"            auto nextX = rowX + {order} * matrices;",
    ]
    lines += [
        f"            {chunk_name(order, 'Upper', first)}(lu + wave, rowX, r + wave, row, pitch, count);"
        for first in range(0, order, CHUNK_BLOCK)
    ]
    lines += [
        f"            {chunk_name(order, 'Upper', first)}(lu + wave, nextX, r + wave, "
        "static_cast<uint16_t>(row + 1), pitch, count);"
        for first in range(0, order, CHUNK_BLOCK)
    ]
    lines += [
        f"            {chunk_name(order, 'Lower', first)}(lu + wave, rowX, nextX, pitch, count);"
        for first in reversed(chunk_lower_starts(order))
    ]
    lines.append("        }")
    if order % 2:
        # The last inverse row starts with only its initialized reciprocal;
        # no upper solve is required for row == order - 1.
        lines.append(
            f"        auto lastX = x + {order - 1} * {order} * matrices + wave;"
        )
        lines += [
            f"        {chunk_name(order, 'LowerSingle', first)}(lu + wave, lastX, pitch, count);"
            for first in reversed(range(0, order - 1, CHUNK_BLOCK))
        ]
    return "\n".join(lines + ["    }", "}"])


def emit_small(order):
    width = 4 if order == 4 else 2
    functions = "\n\n".join(
        emit_block(order, row, width) for row in range(0, order, width)
    )
    wrapper = [
        f"__aicore__ inline void SmallLapack{order}VF(LocalTensor<float> factors, LocalTensor<float> inverse,",
        "        LocalTensor<float> reciprocals, int matrices)",
        "{",
        "    auto lu = reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr());",
        "    auto x = reinterpret_cast<__ubuf__ float *>(inverse.GetPhyAddr());",
        "    auto r = reinterpret_cast<__ubuf__ float *>(reciprocals.GetPhyAddr());",
        "    for (int wave = 0; wave < matrices; wave += 64) {",
        "        uint32_t count = matrices - wave < 64 ? matrices - wave : 64;",
    ]
    for row in range(0, order, width):
        wrapper.append(
            f"        SmallLapack{order}Rows{row}VF(lu + wave, x + wave, r + wave, "
            "static_cast<uint16_t>(matrices), count);"
        )
    wrapper += ["    }", "}"]
    return functions + "\n\n" + "\n".join(wrapper)


def inplace_upper_wrapper(order):
    lines = [
        f"__aicore__ inline void SmallLapack{order}InplaceUpper(",
        "        LocalTensor<float> factors, LocalTensor<float> reciprocals, int matrices)",
        "{",
        "    auto lu = reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr());",
        "    auto r = reinterpret_cast<__ubuf__ float*>(reciprocals.GetPhyAddr());",
        "    const uint16_t pitch = static_cast<uint16_t>(matrices);",
        "    for (int wave = 0; wave < matrices; wave += 64) {",
        "        uint32_t count = static_cast<uint32_t>(matrices - wave < 64 ? matrices - wave : 64);",
        f"        for (uint16_t row = 0; row < {order}; ++row) {{",
        f"            auto rowX = lu + static_cast<uint32_t>(row) * {order} * matrices + wave;",
    ]
    lines += [
        f"            {chunk_name(order, 'Upper', first)}(lu + wave, rowX, r + wave, row, pitch, count);"
        for first in range(0, order, CHUNK_BLOCK)
    ]
    return "\n".join(lines + ["        }", "    }", "}"])


def emit_chunked(order):
    pieces = [chunk_upper(order, first) for first in range(0, order, CHUNK_BLOCK)]
    pieces += [chunk_lower(order, first) for first in chunk_lower_starts(order)]
    if order % 2:
        pieces += [
            chunk_lower(order, first, single=True)
            for first in range(0, order - 1, CHUNK_BLOCK)
        ]
    pieces.append(chunk_wrapper(order))
    if order >= 17:
        pieces.append(inplace_upper_wrapper(order))
    return "\n\n".join(pieces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "build/real_lu_generated/real_lu_fixed_vf.hpp",
    )
    output = parser.parse_args().output
    header = """/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
// Generated by generate_real_lu_fixed_vf.py. Named registers avoid unsupported
// VF register-array indexing in CANN 9.0. All operations remain native FLOAT32.
namespace RealLuKernel {
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as generated:
        generated.write(
            header
            + "\n\n".join(
                [emit_small(n) for n in (4, 8)]
                + [emit_chunked(n) for n in range(16, 33)]
            )
            + "\n} // namespace RealLuKernel\n",
        )
