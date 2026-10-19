# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile complete real Cheevj Host blocks; no ACL or device API stubs."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src/cheevj/cheevj_host.cpp"
HEADERS = """#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using Complex = std::complex<float>;
using DoubleComplex = std::complex<double>;
"""


def complete_function(text, prefix):
    start = text.index(prefix)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def actual_host_blocks(text):
    thread_count = re.search(
        r"constexpr (?:int|int32_t) HOST_OMP_THREAD_NUM = (\d+);", text
    )
    assert thread_count is not None
    eigenpair_start = text.index("struct CompactEigenpair\n")
    eigenpair_end = text.index("\nvoid SolveInputTwoByTwoBlockDiagonal(")
    sections = [
        thread_count.group(0),
        complete_function(text, "int64_t CompactColumnMajorIndex("),
        complete_function(text, "Complex LoadHermitianValue("),
        complete_function(text, "void ClearInputMatrix("),
        text[eigenpair_start:eigenpair_end],
        complete_function(text, "void SolveInputTwoByTwoBlockDiagonal("),
        complete_function(text, "void SolveCompactTwoByTwoBlockDiagonal("),
    ]
    return "\n\n".join(sections)


@pytest.mark.parametrize("optimization", ("-O0", "-O2"))
def test_real_host_vectors_and_existing_novector_paths(optimization, tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None, "a real Host C++17 compiler is required"
    cpp = tmp_path / "actual_cheevj_host.cpp"
    cpp.write_text(HEADERS + actual_host_blocks(SOURCE.read_text()) + DRIVER)
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [compiler, "-std=c++17", optimization, str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(binary), str(tmp_path / "eigenvalues.bin")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cases=13356" in result.stdout
    assert "nonfiniteVectors=0" in result.stdout
    assert "badEigenvalues=0" in result.stdout
    assert "inputOrPaddingChanged=0" in result.stdout


DRIVER = r"""
struct ResultStats {
    int cases = 0;
    int nonfiniteVectors = 0;
    int badResidual = 0;
    int badNorm = 0;
    int badOrthogonality = 0;
    int badEigenvalues = 0;
    int inputOrPaddingChanged = 0;
    double maxResidual = 0.0;
};

bool Finite(Complex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

void RecordVectors(const std::array<Complex, 4>& v, float d0, float d1, Complex upper,
                   const float* w, ResultStats* stats) {
    bool finite = std::all_of(v.begin(), v.end(), Finite);
    if (!finite) { ++stats->nonfiniteVectors; return; }
    DoubleComplex a(v[0]), b(v[1]), c(v[2]), d(v[3]), u(upper);
    const double scale = std::max({std::abs(static_cast<double>(d0)),
                                   std::abs(static_cast<double>(d1)), std::abs(u)});
    const double norm0 = std::norm(a) + std::norm(b);
    const double norm1 = std::norm(c) + std::norm(d);
    if (std::abs(norm0 - 1.0) > 2.0e-6 || std::abs(norm1 - 1.0) > 2.0e-6) ++stats->badNorm;
    if (std::abs(std::conj(a) * c + std::conj(b) * d) > 2.0e-6) ++stats->badOrthogonality;
    const double residual = std::max({
        std::abs(static_cast<double>(d0) * a + u * b - static_cast<double>(w[0]) * a),
        std::abs(std::conj(u) * a + static_cast<double>(d1) * b - static_cast<double>(w[0]) * b),
        std::abs(static_cast<double>(d0) * c + u * d - static_cast<double>(w[1]) * c),
        std::abs(std::conj(u) * c + static_cast<double>(d1) * d - static_cast<double>(w[1]) * d)});
    // The absolute component covers float subnormal output quantization.
    if (residual > scale * 2.0e-6 + std::numeric_limits<float>::denorm_min() * 2.0) ++stats->badResidual;
    if (scale > 0.0) stats->maxResidual = std::max(stats->maxResidual, residual / scale);
}

void RecordEigenvalues(float d0, float d1, Complex u, const float* w, ResultStats* stats) {
    // Long-double analytic Hermitian roots, independent of the production hypotenuse expression.
    const long double center = (static_cast<long double>(d0) + d1) / 2;
    const long double half = (static_cast<long double>(d0) - d1) / 2;
    const long double radius = std::sqrt(half * half + static_cast<long double>(u.real()) * u.real()
                                       + static_cast<long double>(u.imag()) * u.imag());
    const long double low = center - radius, high = center + radius;
    const long double tolerance = std::max(std::abs(low), std::abs(high)) * 2.0e-6L
                                + std::numeric_limits<float>::denorm_min() * 2.0L;
    if (!std::isfinite(w[0]) || !std::isfinite(w[1]) || w[0] > w[1]
        || std::abs(w[0] - low) > tolerance || std::abs(w[1] - high) > tolerance) ++stats->badEigenvalues;
}

void RecordInputPath(float d0, float d1, Complex u, bool lower, bool vectors,
                     ResultStats* stats, std::ofstream* eigenvalueBytes) {
    constexpr int64_t lda = 4;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<Complex> a(8, Complex(37.0f, -29.0f));
    a[0] = Complex(d0, 0); a[5] = Complex(d1, 0);
    a[1] = lower ? std::conj(u) : Complex(poison, poison);
    a[4] = lower ? Complex(poison, poison) : u;
    const std::vector<Complex> before = a;
    float w[2];
    SolveInputTwoByTwoBlockDiagonal(a.data(), lda, 2, lower, vectors, w);
    eigenvalueBytes->write(reinterpret_cast<const char*>(w), sizeof(w));
    RecordEigenvalues(d0, d1, u, w, stats);
    if (!vectors) {
        if (std::memcmp(before.data(), a.data(), sizeof(Complex) * a.size()) != 0) ++stats->inputOrPaddingChanged;
    } else {
        RecordVectors({a[0], a[1], a[4], a[5]}, d0, d1, u, w, stats);
        for (size_t index : {2u, 3u, 6u, 7u})
            if (std::memcmp(&before[index], &a[index], sizeof(Complex)) != 0) ++stats->inputOrPaddingChanged;
    }
    ++stats->cases;
}

void RecordCompactPath(float d0, float d1, Complex u, bool vectors,
                       ResultStats* stats, std::ofstream* eigenvalueBytes) {
    std::vector<Complex> a{Complex(d0, 0), std::conj(u), u, Complex(d1, 0)};
    const auto before = a;
    float w[2];
    SolveCompactTwoByTwoBlockDiagonal(&a, 2, vectors, w);
    eigenvalueBytes->write(reinterpret_cast<const char*>(w), sizeof(w));
    RecordEigenvalues(d0, d1, u, w, stats);
    if (!vectors) {
        if (std::memcmp(before.data(), a.data(), sizeof(Complex) * a.size()) != 0) ++stats->inputOrPaddingChanged;
    } else RecordVectors({a[0], a[1], a[2], a[3]}, d0, d1, u, w, stats);
    ++stats->cases;
}

void RecordCase(float d0, float d1, Complex u, ResultStats* stats, std::ofstream* eigenvalueBytes) {
    for (bool vectors : {false, true}) {
        for (bool lower : {false, true}) RecordInputPath(d0, d1, u, lower, vectors, stats, eigenvalueBytes);
        RecordCompactPath(d0, d1, u, vectors, stats, eigenvalueBytes);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ofstream eigenvalueBytes(argv[1], std::ios::binary);
    ResultStats stats;
    for (int exponent = -149; exponent <= 127; ++exponent) {
        float scale = std::ldexp(1.0f, exponent);
        float coupling = std::ldexp(1.0f, exponent - 2);
        for (int direction : {1, -1}) {
            for (Complex phase : {Complex(1, 0), Complex(0, 1), Complex(1, 1), Complex(-1, 0.5f)})
                RecordCase(scale * direction, -scale * direction, phase * coupling, &stats, &eigenvalueBytes);
        }
    }
    for (float scale : {1.0f, 1.0e38f, std::ldexp(1.0f, 127)})
        for (int direction : {1, -1}) RecordCase(scale * direction, -scale * direction,
                                                Complex(1, 1), &stats, &eigenvalueBytes);
    // PR69 zero-diagonal complex-radius controls and the original zero-upper branch.
    for (float scale : {2.0f, 2.0e20f}) RecordCase(0, 0, Complex(scale, scale * 0.5f), &stats, &eigenvalueBytes);
    RecordCase(2, -3, Complex(0, 0), &stats, &eigenvalueBytes);
    RecordCase(-3, 2, Complex(0, 0), &stats, &eigenvalueBytes);
    std::cout << "cases=" << stats.cases << " nonfiniteVectors=" << stats.nonfiniteVectors
              << " badResidual=" << stats.badResidual << " badNorm=" << stats.badNorm
              << " badOrthogonality=" << stats.badOrthogonality << " badEigenvalues=" << stats.badEigenvalues
              << " inputOrPaddingChanged=" << stats.inputOrPaddingChanged
              << " maxResidual=" << stats.maxResidual << '\n';
    return stats.nonfiniteVectors + stats.badResidual + stats.badNorm + stats.badOrthogonality
           + stats.badEigenvalues + stats.inputOrPaddingChanged == 0 ? 0 : 1;
}
"""
