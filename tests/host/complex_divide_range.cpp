/**
 * Copyright (c) 2026 Xztxzt
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

#include "cann_ops_solver_common.h"
#define __simt_callee__
#define __aicore__
struct Lane
{
    int x;
};
static Lane threadIdx{0}, blockDim{256};
inline void asc_syncthreads() {}
inline float asc_shfl(float value, int, int) { return value; }
#include "production.inc"

int failures = 0, cases = 0;
// The compensated result is checked componentwise within 4 representable FP32 steps.
// No relative tolerance that would hide cancellation or tiny representable components.
bool Close(float actual, float expected)
{
    if (!std::isfinite(actual)) return false;
    if (actual == expected) return true;
    for (int i = 0; i < 4; ++i)
    {
        actual = std::nextafter(actual, expected);
        if (actual == expected) return true;
    }
    return false;
}
void Check(aclFloatComplex a, aclFloatComplex b)
{
    std::complex<long double> reference =
        std::complex<long double>(a.real, a.imag) / std::complex<long double>(b.real, b.imag);
    float real = static_cast<float>(reference.real()), imag = static_cast<float>(reference.imag());
    if (!std::isfinite(real) || !std::isfinite(imag)) return;  // finite representable quotient contract
    auto direct = Divide(a, b);
    auto prepared = PreparedDivide(a, BroadcastDivisor<32>(PrepareDivisor(b), 0));
    aclFloatComplex column[2] = {b, a};
    DivideColumn(column, 2, 2, 0);
    auto divisor = PrepareDivisor(b);
    float inverseScale = divisor.scale >= 0x1p-126f ? 1.0f / divisor.scale : 0.0f;
    auto cached = NormalizeCachedValue(a, divisor, inverseScale);
    for (auto got : {direct, prepared, column[1], cached})
    {
        ++cases;
        if (!Close(got.real, real) || !Close(got.imag, imag))
        {
            if (failures < 12)
                printf("a=(%a,%a) b=(%a,%a) expected=(%a,%a) got=(%a,%a)\n", double(a.real), double(a.imag),
                       double(b.real), double(b.imag), double(real), double(imag), double(got.real), double(got.imag));
            ++failures;
        }
    }
}
void CheckRangeBoundaries()
{
    float maximum = std::numeric_limits<float>::max(), tiny = std::numeric_limits<float>::denorm_min();
    for (float ar : {0.f, 1.f, -1.f, 0x1p126f, 0x1p127f, -0x1p127f, maximum, -maximum, tiny, -tiny})
        for (float ai : {0.f, 1.f, -1.f, 0x1p126f, 0x1p127f, -0x1p127f, maximum, -maximum, tiny, -tiny})
            for (auto b : {aclFloatComplex{1, 1},
                           {1, -1},
                           {-1, 1},
                           {0, 1},
                           {1, 0},
                           {2, 1},
                           {1, 2},
                           {0x1p-126f, 0},
                           {0x1p-126f, 0x1p-126f},
                           {0x1p126f, 0x1p126f},
                           {maximum, maximum},
                           {tiny, tiny}})
                Check({ar, ai}, b);
    // Cancellation and exponent imbalance, including a large numerator with a tiny real answer.
    for (int e : {-100, -60, -10, 0, 10, 60, 100, 126})
    {
        float v = std::ldexp(1.f, e);
        Check({0x1p127f, v}, {v, 1});
        Check({0x1p127f, -v}, {v, 1});
        Check({maximum, std::nextafter(maximum, 0.f)}, {1, 1});
        Check({v, -v}, {v, v});
    }
}

void CheckRandomInputs()
{
    std::mt19937 rng(48291);
    for (int i = 0; i < 20000; ++i)
    {
        auto sample = [&]()
        {
            int e = int(rng() % 201) - 100;
            float m = float(int(rng() % 65535) - 32767) / 32768.f;
            return std::ldexp(m, e);
        };
        aclFloatComplex a{sample(), sample()}, b{sample(), sample()};
        if (b.real != 0 || b.imag != 0) Check(a, b);
    }
    for (int i = 0; i < 20000; ++i)
    {
        auto sample = [&]()
        {
            int e = int(rng() % 277) - 148;
            float m = float(int(rng() % 65535) - 32767) / 32768.f;
            return std::ldexp(m, e);
        };
        aclFloatComplex a{sample(), sample()}, b{sample(), sample()};
        if (b.real != 0 || b.imag != 0) Check(a, b);
    }
}

int main()
{
    CheckRangeBoundaries();
    CheckRandomInputs();
    const float tiny = std::numeric_limits<float>::denorm_min();
    for (float v : {tiny, 3 * tiny, 127 * tiny, 0x1p-126f})
    {
        Check({v, v}, {v, 3 * v});
        Check({3 * v, -v}, {v, 3 * v});
    }
    printf("componentwise FP32 cases=%d failures=%d (four production paths); CPU scalar execution only\n", cases,
           failures);
    return failures != 0;
}
