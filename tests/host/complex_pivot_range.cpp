/**
 * Copyright (c) 2026 Xztxzt
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pthread.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#include "cann_ops_solver_common.h"
#define __simt_callee__
#define __aicore__
#define __gm__
#define __ubuf__
struct Lane
{
    int x;
};
static thread_local Lane threadIdx{0};
static Lane blockDim{32};
static pthread_barrier_t barrier;
inline void asc_syncthreads() { pthread_barrier_wait(&barrier); }
template <class T>
T* Exchange()
{
    static T values[32];
    return values;
}
template <class T>
T asc_shfl(T value, int lane, int width = 32)
{
    if (width <= 0 || width > 32) std::abort();
    T* values = Exchange<T>();
    values[threadIdx.x] = value;
    asc_syncthreads();
    T got = values[threadIdx.x / width * width + lane];
    asc_syncthreads();
    return got;
}
template <class T>
T asc_shfl_down(T value, int offset, int width = 32)
{
    if (width <= 0 || width > 32) std::abort();
    T* values = Exchange<T>();
    values[threadIdx.x] = value;
    asc_syncthreads();
    int lane = threadIdx.x % width;
    T got = values[lane + offset < width ? threadIdx.x + offset : threadIdx.x];
    asc_syncthreads();
    return got;
}
template <class T>
T Reduce(T value, bool maximum)
{
    T* values = Exchange<T>();
    values[threadIdx.x] = value;
    asc_syncthreads();
    T got = values[0];
    for (int i = 1; i < 32; ++i) got = maximum ? std::max(got, values[i]) : std::min(got, values[i]);
    asc_syncthreads();
    return got;
}
template <class T>
T asc_reduce_max(T value)
{
    return Reduce(value, true);
}
template <class T>
T asc_reduce_min(T value)
{
    return Reduce(value, false);
}
#include "production.inc"
template <class T>
T ScoreType(void (*)(T&, int&));
using Score = decltype(ScoreType(&ReducePivot));

struct Case
{
    std::vector<aclFloatComplex> values;
    int winner;
};
std::vector<Case> PivotCases()
{
    std::vector<Case> cases;
    float max = std::numeric_limits<float>::max(), tiny = std::numeric_limits<float>::denorm_min();
    for (int e : {4, 126, 127})
    {
        float v = std::ldexp(1.f, e);
        cases.push_back({{{v, v}, {1.5f * v, 1.5f * v}}, 1});
        cases.push_back({{{1.5f * v, 1.5f * v}, {v, v}}, 0});
        cases.push_back({{{v, v}, {v, -v}}, 0});
    }
    cases.push_back({{{max, 0}, {max, max}}, 1});
    cases.push_back({{{tiny, 0}, {2 * tiny, 0}}, 1});
    cases.push_back({{{0, tiny}, {tiny, 0}}, 0});
    cases.push_back({{{0, 0}, {0, 0}}, 0});
    cases.push_back({{{1, 2}, {2, 1}}, 0});
    cases.push_back({{{max, 1}, {1, max}}, 0});
    for (auto nonzero : {aclFloatComplex{tiny, 0}, {1, 0}, {max, max}})
    {
        cases.push_back({{{-0.0f, -0.0f}, nonzero}, 1});
        cases.push_back({{nonzero, {-0.0f, -0.0f}}, 0});
    }
    cases.push_back({{{-0.0f, -0.0f}, {0.0f, 0.0f}}, 0});
    cases.push_back({{{0.0f, 0.0f}, {-0.0f, -0.0f}}, 0});
    // Winner across warp iteration boundaries, including the last partial iteration.
    for (int n : {33, 65})
    {
        Case c{std::vector<aclFloatComplex>(n, {tiny, 0}), n - 1};
        c.values.back() = {2 * tiny, 0};
        cases.push_back(c);
    }
    return cases;
}

std::array<int, 5> RunPivotPaths(Case& c)
{
    Score scores[1]{};
    int rows[1]{};
    std::array<int, 5> got{};
    // Shared layout for RegisterPivot<4>: row0,col0 at0; row1,col0 at1.
    aclFloatComplex shared[20]{};
    for (int i = 0; i < std::min(4, int(c.values.size())); ++i) shared[i * 5] = c.values[i];
    std::vector<std::thread> workers;
    for (int lane = 0; lane < 32; ++lane)
        workers.emplace_back(
            [&, lane]
            {
                threadIdx.x = lane;
                int p = FindPivot(c.values.data(), int(c.values.size()), int(c.values.size()), 0, scores, rows);
                if (lane == 0) got[0] = p;
                p = FindCachedPivot(c.values.data(), int(c.values.size()), 0, scores, rows);
                if (lane == 0) got[1] = p;
                if (c.values.size() == 2)
                {
                    LaneMatrix<2> small;
                    small.Set(0, c.values[0]);
                    small.Set(1, c.values[1]);
                    p = RegisterPivot<2>(small, 2, 0, lane % 2);
                    if (lane == 0) got[2] = p;
                    LaneMatrix<4> matrix;
                    matrix.column = shared + lane % 4;
                    p = RegisterPivot<4>(matrix, 2, 0, lane % 4);
                    if (lane == 0) got[3] = p;
                    RegisterMatrix<4> row;
                    row.Set(0, lane % 4 < 2 ? c.values[lane % 4] : aclFloatComplex{0, 0});
                    p = DirectRowPivot<4>(row, 0, 2, lane % 4);
                    if (lane == 0) got[4] = p;
                }
            });
    for (auto& t : workers) t.join();
    return got;
}

void CheckPivots(int& checks, int& failures)
{
    auto cases = PivotCases();
    for (auto& c : cases)
    {
        auto got = RunPivotPaths(c);
        int paths = c.values.size() == 2 ? 5 : 2;
        for (int path = 0; path < paths; ++path)
        {
            ++checks;
            if (got[path] != c.winner)
            {
                ++failures;
                printf("pivot case=%d path=%d expected=%d got=%d\n", int(&c - cases.data()), path, c.winner, got[path]);
            }
        }
    }
}

void CheckFactorization(int& checks, int& failures)
{
    for (int e : {4, 126, 127})
    {
        float v = std::ldexp(1.f, e);
        aclFloatComplex matrix[4] = {{v, v}, {1.5f * v, 1.5f * v}, {v, 0}, {-v, 0}};
        Score scores[1]{};
        int rows[1]{}, pivots[2]{}, info = 0;
        std::vector<std::thread> workers;
        for (int lane = 0; lane < 32; ++lane)
            workers.emplace_back(
                [&, lane]
                {
                    threadIdx.x = lane;
                    for (int col = 0; col < 2; ++col)
                        FactorColumn(matrix, 2, 2, 2, col, 2, pivots, &info, scores, rows);
                });
        for (auto& t : workers) t.join();
        bool finite = true;
        for (auto z : matrix) finite &= std::isfinite(z.real) && std::isfinite(z.imag);
        ++checks;
        if (pivots[0] != 2 || pivots[1] != 2 || info != 0 || !finite)
        {
            ++failures;
            printf("full factor exponent=%d pivot0=%d finite=%d\n", e, pivots[0], finite);
        }
    }
}

int main()
{
    int failures = 0, checks = 0;
    pthread_barrier_init(&barrier, nullptr, 32);
    CheckPivots(checks, failures);
    CheckFactorization(checks, failures);
    pthread_barrier_destroy(&barrier);
    printf("pivot checks=%d failures=%d; five production paths with CPU thread exchange only\n", checks, failures);
    return failures != 0;
}
