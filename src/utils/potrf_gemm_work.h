/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file potrf_gemm_work.h
 * \brief cpotrf/spotrf host 侧 GEMM 分块调度共用的数据结构与入队逻辑（单一来源），
 * 消除两份 host 实现之间的重复定义与重复代码块（issue #336 CodeCheck 整改）。
 */

#ifndef ACLSOLVER_UTILS_POTRF_GEMM_WORK_H
#define ACLSOLVER_UTILS_POTRF_GEMM_WORK_H

#include <queue>
#include <vector>

// GEMM 分块调度单元：x 为次对角 tile 行号，y 为列组下标，z 为 tile 高度
struct L
{
    int x, y, z;
    // 优先队列为小顶堆：x 越小优先级越高，x 相同时 y 越小优先级越高
    bool operator>(const L &other) const
    {
        if (x != other.x) return x > other.x;
        return y > other.y;
    }
};

// 将第 i 列的次对角 tile（t = i+1 .. n-1）以 (t, i-1, n-t) 入队，供各列配额消费
inline void PushColumnTiles(std::priority_queue<L, std::vector<L>, std::greater<L>> &pq, int n, int i)
{
    for (int t = i + 1; t <= n - 1; ++t)
    {
        L item;
        item.x = t;
        item.y = i - 1;
        item.z = n - t;
        pq.push(item);
    }
}

#endif  // ACLSOLVER_UTILS_POTRF_GEMM_WORK_H
