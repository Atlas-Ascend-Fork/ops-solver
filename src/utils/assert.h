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
 * \file assert.h
 * \brief error-check macros shared by solver host implementations.
 */

#ifndef ACLSOLVER_UTILS_ASSERT_H
#define ACLSOLVER_UTILS_ASSERT_H

#include <cstdio>
#include <iostream>

#define SOLVER_ECHECK(condition, logExpr, handleExpr)                                   \
    do                                                                                  \
    {                                                                                   \
        if (!(condition))                                                               \
        {                                                                               \
            flockfile(stderr);                                                          \
            fprintf(stderr, "[ERROR] %s | Return value: %s\n", (logExpr), #handleExpr); \
            fflush(stderr);                                                             \
            funlockfile(stderr);                                                        \
            return handleExpr;                                                          \
        }                                                                               \
    } while (0)

// 注意（issue #181）：SOLVER_CHECK 与下方 CHECK_RET 当前在全库无调用点，属预留宏。
// 与 SOLVER_ECHECK 的关键差异：SOLVER_CHECK 条件失败时仅打印日志并执行 handleExpr，
// **不做 return**，失败后代码继续向下执行——不得将其误用作"失败即返回"的参数校验；
// 需要失败即返回时请使用 SOLVER_ECHECK。若确认永不使用可整体删除。
#define SOLVER_CHECK(condition, logExpr, handleExpr)    \
    do                                                  \
    {                                                   \
        if (!(condition))                               \
        {                                               \
            flockfile(stderr);                          \
            fprintf(stderr, "[ERROR] %s\n", (logExpr)); \
            fflush(stderr);                             \
            funlockfile(stderr);                        \
            handleExpr;                                 \
        }                                               \
    } while (0)

// 注意（issue #181）：CHECK_RET 当前在全库无调用点，属预留宏；条件失败时执行
// return_expr——仅适用于"条件不满足则离开当前函数"的语义，命名未体现返回动作，
// 使用时注意与 SOLVER_ECHECK（自带日志）区分。
#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define CHECK_ACLRT(func, cleanup_action)                                                                            \
    {                                                                                                                \
        aclError status = (func);                                                                                    \
        if (status != ACL_SUCCESS)                                                                                   \
        {                                                                                                            \
            std::cerr << "ACL Runtime Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" \
                      << std::endl;                                                                                  \
            cleanup_action;                                                                                          \
            return status;                                                                                           \
        }                                                                                                            \
    }

#endif  // ASSERT_H
