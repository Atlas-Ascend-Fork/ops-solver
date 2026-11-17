# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


if(NOT SOC_VERSION_LOWER MATCHES "^ascend950")
    # 复数 LU 为 950 专属（文档契约：其他架构返回 ARCH_MISMATCH），测试断言
    # SUCCESS 必失败：对齐 #295 门控记 SKIPPED（issue #491）。cgetrs 为 A2/A3
    # legacy 接口，不在此列。
    message(WARNING "cmatinv_batched tests require Ascend950, skipping on ${SOC_VERSION_LOWER}")
    return()
endif()

add_executable(cmatinv_batched_test
    cmatinv_batched_test.cpp
)

target_include_directories(cmatinv_batched_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(cmatinv_batched_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
