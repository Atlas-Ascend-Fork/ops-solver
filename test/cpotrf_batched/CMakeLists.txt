# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


if(NOT SOC_VERSION_LOWER MATCHES "^ascend950")
    # 复数 Cholesky 为 950 专属（文档产品表），非 950 构建走 cholesky_unsupported
    # 桩（ARCH_MISMATCH），测试运行必失败：对齐 #295 门控记 SKIPPED（issue #489）。
    message(WARNING "cpotrf_batched tests require Ascend950, skipping on ${SOC_VERSION_LOWER}")
    return()
endif()

add_executable(cpotrf_batched_test
    cpotrf_batched_test.cpp
    ../utils/cholesky/fixture_runner.cpp
)
target_include_directories(cpotrf_batched_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)
target_link_libraries(cpotrf_batched_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
