# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# A2/A3 legacy 接口（test/sgetrs/README.md「支持的产品」仅 Atlas A2/A3，
# 接口文档产品表 950 为 ×）。950 构建不生成测试，按 build.sh 的 configure
# WARNING 约定计入 SKIPPED，避免在 950 上误报算子执行失败。
if(SOC_VERSION_LOWER MATCHES "^ascend950")
    message(WARNING "sgetrs tests require Atlas A2/A3, skipping on ${SOC_VERSION_LOWER}")
    return()
endif()

add_executable(sgetrs_test
    sgetrs_test.cpp
)

target_include_directories(sgetrs_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(sgetrs_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)