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
    # 五算子源码全部位于 arch35/（仅 ascend950* 映射，见根 CMakeLists get_soc_arch_dirs），非 950 下
    # 库中无算子符号、测试链接必然失败：对齐 #295 门控跳过而非中断（issue #488）。
    message(WARNING "xlarft tests require Ascend950, skipping on ${SOC_VERSION_LOWER}")
    return()
endif()

add_executable(xlarft_test
    xlarft_test.cpp
)

target_include_directories(xlarft_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(xlarft_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)

# 全量 case 清单（取自任务包 cases.json）要和二进制同目录，sweep 模式默认读 ./data/
add_custom_command(TARGET xlarft_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data $<TARGET_FILE_DIR:xlarft_test>/data
)
