# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# 收集当前目录下的基础源文件
file(GLOB BASE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 自动收集所有算子目录的源文件（支持子目录中的架构特定目录）
set(OP_SRC_FILES "")
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        # 递归收集非架构特定源码，避免与 archXX 目录重复收集
        file(GLOB_RECURSE dir_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${child}/*.cpp)
        foreach(src_file ${dir_srcs})
            set(is_arch_specific FALSE)
            foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
                if(src_file MATCHES "/${arch_dir}/")
                    set(is_arch_specific TRUE)
                    break()
                endif()
            endforeach()
            if(NOT is_arch_specific)
                list(APPEND OP_SRC_FILES ${src_file})
            endif()
        endforeach()
    endif()
endforeach()

# 收集子目录中的架构特定源文件
set(ARCH_SRC_FILES "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    file(GLOB_RECURSE arch_dir_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/*/${arch_dir}/*.cpp
    )
    list(APPEND ARCH_SRC_FILES ${arch_dir_srcs})
endforeach()

# Select the unsupported-architecture ABI source in complex_qr.cmake.
list(FILTER OP_SRC_FILES EXCLUDE REGEX "/(cgeqrf|cungqr|cunmqr|ccgels)_unsupported[.]cpp$")

set(ALL_SOLVER_SRC_FILES
    ${BASE_SRC_FILES}
    ${OP_SRC_FILES}
    ${ARCH_SRC_FILES}
    PARENT_SCOPE
)
