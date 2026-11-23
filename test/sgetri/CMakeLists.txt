# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)

# 实数 LU 为 950 专属实现（test/sgetri/README.md「支持的产品」仅 Ascend 950PR，
# 接口文档产品表 A2/A3 为待验证）。非 950 构建下 kernel 不可执行，测试断言
# SUCCESS 必失败：对齐 #488/#491 的门控口径，configure 期告警后计入 SKIPPED。
# 独立构建（未定义 SOC_VERSION_LOWER）不门控，由调用方提供匹配架构的固定库。
if(DEFINED SOC_VERSION_LOWER AND NOT SOC_VERSION_LOWER MATCHES "^ascend950")
    message(WARNING "sgetri tests require Ascend950, skipping on ${SOC_VERSION_LOWER}")
    return()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sgetri_tests LANGUAGES CXX)
    set(OPS_SOLVER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "ops-solver source root")
    set(OPS_SOLVER_LIBRARY "" CACHE FILEPATH "fixed libops_solver.so")
    if(NOT EXISTS "${OPS_SOLVER_LIBRARY}")
        message(FATAL_ERROR "Set OPS_SOLVER_LIBRARY to the fixed libops_solver.so")
    endif()
    set(OPS_SOLVER "${OPS_SOLVER_LIBRARY}")
elseif(NOT DEFINED OPS_SOLVER_SOURCE_DIR)
    set(OPS_SOLVER_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
endif()

find_path(sgetri_ACL_INCLUDE acl/acl.h
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/include" "$ENV{ASCEND_HOME_PATH}/include" REQUIRED)
find_library(sgetri_ACL_LIBRARY ascendcl
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" "$ENV{ASCEND_HOME_PATH}/lib64" REQUIRED)

add_library(sgetri_adapter SHARED sgetri_adapter.cpp)
target_compile_features(sgetri_adapter PRIVATE cxx_std_17)
target_include_directories(sgetri_adapter PRIVATE "${OPS_SOLVER_SOURCE_DIR}/include" "${sgetri_ACL_INCLUDE}")
target_link_libraries(sgetri_adapter PRIVATE ${OPS_SOLVER} "${sgetri_ACL_LIBRARY}")
set_target_properties(sgetri_adapter PROPERTIES OUTPUT_NAME sgetri_adapter INSTALL_RPATH "$ORIGIN")
# Standalone-only：in-tree 构建（build.sh --pkg）不得把测试 adapter 装进发布包（issue #355）
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    install(TARGETS sgetri_adapter LIBRARY DESTINATION lib64)
endif()

add_executable(sgetri_test sgetri_test.cpp)
target_compile_features(sgetri_test PRIVATE cxx_std_17)
target_include_directories(sgetri_test PRIVATE "${OPS_SOLVER_SOURCE_DIR}/include" "${sgetri_ACL_INCLUDE}")
target_link_libraries(sgetri_test PRIVATE sgetri_adapter ${OPS_SOLVER} "${sgetri_ACL_LIBRARY}")
target_compile_options(sgetri_test PRIVATE -fno-fast-math -ffp-contract=off)
