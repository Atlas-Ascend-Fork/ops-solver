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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sgetrf_tests LANGUAGES CXX)
    set(OPS_SOLVER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "ops-solver source root")
    set(OPS_SOLVER_LIBRARY "" CACHE FILEPATH "fixed libops_solver.so")
    if(NOT EXISTS "${OPS_SOLVER_LIBRARY}")
        message(FATAL_ERROR "Set OPS_SOLVER_LIBRARY to the fixed libops_solver.so")
    endif()
    set(OPS_SOLVER "${OPS_SOLVER_LIBRARY}")
elseif(NOT DEFINED OPS_SOLVER_SOURCE_DIR)
    set(OPS_SOLVER_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
endif()

find_path(sgetrf_ACL_INCLUDE acl/acl.h
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/include" "$ENV{ASCEND_HOME_PATH}/include" REQUIRED)
find_library(sgetrf_ACL_LIBRARY ascendcl
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" "$ENV{ASCEND_HOME_PATH}/lib64" REQUIRED)

add_library(sgetrf_adapter SHARED sgetrf_adapter.cpp)
target_compile_features(sgetrf_adapter PRIVATE cxx_std_17)
target_include_directories(sgetrf_adapter PRIVATE "${OPS_SOLVER_SOURCE_DIR}/include" "${sgetrf_ACL_INCLUDE}")
target_link_libraries(sgetrf_adapter PRIVATE ${OPS_SOLVER} "${sgetrf_ACL_LIBRARY}")
set_target_properties(sgetrf_adapter PROPERTIES OUTPUT_NAME sgetrf_adapter INSTALL_RPATH "$ORIGIN")
install(TARGETS sgetrf_adapter LIBRARY DESTINATION lib64)

add_executable(sgetrf_test sgetrf_test.cpp)
target_compile_features(sgetrf_test PRIVATE cxx_std_17)
target_include_directories(sgetrf_test PRIVATE "${OPS_SOLVER_SOURCE_DIR}/include" "${sgetrf_ACL_INCLUDE}")
target_link_libraries(sgetrf_test PRIVATE sgetrf_adapter ${OPS_SOLVER} "${sgetrf_ACL_LIBRARY}")
target_compile_options(sgetrf_test PRIVATE -fno-fast-math -ffp-contract=off)

add_executable(sgetrf_subnormal_pivot_test sgetrf_subnormal_pivot_test.cpp)
target_compile_features(sgetrf_subnormal_pivot_test PRIVATE cxx_std_17)
target_include_directories(sgetrf_subnormal_pivot_test PRIVATE "${OPS_SOLVER_SOURCE_DIR}/include" "${sgetrf_ACL_INCLUDE}")
target_link_libraries(sgetrf_subnormal_pivot_test PRIVATE ${OPS_SOLVER} "${sgetrf_ACL_LIBRARY}")
target_compile_options(sgetrf_subnormal_pivot_test PRIVATE -fno-fast-math -ffp-contract=off)
