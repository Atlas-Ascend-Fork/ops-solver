# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_executable(cheevj_test
    cheevj_test.cpp
)

add_executable(cheevj_benchmark
    cheevj_benchmark.cpp
)

add_executable(cheevj_contract_test
    cheevj_contract_test.cpp
)

# TARGET_DIRECTORY makes source properties visible in the directory where
# ops_solver is defined. It is optional here because CMake added it in 3.18.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_property(SOURCE
        ${CMAKE_SOURCE_DIR}/src/cheevj/cheevj_panel_kernel.cpp
        TARGET_DIRECTORY ${OPS_SOLVER}
        PROPERTY COMPILE_OPTIONS "-O2"
    )
    set_property(SOURCE
        ${CMAKE_SOURCE_DIR}/src/cheevj/cheevj_host.cpp
        TARGET_DIRECTORY ${OPS_SOLVER}
        PROPERTY COMPILE_OPTIONS "-O3"
    )
endif()

target_include_directories(cheevj_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_include_directories(cheevj_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_include_directories(cheevj_contract_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(cheevj_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)

target_link_libraries(cheevj_benchmark PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)

target_link_libraries(cheevj_contract_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)

add_executable(cheevj_small_scale_host_test
    cheevj_small_scale_host_test.cpp
)

target_include_directories(cheevj_small_scale_host_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(cheevj_small_scale_host_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
