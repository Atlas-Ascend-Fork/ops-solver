# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT "arch35" IN_LIST SOC_ARCH_DIRS)
    add_executable(complex_qr_arch_mismatch arch_mismatch.c)
    target_include_directories(complex_qr_arch_mismatch PRIVATE
        ${CMAKE_SOURCE_DIR}/include ${ASCEND_CANN_PACKAGE_PATH}/include)
    target_link_libraries(complex_qr_arch_mismatch PRIVATE ${OPS_SOLVER})
    add_test(NAME complex_qr_arch_mismatch COMMAND complex_qr_arch_mismatch)
    return()
endif()

find_package(LAPACK REQUIRED)
add_executable(complex_qr_public_abi public_abi.c)
add_executable(complex_qr_profile_case profile_case.cpp)
foreach(target complex_qr_public_abi complex_qr_profile_case)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include
    )
    target_link_libraries(${target} PRIVATE
        ${OPS_SOLVER}
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so
    )
endforeach()
add_test(NAME complex_qr_supplementary
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/run_tests.sh ${CMAKE_BINARY_DIR})
set_tests_properties(complex_qr_supplementary PROPERTIES
    RESOURCE_LOCK solver_npu TIMEOUT 1800 LABELS "npu;complex_qr")
target_link_libraries(complex_qr_profile_case PRIVATE ${LAPACK_LIBRARIES})

foreach(op cgeqrf cungqr cunmqr ccgels)
    set(QR_ADAPTER_TARGET ${op})
    include(${CMAKE_CURRENT_LIST_DIR}/adapter.cmake)
endforeach()
