# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if("arch35" IN_LIST SOC_ARCH_DIRS AND NOT TARGET ${QR_ADAPTER_TARGET}_adapter)
    add_library(${QR_ADAPTER_TARGET}_adapter SHARED
        ${CMAKE_SOURCE_DIR}/test/${QR_ADAPTER_TARGET}/${QR_ADAPTER_TARGET}_adapter.cpp)
    target_include_directories(${QR_ADAPTER_TARGET}_adapter PRIVATE
        ${CMAKE_SOURCE_DIR}/include ${ASCEND_CANN_PACKAGE_PATH}/include)
    target_link_libraries(${QR_ADAPTER_TARGET}_adapter PRIVATE
        ${OPS_SOLVER} ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so)
    target_link_options(${QR_ADAPTER_TARGET}_adapter PRIVATE -Wl,--no-undefined)
    install(TARGETS ${QR_ADAPTER_TARGET}_adapter LIBRARY DESTINATION lib)
endif()

if(NOT "arch35" IN_LIST SOC_ARCH_DIRS AND NOT TARGET ${QR_ADAPTER_TARGET}_arch_mismatch)
    add_executable(${QR_ADAPTER_TARGET}_arch_mismatch ${CMAKE_SOURCE_DIR}/test/complex_qr/arch_mismatch.c)
    target_include_directories(${QR_ADAPTER_TARGET}_arch_mismatch PRIVATE
        ${CMAKE_SOURCE_DIR}/include ${ASCEND_CANN_PACKAGE_PATH}/include)
    target_link_libraries(${QR_ADAPTER_TARGET}_arch_mismatch PRIVATE ${OPS_SOLVER})
endif()

if("arch35" IN_LIST SOC_ARCH_DIRS AND NOT TEST ${QR_ADAPTER_TARGET}_independent)
    add_test(NAME ${QR_ADAPTER_TARGET}_independent
        COMMAND ${CMAKE_COMMAND} -E env --unset=COMPLEX_QR_CASES_ROOT
            bash ${CMAKE_SOURCE_DIR}/test/${QR_ADAPTER_TARGET}/run_tests.sh ${CMAKE_BINARY_DIR})
    set_tests_properties(${QR_ADAPTER_TARGET}_independent PROPERTIES
        RESOURCE_LOCK solver_npu TIMEOUT 600 LABELS "npu;operator")
endif()
