# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_executable(cholesky_destroy_test destroy_test.cpp ${CMAKE_SOURCE_DIR}/src/utils/aclsolver_auxiliary.cpp)
target_include_directories(cholesky_destroy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${ASCEND_CANN_PACKAGE_PATH}/include)
add_test(NAME cholesky_destroy_ownership COMMAND cholesky_destroy_test)
set_tests_properties(cholesky_destroy_ownership PROPERTIES LABELS "cholesky;cpu" TIMEOUT 30)

if(NOT SOC_VERSION_LOWER MATCHES "^ascend950")
    add_executable(cholesky_test unsupported_test.cpp)
    target_include_directories(cholesky_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include)
    target_link_libraries(cholesky_test PRIVATE ${OPS_SOLVER})
    add_test(NAME cholesky_arch_mismatch COMMAND cholesky_test)
    set_tests_properties(cholesky_arch_mismatch PROPERTIES LABELS "cholesky;cpu;abi" TIMEOUT 30)
    return()
endif()

add_executable(cholesky_test cholesky_test.cpp native_benchmark.cpp)
target_include_directories(cholesky_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_link_directories(cholesky_test PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(cholesky_test PRIVATE ${OPS_SOLVER} ascendcl dl)
add_test(NAME cholesky_c_api_smoke COMMAND cholesky_test --smoke)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME cholesky_lapack_residual
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_lapack_residual.py)

add_test(NAME cholesky_numerical_boundaries
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/runner.py
        --library $<TARGET_FILE:${OPS_SOLVER}>
        --sizes 1,8,9,17,32,33,64,65,128,129,141,201,257,509,513,1025,2048
        --batch 7 --nrhs 17
        --output ${CMAKE_CURRENT_BINARY_DIR}/numerical-boundaries.json)
add_test(NAME cholesky_numerical_large
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/runner.py
        --library $<TARGET_FILE:${OPS_SOLVER}>
        --sizes 4096 --batch 1 --nrhs 128 --padding 8
        --output ${CMAKE_CURRENT_BINARY_DIR}/numerical-large.json)
add_test(NAME cholesky_contracts
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/contracts.py
        --library $<TARGET_FILE:${OPS_SOLVER}>
        --output ${CMAKE_CURRENT_BINARY_DIR}/contracts.json)
add_library(cholesky_runtime_faults SHARED runtime_faults.cpp)
target_include_directories(cholesky_runtime_faults PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_libraries(cholesky_runtime_faults PRIVATE dl)
add_test(NAME cholesky_runtime_failures
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:cholesky_runtime_faults>"
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime_failures.py
        --library $<TARGET_FILE:${OPS_SOLVER}>
        --output ${CMAKE_CURRENT_BINARY_DIR}/runtime-failures.json)
set_tests_properties(cholesky_c_api_smoke cholesky_numerical_boundaries
    cholesky_numerical_large cholesky_contracts cholesky_runtime_failures PROPERTIES
    LABELS "cholesky;npu"
    RESOURCE_LOCK cholesky_npu
    TIMEOUT 1200
    ENVIRONMENT "OPENBLAS_NUM_THREADS=8;OMP_NUM_THREADS=8")
set_tests_properties(cholesky_lapack_residual PROPERTIES
    LABELS "cholesky;cpu" TIMEOUT 120)
