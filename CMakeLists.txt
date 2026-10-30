# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(ops_solver VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========= 基本配置 =========
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  set(SOC_VERSION ascend910b)
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
# SOC_VERSION -> NPU_ARCH 映射（ascend910b->dav-2201, ascend950->dav-3510）
if(SOC_VERSION_LOWER MATCHES "^ascend910b")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-2002")
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*")
endif()
message(STATUS "SOC_VERSION=${SOC_VERSION}, NPU_ARCH=${NPU_ARCH}")

# ========= SOC 架构目录配置 =========
# 根据 SOC 返回需要编译的架构目录列表
function(get_soc_arch_dirs soc_version arch_dirs)
  string(TOLOWER "${soc_version}" soc_lower)
  set(dirs "")

  if(soc_lower MATCHES "^ascend950")
    list(APPEND dirs "arch35")
  elseif(soc_lower MATCHES "^ascend310p")
    list(APPEND dirs "arch20")
  endif()
  # 其他 SOC 可以根据需要添加

  set(${arch_dirs} ${dirs} PARENT_SCOPE)
endfunction()

# 获取当前 SOC 对应的架构目录
get_soc_arch_dirs(${SOC_VERSION} SOC_ARCH_DIRS)
message(STATUS "SOC_ARCH_DIRS=${SOC_ARCH_DIRS}")

# 架构特定目录列表（全局定义）
set(ARCH_SPECIFIC_DIRS arch35 arch20 CACHE INTERNAL "Architecture specific directories")

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/build_out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)

# ========= 构建动态库 =========
find_package(ASC REQUIRED)
set(CMAKE_SHARED_LIBRARY_RUNTIME_ASC_FLAG "-Wl,-rpath,")
set(CMAKE_SHARED_LIBRARY_RUNTIME_ASC_FLAG_SEP ":")

# solver
set(ALL_SOLVER_SRC_FILES "")
set(OPS_SOLVER ops_solver)

add_subdirectory(src)

project(${OPS_SOLVER} LANGUAGES ASC CXX)

if(SOC_VERSION_LOWER MATCHES "^ascend950")
  list(FILTER ALL_SOLVER_SRC_FILES EXCLUDE REGEX "/cholesky/cholesky_unsupported.cpp$")
  # Pure SIMT has its own device link mode; keep it separate from SIMD/Cube code.
  list(FILTER ALL_SOLVER_SRC_FILES EXCLUDE REGEX "/cholesky/arch35/cholesky_simt.cpp$")
  add_library(cholesky_simt SHARED src/cholesky/arch35/cholesky_simt.cpp)
  set_source_files_properties(src/cholesky/arch35/cholesky_simt.cpp PROPERTIES LANGUAGE ASC)
  set_target_properties(cholesky_simt PROPERTIES LINKER_LANGUAGE ASC BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
  target_include_directories(cholesky_simt PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_compile_options(cholesky_simt PRIVATE --npu-arch=${NPU_ARCH} --enable-simt -O2)
  target_link_options(cholesky_simt PRIVATE --npu-arch=${NPU_ARCH} --enable-simt)
  target_link_directories(cholesky_simt PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
  target_link_libraries(cholesky_simt PRIVATE ascendcl)
endif()

add_library(${OPS_SOLVER} SHARED ${ALL_SOLVER_SRC_FILES})
set_target_properties(${OPS_SOLVER} PROPERTIES LINKER_LANGUAGE ASC BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

set_source_files_properties(
    ${ALL_SOLVER_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

# Real LU accepts every finite FLOAT32 value, including subnormal pivots.
# The 950 compiler defaults to flushing these values to zero.
if(SOC_VERSION_LOWER MATCHES "^ascend950")
    # CANN 9.0 cannot dynamically index VF register arrays. Generate the small
    # straight-line variants in the build tree instead of storing expanded code.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(REAL_LU_VF_DIR "${CMAKE_CURRENT_BINARY_DIR}/real_lu_generated")
    set(REAL_LU_VF_HEADER "${REAL_LU_VF_DIR}/real_lu_fixed_vf.hpp")
    add_custom_command(OUTPUT "${REAL_LU_VF_HEADER}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_real_lu_fixed_vf.py"
                --output "${REAL_LU_VF_HEADER}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_real_lu_fixed_vf.py"
        VERBATIM)
    add_custom_target(real_lu_fixed_vf DEPENDS "${REAL_LU_VF_HEADER}")
    add_dependencies(${OPS_SOLVER} real_lu_fixed_vf)
    target_include_directories(${OPS_SOLVER} PRIVATE "${REAL_LU_VF_DIR}")
    set(REAL_LU_VF_STACK_BYTES 16384)
    target_compile_definitions(${OPS_SOLVER} PRIVATE
        REAL_LU_REGISTER_LAYOUT_950=1 REAL_LU_VF_STACK_BYTES=${REAL_LU_VF_STACK_BYTES})
    set_property(SOURCE
        src/sgetrf/sgetrf_kernel.cpp
        src/sgetrf/real_lu_blocked_kernel.cpp
        src/sgetri/sgetri_kernel.cpp
        src/sgetri/real_lu_blocked_inverse_kernel.cpp
        APPEND PROPERTY COMPILE_OPTIONS --cce-ftz=false
    )
    set_property(SOURCE src/sgetri/sgetri_kernel.cpp APPEND PROPERTY COMPILE_OPTIONS
        "-mllvm" "-cce-vf-stack-size=${REAL_LU_VF_STACK_BYTES}")
endif()

if(SOC_VERSION_LOWER MATCHES "^ascend950")
    set_source_files_properties(src/cpotrf/arch35/cholesky_kernels.cpp
        src/cpotrf/arch35/cholesky_factor.cpp
        src/cpotrf/arch35/cholesky_panel.cpp
        src/cpotrf/arch35/cholesky_native.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;--cce-long-scbz=true")
    if(NOT DEFINED CANN_3RD_LIB_PATH)
        set(CANN_3RD_LIB_PATH "${CMAKE_BINARY_DIR}/third_party")
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/catlass-fetch.cmake)
    target_link_libraries(${OPS_SOLVER} PRIVATE OPS_SOLVER::catlass)
    set_source_files_properties(src/cpotrf/arch35/cholesky_schur.cpp
        PROPERTIES COMPILE_OPTIONS "-O3")
endif()
set_source_files_properties(
    src/utils/aclsolver_auxiliary.cpp
    src/cholesky/arch35/cholesky_host.cpp
    src/cholesky/cholesky_unsupported.cpp
    PROPERTIES LANGUAGE CXX)

target_include_directories(${OPS_SOLVER} BEFORE PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
)

target_compile_options(${OPS_SOLVER} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    $<$<COMPILE_LANGUAGE:ASC>:-iquote${CMAKE_CURRENT_LIST_DIR}/include>
)
target_link_options(${OPS_SOLVER} PRIVATE --npu-arch=${NPU_ARCH})

if(SOC_VERSION_LOWER MATCHES "^ascend950")
  target_link_libraries(${OPS_SOLVER} PRIVATE cholesky_simt)
  set_source_files_properties(
    src/cholesky/arch35/cholesky_cube.cpp
    src/cholesky/arch35/cholesky_vector.cpp
    PROPERTIES COMPILE_OPTIONS "-O3")
endif()

target_link_libraries(${OPS_SOLVER} PRIVATE
    tiling_api
    platform
    register
    ascendcl
)
target_link_directories(${OPS_SOLVER} PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)

# ========= 安装规则（CPack 打包时放入 lib64 和 include） =========
install(TARGETS ${OPS_SOLVER}
    LIBRARY DESTINATION lib64
)
if(TARGET cholesky_simt)
  install(TARGETS cholesky_simt LIBRARY DESTINATION lib64)
endif()
install(FILES
    include/cann_ops_solver.h
    include/cann_ops_solver_common.h
    DESTINATION include
)

# ========= 构建测试程序 =========
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    add_dependencies(${OPS_SOLVER} gen_ops_solver_version_info)
    pack(${SOC_VERSION})
endif()
