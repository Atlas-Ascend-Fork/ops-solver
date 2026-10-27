# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include_guard(GLOBAL)

set(CATLASS_COMMIT "e420328d38945dea53ac8ee56795215db20b943b")
set(CATLASS_PATH "${CANN_3RD_LIB_PATH}/catlass")
set(CATLASS_URL "https://raw.gitcode.com/cann/catlass/archive/refs/heads/${CATLASS_COMMIT}.tar.gz")
set(CATLASS_ARCHIVE "${CMAKE_SOURCE_DIR}/third_party/cache/catlass-${CATLASS_COMMIT}.tar.gz")
if(EXISTS "${CATLASS_ARCHIVE}")
    set(CATLASS_URL "${CATLASS_ARCHIVE}")
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(catlass
    URL "${CATLASS_URL}"
    URL_HASH SHA256=1718ad8b2a48bb7b7fbb18bbec110193ba6226a625793ad1346e57650a3df87a
    SOURCE_DIR "${CATLASS_PATH}"
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(catlass)

add_library(ops_solver_catlass_headers INTERFACE)
add_library(OPS_SOLVER::catlass ALIAS ops_solver_catlass_headers)
target_include_directories(ops_solver_catlass_headers INTERFACE "${CATLASS_PATH}/include")
target_compile_definitions(ops_solver_catlass_headers INTERFACE CATLASS_ARCH=3510)
