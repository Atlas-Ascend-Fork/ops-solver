# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if("arch35" IN_LIST SOC_ARCH_DIRS)
    # Shared family sources are nested below utils/complex_qr, outside the
    # repository's direct src/<operator>/arch35 discovery pattern.
    file(GLOB COMPLEX_QR_SHARED_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/../complex_qr/arch35/*.cpp")
    target_sources(${OPS_SOLVER} PRIVATE ${COMPLEX_QR_SHARED_SOURCES})
    set_source_files_properties(${COMPLEX_QR_SHARED_SOURCES} PROPERTIES LANGUAGE ASC)
    # The paired CANN profiler requires bounded panel VF loops to be materialized.
    # Apply this only to the affected sources; retain shared optimization flags.
    set_property(SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../cgeqrf/arch35/cgeqrf_panel_kernel.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../../cunmqr/arch35/complex_qr_pack_reflectors_kernel.cpp"
        APPEND PROPERTY COMPILE_OPTIONS
        -Xaicore-start -mllvm -enable-hiipu-vf-loop-unroll=true -Xaicore-end)
    target_link_libraries(${OPS_SOLVER} PRIVATE register)
else()
    # A native Host source retains the public ABI without compiling arch35 kernels.
    target_include_directories(${OPS_SOLVER} PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
    set(COMPLEX_QR_UNSUPPORTED_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/../../cgeqrf/cgeqrf_unsupported.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../../cungqr/cungqr_unsupported.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../../cunmqr/cunmqr_unsupported.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../../ccgels/ccgels_unsupported.cpp")
    target_sources(${OPS_SOLVER} PRIVATE ${COMPLEX_QR_UNSUPPORTED_SOURCES})
    set_source_files_properties(${COMPLEX_QR_UNSUPPORTED_SOURCES} PROPERTIES LANGUAGE CXX)
    # Keep the ASC link driver and its runtime search paths when adding native C++.
    set_property(TARGET ${OPS_SOLVER} PROPERTY LINKER_LANGUAGE ASC)
endif()
