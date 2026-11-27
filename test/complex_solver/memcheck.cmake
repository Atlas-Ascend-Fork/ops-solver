# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Some mssanitizer versions return zero even when the application fails.
execute_process(
    COMMAND mssanitizer --tool=memcheck -- "${TEST_EXECUTABLE}" --memory-regression
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
)
set(report "${output}\n${errors}")
message("${report}")
string(JOIN "|" _sanitizer_diag_re
    "====== (ERROR|WARNING|FATAL)"
    "[[]warning[]]"
    "[[]mssanitizer[]] (ERROR|WARN|WARNING|FATAL):"
    "See all detected errors above"
    "detected memory leaks"
    "[1-9][0-9]* byte([(]s[)]|s)? leaked"
    "exited abnormally"
)
if(NOT "${status}" STREQUAL "0" OR
   report MATCHES "${_sanitizer_diag_re}" OR
   NOT report MATCHES "complex solver focused regression passed")
    message(FATAL_ERROR "Memory regression failed or sanitizer coverage is incomplete; inspect the report above.")
endif()
foreach(kernel complex_factor_distributed_panel complex_tiny_inverse)
    if(NOT report MATCHES "Sanitizer finished on kernel[^\n]*${kernel}[^\n]*[.] No error detected[.]")
        message(FATAL_ERROR "Missing clean sanitizer coverage for ${kernel}.")
    endif()
endforeach()
