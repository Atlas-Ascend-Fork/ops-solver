# PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
#
# Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
# All rights reserved.
#
# This software is dual-licensed:
#   1) Community license: use, modify, and redistribute freely under the
#      GNU General Public License v3 or later (GPL-3.0-or-later);
#   2) Commercial license: for closed-source or commercial use / redistribution,
#      contact the copyright holder for a separate license.
# You may choose either. Default terms are GPL-3.0-or-later.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you may redistribute it and/or modify it under
# the terms of the GPL. It is distributed "as is", without warranty.
# See COPYING-PACO (full GPL) and LICENSE-PACO (dual licensing notice) at the repo root.
# Commercial licensing contact: yuantang@fudan.edu.cn
#
# -------------------------------------------------------------------------
# 中文版 (Chinese):
#
# PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
#
# 本软件采用双重许可（dual licensing）：
#   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
#      （GPL-3.0-or-later）自由使用、修改与再分发；
#   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
# 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
#
# 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
# 详情见仓库根目录 COPYING-PACO（GPL 全文）与 LICENSE-PACO（双重许可说明）。
# 商业授权联系：yuantang@fudan.edu.cn


# 原 IN_LIST 门控极性反了：NOT arch35 AND NOT arch20 时才 add_executable——
# 恰好只在无 arch35/arch20 的 SoC（如 910b，走 unsupported 桩）上构建 cgeqrf_test，
# 950 上反而缺失（build.sh --run 记 "Test binary not found" FAILED）。本次按
# #295 口径重写：950 构建，非 950 发 WARNING 记 SKIPPED（issue #490）。
if(NOT SOC_VERSION_LOWER MATCHES "^ascend950")
    message(WARNING "cgeqrf tests require Ascend950, skipping on ${SOC_VERSION_LOWER}")
else()
add_executable(cgeqrf_test
    cgeqrf_test.cpp
)

target_include_directories(cgeqrf_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(cgeqrf_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
endif()

set(QR_ADAPTER_TARGET cgeqrf)
include(${CMAKE_CURRENT_LIST_DIR}/../complex_qr/adapter.cmake)
