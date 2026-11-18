# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Fault injection against complete getrs Host sources; no device execution."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
ACL_STUB = r'''#pragma once
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <cstdint>
using aclError = int;
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0, ACL_ERROR_INVALID_PARAM = 1, ACL_ERROR_RT_FEATURE_NOT_SUPPORT = 2;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0, ACL_MEMCPY_HOST_TO_DEVICE = 0, ACL_MEMCPY_DEVICE_TO_HOST = 1;
constexpr int ACL_RT_THREAD_LEVEL = 0;
inline std::string failure;
inline int failureIndex = 0, allocations = 0, copies = 0, frees = 0, duplicateFrees = 0, unsafeFrees = 0;
inline bool inFlight = false;
inline std::set<void*> live;
inline int aclrtMalloc(void** pointer, size_t size, int) {
    if (++allocations == failureIndex && failure == "allocation") return 91;
    *pointer = std::malloc(size);
    if (!*pointer) return 91;
    live.insert(*pointer);
    return ACL_SUCCESS;
}
inline int aclrtMallocHost(void** pointer, size_t size) { return aclrtMalloc(pointer, size, 0); }
inline int aclrtFree(void* pointer) {
    ++frees;
    if (inFlight) ++unsafeFrees;
    if (!live.erase(pointer)) { ++duplicateFrees; return 92; }
    std::free(pointer);
    return ACL_SUCCESS;
}
inline int aclrtFreeHost(void* pointer) { return aclrtFree(pointer); }
inline int aclrtMemcpy(void* dst, size_t, const void* src, size_t count, int) {
    if (++copies == failureIndex && failure == "copy") return 94;
    std::memcpy(dst, src, count);
    return ACL_SUCCESS;
}
inline int aclrtMemset(void* dst, size_t, int value, size_t count) {
    if (failure == "memset") return 95;
    std::memset(dst, value, count);
    return ACL_SUCCESS;
}
inline int aclrtSynchronizeStream(aclrtStream) {
    if (failure == "sync" || failure == "enqueue_sync") return 93;
    inFlight = false;
    return ACL_SUCCESS;
}
inline int aclrtPeekAtLastError(int) {
    return failure == "enqueue" || failure == "enqueue_sync" ? 96 : ACL_SUCCESS;
}
'''
DRIVER = r'''
extern "C" aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t, aclrtStream* stream) {
    *stream = nullptr;
    return ACLSOLVER_STATUS_SUCCESS;
}
void OPERATOR_kernel_do(int, int, int, int, int, int, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                       uint8_t*, uint8_t*COMPLEX_ARGUMENT, uint32_t, void*) { inFlight = true; }
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    failure = argv[1]; failureIndex = std::atoi(argv[2]);
    VALUE_TYPE a{}, b{};
    int32_t pivot = 1, info = -99;
    auto status = API(nullptr, 0, 1, 1, &a, 1, &pivot, &b, 1, &info);
    std::cout << "RESULT " << status << " " << allocations << " " << frees << " " << duplicateFrees
              << " " << unsafeFrees << " " << live.size() << " " << info << std::endl;
    // Test teardown only: retained allocations are owned by this process, never a device.
    for (auto pointer : live) std::free(pointer);
}
'''


@pytest.fixture(scope="module", params=["sgetrs", "cgetrs"])
def host_binary(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ is required for host fault injection")
    operator = request.param
    directory = tmp_path_factory.mktemp(operator)
    stubs = directory / "stubs"
    for folder in [stubs / "acl", stubs / "tiling/platform"]:
        folder.mkdir(parents=True)
    (stubs / "acl/acl.h").write_text(ACL_STUB)
    (stubs / "acl/acl_rt.h").write_text('#include "acl.h"\n')
    (stubs / "securec.h").write_text("#pragma once\n")
    (stubs / "tiling/platform/platform_ascendc.h").write_text(r'''#pragma once
namespace platform_ascendc {
struct PlatformAscendCManager {
    static PlatformAscendCManager* GetInstance() { static PlatformAscendCManager instance; return &instance; }
    unsigned GetCoreNumAic() { return 2; }
};
}
''')
    source = directory / "check.cpp"
    body = DRIVER.replace("OPERATOR", operator).replace("API", "aclsolver" + operator.capitalize())
    body = body.replace("VALUE_TYPE", "float" if operator == "sgetrs" else "aclFloatComplex")
    body = body.replace("COMPLEX_ARGUMENT", "" if operator == "sgetrs" else ", uint8_t*")
    source.write_text(f'#include "src/{operator}/{operator}_host.cpp"\n' + body)
    binary = directory / "check"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(stubs), "-I" + str(ROOT / "include"),
                    "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    return operator, binary


@pytest.mark.parametrize("failure,index", [
    *[("allocation", i) for i in range(1, 9)],
    *[("copy", i) for i in range(1, 6)],
    ("memset", 1), ("enqueue", 1), ("sync", 1), ("enqueue_sync", 1), ("none", 0),
])
def test_cleanup_has_one_owner(host_binary, failure, index):
    operator, binary = host_binary
    total_allocations = 6 if operator == "sgetrs" else 8
    total_copies = 4 if operator == "sgetrs" else 5
    stage_limits = {"allocation": total_allocations, "copy": total_copies}
    if failure in stage_limits and index > stage_limits[failure]:
        pytest.skip("This allocation/copy stage exists only in Cgetrs")
    run = subprocess.run([str(binary), failure, str(index)], capture_output=True, text=True, check=True)
    line = next(line for line in run.stdout.splitlines() if line.startswith("RESULT "))
    status, allocations, frees, duplicate, unsafe, live, info = map(int, line.split()[1:])
    expected_status = {"allocation": 91, "copy": 94, "memset": 95, "enqueue": 96,
                       "sync": 93, "enqueue_sync": 96, "none": 0}[failure]
    assert status == expected_status, run.stdout + run.stderr
    assert duplicate == 0, line
    assert unsafe == 0, line
    retained = total_allocations if failure in ("sync", "enqueue_sync") else 0
    assert live == retained, line
    successful_allocations = allocations - (failure == "allocation")
    assert frees == successful_allocations - retained, line
    assert info == (0 if failure == "none" else -99), line
