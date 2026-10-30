/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Test-only LD_PRELOAD failure injection; never installed with the solver.
#include <acl/acl.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>

namespace
{
std::atomic<int> failNext{0};
std::atomic<uint64_t> mallocCalls{0};
bool Take(int kind) { return failNext.compare_exchange_strong(kind, 0); }
}  // namespace
extern "C" void choleskyTestArm(int kind)
{
    mallocCalls = 0;
    failNext = kind;
}
extern "C" uint64_t choleskyTestMallocCount() { return mallocCalls.load(); }
extern "C" aclError aclrtMalloc(void **p, size_t bytes, aclrtMemMallocPolicy policy)
{
    ++mallocCalls;
    if (Take(1)) return static_cast<aclError>(1);
    using Fn = aclError (*)(void **, size_t, aclrtMemMallocPolicy);
    static Fn original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtMalloc"));
    return original ? original(p, bytes, policy) : static_cast<aclError>(1);
}
extern "C" aclError aclrtCreateEvent(aclrtEvent *event)
{
    if (Take(2)) return static_cast<aclError>(1);
    using Fn = aclError (*)(aclrtEvent *);
    static Fn original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtCreateEvent"));
    return original ? original(event) : static_cast<aclError>(1);
}
extern "C" aclError aclrtPeekAtLastError(aclrtLastErrLevel level)
{
    if (Take(3)) return static_cast<aclError>(1);
    using Fn = aclError (*)(aclrtLastErrLevel);
    static Fn original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtPeekAtLastError"));
    return original ? original(level) : static_cast<aclError>(1);
}
