/**
 * Copyright (c) 2026 Xztxzt
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CPU transport fixture only. No device or simulator execution.
#pragma once
#include <algorithm>
#include <cstdlib>
using aclError = int;
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0, ACL_MEM_MALLOC_HUGE_FIRST = 0, ACL_MEMCPY_HOST_TO_DEVICE = 0,
              ACL_MEMCPY_DEVICE_TO_HOST = 1;
inline int aclrtMalloc(void** p, size_t n, int)
{
    *p = std::malloc(n ? n : 1);
    return *p ? 0 : 1;
}
inline int aclrtFree(void* p)
{
    std::free(p);
    return 0;
}
inline int aclrtMemcpy(void* d, size_t capacity, const void* s, size_t n, int)
{
    if (!d || !s || n > capacity) return 1;
    std::copy_n(static_cast<const unsigned char*>(s), n, static_cast<unsigned char*>(d));
    return 0;
}
inline int aclrtMemset(void* d, size_t capacity, int v, size_t n)
{
    if (!d || n > capacity) return 1;
    std::fill_n(static_cast<unsigned char*>(d), n, static_cast<unsigned char>(v));
    return 0;
}
inline int aclrtSynchronizeStream(aclrtStream) { return 0; }
int aclInit(const char*);
int aclFinalize();
int aclrtSetDevice(int);
int aclrtResetDevice(int);
int aclrtCreateStream(aclrtStream*);
int aclrtDestroyStream(aclrtStream);
