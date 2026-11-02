/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stddef.h>

#include "cann_ops_solver.h"

_Static_assert(sizeof(aclsolverFloatComplex) == 8, "complex64 must occupy eight bytes");
_Static_assert(_Alignof(aclsolverFloatComplex) == _Alignof(float), "complex64 uses FP32 alignment");
_Static_assert(offsetof(aclsolverFloatComplex, real) == 0, "real component offset");
_Static_assert(offsetof(aclsolverFloatComplex, imag) == 4, "imaginary component offset");
_Static_assert(ACLSOLVER_OP_N == 0 && ACLSOLVER_OP_T == 1 && ACLSOLVER_OP_C == 2, "operation enum ABI");
_Static_assert(ACLSOLVER_SIDE_LEFT == 0 && ACLSOLVER_SIDE_RIGHT == 1, "side enum ABI");

int main(void)
{
    /* Volatile storage retains relocations even in an optimized C build.
     * The converted function pointers are inspected, never called. */
    void (*volatile symbols[])(void) = {(void (*)(void))aclsolverCgeqrf, (void (*)(void))aclsolverCgeqrf_bufferSize,
                                        (void (*)(void))aclsolverCungqr, (void (*)(void))aclsolverCungqr_bufferSize,
                                        (void (*)(void))aclsolverCunmqr, (void (*)(void))aclsolverCunmqr_bufferSize,
                                        (void (*)(void))aclsolverCCgels, (void (*)(void))aclsolverCCgels_bufferSize};
    for (size_t i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i)
    {
        if (symbols[i] == NULL)
        {
            return 1;
        }
    }
    return 0;
}
