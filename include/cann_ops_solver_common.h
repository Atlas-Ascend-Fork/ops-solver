/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cann_ops_solver_common.h
 * \brief aclsolver common types and status codes.
 */

#pragma once

#include <acl/acl.h>
#include <stddef.h>

#ifdef __cplusplus
#include <complex>
#include <type_traits>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /*! \brief aclsolver status codes definition */
    typedef enum aclsolverStatus
    {
        ACLSOLVER_STATUS_SUCCESS = 0,           /**< Function succeeds */
        ACLSOLVER_STATUS_NOT_INITIALIZED = 1,   /**< aclsolver library not initialized */
        ACLSOLVER_STATUS_ALLOC_FAILED = 2,      /**< resource allocation failed */
        ACLSOLVER_STATUS_INVALID_VALUE = 3,     /**< unsupported numerical value was passed to function */
        ACLSOLVER_STATUS_MAPPING_ERROR = 4,     /**< access to memory space failed */
        ACLSOLVER_STATUS_EXECUTION_FAILED = 5,  /**< program failed to execute */
        ACLSOLVER_STATUS_INTERNAL_ERROR = 6,    /**< an internal aclsolver operation failed */
        ACLSOLVER_STATUS_NOT_SUPPORTED = 7,     /**< function not implemented */
        ACLSOLVER_STATUS_ARCH_MISMATCH = 8,     /**< architecture mismatch */
        ACLSOLVER_STATUS_HANDLE_IS_NULLPTR = 9, /**< aclsolver handle is null pointer */
        ACLSOLVER_STATUS_INVALID_ENUM = 10,     /**< unsupported enum value was passed to function */
        ACLSOLVER_STATUS_UNKNOWN = 11           /**< back-end returned an unsupported status code */
    } aclsolverStatus_t;

    typedef void* aclsolverHandle_t;

    /*! \brief Two consecutive FP32 components, matching std::complex<float> storage.
     * Device buffers may be reused by C and C++ APIs; host objects must not be accessed
     * through an unrelated type. Copy host values or initialize components explicitly.
     */
    typedef struct
    {
        float real;
        float imag;
    } aclsolverFloatComplex;

    /*! \brief Triangle used by Hermitian solver routines. */
    typedef enum
    {
        ACLSOLVER_FILL_MODE_LOWER = 0,
        ACLSOLVER_FILL_MODE_UPPER = 1
    } aclsolverFillMode_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(aclsolverFloatComplex) == 2 * sizeof(float), "complex size must be two FP32 components");
static_assert(alignof(aclsolverFloatComplex) == alignof(float), "complex alignment must match FP32");
static_assert(offsetof(aclsolverFloatComplex, real) == 0, "real must be the first component");
static_assert(offsetof(aclsolverFloatComplex, imag) == sizeof(float), "imag must follow real");
static_assert(sizeof(aclsolverFloatComplex) == sizeof(std::complex<float>) &&
                  alignof(aclsolverFloatComplex) == alignof(std::complex<float>),
              "C and C++ complex storage must have identical size and alignment");
static_assert(std::is_standard_layout<aclsolverFloatComplex>::value &&
                  std::is_trivially_copyable<aclsolverFloatComplex>::value,
              "complex values must remain standard layout and trivially copyable");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(aclsolverFloatComplex) == 2 * sizeof(float), "complex size must be two FP32 components");
_Static_assert(_Alignof(aclsolverFloatComplex) == _Alignof(float), "complex alignment must match FP32");
_Static_assert(offsetof(aclsolverFloatComplex, real) == 0, "real must be the first component");
_Static_assert(offsetof(aclsolverFloatComplex, imag) == sizeof(float), "imag must follow real");
#endif
