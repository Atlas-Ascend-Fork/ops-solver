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
 * \file cheevj_tridiag_vectors_reset.hpp
 * \brief Shared persistent tridiagonal workspace reset.
 */

#ifndef CHEEVJ_C64_TRIDIAG_VECTORS_RESET_HPP
#define CHEEVJ_C64_TRIDIAG_VECTORS_RESET_HPP

#include <cstdint>

#include "kernel_operator.h"

namespace Cheevj
{

template <int BarrierInts, int CommandInts>
class CheevjTridiagVectorsReset
{
   public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR commandWorkspace, GM_ADDR barrierWorkspace)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        commandGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commandWorkspace));
        barrierGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(barrierWorkspace));
        pipe->InitBuffer(zeroBuf, BarrierInts * sizeof(int32_t));
#else
        (void)pipe;
        (void)commandWorkspace;
        (void)barrierWorkspace;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        if (AscendC::GetBlockIdx() != 0)
        {
            return;
        }
        auto zero = zeroBuf.Get<int32_t>();
        AscendC::Duplicate(zero, static_cast<int32_t>(0), BarrierInts);
        AscendC::PipeBarrier<PIPE_ALL>();
        Store(commandGlobal, zero, CommandInts);
        Store(barrierGlobal, zero, BarrierInts);
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

   private:
    AscendC::GlobalTensor<int32_t> commandGlobal;
    AscendC::GlobalTensor<int32_t> barrierGlobal;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;

    __aicore__ inline void Store(AscendC::GlobalTensor<int32_t> destination,
                                 const AscendC::LocalTensor<int32_t> &source, int count)
    {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(destination, source, copy);
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_TRIDIAG_VECTORS_RESET_HPP
