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
 * \file cast.hpp
 * \brief
 */

#ifndef _CAST_HPP_
#define _CAST_HPP_

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

template <typename IN_DTYPE, typename OUT_DTYPE>
class CastDtype
{
    static constexpr int64_t BUFFER_NUM = 2;
    static constexpr int32_t ELENUM_REPEAT_FP32 = 64;
    static constexpr int32_t ELENUM_REPEAT_FP16 = 128;
    static constexpr int64_t MAX_CAST_ELENUM = 8192;

    GlobalTensor<IN_DTYPE> srcGm;
    GlobalTensor<OUT_DTYPE> dstGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    int64_t totalElenum;

   public:
    __aicore__ inline CastDtype() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, int64_t elenum, GM_ADDR dst, GM_ADDR src)
    {
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(src));
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(dst));

        pipe->InitBuffer(inQueue, BUFFER_NUM, MAX_CAST_ELENUM * sizeof(IN_DTYPE));
        pipe->InitBuffer(outQueue, BUFFER_NUM, MAX_CAST_ELENUM * sizeof(OUT_DTYPE));
        this->totalElenum = elenum;
    }
    __aicore__ inline void SingleProcess(uint64_t eleOffset, int32_t curCalElenum)
    {
        LocalTensor<IN_DTYPE> inLocal = inQueue.AllocTensor<IN_DTYPE>();

        uint32_t inBlockLen = curCalElenum * sizeof(IN_DTYPE);
        DataCopyExtParams copyInParams{1, inBlockLen, 0, 0, 0};
        DataCopyPadExtParams<IN_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal, this->srcGm[eleOffset], copyInParams, padParams);
        inQueue.EnQue<IN_DTYPE>(inLocal);

        LocalTensor<IN_DTYPE> srcLocal = inQueue.DeQue<IN_DTYPE>();
        LocalTensor<OUT_DTYPE> dstLocal = outQueue.AllocTensor<OUT_DTYPE>();

        int32_t repeatNum = (curCalElenum - 1) / ELENUM_REPEAT_FP16 + 1;

        Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, repeatNum * ELENUM_REPEAT_FP16);
        outQueue.EnQue<OUT_DTYPE>(dstLocal);
        inQueue.FreeTensor(srcLocal);

        LocalTensor<OUT_DTYPE> outLocal = outQueue.DeQue<OUT_DTYPE>();
        uint32_t outBlockLen = curCalElenum * sizeof(OUT_DTYPE);
        DataCopyExtParams copyOutParams{1, outBlockLen, 0, 0, 0};
        DataCopyPad(this->dstGm[eleOffset], outLocal, copyOutParams);
        outQueue.FreeTensor(outLocal);
    }
    __aicore__ inline void Process()
    {
        int64_t vecIdx = GetBlockIdx();
        int64_t vecNum = GetBlockNum() * GetSubBlockNum();

        int64_t loopNum = this->totalElenum / MAX_CAST_ELENUM;
        int64_t tailNum = this->totalElenum % MAX_CAST_ELENUM;
        if (tailNum > 0)
        {
            loopNum += 1;
        }

        for (int64_t loopIdx = 0; loopIdx < loopNum; loopIdx++)
        {
            uint64_t eleOffset = static_cast<uint64_t>(loopIdx * MAX_CAST_ELENUM);
            if (loopIdx % vecNum != vecIdx)
            {
                continue;
            }
            int32_t curCalElenum = MAX_CAST_ELENUM;
            if (loopIdx == loopNum - 1 && tailNum > 0)
            {
                curCalElenum = tailNum;
            }

            SingleProcess(eleOffset, curCalElenum);
        }
    }
};

__aicore__ inline void cast_up(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, GM_ADDR dst, GM_ADDR src)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CastDtype<half, float> opup;
    opup.Init(&tbufPool, orgM * orgN * 2, dst, src);
    opup.Process();

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

__aicore__ inline void cast_down(TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN, GM_ADDR dst,
                                 GM_ADDR src)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CastDtype<float, half> opdown;
    opdown.Init(&tbufPool, orgM * orgN * 2, dst, src);
    opdown.Process();

    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

#endif  // _CAST_HPP_
