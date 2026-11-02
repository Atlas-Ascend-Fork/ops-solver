/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>

#include "../utils/complex_solver_test.h"

int main(int argc, char** argv)
{
    using namespace ComplexSolverTest;
    try
    {
        int device = argc > 1 ? std::atoi(argv[1]) : 0;
        int m = argc > 2 ? std::atoi(argv[2]) : 3;
        int n = m;
        int batch = argc > 3 ? std::atoi(argv[3]) : 2;
        Require(m > 0 && n > 0 && batch > 0, "positive dimensions required");
        Session session(device);
        RunBatched(session, true, n, batch, "cmatinv_batched");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
