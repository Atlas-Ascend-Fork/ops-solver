# Cholesky 求解（COMPLEX64）


## 产品支持情况


| 产品 | 支持情况 |
| --- | --- |
| Ascend950PR（dav-3510 / arch35） | √ |
| 其他架构 | 本实现未提供 |


## 功能说明


输入已经分解的三角因子，保持 A 不变，原地将 B 更新为 X。调用前需完成 Cholesky 分解；本接口不包含前置分解。


$$
AX=B,\quad A=LL^{H}\ \text{或}\ U^{H}U
$$


## 函数原型


```cpp
#include "cann_ops_solver.h"



aclsolverStatus_t aclsolverCpotrs(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
    const aclsolverFloatComplex* A, int lda, aclsolverFloatComplex* B, int ldb, int* devInfo);


```

## 参数说明

| 参数 | 方向与位置 | 说明 |
| --- | --- | --- |

| handle | 输入 / host | 由 aclsolverCreate 创建；使用其绑定的调用方 stream。 |

| uplo | 输入 / host | ACLSOLVER_FILL_MODE_LOWER 或 ACLSOLVER_FILL_MODE_UPPER。 |

| n | 输入 / host | int，矩阵阶数，0≤n≤4096。 |

| nrhs | 输入 / host | int，0≤nrhs≤128；n=0 或 nrhs=0 为空问题。 |

| A | 输入 / device | 列主序 aclsolverFloatComplex 矩阵，元素由 FP32 real、imag 组成；求解时保持全部字节不变。 |

| lda | 输入 / host | int，A 的相邻两列起点间距，以复数元素计，至少为 max(1,n)；填充区保持不变。 |

| B | 输入/输出 / device | n×nrhs 的设备矩阵地址；列主序 COMPLEX64，原地更新为解。 |

| ldb | 输入 / host | int，B 的列间距，以复数元素计，至少为 max(1,n)；填充区保持不变。 |

| devInfo | 输出 / device | 一个 int；异步写入，在调用 stream 完成后读取。 |


## 约束与状态


矩阵采用列主序。LOWER 使用 L，UPPER 使用 U；未选择的三角不参与数学计算。只对所选三角承诺计算结果，lda/ldb 填充区保持不变。

计算在调用方 stream 上异步执行；查询接口不启动设备任务。调用方应等待 stream 完成后读取输出，并先销毁 handle，再销毁 stream。

返回值使用 aclsolverStatus_t。成功为 ACLSOLVER_STATUS_SUCCESS；非法参数为 ACLSOLVER_STATUS_INVALID_VALUE；内部工作区分配失败为 ACLSOLVER_STATUS_ALLOC_FAILED。失败路径提供主机 ERROR 日志。分配、初始化或执行失败时 info 未定义，不可用 info=0 判成功。

先检查 host 返回状态；SUCCESS 且 stream 成功完成后，info=0 表示成功；非法参数且 handle、stream、info 有效时，负值为错误参数的一基位置，不计 handle。分解时正值为最早失败主子式索引；求逆时为第一个零实部对角索引；求解仅报告参数错误。

A 必须是有效的 Cholesky 因子。正定性由前置分解验证，求解接口不重新执行分解。

内部工作区按 handle、stream 复用，扩容保留旧块至 handle 销毁；其 stream 必须存活至销毁完成。

空规模允许无需读取的矩阵地址为空，但仍要求 lda/ldb 至少为 1，以及可写 info。


## 调用说明


原生入口：[test/cpotrs/cpotrs_test.cpp](../../test/cpotrs/cpotrs_test.cpp)。公开 C API 的完整调用流程由[共享 runner](../../test/utils/cholesky/fixture_runner.cpp) 中的 `InvokeFixture()` 和 `RunCase()` 实现，包含 workspace 查询、stream 等待及输出读取。


```bash
bash build.sh --soc=ascend950 --ops=cpotrs --run
```


标准入口、CPU 参考、padding 和确定性校验见[测试指导](../../test/utils/cholesky/README.md)。所有接口共用的布局、生命周期和 info 约束见[共享说明](cholesky.md)。
