# Cholesky 求逆（COMPLEX64）

## 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| 其他架构 | 本实现未提供 |

## 功能说明

输入已经分解的三角因子，原地生成原矩阵逆的所选三角。输入因子的零实部对角通过 info 报告。

$$
A^{-1}=(LL^{H})^{-1}\quad\text{或}\quad (U^{H}U)^{-1}
$$

## 函数原型

```cpp
#include "cann_ops_solver.h"

aclsolverStatus_t aclsolverCpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclsolverFloatComplex* A, int lda, int* Lwork);

aclsolverStatus_t aclsolverCpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, aclsolverFloatComplex* A,
    int lda, aclsolverFloatComplex* Workspace, int Lwork, int* devInfo);

```

## 参数说明

| 参数 | 方向与位置 | 说明 |
| --- | --- | --- |
| handle | 输入 / host | 由 aclsolverCreate 创建；使用其绑定的调用方 stream。 |
| uplo | 输入 / host | ACLSOLVER_FILL_MODE_LOWER 或 ACLSOLVER_FILL_MODE_UPPER。 |
| n | 输入 / host | int，矩阵阶数，0≤n≤4096。 |
| A | 输入/输出 / device | 列主序 aclsolverFloatComplex 矩阵，读取并写入所选三角；每元素 8 字节。 |
| lda | 输入 / host | int，A 的相邻两列起点间距，以复数元素计，至少为 max(1,n)；填充区保持不变。 |
| Workspace | 输入 / device | 调用方分配的工作区；实际元素数至少为查询返回的 Lwork。 |
| Lwork | 查询输出 / host；计算输入 / host | int，工作区大小，以 aclsolverFloatComplex 元素计；不是字节数。 |
| devInfo | 输出 / device | 一个 int；异步写入，在调用 stream 完成后读取。 |

## 约束与状态

矩阵采用列主序。LOWER 使用 L，UPPER 使用 U；未选择的三角不参与数学计算。只对所选三角承诺计算结果，lda/ldb 填充区保持不变。

计算在调用方 stream 上异步执行；查询接口不启动设备任务。调用方应等待 stream 完成后读取输出，并先销毁 handle，再销毁 stream。

返回值使用 aclsolverStatus_t。成功为 ACLSOLVER_STATUS_SUCCESS；非法参数为 ACLSOLVER_STATUS_INVALID_VALUE；内部工作区分配失败为 ACLSOLVER_STATUS_ALLOC_FAILED。失败路径提供主机 ERROR 日志。分配、初始化或执行失败时 info 未定义，不可用 info=0 判成功。

先检查 host 返回状态；SUCCESS 且 stream 成功完成后，info=0 表示成功；非法参数且 handle、stream、info 有效时，负值为错误参数的一基位置，不计 handle。分解时正值为最早失败主子式索引；求逆时为第一个零实部对角索引；求解仅报告参数错误。

空规模允许无需读取的矩阵地址为空，但仍要求 lda/ldb 至少为 1，以及可写 info。

先调用对应 bufferSize 查询，再分配 Lwork×sizeof(aclsolverFloatComplex) 字节；查询的 Lwork 为主机输出指针，计算的 Workspace 为设备地址。

## 调用说明

原生入口：[test/cpotri/cpotri_test.cpp](../../test/cpotri/cpotri_test.cpp)。公开 C API 的完整调用流程由[共享 runner](../../test/utils/cholesky/fixture_runner.cpp) 中的 `InvokeFixture()` 和 `RunCase()` 实现，包含 workspace 查询、stream 等待及输出读取。

```bash
bash build.sh --soc=ascend950 --ops=cpotri --run
```

标准入口、CPU 参考、padding 和确定性校验见[测试指导](../../test/utils/cholesky/README.md)。所有接口共用的布局、生命周期和 info 约束见[共享说明](cholesky.md)。
