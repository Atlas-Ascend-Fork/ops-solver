# 批量 Cholesky 求解（COMPLEX64）

## 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| 其他架构 | 本实现未提供 |

## 功能说明

Aarray、Barray 是设备上的矩阵指针数组。因子保持不变，各 B 原地更新为 X。非空问题仅支持一个右端项；info 为设备标量。

$$
A_iX_i=B_i,\quad A_i=L_iL_i^{H}\ \text{或}\ U_i^{H}U_i
$$

## 函数原型

```cpp
#include "cann_ops_solver.h"

aclsolverStatus_t aclsolverCpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
    const aclsolverFloatComplex* const Aarray[], int lda, aclsolverFloatComplex* Barray[], int ldb,
    int* info, int batchSize);

```

## 参数说明

| 参数 | 方向与位置 | 说明 |
| --- | --- | --- |
| handle | 输入 / host | 由 aclsolverCreate 创建；使用其绑定的调用方 stream。 |
| uplo | 输入 / host | ACLSOLVER_FILL_MODE_LOWER 或 ACLSOLVER_FILL_MODE_UPPER。 |
| n | 输入 / host | int，矩阵阶数，0≤n≤4096。 |
| nrhs | 输入 / host | int，n>0 且 batchSize>0 时必须为 1；空问题接受 0≤nrhs≤128。 |
| Aarray | 输入 / device | 只读设备指针数组，各元素为只读的列主序 aclsolverFloatComplex 矩阵地址；支持独立分配。 |
| lda | 输入 / host | int，A 的相邻两列起点间距，以复数元素计，至少为 max(1,n)；填充区保持不变。 |
| Barray | 输入/输出 / device | 设备指针数组，各元素指向 n×1 的列主序 COMPLEX64 矩阵；矩阵原地更新为解，指针表不修改。 |
| ldb | 输入 / host | int，B 的列间距，以复数元素计，至少为 max(1,n)；填充区保持不变。 |
| info | 输出 / device | 一个 int；异步写入，在调用 stream 完成后读取。 |
| batchSize | 输入 / host | int，0≤batchSize≤1000000。 |

## 约束与状态

矩阵采用列主序。LOWER 使用 L，UPPER 使用 U；未选择的三角不参与数学计算。只对所选三角承诺计算结果，lda/ldb 填充区保持不变。

计算在调用方 stream 上异步执行；查询接口不启动设备任务。调用方应等待 stream 完成后读取输出，并先销毁 handle，再销毁 stream。

返回值使用 aclsolverStatus_t。成功为 ACLSOLVER_STATUS_SUCCESS；非法参数为 ACLSOLVER_STATUS_INVALID_VALUE；内部工作区分配失败为 ACLSOLVER_STATUS_ALLOC_FAILED。失败路径提供主机 ERROR 日志。分配、初始化或执行失败时 info 未定义，不可用 info=0 判成功。

先检查 host 返回状态；SUCCESS 且 stream 成功完成后，info=0 表示成功；非法参数且 handle、stream、info 有效时，负值为错误参数的一基位置，不计 handle。分解时正值为最早失败主子式索引；求逆时为第一个零实部对角索引；求解仅报告参数错误。

A 必须是有效的 Cholesky 因子。正定性由前置分解验证，求解接口不重新执行分解。

指针数组及非空矩阵必须是可访问的设备内存。空规模仍需合法枚举、尺寸、leading dimension 和可写 info；空批量允许矩阵指针数组为空。内部工作区按 handle、stream 复用，扩容保留旧块至 handle 销毁。

## 调用说明

原生入口：[test/cpotrs_batched/cpotrs_batched_test.cpp](../../test/cpotrs_batched/cpotrs_batched_test.cpp)。公开 C API 的完整调用流程由[共享 runner](../../test/utils/cholesky/fixture_runner.cpp) 中的 `InvokeFixture()` 和 `RunCase()` 实现，包含 workspace 查询、stream 等待及输出读取。

```bash
bash build.sh --soc=ascend950 --ops=cpotrs_batched --run
```

标准入口、CPU 参考、padding 和确定性校验见[测试指导](../../test/utils/cholesky/README.md)。所有接口共用的布局、生命周期和 info 约束见[共享说明](cholesky.md)。
