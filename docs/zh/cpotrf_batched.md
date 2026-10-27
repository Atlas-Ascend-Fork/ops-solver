# 批量 Cholesky 分解（COMPLEX64）


## 产品支持情况


| 产品 | 支持情况 |
| --- | --- |
| Ascend950PR（dav-3510 / arch35） | √ |
| 其他架构 | 本实现未提供 |


## 功能说明


Aarray 是设备上的矩阵指针数组；各矩阵独立分解，infoArray 为逐矩阵设备状态数组。支持非连续矩阵地址。


$$
A_i=L_iL_i^{H}\quad\text{或}\quad A_i=U_i^{H}U_i
$$


## 函数原型


```cpp
#include "cann_ops_solver.h"



aclsolverStatus_t aclsolverCpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclsolverFloatComplex* Aarray[], int lda, int* infoArray, int batchSize);


```

## 参数说明

| 参数 | 方向与位置 | 说明 |
| --- | --- | --- |

| handle | 输入 / host | 由 aclsolverCreate 创建；使用其绑定的调用方 stream。 |

| uplo | 输入 / host | ACLSOLVER_FILL_MODE_LOWER 或 ACLSOLVER_FILL_MODE_UPPER。 |

| n | 输入 / host | int，矩阵阶数，0≤n≤4096。 |

| Aarray | 输入/输出 / device | 设备指针数组，各元素为列主序 aclsolverFloatComplex 矩阵地址；支持独立分配。 |

| lda | 输入 / host | int，A 的相邻两列起点间距，以复数元素计，至少为 max(1,n)；填充区保持不变。 |

| infoArray | 输出 / device | 批量分解为 batchSize 个 int；参数错误仅写首元素，空批量仍需可写首元素。 |

| batchSize | 输入 / host | int，0≤batchSize≤1000000。 |


## 约束与状态


矩阵采用列主序。LOWER 使用 L，UPPER 使用 U；未选择的三角不参与数学计算。只对所选三角承诺计算结果，lda/ldb 填充区保持不变。

计算在调用方 stream 上异步执行；查询接口不启动设备任务。调用方应等待 stream 完成后读取输出，并先销毁 handle，再销毁 stream。

同一 handle 不支持跨 Host 线程并发使用；调用方须串行化查询、SetStream、计算和 Destroy。
Destroy 不能与任何其他线程访问同一 handle 重叠。不同 handle 可独立使用。

返回值使用 aclsolverStatus_t。成功为 ACLSOLVER_STATUS_SUCCESS；非法参数为 ACLSOLVER_STATUS_INVALID_VALUE；内部工作区分配失败为 ACLSOLVER_STATUS_ALLOC_FAILED。失败路径提供主机 ERROR 日志。分配、初始化或执行失败时 info 未定义，不可用 info=0 判成功。

先检查 host 返回状态；SUCCESS 且 stream 成功完成后，info=0 表示成功；非法参数且 handle、stream、info 有效时，负值为错误参数的一基位置，不计 handle。分解时正值为最早失败主子式索引；求逆时为第一个零实部对角索引；求解仅报告参数错误。

指针数组及非空矩阵必须是可访问的设备内存。空规模仍需合法枚举、尺寸、leading dimension 和可写 info；空批量允许矩阵指针数组为空。内部工作区按 handle、stream 复用，扩容保留旧块至 handle 销毁。

`n` 和 `batchSize` 的上限独立，不保证大阶数与最大 batch 的组合可执行。
输入、工作区及其他存活分配的总量必须适合实际 HBM；容量公式和 128 GiB 设备示例见
[批量分解的容量约束](cholesky.md#批量分解的容量约束)。分配失败返回 ALLOC_FAILED，不能依据 info=0 判成功。


## 调用说明


原生入口：[test/cpotrf_batched/cpotrf_batched_test.cpp](../../test/cpotrf_batched/cpotrf_batched_test.cpp)。公开 C API 的完整调用流程由[共享 runner](../../test/utils/cholesky/fixture_runner.cpp) 中的 `InvokeFixture()` 和 `RunCase()` 实现，包含 workspace 查询、stream 等待及输出读取。


```bash
bash build.sh --soc=ascend950 --ops=cpotrf_batched --run
```


标准入口、CPU 参考、padding 和确定性校验见[测试指导](../../test/utils/cholesky/README.md)。所有接口共用的布局、生命周期和 info 约束见[共享说明](cholesky.md)。
