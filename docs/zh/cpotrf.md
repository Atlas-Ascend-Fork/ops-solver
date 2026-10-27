# Cpotrf

本页包含两种独立的分解接口，请按产品和内存布局选择，参数与返回值不可混用。

| 接口 | 产品 | 矩阵 | 工作区和 info |
| --- | --- | --- | --- |
| C++ `aclsolverCpotrf(handle, char, n, A, lda, info)` | A2/A3 | Host 行主序 `std::complex<float>` | 内部分配工作区；沿用下文 C++ 接口约束 |
| C `aclsolverCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)` | Ascend950PR | Device 列主序 `aclsolverFloatComplex` | 显式 workspace；Device info 异步写入 |

## C++ 接口：Host 行主序

### 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Ascend 950PR/Ascend 950DT</term>|×|

### 功能说明

- 接口功能\
  `aclsolverCpotrf`：对 Hermitian 正定复数矩阵进行 Cholesky 分解。

- 计算公式

  当输入参数 `uplo` 为 `'L'` 时，计算下三角 Cholesky 因子 $L$：

  $$
  A=L\cdot L^{H}
  $$

  当输入参数 `uplo` 为 `'U'` 时，计算上三角 Cholesky 因子 $U$：

  $$
  A=U^{H}\cdot U
  $$

  其中：

  - $A$ 为 $n\times n$ 阶 Hermitian 正定复数矩阵；
  - $L$ 为对角元素为正实数的下三角矩阵；
  - $U$ 为对角元素为正实数的上三角矩阵；
  - $L^{H}$ 和 $U^{H}$ 表示矩阵的共轭转置。

- 数据存储说明

  输入矩阵 `A` 使用 `COMPLEX64` 数据类型，每个元素由一个单精度实部和一个单精度虚部组成。

  当 `uplo` 为 `'L'` 时：

  - 输入使用矩阵 `A` 的下三角部分；
  - 输出的下三角部分保存 Cholesky 因子 $L$；
  - 输出的上三角部分不属于有效结果，不应使用。

  当 `uplo` 为 `'U'` 时：

  - 输入使用矩阵 `A` 的上三角部分；
  - 输出的上三角部分保存 Cholesky 因子 $U$；
  - 输出的下三角部分不属于有效结果，不应使用。

- 示例

  输入 Hermitian 正定矩阵 `A` 为：

  ```
  [ 4+0i,   2-2i,   4+2i
    2+2i,  11+0i,  -2-3i
    4-2i,  -2+3i,  14+0i]
  ```

  输入 `n` 为：

  ```
  3
  ```

  当输入 `uplo` 为 `'L'` 时，调用 `aclsolverCpotrf` 后，矩阵 `A` 的下三角部分保存矩阵 $L$：

  ```
  [ 2+0i,      *,      *
    1+1i,   3+0i,      *
    2-1i,  -1+2i,   2+0i]
  ```

  此时：

  ```
  L =
  [ 2+0i,   0+0i,   0+0i
    1+1i,   3+0i,   0+0i
    2-1i,  -1+2i,   2+0i]
  ```

  并满足：

  $$
  A=L\cdot L^{H}
  $$

  当输入 `uplo` 为 `'U'` 时，调用 `aclsolverCpotrf` 后，矩阵 `A` 的上三角部分保存矩阵 $U$：

  ```
  [ 2+0i,   1-1i,   2+1i
       *,   3+0i,  -1-2i
       *,      *,   2+0i]
  ```

  此时：

  ```
  U =
  [2+0i,  1-1i,   2+1i
   0+0i,  3+0i,  -1-2i
   0+0i,  0+0i,   2+0i]
  ```

  并满足：

  $$
  A=U^{H}\cdot U
  $$

  上述输出中的 `*` 表示该位置不属于有效输出区域，其结果未定义。

### 函数原型

- 函数定义

  ```cpp
  aclError aclsolverCpotrf(
      aclsolverHandle_t handle,
      const char uplo,
      const int64_t n,
      std::complex<float> *A,
      const int64_t lda,
      int32_t *info);
  ```

- 参数说明

  <table>
  <tr>
      <td align="center">参数名</td>
      <td align="center">输入输出</td>
      <td align="center">描述</td>
  </tr>
  <tr>
      <td align="center">handle</td>
      <td align="center">输入</td>
      <td align="left">Solver handle，通过<code>aclsolverCreate</code>创建，并通过<code>aclsolverSetStream</code>设置执行流。</td>
  </tr>
  <tr>
      <td align="center">uplo</td>
      <td align="center">输入</td>
      <td align="left">
          指定输出为矩阵A的上三角部分还是下三角部分。<br>
          取值为：<br>
          <code>'L'</code>：输出下三角 Cholesky 因子L；<br>
          <code>'U'</code>：输出上三角 Cholesky 因子U。
      </td>
  </tr>
  <tr>
      <td align="center">n</td>
      <td align="center">输入</td>
      <td align="left">矩阵A的阶数。矩阵A的shape为[n, n]。</td>
  </tr>
  <tr>
      <td align="center">A</td>
      <td align="center">输入/输出</td>
      <td align="left">
          输入为n×n阶 Hermitian 正定复数矩阵A，必须有上三角元素。<br>
          当uplo为<code>'L'</code>时，输出的下三角部分保存矩阵L；<br>
          当uplo为<code>'U'</code>时，输出的上三角部分保存矩阵U。<br>
          数据类型仅支持COMPLEX64，数据格式支持ND，逻辑shape为[n, n]。<br>
          每个COMPLEX64元素由一个float实部和一个float虚部组成。
      </td>
  </tr>
  <tr>
      <td align="center">lda</td>
      <td align="center">输入</td>
      <td align="left">
          矩阵A中相邻两行起始地址之间的元素跨度（leading dimension），满足<code>lda &gt;= max(1, n)</code>。<br>
          接口内部会将带步长的n×lda存储转换为连续的n×n矩阵进行计算。
      </td>
  </tr>
  <tr>
      <td align="center">info</td>
      <td align="center">输出</td>
      <td align="left">
          分解结果信息，数据类型为int32_t。<br>
          按照POTRF语义：<br>
          info=0表示分解成功；<br>
          info&gt;0表示对应阶数的顺序主子式不是正定矩阵，分解未完成；<br>
          info&lt;0表示对应位置的参数不合法。<br>
          当前版本中该参数实际未启用，调用时仍需传入有效的非空指针。
      </td>
  </tr>
  </table>

- 算子约束

  - `uplo` 仅支持 `'L'` 和 `'U'`。
  - `n` 必须大于 0。
  - `lda` 必须大于等于 `max(1, n)`，单位为 `COMPLEX64` 元素。
  - 输入矩阵 `A` 必须为 Hermitian 正定复数矩阵。
  - Hermitian 矩阵满足：

    $$
    A=A^{H}
    $$

    即：

    $$
    A_{i,j}=\overline{A_{j,i}}
    $$

  - 矩阵 `A` 的对角元素应为实数，其虚部应为 0。
  - `A` 的数据类型仅支持 `COMPLEX64`，即 `std::complex<float>`。
  - 当前版本中 `info` 参数实际未启用，调用时需要提供有效的非空指针，但分解状态应以接口返回值为准。
  - 当前实现建议输入完整的 Hermitian 矩阵，即同时初始化矩阵的上三角和下三角区域。
  - 当 `uplo` 为 `'L'` 时，仅输出矩阵 `A` 的下三角部分有效。
  - 当 `uplo` 为 `'U'` 时，仅输出矩阵 `A` 的上三角部分有效。
  - 不应读取或依赖输出矩阵中非目标三角区域的数据。
  - 如果输入矩阵不是正定矩阵，输出结果无效。

- 调用实现

  使用内核调用符 `<<<>>>` 调用核函数。

### 调用示例

- 完整代码示例：[aclsolverCpotrf复数矩阵Cholesky分解示例](../../test/cpotrf/cpotrf_test.cpp)

- 核心调用步骤

    ```cpp
    #include <complex>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    int32_t main(int32_t argc, char *argv[])
    {
        // 固定写法，acl初始化
        int32_t deviceId = 0;
        aclrtStream stream = nullptr;
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        aclrtCreateStream(&stream);

        // 创建solver handle并设置stream
        aclsolverHandle_t handle = nullptr;
        CHECK_ACL(aclsolverCreate(&handle));
        CHECK_ACL(aclsolverSetStream(handle, stream));

        // 构造输入数据
        int64_t n = 4;
        size_t aMatrixFileSize = n * n * sizeof(float) * 2;
        std::complex<float>* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t infoVal = 0;
        int32_t *info = &infoVal;

        // 调用 aclsolverCpotrf
        auto ret = aclsolverCpotrf(handle, 'U', n, A, n, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCpotrf failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclrtFreeHost(A);
        aclrtDestroyStream(stream);
        aclsolverDestroy(handle);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```

## Ascend950 C API

### 产品支持情况


| 产品 | 支持情况 |
| --- | --- |
| Ascend950PR（dav-3510 / arch35） | √ |
| 其他架构 | 本实现未提供 |


### 功能说明


读取 Hermitian 正定矩阵的所选三角，原地生成 L 或 U。非正定时返回最早失败主子式的一基索引。


$$
A = LL^{H}\quad\text{或}\quad A = U^{H}U
$$


### 函数原型


```cpp
#include "cann_ops_solver.h"



aclsolverStatus_t aclsolverCpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclsolverFloatComplex* A, int lda, int* Lwork);


aclsolverStatus_t aclsolverCpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, aclsolverFloatComplex* A,
    int lda, aclsolverFloatComplex* Workspace, int Lwork, int* devInfo);


```

### 参数说明

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


### 约束与状态


矩阵采用列主序。LOWER 使用 L，UPPER 使用 U；未选择的三角不参与数学计算。只对所选三角承诺计算结果，lda/ldb 填充区保持不变。

计算在调用方 stream 上异步执行；查询接口不启动设备任务。调用方应等待 stream 完成后读取输出，并先销毁 handle，再销毁 stream。

返回值使用 aclsolverStatus_t。成功为 ACLSOLVER_STATUS_SUCCESS；非法参数为 ACLSOLVER_STATUS_INVALID_VALUE；内部工作区分配失败为 ACLSOLVER_STATUS_ALLOC_FAILED。失败路径提供主机 ERROR 日志。分配、初始化或执行失败时 info 未定义，不可用 info=0 判成功。

先检查 host 返回状态；SUCCESS 且 stream 成功完成后，info=0 表示成功；非法参数且 handle、stream、info 有效时，负值为错误参数的一基位置，不计 handle。分解时正值为最早失败主子式索引；求逆时为第一个零实部对角索引；求解仅报告参数错误。

空规模允许无需读取的矩阵地址为空，但仍要求 lda/ldb 至少为 1，以及可写 info。

先调用对应 bufferSize 查询，再分配 Lwork×sizeof(aclsolverFloatComplex) 字节；查询的 Lwork 为主机输出指针，计算的 Workspace 为设备地址。


### 调用说明


原生入口：[test/cpotrf_950/cpotrf_950_test.cpp](../../test/cpotrf_950/cpotrf_950_test.cpp)。公开 C API 的完整调用流程由[共享 runner](../../test/utils/cholesky/fixture_runner.cpp) 中的 `InvokeFixture()` 和 `RunCase()` 实现，包含 workspace 查询、stream 等待及输出读取。


```bash
bash build.sh --soc=ascend950 --ops=cpotrf_950 --run
```


标准入口、CPU 参考、padding 和确定性校验见[测试指导](../../test/utils/cholesky/README.md)。所有接口共用的布局、生命周期和 info 约束见[共享说明](cholesky.md)。
