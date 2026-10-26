## Cpotrf

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Ascend 950PR/Ascend 950DT</term>|×|

## 功能说明

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

## 函数原型

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

## 调用示例

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