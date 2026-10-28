## Spotrf

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |


## 功能说明

- 接口功能\
    aclsolverSpotrf：对实数正定对称矩阵进行cholesky分解。

- 计算公式
    $$
    A = L \cdot L^T
    $$
    其中$A$为$n \times n$阶实数正定对称矩阵，$L$为单位下三角矩阵，$L^T$为上三角矩阵。

- 示例 \
    输入"A"为：
    ```
    [1.000000 0.109793 -2.551319 -1.269791
     0.109793 81.012054 7.577830 12.328438
     -2.551319 7.577830 32.271538 3.670894
     -1.269791 12.328438 3.670894 103.555702]
    ```
    输入"n"为： 4\
    调用"aclsolverSpotrf"算子后，\
    输出"A"为L^T存储：
    ```
    [1.000000 0.109793 -2.551319 -1.269791
     0.000000 9.000000 0.873105 1.385317
     0.000000 0.000000 5.000000 -0.155655
     0.000000 0.000000 0.000000 10.000000]
    ```


## 函数原型
- 函数定义
    ```
    aclError aclsolverSpotrf(
        aclsolverHandle_t handle,
        cost char uplo,
        const int64_t n,
        float *A,
        const int64_t lda,
        int32_t *info);
    ```
- 参数说明：
    <table>
    <tr>
        <td align="center">参数名</td>
        <td align="center">输入输出</td>
        <td align="center">描述</td>
    </tr>
    <tr>
        <td align="center">handle</td>
        <td align="center">输入</td>
        <td align="left">solver handle，通过aclsolverCreate创建</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的行数，当前约束为m == n（Cholesky要求方阵）</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的列数，取值范围[1, 8192]</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵A，输出为L^T矩阵<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A左右相邻元素间的内存地址偏移量（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">分解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - m必须等于n（Cholesky要求方阵）；n取值范围为[1, 8192]，超出返回ACL_ERROR_INVALID_PARAM。
  - lda >= n且lda <= 16384（kernel以uint16_t字节计数承载行间隙，'U'因子成员列数逐轮递减，最坏行间隙由lda本身决定而非lda - n）。
  - info须为有效指针（当前版本未解引用，预留扩展）。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverSpotrf矩阵cholesky分解示例](../../test/spotrf/spotrf_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
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
        aclsolverCreate(&handle);
        aclsolverSetStream(handle, stream);

        // 构造输入数
        int64_t n = 4;
        size_t aMatrixFileSize = n * n * sizeof(float);
        float* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t *info;

        // 调用 aclsolverSpotrf
        auto ret = aclsolverSpotrf(handle, uplo, n, n, A, n, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSpotrf failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclrtFreeHost(A);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
