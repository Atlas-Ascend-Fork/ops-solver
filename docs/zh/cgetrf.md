## Cgetrf

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |


注：表中"Atlas 训练系列产品"指非A2代际的历史训练产品；Atlas A2训练/推理系列产品（含910B等）单独标注支持情况。

## 功能说明

- 接口功能\
    aclsolverCgetrf：对复数矩阵进行LU分解。

- 计算公式
    $$
    A = P \cdot L \cdot U
    $$
    其中$A$为$m \times n$阶复数矩阵，$P$为置换矩阵，$L$为单位下三角矩阵，$U$为上三角矩阵。

- 示例 \
    输入"A"为：
    ```
    [4+0i, 3+0i, 2+0i, 1+0i
     3+0i, 4+0i, 3+0i, 2+0i
     2+0i, 3+0i, 4+0i, 3+0i
     1+0i, 2+0i, 3+0i, 4+0i]
    ```
    输入"m"为： 4\
    输入"n"为： 4\
    调用"aclsolverCgetrf"算子后，\
    输出"A"为L和U合并存储：
    ```
    [4+0i, 3+0i, 2+0i, 1+0i
     0.75+0i, 1.75+0i, 1.5+0i, 1.25+0i
     0.5+0i, 0.8571429+0i, 1.7142857+0i, 1.4285714+0i
     0.25+0i, 0.7142857+0i, 0.8333333+0i, 1.6666667+0i]
    ```
    输出"ipiv"为pivot信息：
    ```
    [1, 2, 3, 4]
    ```

## 函数原型
- 函数定义
    ```
    aclError aclsolverCgetrf(
        aclsolverHandle_t handle,
        const int64_t m,
        const int64_t n,
        std::complex<float> *A,
        const int64_t lda,
        int32_t *ipiv,
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
        <td align="left">矩阵A的行数，范围为[1, 8192]</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的列数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵A，输出为L和U矩阵<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[m, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">ipiv</td>
        <td align="center">输出</td>
        <td align="left">置换矩阵的pivot信息<br>数据类型支持int32_t，数据格式支持ND，shape为[min(m,n), 1]</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">分解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - lda、info参数在当前版本实际未启用。
  - lda当前必须等于n（矩阵按m*n/n*n连续布局处理），传入其它值将直接返回参数错误。
  - info参数当前版本不会写入任何返回值，传入可丢弃的变量即可。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverCgetrf复数矩阵LU分解示例](../../test/cgetrf/cgetrf_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
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
        aclsolverCreate(&handle);
        aclsolverSetStream(handle, stream);

        // 构造输入数据
        int64_t m = 4;
        int64_t n = 4;
        size_t aMatrixFileSize = m * n * sizeof(float) * 2;
        std::complex<float>* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t *ipiv = new int32_t[std::min(m, n)];
        int32_t infoVal = 0;
        int32_t *info = &infoVal;

        // 调用 aclsolverCgetrf
        auto ret = aclsolverCgetrf(handle, m, n, A, n, ipiv, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCgetrf failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        delete[] ipiv;
        aclrtFreeHost(A);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
