# Sgetrs

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
    aclsolverSgetrs：利用LU分解结果求解实数线性方程组。

- 计算公式

    当`trans`为0时：
    $$
    A \cdot X = B
    $$

    当`trans`不为0时：
    $$
    A^T \cdot X = B
    $$

    其中$A$为$n \times n$阶实数矩阵的LU分解结果，$X$为求解结果矩阵，$B$为输入的右端项矩阵。

- 示例 \
    输入"A"为经过LU分解后的矩阵：
    ```
    [4.0, 3.0, 2.0, 1.0
     0.75, 1.75, 1.5, 1.25
     0.5, 0.4286, 2.0, 2.0
     0.25, 0.2857, 0.5, 2.0]
    ```

    输入"ipiv"为：
    ```
    [1, 2, 3, 4]
    ```

    输入"B"为：
    ```
    [10.0
      8.0
      8.0
     10.0]
    ```

    输入"trans"为：0\
    输入"n"为：4\
    输入"nrhs"为：1\
    调用"aclsolverSgetrs"算子后，\
    输出"B"为求解结果：
    ```
    [ 2.9107
      -0.4821
      -1.5893
      2.9821]
    ```


## 函数原型

- 函数定义
    ```
    aclError aclsolverSgetrs(
        aclsolverHandle_t handle,
        const uint64_t trans,
        const int64_t n,
        const int64_t nrhs,
        float *A,
        const int64_t lda,
        int32_t *ipiv,
        float *B,
        const int64_t ldb,
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
        <td align="center">trans</td>
        <td align="center">输入</td>
        <td align="left">矩阵运算类型。<br>0：求解A·X=B。<br>1：求解A<sup>T</sup>·X=B。</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的阶数</td>
    </tr>
    <tr>
        <td align="center">nrhs</td>
        <td align="center">输入</td>
        <td align="left">右端项矩阵B的列数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入</td>
        <td align="left">输入为经过LU分解后的矩阵A。<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A上下相邻元素间的内存地址偏移量</td>
    </tr>
    <tr>
        <td align="center">ipiv</td>
        <td align="center">输入</td>
        <td align="left">置换矩阵的pivot信息，由aclsolverSgetrf输出。<br>数据类型支持int32_t，数据格式支持ND，shape为[n, 1]</td>
    </tr>
    <tr>
        <td align="center">B</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为右端项矩阵B，输出为求解结果矩阵X。<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[n, nrhs]</td>
    </tr>
    <tr>
        <td align="center">ldb</td>
        <td align="center">输入</td>
        <td align="left">B上下相邻元素间的内存地址偏移量</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">求解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - 输入矩阵A必须为aclsolverSgetrf分解后的LU矩阵。
  - ipiv必须为aclsolverSgetrf输出的pivot信息。
  - 当前仅支持FLOAT32数据类型。
  - lda、ldb、info参数在当前版本实际未启用。

- 调用实现

    使用内核调用符<<<>>>调用核函数。

## 调用示例

- 完整代码示例：[aclsolverSgetrs线性方程组求解示例](../../test/sgetrs/sgetrs_test.cpp)
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
    
        // 构造输入数据
        int64_t n = 4;
        int64_t nrhs = 1;
        size_t aMatrixFileSize = n * n * sizeof(float);
        size_t bMatrixFileSize = n * nrhs * sizeof(float);
    
        float* A;
        float* B;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        aclrtMallocHost((void**)(&B), bMatrixFileSize);
    
        // 填充A矩阵数据...
        // 填充B矩阵数据...
    
        int32_t *ipiv = new int32_t[n];
        int32_t *info;
    
        // 调用 aclsolverSgetrs
        auto ret = aclsolverSgetrs(handle, 0, n, nrhs, 0, n, nrhs, A, n, ipiv, B, nrhs, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgetrs failed. ERROR: %d\n", ret); return ret);
    
        // 释放资源
        delete[] ipiv;
        aclrtFreeHost(A);
        aclrtFreeHost(B);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    
        return 0;
    }
    ```