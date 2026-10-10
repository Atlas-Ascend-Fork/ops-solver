## Sgetri

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
    aclsolverSgetri：计算实数矩阵的逆矩阵。

- 计算公式
    $$
    A^{-1}A = I
    $$
    其中$A$为$n \times n$阶非奇异实数方阵，$I$为$n$阶单位矩阵。

- 示例 \
    输入"A"为：
    ```
    [4.0, 3.0, 2.0, 1.0
     3.0, 4.0, 3.0, 2.0
     2.0, 3.0, 4.0, 3.0
     1.0, 2.0, 3.0, 4.0]
    ```
    输入"n"为： 4\
    调用"aclsolverSgetri"算子后，\
    输出"A"为：
    ```
    [0.6, -0.5, 0, 0.1
     -0.5, 1, -0.5, 0
     0, -0.5, 1, -0.5
     0.1, 0, -0.5, 0.6]
    ```

## 函数原型
- 函数定义
    ```
    aclError aclsolverSgetri(
        aclsolverHandle_t handle,
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
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的行数和列数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵A（通常为Sgetrf分解后的L和U矩阵），输出为逆矩阵<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">求逆结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - n 的有效范围为 [1, 8192]（kernel 列暂存区设计上限，超出将返回参数错误）。
  - lda、info参数在当前版本实际未启用。
  - lda当前必须等于n（矩阵按m*n/n*n连续布局处理），传入其它值将直接返回参数错误。
  - info参数当前版本不会写入任何返回值，传入可丢弃的变量即可。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverSgetri矩阵求逆示例](../../test/sgetri/sgetri_test.cpp)
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
        size_t aMatrixFileSize = n * n * sizeof(float);
        float* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t infoVal = 0;
        int32_t *info = &infoVal;

        // 调用 aclsolverSgetri
        auto ret = aclsolverSgetri(handle, n, A, n, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgetri failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclrtFreeHost(A);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
