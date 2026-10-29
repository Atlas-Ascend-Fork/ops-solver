## Sgetrf

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   | 待验证 |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     | 待验证 |
|  <term>Ascend 950PR</term>   |     √    |
|  <term>Ascend 950DT</term>   |  待验证  |


注：表中"Atlas 训练系列产品"指非A2代际的历史训练产品；Atlas A2训练/推理系列产品（含910B等）单独标注支持情况。

本次实数Device接口在Ascend 950PR及仓库README列出的配套CANN 9.0.0环境完成自验证；其余标为“待验证”的产品尚需对应平台回归。

## 功能说明

- 接口功能\
    aclsolverSgetrf：对实数矩阵进行LU分解。

- 计算公式
    $$
    P \cdot A = L \cdot U
    $$
    其中$P$表示按返回的主元顺序执行行交换；$A$为$m \times n$阶实数矩阵，$P$为置换矩阵，$L$为单位下三角矩阵，$U$为上三角矩阵。

- 示例 \
    输入"A"为：
    ```
    [4.0, 3.0, 2.0, 1.0
     3.0, 4.0, 3.0, 2.0
     2.0, 3.0, 4.0, 3.0
     1.0, 2.0, 3.0, 4.0]
    ```
    输入"m"为： 4\
    输入"n"为： 4\
    调用"aclsolverSgetrf"算子后，\
    输出"A"为L和U合并存储：
    ```
    [4, 3, 2, 1
     0.75, 1.75, 1.5, 1.25
     0.5, 0.8571429, 1.7142857, 1.4285714
     0.25, 0.7142857, 0.8333333, 1.6666667]
    ```
    输出"ipiv"为pivot信息：
    ```
    [1, 2, 3, 4]
    ```

## 函数原型

- 函数定义

    公开头文件为 `cann_ops_solver.h`。

    ```cpp
    aclsolverStatus_t aclsolverSgetrf_bufferSize(
        aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork);

    aclsolverStatus_t aclsolverSgetrf(
        aclsolverHandle_t handle, int m, int n, float *A, int lda,
        float *Workspace, int *devIpiv, int *devInfo);
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
        <td align="left">solver handle，通过aclsolverCreate创建，并通过aclsolverSetStream设置stream</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的行数，0≤m≤8192；本任务数值验收覆盖m≤4096</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的列数，n≥0；workspace元素数须能以int表示，且总分配须满足设备内存容量；本任务数值验收覆盖n≤4096</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">Device FLOAT32，逻辑shape为[m,n]；列主序A[c*lda+r]，分配至少lda*n个元素；输入原始矩阵，输出packed LU，L的单位对角不存储</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">相邻两列起始元素的步长，以FLOAT32元素计，lda≥max(1,m)</td>
    </tr>
    <tr>
        <td align="center">Workspace</td>
        <td align="center">输入/临时</td>
        <td align="left">Device FLOAT32工作区，与A、主元和INFO不重叠；容量至少为bufferSize返回的Lwork个元素</td>
    </tr>
    <tr>
        <td align="center">devIpiv</td>
        <td align="center">输出</td>
        <td align="left">Device INT32，1-based行交换索引；nullptr表示不选主元，长度为min(m,n)</td>
    </tr>
    <tr>
        <td align="center">devInfo</td>
        <td align="center">输出</td>
        <td align="left">Device INT32，长度为1；成功为0，正值k表示首个U(k,k)=0；参数错误在可安全写回时填-i，编号不计handle</td>
    </tr>
    <tr>
        <td align="center">Lwork</td>
        <td align="center">查询输出</td>
        <td align="left">bufferSize的Host int输出，单位为FLOAT32元素，实际字节数为Lwork*sizeof(float)</td>
    </tr>
    </table>

- 算子约束：

  - m、n可为0；空问题成功返回并写devInfo=0，devInfo仍须为有效Device INT32。
  - devIpiv为nullptr时不执行主元交换；非空时返回min(m,n)个1-based行交换索引。
  - 先调用aclsolverSgetrf_bufferSize，按返回的FLOAT32元素数分配Device workspace；查询时A可为nullptr，Lwork须指向Host int。
  - m、n、lda和分配容量须满足上表约束；workspace查询拒绝超过int可表示范围的元素数。
  - 有效矩阵元素要求为有限FLOAT32数；A、主元和INFO须满足各自类型的对齐与容量要求。
  - INFO为Device输出；返回码报告参数、资源或执行状态。读取INFO及数值结果前同步handle关联的stream，并同时检查返回码和INFO。
  - 部分Device元数据和有限值校验会同步调用方stream；矩阵数值计算在AI Core执行。
  - 输入、输出、主元、指针数组和workspace在计算完成前保持有效；相同输入在同一stream串行重复调用时保持bit-wise一致。

- 调用实现

    Host完成参数校验并在handle关联的stream上启动Ascend C核函数；查询接口只返回所需workspace容量。

- 返回值

    成功提交返回 `ACLSOLVER_STATUS_SUCCESS`；handle为空返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`，非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`。资源或运行时失败返回相应 `aclsolverStatus_t`，状态定义见 [cann_ops_solver_common.h](../../include/cann_ops_solver_common.h)。

## 调用示例

- 完整代码示例：[Sgetrf基础测试](../../test/sgetrf/sgetrf_test.cpp)；构建和测试说明见[test/sgetrf](../../test/sgetrf/README.md)。

- 核心调用步骤：

    调用方先完成ACL初始化、设备选择、handle与stream创建。以下函数设置handle关联的stream，分配Device资源、调用公开接口并回传结果；函数返回后由调用方销毁handle和stream。

    ```cpp
    #include <algorithm>
    #include <stdexcept>
    #include <vector>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    static void Require(bool value)
    {
        if (!value)
        {
            throw std::runtime_error("solver example failed");
        }
    }

    // Synchronize before releasing Device memory used by the supplied stream.
    struct DeviceBuffer
    {
        void *data = nullptr;
        aclrtStream stream;
        DeviceBuffer(size_t bytes, aclrtStream s) : stream(s)
        {
            Require(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        }
        ~DeviceBuffer()
        {
            if (data != nullptr && aclrtSynchronizeStream(stream) == ACL_SUCCESS)
            {
                aclrtFree(data);
            }
        }
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;
        template <typename T> T *As() { return static_cast<T *>(data); }
    };

    // The caller initializes ACL, selects a device, creates handle/stream,
    // and destroys them after this function returns.
    std::vector<float> SgetrfExample(aclsolverHandle_t handle, aclrtStream stream)
    {
        Require(aclsolverSetStream(handle, stream) == ACLSOLVER_STATUS_SUCCESS);
        const int n = 4, lda = n;
        // Column-major input for the matrix in the example above.
        std::vector<float> hostA = {4, 3, 2, 1, 3, 4, 3, 2,
                                    2, 3, 4, 3, 1, 2, 3, 4};
        const size_t bytes = hostA.size() * sizeof(float);
        DeviceBuffer a(bytes, stream), piv(n * sizeof(int), stream), info(sizeof(int), stream);
        Require(aclrtMemcpy(a.data, bytes, hostA.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
        int factorWork = 0;
        Require(aclsolverSgetrf_bufferSize(handle, n, n, a.As<float>(), lda, &factorWork)
                == ACLSOLVER_STATUS_SUCCESS);
        DeviceBuffer workspace(static_cast<size_t>(factorWork) * sizeof(float), stream);
        Require(aclsolverSgetrf(handle, n, n, a.As<float>(), lda, workspace.As<float>(),
                               piv.As<int>(), info.As<int>()) == ACLSOLVER_STATUS_SUCCESS);
        Require(aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        int hostInfo = -1;
        Require(aclrtMemcpy(&hostInfo, sizeof(hostInfo), info.data, sizeof(hostInfo),
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
        Require(hostInfo == 0);

        Require(aclrtMemcpy(hostA.data(), bytes, a.data, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
        return hostA;
    }
    ```

    示例中的资源对象在释放Device内存前同步stream；同步失败时保留相应分配。完整环境初始化和测试异常处理见上述源码。
