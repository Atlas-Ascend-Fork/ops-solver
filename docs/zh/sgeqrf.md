## Sgeqrf

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |


注：本算子的 kernel 位于`src/sgeqrf/arch35/`，仅在 SOC 为 ascend950 时参与编译（见`src/CMakeLists.txt`
的架构目录机制），其他产品不提供该接口的实现。实现与验证在<term>Ascend 950PR</term>上完成，
同架构的<term>Ascend 950DT</term>复用同一份 arch35 代码。

## 功能说明

- 接口功能\
    aclsolverSgeqrf：对单精度实数矩阵做 QR 分解。

- 计算公式
    $$
    A = Q \cdot R
    $$
    其中 $A$ 为 $m \times n$ 阶单精度实数矩阵，$Q$ 为 $m \times m$ 正交矩阵，$R$ 为 $m \times n$ 上三角矩阵。
    $Q$ 不显式输出，而是以 $k = \min(m, n)$ 个 Householder 反射子
    $H(i) = I - \tau_i v_i v_i^T$ 的形式给出，$Q = H(1) H(2) \cdots H(k)$。

- 示例 \
    输入"A"为（列主序，$m = n = 2$，$lda = 2$）：
    ```
    [ 3.0,  4.0
      4.0, -3.0]
    ```
    调用"aclsolverSgeqrf"算子后，\
    输出"A"的上三角为 $R$、下梯形为反射子 $v$ 的非对角分量（对角上的 1 隐含不存）：
    ```
    R = [-5.0, 0.0
          0.0, -5.0]
    ```
    输出"TAU"为 2 个反射子标量。\
    符号约定与 LAPACK sgeqrf 一致，因此同一输入重复调用的结果<strong>逐位相同</strong>。

## 函数原型
- 函数定义
    ```
    aclsolverStatus_t aclsolverSgeqrf_bufferSize(
        aclsolverHandle_t handle,
        int m,
        int n,
        float *A,
        int lda,
        int *Lwork);

    aclsolverStatus_t aclsolverSgeqrf(
        aclsolverHandle_t handle,
        int m,
        int n,
        float *A,
        int lda,
        float *TAU,
        float *Workspace,
        int Lwork,
        int *devInfo);
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
        <td align="left">solver handle，通过 aclsolverCreate 创建；计算在 aclsolverSetStream 绑定的 stream 上下发</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">矩阵 A 的行数，需满足 m >= 0；受 UB 容量约束，当前配置支持 m <= 8192（超限返回 ACLSOLVER_STATUS_NOT_SUPPORTED）</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵 A 的列数，需满足 n >= 0</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵 A，输出上三角为 R、下梯形为 Householder 向量<br>数据类型仅支持 FLOAT32，<strong>列主序</strong>存放，shape 为 [m, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A 的 leading dimension，需满足 lda >= max(1, m)；<strong>允许 lda > m</strong>（padding）</td>
    </tr>
    <tr>
        <td align="center">TAU</td>
        <td align="center">输出</td>
        <td align="left">min(m, n) 个反射子标量<br>数据类型仅支持 FLOAT32</td>
    </tr>
    <tr>
        <td align="center">Workspace</td>
        <td align="center">输入</td>
        <td align="left">device 侧 workspace，元素数不少于 Lwork</td>
    </tr>
    <tr>
        <td align="center">Lwork</td>
        <td align="center">输入/输出</td>
        <td align="left">_bufferSize 时为输出、主入口时为输入；workspace 的单位是 <strong>float 元素个数</strong>（不是字节），由对应的 _bufferSize 返回</td>
    </tr>
    <tr>
        <td align="center">devInfo</td>
        <td align="center">输出</td>
        <td align="left">device 侧 int32：成功写 0，第 i 个参数非法写 -i（handle 不计入）<br>允许传 nullptr（不关心状态时）</td>
    </tr>
    </table>

- 算子约束：
  - **元素量级窗口**：要求 `5e-20 < max|A| < 1.8e19`。面板分解为了性能把反射子的归一化推迟到最后，这让可用范围**两端都有界**：面板内点积在 ~max|A| 量级的向量上做、乘积达 max|A|² （上界由此而来）；而存下来的反射子标量量级是 1/max|A|²（下界由此而来）。窗口外的输出含非有限值、调用方可检测，但不返回错误码。两个边界都是在 <term>Ascend 950PR</term> 上实测的，数量级分别约为 √FLT_MAX 与 1/√FLT_MAX。
  - m、n 均允许为 0（空问题）：此时 _bufferSize 返回 Lwork = 0，主入口写 devInfo = 0 并直接返回成功，不下发 kernel。
  - _bufferSize 不读 A，允许传 nullptr：调用方通常先问尺寸再分配。
  - devInfo 的 -i 语义按 LAPACK 约定，i 为参数序号（不含 handle）。
  - 本实现为融合的单个 MIX kernel（AIV 做面板分解与 T 递推、AIC 做尾随更新），不依赖显式 Q。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverSgeqrf示例](../../test/sgeqrf/sgeqrf_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    #include <cstdio>

    #define CHECK_RET(cond, return_expr) \
        do {                            \
            if (!(cond)) {              \
                return_expr;            \
            }                           \
        } while (0)

    #define LOG_PRINT(message, ...) \
        std::printf(message, ##__VA_ARGS__)

    int32_t main(int32_t argc, char *argv[])
    {
        // 固定写法，acl 初始化
        int32_t deviceId = 0;
        aclrtStream stream = nullptr;
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        aclrtCreateStream(&stream);

        // 创建 solver handle 并设置 stream
        aclsolverHandle_t handle = nullptr;
        aclsolverCreate(&handle);
        aclsolverSetStream(handle, stream);

        // 构造输入数据（列主序）
        int m = 4;
        int n = 4;
        int lda = m;
        int Lwork = 0;
        aclsolverSgeqrf_bufferSize(handle, m, n, nullptr, lda, &Lwork);

        void *dA = nullptr, *dTau = nullptr, *dWork = nullptr, *dInfo = nullptr;
        aclrtMalloc(&dA, static_cast<size_t>(lda) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dTau, static_cast<size_t>(std::min(m, n)) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dWork, static_cast<size_t>(Lwork) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dInfo, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        // 把 A 拷到 dA ...

        // 调用 aclsolverSgeqrf
        auto ret = aclsolverSgeqrf(handle, m, n, reinterpret_cast<float *>(dA), lda,
                                   reinterpret_cast<float *>(dTau), reinterpret_cast<float *>(dWork), Lwork,
                                   reinterpret_cast<int *>(dInfo));
        CHECK_RET(ret == ACLSOLVER_STATUS_SUCCESS, LOG_PRINT("aclsolverSgeqrf failed. ERROR: %d\n", ret); return ret);
        aclrtSynchronizeStream(stream);

        aclrtFree(dA); aclrtFree(dTau); aclrtFree(dWork); aclrtFree(dInfo);


        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
