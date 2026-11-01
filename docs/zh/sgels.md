## SSgels

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |


注：本算子的 kernel 位于`src/sgels/arch35/`，仅在 SOC 为 ascend950 时参与编译（见`src/CMakeLists.txt`
的架构目录机制），其他产品不提供该接口的实现。实现与验证在<term>Ascend 950PR</term>上完成，
同架构的<term>Ascend 950DT</term>复用同一份 arch35 代码。

## 功能说明

- 接口功能\
    aclsolverSSgels：求解超定线性最小二乘问题。

- 计算公式
    $$
    \min_X \lVert A X - B \rVert_2, \qquad A \in \mathbb{R}^{m \times n},\ m \ge n
    $$
    解法为 QR 分解：$A = Q R$，再 $X = R^{-1} (Q^T B)(1{:}n, :)$。

- 示例 \
    输入"dA"为 $m \times n$ 系数矩阵、"dB"为 $m \times nrhs$ 右端项，\
    调用"aclsolverSSgels"算子后，"dX"为 $n \times nrhs$ 的最小二乘解，\
    "dA" 的内容被分解结果覆盖，"dB" <strong>不被修改</strong>，"niter" 恒为 0。

## 函数原型
- 函数定义
    ```
    aclsolverStatus_t aclsolverSSgels_bufferSize(
        aclsolverHandle_t handle,
        int m,
        int n,
        int nrhs,
        float *dA,
        int ldda,
        float *dB,
        int lddb,
        float *dX,
        int lddx,
        void *dwork,
        size_t *lwork_bytes);

    aclsolverStatus_t aclsolverSSgels(
        aclsolverHandle_t handle,
        int m,
        int n,
        int nrhs,
        float *dA,
        int ldda,
        float *dB,
        int lddb,
        float *dX,
        int lddx,
        void *dWorkspace,
        size_t lwork_bytes,
        int *niter,
        int *dinfo);
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
        <td align="left">A 的行数，需满足 m >= n >= 0 且 m <= 4096</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">A 的列数，需满足 n >= 0 且 n <= 4096</td>
    </tr>
    <tr>
        <td align="center">nrhs</td>
        <td align="center">输入</td>
        <td align="left">右端项个数，需满足 nrhs >= 0</td>
    </tr>
    <tr>
        <td align="center">dA</td>
        <td align="center">输入/输出</td>
        <td align="left">系数矩阵，<strong>内容会被分解结果破坏</strong><br>数据类型仅支持 FLOAT32，<strong>列主序</strong>，shape 为 [m, n]</td>
    </tr>
    <tr>
        <td align="center">ldda</td>
        <td align="center">输入</td>
        <td align="left">dA 的 leading dimension，需满足 ldda >= max(1, m)</td>
    </tr>
    <tr>
        <td align="center">dB</td>
        <td align="center">输入</td>
        <td align="left">右端项矩阵，<strong>只读、不被修改</strong><br>数据类型仅支持 FLOAT32，<strong>列主序</strong>，shape 为 [m, nrhs]</td>
    </tr>
    <tr>
        <td align="center">lddb</td>
        <td align="center">输入</td>
        <td align="left">dB 的 leading dimension，需满足 lddb >= max(1, m)</td>
    </tr>
    <tr>
        <td align="center">dX</td>
        <td align="center">输出</td>
        <td align="left">最小二乘解，shape 为 [n, nrhs]，<strong>列主序</strong><br>不得与 dB 同址（同址返回 -8）</td>
    </tr>
    <tr>
        <td align="center">lddx</td>
        <td align="center">输入</td>
        <td align="left">dX 的 leading dimension，需满足 lddx >= max(1, n)</td>
    </tr>
    <tr>
        <td align="center">dWorkspace</td>
        <td align="center">输入</td>
        <td align="left">device 侧 workspace，字节数不少于 lwork_bytes</td>
    </tr>
    <tr>
        <td align="center">lwork_bytes</td>
        <td align="center">输入/输出</td>
        <td align="left">_bufferSize 时为输出、主入口时为输入；单位是<strong>字节</strong></td>
    </tr>
    <tr>
        <td align="center">niter</td>
        <td align="center">输出</td>
        <td align="left">host 侧 int32：精化迭代次数。本实现为直接 QR 解、<strong>恒为 0</strong></td>
    </tr>
    <tr>
        <td align="center">dinfo</td>
        <td align="center">输出</td>
        <td align="left">device 侧 int32：成功写 0，第 i 个参数非法写 -i（handle 不计入）<br>允许传 nullptr</td>
    </tr>
    </table>

- 算子约束：
  - **元素量级窗口**：内部 QR 与 aclsolverSgeqrf 同一约束，要求 `5e-20 < max|dA| < 1.8e19`。
  - 仅支持超定/方阵情形 <strong>m >= n</strong>；m < n 返回参数错误。
  - m、n、nrhs 任一为 0 时 lwork_bytes = 0，主入口写 dinfo = 0 后返回成功。
  - <strong>dX 与 dB 不得同址</strong>（会返回 -8）；dB 是只读输入，内部先整块拷入 workspace。
  - 内部按 aclsolverSgeqrf -> aclsolverSormqr(LEFT, OP_T) -> 上三角回代三步实现，前两步直接调用本仓公开接口。
  - 多个接口串联调用时<strong>不要共用同一块 workspace</strong>：它们在同一条流上重叠执行，后者会冲掉前者的中间结果。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverSSgels示例](../../test/sgels/sgels_test.cpp)
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
        int m = 8, n = 4, nrhs = 1;
        int ldda = m, lddb = m, lddx = n;
        size_t lworkBytes = 0;
        aclsolverSSgels_bufferSize(handle, m, n, nrhs, nullptr, ldda, nullptr, lddb, nullptr, lddx, nullptr,
                                   &lworkBytes);

        void *dA = nullptr, *dB = nullptr, *dX = nullptr, *dWork = nullptr, *dInfo = nullptr;
        aclrtMalloc(&dA, static_cast<size_t>(ldda) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dB, static_cast<size_t>(lddb) * nrhs * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dX, static_cast<size_t>(lddx) * nrhs * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dWork, lworkBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dInfo, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

        // 调用 aclsolverSSgels
        int niter = 0;
        auto ret = aclsolverSSgels(handle, m, n, nrhs, reinterpret_cast<float *>(dA), ldda,
                                   reinterpret_cast<float *>(dB), lddb, reinterpret_cast<float *>(dX), lddx,
                                   dWork, lworkBytes, &niter, reinterpret_cast<int *>(dInfo));
        CHECK_RET(ret == ACLSOLVER_STATUS_SUCCESS, LOG_PRINT("aclsolverSSgels failed. ERROR: %d\n", ret); return ret);
        aclrtSynchronizeStream(stream);

        aclrtFree(dA); aclrtFree(dB); aclrtFree(dX); aclrtFree(dWork); aclrtFree(dInfo);


        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
