# Sormqr

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |


注：本算子的 kernel 位于`src/sormqr/arch35/`，仅在 SOC 为 ascend950 时参与编译（见`src/CMakeLists.txt`
的架构目录机制），其他产品不提供该接口的实现。实现与验证在<term>Ascend 950PR</term>上完成，
同架构的<term>Ascend 950DT</term>复用同一份 arch35 代码。

## 上下游接口

输入反射器来自 [Sgeqrf 分解](sgeqrf.md)；构造显式 Q 见 [Sorgqr](sorgqr.md)；最小二乘直接求解见 [Sgels](sgels.md)。

## 功能说明

- 接口功能\
    aclsolverSormqr：计算 $op(Q) \cdot C$（side = LEFT）或 $C \cdot op(Q)$（side = RIGHT），就地覆盖 C。

- 计算公式
    $$
    C \leftarrow op(Q) \cdot C \;(\text{LEFT}), \qquad C \leftarrow C \cdot op(Q) \;(\text{RIGHT})
    $$
    其中 $Q = H(1) H(2) \cdots H(k)$ 由 Sgeqrf 的输出给出；$op(Q) = Q$（trans = N）或 $Q^T$（trans = T / C，实数下两者等价）。

- 示例 \
    输入"A"、"tau"为 aclsolverSgeqrf 的输出，"C"为待作用的 $m \times n$ 矩阵，\
    取 side = ACLSOLVER_SIDE_LEFT、trans = ACLSOLVER_OP_T，\
    调用"aclsolverSormqr"算子后，"C"被覆盖为 $Q^T C$。\
    $k = 0$ 时 $op(Q) = I$，C 原样不变。

## 函数原型
- 函数定义
    ```
    aclsolverStatus_t aclsolverSormqr_bufferSize(
        aclsolverHandle_t handle,
        aclsolverSideMode_t side,
        aclsolverOperation_t trans,
        int m,
        int n,
        int k,
        const float *A,
        int lda,
        const float *tau,
        const float *C,
        int ldc,
        int *lwork);

    aclsolverStatus_t aclsolverSormqr(
        aclsolverHandle_t handle,
        aclsolverSideMode_t side,
        aclsolverOperation_t trans,
        int m,
        int n,
        int k,
        const float *A,
        int lda,
        const float *tau,
        float *C,
        int ldc,
        float *work,
        int lwork,
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
        <td align="center">side</td>
        <td align="center">输入</td>
        <td align="left">ACLSOLVER_SIDE_LEFT 从左施加，ACLSOLVER_SIDE_RIGHT 从右施加；其他值返回参数错误</td>
    </tr>
    <tr>
        <td align="center">trans</td>
        <td align="center">输入</td>
        <td align="left">ACLSOLVER_OP_N 施加 Q；ACLSOLVER_OP_T 与 ACLSOLVER_OP_C 均施加 Q 的转置（实数下等价，结果逐位相同）</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">C 的行数，需满足 m >= 0</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">C 的列数，需满足 n >= 0</td>
    </tr>
    <tr>
        <td align="center">k</td>
        <td align="center">输入</td>
        <td align="left">反射子个数，需满足 0 <= k <= nq，其中 nq = m（LEFT）或 n（RIGHT）。两个入口均要求 nq <= 4096；当 m、n、k 均非零且 nq 超过当前 kernel 上限 4096 时，_bufferSize 与执行接口均返回 ACLSOLVER_STATUS_NOT_SUPPORTED（主入口另写 devInfo = -3/-4）。仅在 m、n 或 k 为 0 的退化情形下，_bufferSize 在 nq 检查前返回 lwork = 0，执行接口直接成功返回。</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入</td>
        <td align="left">Sgeqrf 输出的 Householder 表示<br>数据类型仅支持 FLOAT32，<strong>列主序</strong>存放</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A 的 leading dimension，需满足 lda >= max(1, nq)</td>
    </tr>
    <tr>
        <td align="center">tau</td>
        <td align="center">输入</td>
        <td align="left">Sgeqrf 输出的 k 个反射子标量</td>
    </tr>
    <tr>
        <td align="center">C</td>
        <td align="center">输入/输出</td>
        <td align="left">被 op(Q) 作用的矩阵，就地覆盖<br>数据类型仅支持 FLOAT32，<strong>列主序</strong>存放，shape 为 [m, n]</td>
    </tr>
    <tr>
        <td align="center">ldc</td>
        <td align="center">输入</td>
        <td align="left">C 的 leading dimension，需满足 ldc >= max(1, m)</td>
    </tr>
    <tr>
        <td align="center">work</td>
        <td align="center">输入</td>
        <td align="left">device 侧 workspace，元素数不少于 lwork</td>
    </tr>
    <tr>
        <td align="center">lwork</td>
        <td align="center">输入/输出</td>
        <td align="left">_bufferSize 时为输出、主入口时为输入；workspace 的单位是 <strong>float 元素个数</strong>（不是字节），由对应的 _bufferSize 返回</td>
    </tr>
    <tr>
        <td align="center">devInfo</td>
        <td align="center">输出</td>
        <td align="left">device 侧 int32 输出指针（<strong>必须非空，不允许传 nullptr</strong>）：成功写 0，第 i 个参数非法写 -i（handle 不计入）；为空返回 ACLSOLVER_STATUS_INVALID_VALUE</td>
    </tr>
    </table>

- 算子约束：
  - m、n、k 任一为 0 时 lwork = 0；特别地 k = 0 表示 op(Q) = I，主入口写 devInfo = 0 后直接返回成功、不下发 kernel，C 保持原值。
  - _bufferSize 不读 A、tau、C，允许传 nullptr。
  - trans = ACLSOLVER_OP_C 与 ACLSOLVER_OP_T 在实数下走同一条计算路径，结果逐位相同（测试中对此有逐位比对用例）。
  - 与 Sorgqr 相同，窄 GEMM 走确定性分 K，保证重复运行逐位一致。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 返回值

两个入口均返回 `aclsolverStatus_t`：

| 返回值 | 触发条件 |
|---|---|
| `ACLSOLVER_STATUS_SUCCESS` | 查询返回 lwork 或变换已提交；不代表异步数值一定成功（devInfo 须同步后读取） |
| `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR` | 仅主入口：handle 为 nullptr |
| `ACLSOLVER_STATUS_INVALID_VALUE` | 参数非法（`lwork` 为 nullptr、`CheckShape` 形状不满足、lwork 小于查询值等；主入口另写 `devInfo = -i`，handle 不计入编号） |
| `ACLSOLVER_STATUS_INTERNAL_ERROR` | 仅主入口：平台资源查询失败（核数为 0 等） |
| `ACLSOLVER_STATUS_EXECUTION_FAILED` | 仅主入口：kernel 下发后线程级错误 |
| `ACLSOLVER_STATUS_MAPPING_ERROR` | 仅主入口：WriteInfo 写 device info 失败 |

`_bufferSize` 显式 `(void)handle`（尺寸查询与 handle/stream 无关，sormqr_host.cpp:153），handle 为空仍可查询成功；其无 devInfo 参数，参数错误仅以返回码表达。

## 调用示例
- 完整代码示例：[aclsolverSormqr示例](../../test/sormqr/sormqr_test.cpp)
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

        // A、tau 来自先前的 aclsolverSgeqrf 调用
        int m = 4, n = 2, k = 4;
        int lda = m, ldc = m;
        aclsolverSideMode_t side = ACLSOLVER_SIDE_LEFT;
        aclsolverOperation_t trans = ACLSOLVER_OP_T;
        int lwork = 0;
        aclsolverSormqr_bufferSize(handle, side, trans, m, n, k, nullptr, lda, nullptr, nullptr, ldc, &lwork);

        void *dA = nullptr, *dTau = nullptr, *dC = nullptr, *dWork = nullptr, *dInfo = nullptr;
        aclrtMalloc(&dA, static_cast<size_t>(lda) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dTau, static_cast<size_t>(k) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dC, static_cast<size_t>(ldc) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dWork, static_cast<size_t>(lwork) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dInfo, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

        // 调用 aclsolverSormqr
        auto ret = aclsolverSormqr(handle, side, trans, m, n, k, reinterpret_cast<const float *>(dA), lda,
                                   reinterpret_cast<const float *>(dTau), reinterpret_cast<float *>(dC), ldc,
                                   reinterpret_cast<float *>(dWork), lwork, reinterpret_cast<int *>(dInfo));
        CHECK_RET(ret == ACLSOLVER_STATUS_SUCCESS, LOG_PRINT("aclsolverSormqr failed. ERROR: %d\n", ret); return ret);
        aclrtSynchronizeStream(stream);

        aclrtFree(dA); aclrtFree(dTau); aclrtFree(dC); aclrtFree(dWork); aclrtFree(dInfo);


        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
