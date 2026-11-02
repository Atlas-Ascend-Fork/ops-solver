# Cungqr

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

已实测设备为 Ascend 950PR。表中 Atlas 训练系列产品指非 A2 代际的历史产品。
使用仓库 README 中配套的 CANN Toolkit。其他架构保留公开接口，非空 handle
返回 `ACLSOLVER_STATUS_ARCH_MISMATCH`，空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`，不访问输入或输出。

## 功能说明

从 [Cgeqrf](cgeqrf.md) 产生的前 k 个 Householder 反射器构造显式 `m × n` 酉矩阵 Q，
原地覆盖 A。要求 `m >= n >= k >= 0`，支持 k 小于 n；`k=0` 时输出单位矩阵的前 n 列。
Q 的列满足 `QᴴQ = I`（浮点误差范围内）。

## 函数原型

```c
aclsolverStatus_t aclsolverCungqr_bufferSize(
    aclsolverHandle_t handle, int m, int n, int k,
    const aclsolverFloatComplex *A, int lda, const aclsolverFloatComplex *tau, int *lwork);

aclsolverStatus_t aclsolverCungqr(
    aclsolverHandle_t handle, int m, int n, int k,
    aclsolverFloatComplex *A, int lda, const aclsolverFloatComplex *tau,
    aclsolverFloatComplex *work, int lwork, int *devInfo);
```

## 参数说明

| 参数 | 归属 | 说明 |
| --- | --- | --- |
| handle | Host | 已创建的 solver handle；使用其 stream |
| m、n、k | Host 标量 | Q 行数、列数和使用的反射器数，`m >= n >= k >= 0` |
| A | Device 输入/输出 | 列主序 `lda × n`；前 k 列下三角为反射器，返回前 n 列 Q |
| lda | Host 标量 | `lda >= max(1,m)`，不要求等于 m；padding 保持不变 |
| tau | Device 输入 | 长度至少 k；k=0 时允许为空，计算不修改 tau |
| work | Device 工作区 | 使用 `aclrtMalloc` 按 bufferSize 结果分配，不与 A 或 tau 重叠 |
| lwork | Host | 查询时为输出指针；计算时为复数元素个数，非字节数 |
| devInfo | Device 输出 | 一个 INT32；成功 0，参数非法为 `-i`，不计 handle |

## 约束与状态

- 类型为 `aclsolverFloatComplex` / COMPLEX64；只接受列主序和有限的反射器、tau。
  反射器对角为隐含的 1，上三角不参与 Q 构造。
- 任务保证 `1 <= m <= 4096` 内全部合法 n、k 及空问题。
- `m=0` 或 `n=0` 时不访问矩阵并成功写入 info=0。非空且 k=0 时仍须提供可写 A。
- 非法维数、lda、指针、非有限有效输入或不足的 workspace 返回
  `ACLSOLVER_STATUS_INVALID_VALUE`；空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。
- bufferSize 的 lwork 位于 Host；A、tau、work、devInfo 均位于 Device。
- 复用 handle 的 stream。输入有限性检查需要同步以返回状态；后续 Q 构造在同一 stream
  上执行，读取或释放输出前须同步。不会调用 CPU QR 或 Q 构造作为计算回退。
- 相同输入在同一 stream 串行重复执行采用固定计算顺序。A 为原地输出，重复测试必须恢复输入。

## 输入检查与同步开销

非空计算路径在下发主计算前，先由设备端扫描实际使用的输入，检查 NaN/Inf，
随后同步 handle 对应的 stream，并将 4 字节检查结果复制到 Host，以同步返回
`ACLSOLVER_STATUS_INVALID_VALUE`。该同步也会等待同一 stream 上此前排队的任务，
因此调用不是完全异步；主计算仍在同一 stream 上下发，API 返回不代表主计算完成。
空问题与参数拒绝路径不执行这次扫描。

扫描增加与有效输入元素数相关的读取开销；leading-dimension padding 及反射器的
隐式对角/上三角不属于有效输入。大矩阵应同时评估扫描 kernel 占比和完整调用耗时。
kernel 性能统计应包含检查 kernel；仅统计主计算会低估成本。kernel 耗时不包含
Host 等待和 D2H 回读的全部开销，不能据此宣称端到端延迟相同。
当前不提供跳过检查的开关，以保持非有限输入的错误返回契约。

## 调用与测试

保留 geqrf 输出的反射器和 tau，查询本次 lwork 并分配 Device 工作区，调用接口、同步
stream 后检查 devInfo。若后续还要调用 unmqr，应另存反射器，因为显式 Q 会覆盖 A。
可运行 [profile_case.cpp](../../test/complex_qr/profile_case.cpp) 的 `--operator cungqr` 入口；
完整精度、确定性和性能命令见 [测试说明](../../test/complex_qr/README.md)。
