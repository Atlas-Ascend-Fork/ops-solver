# Cunmqr

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

使用 [Cgeqrf](cgeqrf.md) 返回的 Householder 反射器对矩阵 C 施加酉变换：

- LEFT：`C = op(Q) C`，Q 的阶数为 m。
- RIGHT：`C = C op(Q)`，Q 的阶数为 n。
- N：使用 Q；C：使用共轭转置 Qᴴ。**T（普通转置）不支持并返回错误。**

矩阵 C 原地更新；输入 A 和 tau 保持不变，不需要调用方预先构造显式 Q。

## 函数原型

```c
aclsolverStatus_t aclsolverCunmqr_bufferSize(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclsolverFloatComplex *A, int lda, const aclsolverFloatComplex *tau,
    const aclsolverFloatComplex *C, int ldc, int *lwork);

aclsolverStatus_t aclsolverCunmqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclsolverFloatComplex *A, int lda, const aclsolverFloatComplex *tau,
    aclsolverFloatComplex *C, int ldc, aclsolverFloatComplex *work, int lwork, int *devInfo);
```

## 参数说明

| 参数 | 归属 | 说明 |
| --- | --- | --- |
| handle | Host | 已创建的 solver handle |
| side | Host 标量 | `ACLSOLVER_SIDE_LEFT=0` 或 `ACLSOLVER_SIDE_RIGHT=1` |
| trans | Host 标量 | `ACLSOLVER_OP_N=0` 或 `ACLSOLVER_OP_C=2`；T=1 非法 |
| m、n | Host 标量 | C 的非负行数和列数 |
| k | Host 标量 | `0 <= k <= nq`，nq 为 LEFT 时 m、RIGHT 时 n |
| A、lda | Device 输入 / Host 标量 | 列主序 `lda × k` 反射器，`lda >= max(1,nq)` |
| tau | Device 输入 | 至少 k 个复数系数 |
| C、ldc | Device 输入/输出 / Host 标量 | 列主序 `ldc × n`，有效区域 `m × n`，`ldc >= max(1,m)` |
| work | Device 工作区 | 本次 bufferSize 查询所需工作区，不与输入、输出重叠 |
| lwork | Host | 查询时为输出指针，计算时为复数元素个数，非字节数 |
| devInfo | Device 输出 | 一个 INT32；0 成功，`-i` 为参数非法，编号不计 handle |

## 约束与状态

- COMPLEX64、列主序；不支持广播。lda、ldc 可含合法 padding，C 的 padding 保持不变。
- 任务保证 `m,n` 从 1 到 4096 内的合法组合，以及空问题；四种 side/trans 组合均实现。
- `m=0`、`n=0` 或 `k=0` 成功返回 info=0，不施加变换。维数、枚举和 leading dimension
  仍须合法；空问题允许矩阵指针为空，devInfo 必须可写。
- 有效反射器、tau 和 C 必须为有限复数。非法参数或不足的 workspace 返回
  `ACLSOLVER_STATUS_INVALID_VALUE`；T 的参数编号为 2，即 info=-2。
- 空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。bufferSize 不读取 Device 数据。
- 使用调用方 stream；输入检查为返回非有限值错误需要同步，后续变换在同一 stream
  执行。读取结果或释放内存前应同步 stream。
- 核心运算由 AI Core 完成。右乘在分块打包和写回中按 Cᴴ 的逻辑布局处理，
  所需缓冲已计入查询工作区，不要求调用方进行转置。

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

传入尚未被 ungqr 覆盖的反射器及 tau，查询并分配 work 后调用接口。
示例入口为 [profile_case.cpp](../../test/complex_qr/profile_case.cpp) 的 `--operator cunmqr`。
验收参考链使用同一份 NPU 反射器构造显式 Q；因此先通过 Cungqr 验证，再检查本接口。
完整命令见 [测试说明](../../test/complex_qr/README.md)。
