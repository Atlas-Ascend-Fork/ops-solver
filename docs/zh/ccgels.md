# CCgels

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

以 COMPLEX64 主精度分解并求解 `min ||A X - B||_F`，要求 `m >= n`。
A 为 `m × n`，B 为 `m × nrhs`，X 为 `n × nrhs`。X 写入独立输出缓冲区，B 保持不变，
A 的有效区域可被分解结果覆盖。通过 QR、QᴴB 和上三角回代求解；直接求解返回 niter=0。

## 函数原型

```c
aclsolverStatus_t aclsolverCCgels_bufferSize(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    aclsolverFloatComplex *dA, int ldda, aclsolverFloatComplex *dB, int lddb,
    aclsolverFloatComplex *dX, int lddx, void *dwork, size_t *lwork_bytes);

aclsolverStatus_t aclsolverCCgels(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    aclsolverFloatComplex *dA, int ldda, aclsolverFloatComplex *dB, int lddb,
    aclsolverFloatComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes,
    int *niter, int *dinfo);
```

## 参数说明

| 参数 | 归属 | 说明 |
| --- | --- | --- |
| handle | Host | 已创建的 solver handle |
| m、n、nrhs | Host 标量 | `m >= n >= 0`，`nrhs >= 0` |
| dA、ldda | Device 输入/输出 / Host 标量 | 列主序 `ldda × n`，`ldda >= max(1,m)` |
| dB、lddb | Device 只读输入 / Host 标量 | 列主序 `lddb × nrhs`，`lddb >= max(1,m)`；调用后保持不变 |
| dX、lddx | Device 输出 / Host 标量 | 独立的列主序 `lddx × nrhs`，`lddx >= max(1,n)` |
| dwork | Device 参数 | bufferSize 为接口兼容保留的参数，可传 NULL，不读取或写入 |
| dWorkspace | Device 工作区 | 按查询结果使用 `aclrtMalloc` 分配，不与 A、B、X 重叠 |
| lwork_bytes | Host | 查询时为 size_t 输出指针；计算时为字节数，不是复数元素个数 |
| niter | Host 输出 | 可写 int 指针，当前直接求解算法返回 0 |
| dinfo | Device 输出 | 可写 INT32；0 成功，`-i` 为参数非法，编号不计 handle；正值 j 表示分解后的 R(j,j) 精确为零（1-based） |

## 约束与状态

- 类型为 `aclsolverFloatComplex` / COMPLEX64；仅列主序，不支持广播。
- 任务范围为 `1 <= n <= m <= 4096`、`1 <= nrhs <= 128`，并处理空问题。范围拒绝仅作用于 `m`、`n`、`nrhs` **均非零**的有效问题：此时超出该范围（`m`/`n` > 4096 或 `nrhs` > 128）并非未定义行为，执行与 `bufferSize` 查询两个入口共用同一校验并确定性地返回 `ACLSOLVER_STATUS_INVALID_VALUE`（`m`/`n` 超限写 `dinfo = -1`，`nrhs` 超限写 `dinfo = -3`，编号不计 handle）。任一维度为 0 的空问题不受本范围约束，按下方空问题条目短路成功返回（检视响应：与 `m=8192,n=0` 等组合的实际行为对齐，避免与空问题条目互斥）。
- `m<n` 不支持并返回 `ACLSOLVER_STATUS_INVALID_VALUE`。A 应满足满列秩求解条件。
- 若 QR 后存在精确零对角，按已评审设计返回首个位置 `dinfo=j>0`，不执行回代且 X 保持不变。
  Host 状态仍表示调用是否成功下发；同步后必须读取 dinfo。此检查不按容差推断数值秩，
  不提供秩亏矩阵的最小范数解，也不把很小但非零的对角误判为零。
- 参数首先须满足 `m >= n >= 0`、`nrhs >= 0` 和合法 leading dimension。
  在此前提下，`n=0`（包括 `m=n=0`）或 `nrhs=0` 成功返回；不访问矩阵，
  写入 Host niter=0 和 Device dinfo=0，niter 和 dinfo 仍须可写。
  `m=0,n>0` 属于不支持的 `m<n`，返回 `ACLSOLVER_STATUS_INVALID_VALUE` 和 dinfo=-1，
  不因零维度而跳过参数合法性检查。
- A、B 必须为有限复数。非法参数、非有限有效输入或不足的工作区返回
  `ACLSOLVER_STATUS_INVALID_VALUE`；空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。
- lda padding、B 全部内容及 X padding 保持不变。调用方须提供独立的 A、B、X 缓冲区。
- 使用 handle 中的 stream。输入有限性检查需要同步以返回状态；后续计算在同一 stream
  执行，读取输出或释放工作区前须同步。
- 主计算在 AI Core 完成，不调用 CPU 求解。HF32 块乘采用高低位补偿，精度仍按完整算子残差验证。

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

查询 `lwork_bytes`，分配相应字节的 Device 工作区；niter 在 Host 分配，dinfo 在 Device
分配。调用后同步 stream，复制 dinfo、X 到 Host。每次重复测试须恢复被修改的 A。
示例入口为 [profile_case.cpp](../../test/complex_qr/profile_case.cpp) 的 `--operator cgels`；
多 RHS、padding、空问题、错误参数、精度和性能命令见 [测试说明](../../test/complex_qr/README.md)。

方阵按各 RHS 列的最大归一化残差判定；超定矩阵按正规方程最优性残差判定。
