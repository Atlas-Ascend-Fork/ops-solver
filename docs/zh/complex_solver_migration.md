# 复数求解器接口迁移

## 破坏性变更（Breaking change）

`aclsolverCgetrf`、`aclsolverCgetri`、`aclsolverCgetriBatched`、`aclsolverCmatinvBatched` 已更新签名及内存语义。
旧调用方需要修改源码，并使用同一版本的头文件和库重新编译、链接；旧二进制与新库不兼容。
这四个接口当前仅接受 Ascend950 系列，其他架构返回 `ACLSOLVER_STATUS_ARCH_MISMATCH`。

| 项目 | 旧接口 | 当前接口 |
| --- | --- | --- |
| 返回值 | `aclError`，成功为 `ACL_SUCCESS` | `aclsolverStatus_t`，成功为 `ACLSOLVER_STATUS_SUCCESS` |
| 复数类型 | `std::complex<float>` | `aclFloatComplex { float real; float imag; }` |
| 维度类型 | `int64_t` | `int`，按各接口有效范围检查后传入 |
| 矩阵与状态内存 | Host 指针，内部管理搬运 | 调用方管理的 Device 矩阵、主元、工作空间和 `info` |
| 矩阵布局 | 原接口的连续布局 | 列主序，`A[col * lda + row]`，允许 leading dimension padding |
| 单矩阵求逆输入 | 原始矩阵，内部执行分解 | 已完成的 LU 和匹配主元；原始矩阵先调用 `Cgetrf` |
| 单矩阵工作空间 | 内部管理 | 先查询 `*_bufferSize`，按复数元素数分配 Device 空间 |
| 批量输入/输出 | 连续矩阵缓冲区 | Device 指针数组；数组每项也是 Device 地址 |
| 批量状态 | 旧实现仅清零首项 | 每个矩阵对应一个 Device `int` 状态 |
| 小矩阵直接求逆 | `n ≥ 32` 转调批量接口 | `CmatinvBatched` 支持 `n ≤ 32`，超出范围返回参数错误 |

## 迁移步骤

1. 初始化 CANN，选择 Device，创建 solver handle 并设置调用方 stream。
2. 将原始 Host 数据转换为列主序 `aclFloatComplex`；通过 `aclrtMalloc` 分配 Device 内存并复制输入。仅转换指针类型不会完成数据迁移。
3. 单矩阵先用有效 Device `A` 查询工作空间，`Lwork` 保存在 Host。分配字节数为 `size_t(Lwork) * sizeof(aclFloatComplex)`；查询结果为 0 时可省略该空间。
4. 需要求逆时，先执行 LU 分解，再把分解结果和对应 Device 主元传给 `Cgetri` 或 `CgetriBatched`。无主元分解与求逆均传空主元指针。
5. 批量接口先构造每个矩阵的 Device 地址列表，再把列表复制到 Device。每个输出矩阵独立，输入与输出不重叠。
6. 按 `ACLSOLVER_STATUS_SUCCESS` 检查返回值，再同步 stream 并回读 Device `info`。`info > 0` 表示奇异，`info < 0` 表示非法参数。输入的 `info` 内存本身须有效。
7. 所有 Device 数据在对应 stream 完成前保持有效；完成后由调用方释放。

## 单矩阵调用示例

下例是调用片段：已创建 `handle`、`stream`，设置 `aclsolverSetStream(handle, stream)`，并准备了 `0 < n ≤ 4096` 的列主序方阵 `dA`、`n` 个 Device `int` 主元 `devIpiv` 和 1 个 Device `int` 状态 `devInfo`。
所有 CANN 调用需检查 `ACL_SUCCESS`，solver 调用需检查 `ACLSOLVER_STATUS_SUCCESS`。片段只展示查询、内存分配与接口顺序，完整错误处理及资源释放见[共享测试实现](../../test/utils/complex_solver_test.h)。

```cpp
int luWork = 0;
int inverseWork = 0;
aclsolverStatus_t status = aclsolverCgetrf_bufferSize(handle, n, n, dA, lda, &luWork);
// 检查 status 后再使用查询结果。
status = aclsolverCgetri_bufferSize(handle, n, dA, lda, &inverseWork);
// 检查 status；两次计算复用较大的工作空间。
int workElements = luWork > inverseWork ? luWork : inverseWork;
size_t workBytes = static_cast<size_t>(workElements) * sizeof(aclFloatComplex);
aclFloatComplex* workspace = nullptr;
aclError error = aclrtMalloc(reinterpret_cast<void**>(&workspace), workBytes,
                            ACL_MEM_MALLOC_HUGE_FIRST);
// 检查 error。
status = aclsolverCgetrf(handle, n, n, dA, lda, workspace, devIpiv, devInfo);
// 检查 status；同步并回读 LU 状态，确认非奇异后再求逆。
error = aclrtSynchronizeStream(stream);
int info = 0;
error = aclrtMemcpy(&info, sizeof(info), devInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST);
// 检查 error 和 info == 0。
status = aclsolverCgetri(handle, n, dA, lda, devIpiv, workspace, workElements, devInfo);
// 检查 status；同步并回读求逆状态，再使用结果。
error = aclrtSynchronizeStream(stream);
error = aclrtMemcpy(&info, sizeof(info), devInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST);
// 检查 error 和 info == 0，读取 dA 中的逆矩阵，再释放资源。
error = aclrtFree(workspace);
```

## 与其他接口的关系

本次迁移仅覆盖上述四个复数 LU/求逆接口及两个查询接口。
实数 LU 接口已通过 [PR !116](https://gitcode.com/cann/ops-solver/pull/116) 合入上游，`Sgetrf`、`Sgetri` 及实数批量求逆接口同样使用 Device 列主序、工作空间查询和 Device 主元/状态约定；具体参数及范围应按各接口文档调用。
Hermitian 特征值接口 `Cheevj` 仍使用其旧约定，不属于实数 LU 接口族。

本分支已合并 !116 对应的上游提交，逐段合并公共头文件和接口列表，保留实数与复数接口、公共复数类型及共同的 handle/status 定义。
`Cheevj` 的调用约定统一属于后续独立工作，本次未实施。

## 接口详情

- [Cgetrf 与工作空间查询](cgetrf.md)
- [Cgetri 与工作空间查询](cgetri.md)
- [CgetriBatched](cgetri_batched.md)
- [CmatinvBatched](cmatinv_batched.md)
