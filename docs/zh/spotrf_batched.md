# SpotrfBatched

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Ascend 950PR | √ |
| Ascend 950DT | √ |

## 功能说明

`aclsolverSpotrfBatched` 对 Device 指针数组中的每个单精度实数对称正定矩阵进行分解：

$$
A_b=L_b L_b^T\ \text{或}\ A_b=U_b^T U_b,\qquad 0\le b<\mathrm{batchSize}.
$$

矩阵可以分别分配，允许非连续槽位；所有矩阵使用相同的 `n`、`lda` 和 `uplo`。
每个矩阵原地覆盖选定三角，并返回独立 info。

## 函数原型

```cpp
aclsolverStatus_t aclsolverSpotrfBatched(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *Aarray[], int lda, int *infoArray, int batchSize);
```

## 参数说明

| 参数 | 输入/输出 | 内存位置及说明 |
|:---|:---:|:---|
| `handle` | 输入 | 由 `aclsolverCreate` 创建并关联计算流的句柄 |
| `uplo` | 输入 | 指定读取和覆盖各矩阵的上三角或下三角 |
| `n` | 输入 | 矩阵阶数，`0 <= n <= 4096` |
| `Aarray` | 输入/输出 | Device 指针数组，含 `batchSize` 个 Device `float*`；每个矩阵按列主序存放至少 `lda*n` 个元素；只修改矩阵，不修改指针槽位 |
| `lda` | 输入 | 每个矩阵的列跨度（float 元素数），`lda >= max(1,n)` |
| `infoArray` | 输出 | Device `int*`，含 `batchSize` 个状态值 |
| `batchSize` | 输入 | 非负 `int`，可执行规模受实际内存资源约束 |

## 返回值

接口返回值见[公共约定](cholesky.md#接口公共约定)。
`infoArray[b]=0` 表示该矩阵分解成功，正值表示该矩阵首个非正定主元的位置（从 1 开始），
此前完成的列保留。Device 检测到 `Aarray[b]=nullptr` 时，仅将该槽位的
`infoArray[b]` 写为 `-3`，跳过该矩阵，其余有效矩阵照常分解。
此时 Host 调用成功排队仍返回 `ACLSOLVER_STATUS_SUCCESS`；必须同步后逐项检查 info。
即使所有矩阵槽位均为空，也不改变这一逐槽位约定。

Host 参数错误返回 `ACLSOLVER_STATUS_INVALID_VALUE`；当存在可写槽位时，
`batchSize>0` 时全部 infoArray 槽位写入同一首个非法参数负编号，按
`uplo,n,Aarray,lda,infoArray,batchSize` 编号。`batchSize=0` 不写 info；
batchSize 为负时无法确定数组长度，只在 infoArray 非空时写第一个槽位。
其中 `Aarray` 本身为空也对应 `-3`，应结合 API 返回状态区分 Host 参数错误和
Device 逐矩阵错误，不能仅按 info 的数值或同码槽位数量判断。
负编号均按各接口的参数位置定义；`SpotrsBatched` 中 Aarray 是第 4 个参数，因而使用 `-4`。

## 算子约束

- 仅支持 FLOAT32；只读取和覆盖选定三角，另一三角及 padding 保持不变。
- 需要分解的槽位必须指向有效矩阵，各矩阵的有效存储须相互独立。
  Host 不读取 Device 指针数组；Device 仅对空槽位提供上述逐矩阵报错，
  不检查非空地址是否可访问或矩阵是否重叠。
- `n=0` 或 `batchSize=0` 时允许 `Aarray=nullptr`，其余参数仍校验。
  `batchSize>0` 时 infoArray 仍须提供；`n=0` 会将全部 info 置 0。
  `batchSize=0` 时允许 infoArray 为空且不写 info。
- 无公开 workspace 参数；内部缓冲区由句柄管理。4096 个矩阵的分波上限是调度策略，
  不是接口的 batchSize 上限。
- 指针数组、矩阵和 info 均须保持有效直到关联计算流完成。

## 调用示例

完整的指针数组分配及 Host 到 Device 拷贝见
[`cholesky_test.cpp`](../../test/cholesky/cholesky_test.cpp)。
下面沿用样例的 `Check(int)`；`dAarray` 必须是 Device 指针数组，不能直接传 Host `float**`。
数组内存放各个已上传输入的 Device 地址；`dInfoArray` 有 `batchSize` 个 Device `int`。

```cpp
Check(aclsolverSpotrfBatched(handle, uplo, n, dAarray, lda, dInfoArray, batchSize));
Check(aclrtSynchronizeStream(stream));
std::vector<int> info(batchSize);
Check(aclrtMemcpy(info.data(), info.size() * sizeof(int), dInfoArray,
                  info.size() * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST));
// 分别检查每个 info[b]；失败矩阵不影响其他矩阵的结果检查。
```

## 编译运行

在已完成[环境和依赖准备](../../test/cholesky/README.md#编译运行)的 solver 仓根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=cholesky --run
```

标准样例使用真实 Device 指针数组，覆盖 LOWER/UPPER 及独立矩阵分配。
