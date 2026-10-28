# 单精度复数 Cholesky 接口

本文说明 Ascend950 的 C API；A2/A3 C++ 分解接口的参数及调用示例见 [Cpotrf](cpotrf.md)。

Ascend 950PR/Ascend 950DT 共用 dav-3510 / arch35 实现，构建入口均为 `--soc=ascend950`。

该实现使用公开头文件 `include/cann_ops_solver.h` 的 C ABI。
`aclsolverFloatComplex` 由两个 FP32 分量 real、imag 组成，大小为 8 字节，自然对齐为 4 字节，字段偏移为 0 和 4。
该布局与仓内 C++ 接口的 `std::complex<float>` 大小及对齐一致，公开头文件在编译期检查。
设备缓冲区可复用此存储布局；host 对象应复制分量或按值转换，不能通过无关类型解引用。
矩阵为列主序；`ACLSOLVER_FILL_MODE_LOWER=0`、`UPPER=1`。

| 接口 | 输入与输出 |
| --- | --- |
| `aclsolverCpotrf` | 读取 A 的所选三角，原地生成 L 或 U；对角虚部为零 |
| `aclsolverCpotrs` | 读取 L 或 U，保持 A 不变，原地更新 B；支持 0≤nrhs≤128 |
| `aclsolverCpotri` | 读取 L 或 U，原地生成原矩阵逆的所选三角 |
| `aclsolverCpotrfBatched` | Aarray 为设备上的指针数组，info 为逐矩阵设备数组 |
| `aclsolverCpotrsBatched` | Aarray、Barray 为设备指针数组，非空批量仅支持 nrhs=1，info 为设备标量 |

支持 0≤n≤4096、0≤batchSize≤1000000，以及非连续批量矩阵。
各维度上限独立；输入、输出和内部工作区的总量还必须满足实际可用 HBM。
合法维度组合不代表设备必有足够容量，内部工作区分配失败返回 `ACLSOLVER_STATUS_ALLOC_FAILED`。
lda/ldb 必须至少为 max(1,n)，填充区保持不变。未选择的三角不参与计算。
计算在 handle 绑定的调用方 stream 上异步执行；`nullptr` 表示 CANN 默认 stream。
同一 handle 不支持跨 Host 线程并发使用。查询、SetStream、计算和 Destroy 必须由调用方串行化；
Destroy 不能与其他线程对该 handle 的任何访问重叠，也不能继续使用已销毁的 handle。
不同 handle 可在各自合法的 CANN context 下并发使用。
同一 handle 的创建、计算和销毁须在其所属 CANN Context 中进行，不支持跨 Context 复用。

分解和求逆前分别调用 `aclsolverCpotrf_bufferSize`、`aclsolverCpotri_bufferSize`，
返回的 lwork 以复数元素计数，调用方分配 lwork×sizeof(aclsolverFloatComplex) 字节。
查询不启动设备任务。非批量求解、批量分解和批量求解的内部工作区按 handle、stream 复用；扩容保留旧缓冲区，
由 `aclsolverDestroy` 等待相关任务并释放。调用方应先销毁 handle，再销毁其 stream。
Destroy 记录首个错误并继续回收其他已确认不再使用的块；stream 同步失败时尝试 Device 同步。
不能确认已完成或释放失败的块仍由 handle 持有，不进行未同步释放。
有效 handle 在失败返回后仍由调用方持有，仅供在 Runtime 错误解除后重试 Destroy，不能继续提交计算；
已成功回收的块不会再次释放。即使块已全部回收，失败返回后也须重试以释放 Host handle。

须先检查 host 返回状态：仅当返回 SUCCESS 且 stream 成功完成后，数值 info 才有效。
分配、初始化或执行失败时 info 未定义，即使其值为 0 也不能据此判成功。
在上述成功条件下，info=0 表示成功；分解 info>0 为最早非正定主子式的一基索引，此前列已经完成；
求逆 info>0 为第一个零实部对角元素的一基索引。
非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`，并在 handle、stream 有效且 info 可写时异步记录负参数位置，
位置不计 handle。批量分解的参数错误写入 info[0]。
空规模按接口约束成功返回；非空批量求解的 nrhs!=1 返回 info=-3。

## 批量分解的容量约束

阶数与批量上限分别约束参数，不表示二者的笛卡尔积都能在设备上执行。
非空批量分解的矩阵存储至少为 `8 * lda * n * batchSize` 字节。
当 `n<=32` 或 `n<=64 && batchSize>=16` 时不需要内部工作区；其他路径令
`p=ceil(n/16)*16`，本次工作区需求为 `8 * 130 * p * batchSize` 字节。
还须计入设备指针表、info、已有 handle 缓存、其他调用方缓冲和运行时占用；
剩余 HBM 和连续可分配空间均可能限制实际可执行规模，不能据此承诺固定的最大 batch。

例如 `n=lda=4096、batchSize=1000000` 的参数分别合法，但仅矩阵就需约 134.22 TB，
工作区另需约 4.26 TB，无法同时装入 128 GiB 的 950PR。应按实际可用容量分批提交。
内部 `aclrtMalloc` 失败返回 `ACLSOLVER_STATUS_ALLOC_FAILED` 并记录 ERROR；
此时 `info` 未定义，即使已有异步初始化将其写为 0，也不表示分解成功。

构建、实际设备测试入口见 [test/utils/cholesky/README.md](../../test/utils/cholesky/README.md)。
