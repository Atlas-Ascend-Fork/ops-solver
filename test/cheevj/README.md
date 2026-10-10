# Cheevj validation

本目录包含 `aclsolverCheevj` 的调用样例、API 契约测试和性能测试程序。
测试应在已配置 CANN 9.0.0 的 Ascend 机器上执行。

## Build

```bash
source /path/to/cann/set_env.sh
bash build.sh --ops=cheevj --soc=ascend910b
```

`build.sh` 沿用仓库统一的 `Release` 构建类型。需要调试代码生成时，应在独立构建
目录显式传入 `-DCMAKE_BUILD_TYPE=Debug`，不要修改仓库顶层构建配置。

## Implementation layout

`src/cheevj/` 保持一个算子的 host/kernel 表面：

- `cheevj_host.cpp`：公共 API、参数检查、workspace 和后端选择；
- `cheevj_kernel.cpp`：通用路径和 `512/1024/2048` 固定 shape 的全部设备入口与调度。

固定 shape 共用 prepare、panel、HER2K、finalize 和 eigensolver 设备入口。
Householder、特征向量组装和三对角向量实现分别按算法职责收进一个私有头文件，
`512/1024/2048` 是文件内分支，不再作为独立文件。各阶段不能合成一次物理 launch，
因为 AIV/AIC 任务类型、block 数和全局同步边界不同。

## API contract gate

```bash
LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH} \
  ./build/test/cheevj/cheevj_contract_test 0
```

契约门禁覆盖非法参数、`n=0`、`n=2049` 无人工上限、strided `lda`、U/L 三角隔离、N/V
一致性、N 模式输入不变，以及 V 模式残差和正交性。

## Correctness test

仓库统一构建入口会生成输入、运行调用样例并校验输出：

```bash
bash build.sh --ops=cheevj --soc=ascend910b --run
```

也可通过标准数据脚本指定规模、计算模式、三角区域、随机种子和矩阵类型：

```bash
python3 test/cheevj/data/gen_data.py 128 V L 42 random_hermitian
LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH} ./build/test/cheevj/cheevj_test 0 128 V L
python3 test/cheevj/data/verify_result.py 128 V L random_hermitian
```

## Symbol and loader checks

```bash
nm -D -C build/libops_solver.so | grep 'aclsolverCheevj('
ldd -r build/libops_solver.so
```

`nm` 必须找到公开符号，`ldd -r` 不得报告缺失依赖或未定义符号。

## Performance gate

性能测试应使用已经通过正确性测试的同一构建产物。以下命令采用同进程 public API、
预热 2 次、记录 7 次并输出中位数：

```bash
python3 test/cheevj/data/gen_data.py 2048 N U 42 random_hermitian
LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH} \
  ./build/test/cheevj/cheevj_benchmark 0 2048 N U 2 7
```

## Small-scale Host convergence regression

`cheevj_small_scale_host_test` calls the public API with a null handle and does not initialize an ACL device. It covers
small and scaled `3x3` Hermitian tridiagonal spectra in eigenvalue-only and vector modes, plus zero-matrix controls.
Build and run it with:

```bash
cmake --build build --target cheevj_small_scale_host_test -j2
LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH} ./build/test/cheevj/cheevj_small_scale_host_test
```
