# Cholesky 算子调用样例与测试

## 概述

本目录提供 Cholesky 接口组的原生 C API 样例、通用数值测试和 API 契约回归。
与仓内其他测试一致，使用 `CMakeLists.txt`、`cholesky_test`、数据生成脚本和结果验证脚本。
实现设计见 [Cholesky](../../docs/zh/cholesky.md)。

## 支持的产品

- Ascend 950PR / Ascend 950DT，共用 `arch35` 实现，使用 `--soc=ascend950`。

## 接口与算子描述

矩阵为 FLOAT32、列主序，支持 LOWER/UPPER；设备矩阵、info 和批量指针数组均在 Device。

| 计算接口 | 功能 | 查询与接口说明 |
|---|---|---|
| `aclsolverSpotrf` | 对称正定矩阵原地分解 | [Spotrf](../../docs/zh/spotrf.md)，含 bufferSize |
| `aclsolverSpotrs` | 使用因子求解，原地覆盖 RHS | [Spotrs](../../docs/zh/spotrs.md) |
| `aclsolverSpotri` | 使用因子求逆，覆盖同侧三角 | [Spotri](../../docs/zh/spotri.md)，含 bufferSize |
| `aclsolverSpotrfBatched` | Device 指针数组逐矩阵分解 | [SpotrfBatched](../../docs/zh/spotrf_batched.md) |
| `aclsolverSpotrsBatched` | Device 指针数组单 RHS 求解 | [SpotrsBatched](../../docs/zh/spotrs_batched.md) |

共五个计算接口、两个 workspace 查询接口，全部由同一个样例覆盖。
具体尺寸、空问题、info 和 workspace 约束见对应接口说明。

## 目录结构介绍

```text
cholesky/
├── CMakeLists.txt
├── README.md
├── cholesky_test.cpp        # C API 调用样例及 LOWER/UPPER 检查
├── runtime_faults.cpp      # 仅测试使用的运行时故障注入库，不安装
├── test_runtime_failures.py # 分配/事件失败保留缓存及启动错误传播回归
├── unsupported_test.cpp    # 非 950 库的七接口链接、返回值和输出保持检查
├── native_benchmark.cpp    # 原生基准调用入口
├── runner.py               # 可配置维数、batch、RHS 和 padding 的通用测试
├── destroy_test.cpp         # Host 销毁与混合缓冲区所有权回归
├── contracts.py            # 参数、info、存储隔离和缓冲区生命周期回归
├── lapack_residual.py      # FP64 残差运算及配对 FP32 LAPACK 参考
├── test_lapack_residual.py # 残差计算与批量覆盖回归
└── data/
    ├── gen_data.py
    └── verify_result.py
```

## 编译运行

### 环境和依赖

先按仓内[环境部署](../../docs/zh/install/quick_install.md)配置驱动和 CANN；
本接口在官方 CANN 9.0.0、Ascend 950PR 上验证。标准数据脚本需要 NumPy、SciPy，
CTest 的通用数值和契约测试还需要与 CANN 匹配的 PyTorch/TorchNPU。
缺少这些依赖时对应测试会失败，不会静默跳过；仅编译库和原生样例无需导入 TorchNPU。
选择实际安装路径加载 CANN 环境，所有后续命令均在 **solver 仓根目录**执行：

```bash
source /path/to/cann/set_env.sh
```

CATLASS 仅作编译期头文件依赖，由 `cmake/third_party/catlass-fetch.cmake`
按仓内 FetchContent 方式统一管理，使用 `OPS_SOLVER::catlass` target。
固定提交为 `e420328d38945dea53ac8ee56795215db20b943b`，使用官方完整源码归档，
整包 SHA256 为 `1718ad8b2a48bb7b7fbb18bbec110193ba6226a625793ad1346e57650a3df87a`。
构建优先读取 `third_party/cache/` 的同名归档，否则下载同一官方归档；两条路径使用相同校验。
解压至 `build/third_party/catlass`，只引入 `include/`，不配置或编译 CATLASS 根工程。
运行 solver 库不需要 CATLASS 源码。

离线环境无需 clone 第三方仓；将已有官方归档原样复制到缓存并校验：

```bash
mkdir -p third_party/cache
cp /path/to/offline/catlass-e420328d38945dea53ac8ee56795215db20b943b.tar.gz third_party/cache/
python3 - <<'CHECK_CATLASS'
import hashlib
from pathlib import Path
archive = Path("third_party/cache/catlass-e420328d38945dea53ac8ee56795215db20b943b.tar.gz")
expected = "1718ad8b2a48bb7b7fbb18bbec110193ba6226a625793ad1346e57650a3df87a"
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise SystemExit("CATLASS 官方归档 SHA256 不匹配")
print("CATLASS 官方归档校验通过")
CHECK_CATLASS
```

缓存必须为同一官方归档，不能用重新打包的头文件快照替代。
整包校验绑定固定提交及全部内容；CMake 导入时再次校验。
`build.sh` 会清理 `build/`，但保留仓根缓存，因此可重复离线构建。
在线构建的官方归档地址为：
<https://raw.gitcode.com/cann/catlass/archive/refs/heads/e420328d38945dea53ac8ee56795215db20b943b.tar.gz>。

### 编译

```bash
bash build.sh --soc=ascend950 --ops=cholesky
```

`--ops=cholesky` 对应本目录，在 CMake 中设置 `BUILD_TEST=ON`、`TEST_NAMES=cholesky`。
它选择测试工程，库源码仍按仓内 CMake 的 SOC 规则收集。
不是以 `aclsolverSpotrf` 等 C API 名作为 `--ops` 值；七个公开符号由一次构建共同提供。
`build.sh` 每次清理并重建 `build/`；直接调用 CMake 的方式见仓内 [QuickStart](../../docs/QUICKSTART.md)。

构建与安装产物：

| 路径 | 用途 |
|---|---|
| `build/libops_solver.so` | 公共算子库，导出全部五个计算接口及两个查询接口 |
| `build/libcholesky_simt.so` | 纯 SIMT 设备链接库，主库的运行时依赖 |
| `build/test/cholesky/cholesky_test` | 五接口原生调用样例，包含两个查询接口 |
| `build_out/lib64/` | 上述两个库的安装目录 |
| `build_out/include/` | `cann_ops_solver.h` 和 `cann_ops_solver_common.h` |

使用统一头文件 `cann_ops_solver.h`，链接 `ops_solver` 和 CANN `ascendcl`。
部署时将两个库放在同一目录，并将该目录和 CANN lib64 加入加载路径；
安装后示例为 `export LD_LIBRARY_PATH="$PWD/build_out/lib64:${LD_LIBRARY_PATH}"`。
运行库不需要 CATLASS 源码。

### 标准样例

```bash
bash build.sh --soc=ascend950 --ops=cholesky --run
ctest --test-dir build --output-on-failure
```

`build.sh` 沿用 solver 原有入口。`--run` 依次生成数据、运行二进制、验证结果，
成功后清理生成的数据。默认样例覆盖五个计算接口、LOWER/UPPER、lda padding、
真实 Device 指针数组，以及求解保持因子不变；十份数值输出均执行 LAPACK 残差检查。

五接口的覆盖与输出如下，批量样例检查两个独立物理槽位：

| 接口 | 标准输出（`lower-` / `upper-` 前缀） |
|---|---|
| Spotrf | `factor.bin` |
| Spotrs | `solve.bin` |
| Spotri | `inverse.bin` |
| SpotrfBatched | `batch-factor-0.bin`、`batch-factor-1.bin` |
| SpotrsBatched | `batch-solve-0.bin`、`batch-solve-1.bin` |

成功输出包括：

```text
[Success] Case LAPACK residual verification passed.
[PASS] cholesky_test
```

已编译后，单独重跑三段步骤，无需再次构建：

```bash
python3 test/cholesky/data/gen_data.py
LD_LIBRARY_PATH="build:${LD_LIBRARY_PATH}" build/test/cholesky/cholesky_test
python3 test/cholesky/data/verify_result.py
```

### CTest 与加载检查

`cholesky_test --smoke` 使用内建小矩阵，不依赖生成的数据。
950 构建的 CTest 自动运行以下七组测试，无需另行调用扩展脚本：

| CTest 名称 | 内容 | 是否需要 NPU |
|---|---|:---:|
| `cholesky_c_api_smoke` | 全部五个计算接口、两个查询接口及 LOWER/UPPER；八项实数/复数共用句柄、同流/双流与未同步销毁回归 | 是 |
| `cholesky_destroy_ownership` | 二十项 Host 销毁回归：同步降级、首错、失败重试、LU 上下文恢复、实数池回调与复数缓冲区共存 | 否 |
| `cholesky_lapack_residual` | 六项残差计算回归 | 否 |
| `cholesky_numerical_boundaries` | 1、8、9、17、32、33、64、65、128、129、141、201、257、509、513、1025、2048 阶；五接口、LOWER/UPPER、batch=7、nrhs=17、padding=3，共 170 条 | 是 |
| `cholesky_numerical_large` | 4096 阶；五接口、LOWER/UPPER、batch=1、nrhs=128、padding=8，共 10 条 | 是 |
| `cholesky_runtime_failures` | 真实 NPU 调用配合四项分配/事件/启动状态故障注入；测试库仅通过独立进程 LD_PRELOAD 启用 | 是 |
| `cholesky_contracts` | 非法参数、非连续指针数组、双流缓冲复用和待完成计算时销毁等完整契约回归 | 是 |

五组 NPU 测试共享 CTest 资源锁，使用 `ctest -j` 时不会相互争用同一设备。
每组超时为 1200 秒，Python 测试失败会返回非零退出码；数值和契约结果写入
`build/test/cholesky/` 下的 `numerical-boundaries.json`、`numerical-large.json`、`contracts.json`、`runtime-failures.json`。
这些通用回归不替代任务附件的完整精度、性能及内存验收。

可单独执行某一组：

```bash
ctest --test-dir build -R '^cholesky_numerical_large$' --output-on-failure
ctest --test-dir build -R '^cholesky_contracts$' --output-on-failure
```

```bash
nm -D --defined-only build/libops_solver.so | grep -E 'aclsolverSpot(rf|rs|ri)'
ldd -r build/libops_solver.so
```

应找到七个公开符号，`ldd -r` 不得报告缺失依赖或未定义符号。

### 非 950 构建的接口检查

其他受支持 SOC 的库保留相同七个公开符号，但非空句柄调用返回
`ACLSOLVER_STATUS_ARCH_MISMATCH`，空句柄优先返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`，
均不访问矩阵或改写输出。此分支不需要 CATLASS、SIMT 库或 NPU 初始化。
例如使用仓内支持的 910B 配置构建并检查链接、返回值与输出保持：

```bash
bash build.sh --soc=ascend910b --ops=cholesky
ctest --test-dir build -R '^cholesky_arch_mismatch$' --output-on-failure
```

该配置下 `cholesky_test` 和 `cholesky_arch_mismatch` 仅检查不支持架构的接口行为；
950 的数值测试只在 `--soc=ascend950` 构建中注册。

## 通用数值与契约回归

以下扩展测试需要与 CANN 匹配的 PyTorch/TorchNPU；CPU 只生成输入和参考值，
全部算子计算经公开 C API 在 NPU 上执行。参数可自定义，不读取任务附件或固定 case ID。
在仓根目录执行：

```bash
mkdir -p /tmp/cholesky-tests
python3 test/cholesky/runner.py --library build/libops_solver.so \
  --sizes 1,8,9,17,32,33,64,65,128,129,141,201,257,509,513,1025,2048 \
  --batch 7 --nrhs 17 --output /tmp/cholesky-tests/smoke.json
python3 test/cholesky/contracts.py --library build/libops_solver.so \
  --output /tmp/cholesky-tests/contracts.json
python3 test/cholesky/runner.py --library build/libops_solver.so \
  --sizes 4096 --batch 1 --nrhs 128 --padding 8 \
  --output /tmp/cholesky-tests/large.json
```

契约回归覆盖非法参数优先级、空输入、非正定 info 及已完成列、求逆零对角、
忽略三角、padding、非连续指针数组、确定性、双流缓冲复用及有待完成计算时销毁句柄。

## 验收材料

仓内保留以上可复用测试源码。任务附件、GPU 性能基线、专属性能/内存采集工具、
完整报告和日志在独立验收 ZIP 中交付；它们不参与本目录的标准编译和测试。
完整验收的操作步骤以 ZIP 内 `1 自验证步骤说明.md` 为准。

通用数值回归与大尺寸回归均调用仓内 `lapack_residual.paired/finalize`，
以实际 FP32 输入和配对 FP32 LAPACK 参考构造 FP64 残差，按 3.2.2 判定。
逐元素 max_abs/ratio 只输出为诊断，不作为通过依据。非批量 CPU mean 在当前完整回归组内按算子汇总，
批量 CPU mean 按当前用例全部物理矩阵汇总；完整任务附件仍通过独立验收工具运行。
契约回归包含 1/8 阶 SIMT、大小矩阵混合空槽位的参数优先级、非有限/负对角、污染 workspace。
另以 129、1025、4097 个矩阵覆盖自适应指针检查的线程数及跨步尾部。
批量分解空槽位覆盖 1/8/17/33/65/129/257 阶、LOWER/UPPER、全空数组内容及 4097 个矩阵的分波边界，
另覆盖无空槽位的有限数值哨兵，逐槽检查 info、其他矩阵的残差与逐位一致性、空槽位原存储、忽略三角、padding 和指针数组保持。
另以 Host 空 Aarray 回归区分 INVALID_VALUE/全槽 -3 与成功排队后的逐槽 -3；契约回归共 291 条。
同一 handle 的 Host API 由调用者串行使用；不同流的异步缓冲生命周期仍自动验证。
