# 复数 LU 与求逆回归

使用 Ascend950 和与编译器配套的 CANN 环境。每个测试进程只暴露一张卡，进程内使用逻辑设备 0。

## 精度与多面板专项

```bash
source <cann>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<device>
cmake -S . -B <build> -DSOC_VERSION=ascend950 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON -DTEST_NAMES=complex_solver
cmake --build <build> --target complex_solver_test -j8
<build>/test/complex_solver/complex_solver_test --panel-regression
python3 test/complex_solver/residual_test.py --library <build>/libops_solver.so \
  --op cgetrf --panel-regression --output <results>/panels.jsonl
```

Python 测试依赖 NumPy 和 SciPy。两套专项均覆盖：

| 矩阵 | 路径 |
|---|---|
| 1000×100 | 连续分布式面板、行交换回放、上三角求解、Cube 与尾随更新 |
| 800×64 | 第二面板剩余 768 行，转入普通面板 |
| 1000×40 | 尾部剩余 8 列，转入宽度收缩的普通面板 |

C++ 检查 LU 残差、padding 和重复执行确定性；主元输入强制跨行交换。
Python 对照 SciPy LAPACK 残差，使用两种 padding 和主元/无主元；主元用例使用非对角占优输入。
上述形状也纳入默认 C++ 全套和 Python `--boundary-regression`。

## 内存检查

使用独立构建目录，并对 Device 代码启用插桩。该库使用 ASC 编译源文件，因此通过 `CMAKE_ASC_FLAGS` 传递选项。

```bash
cmake -S . -B <memcheck-build> -DSOC_VERSION=ascend950 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON -DTEST_NAMES=complex_solver \
  -DCMAKE_ASC_FLAGS="--cce-enable-sanitizer"
cmake --build <memcheck-build> --target complex_solver_memcheck -j8
```

底层运行命令（只运行工具，不执行构建目标的报告判定）：

```bash
mssanitizer --tool=memcheck -- <memcheck-build>/test/complex_solver/complex_solver_test --memory-regression
```

专项包含上述面板、单矩阵求逆、批量指针表、反向指针顺序、padding、2×2 批量求逆及非整批尾部。
构建目标会检查工具返回值、错误与告警、专项完成标记，以及分布式面板和 tiny inverse 的采集记录。
检查完整报告中的各路径覆盖和跳过项。
仅退出码为零或残差通过，不代表内存检查通过。工具不支持的指令或采集异常应标为未覆盖。

## 性能验收口径

`residual_test.py --iterations N` 的 `stream_interval_ms` 只用于诊断；它包含 stream 区间，且没有任务书阈值门禁。
其 `passed` 只表示精度、metadata、padding 和确定性检查通过。

任务书性能使用无插桩 Release 库及 `msprof op --application "<单用例程序及参数>"`。
按任务书预热和有效采样设置，读取 `OpBasicInfo*.csv` 的 `Task Duration(us)`；一次接口调用的全部 kernel 耗时相加。
逐 case 与相应任务书基线/倍率换算的上限比较，并保留用例、提交号、环境、采样数与 kernel 覆盖。
缺少 kernel、采集失败或只有 stream event 时间时，性能状态记为未验证，不记通过。
边界回归形状没有对应任务书基线，不对其套用其他形状的阈值。
