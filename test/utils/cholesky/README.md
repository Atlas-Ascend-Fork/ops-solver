# 复数 Cholesky 原生测试

适用于 Ascend 950PR/Ascend 950DT，使用 arch35 实现及 `--soc=ascend950` 构建入口。

使用仓库配套 CANN，安装 NumPy 和 SciPy 后，在仓库根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=cpotrf_950,cpotrs,cpotri,cpotrf_batched,cpotrs_batched --run
```

`build.sh` 按 `test/<op>/data/gen_data.py`、`<op>_test`、`data/verify_result.py` 的顺序运行。
`cpotrf_950` 是 950 C API 的标准测试入口；`cpotrf` 保留主线 A2/A3 C++ 接口的样例。
切换接口时需同时调整调用参数、Host/Device 地址及行列主序，具体区别见 [Cpotrf 接口说明](../../../docs/zh/cpotrf.md)。
五个接口共 259 条测试，覆盖 LOWER/UPPER、非整块维度、合法 lda/ldb padding、空问题、
非连续批量设备指针、非默认 stream、info、只读因子和两次输出逐位一致。
另覆盖病态正定矩阵、非正定分解、批内成功与失败混合、奇异因子求逆的正 info；
数值失败只检验 LAPACK info 契约，成功矩阵仍逐一参加残差门禁。
批量求解新增 511/512/513、1024/1025、2048/2049 阶的上下三角回归，检查分派与工作区边界。
三个非批量接口另覆盖 511/512/513、4095/4096 阶上下三角；两个批量接口以两个独立矩阵覆盖 4095/4096 阶。
非批量求解包含 nrhs=0/8/9/16/17/128，空 RHS 使用空指针并跳过无定义的数值残差。
批量求解的 1100 阶、257 个矩阵用例验证大批次 buffered 分支与同一工作区策略。
批量求解另覆盖 864/865/1023 阶、17 个独立矩阵的上下三角缓冲路径，检查 UB 容量切换和末尾不满整块的情况。
批量分解的 64/73/95 阶用例采用 1025 个独立矩阵，检查高批次矩阵分配、短尾面板行槽及末尾不满组的情况。
批量分解另以 144/145 阶、33 个独立矩阵验证补偿对角状态在 16/17 行尾面板边界的传递。
动态 UB 预算另扣除配套 CANN 内核声明的 8 KiB 共享内存，避免合法尺寸在容量边界启动失败。
`fixtures.py` 按任务书 §3.2.2 以 LAPACK 残差作为最终精度门禁，使用实际 FP32 输入和 CPU FP32 参考，
在 complex128 算术中计算残差。分解和求解门槛为 `max(5*ratio_cpu, 3*ratio_cpu_mean)`，
批量均值仅在当前 case 内统计；求逆门槛为 `max(5*ratio_cpu, 0.1)`，单位舍入取 `2^-23`。
逐元素对比 CPU complex128 参考的结果单列诊断，不替代残差结论。
`fixture_runner.cpp` 共享公开 C API 调用与设备资源管理；各测试目录保留独立的标准入口。
manifest.tsv 前七列为 index/n/nrhs/batch/upper/lda/ldb，第八列 expectedStatus 可选，缺省为 SUCCESS。
非法 n、lda 和批量求解 nrhs=2 的用例同时检查 INVALID_VALUE、负 info、全部操作数保持和重复一致性；
期望 info 沿用各用例的 `*-info.bin`，批量分解只写首槽，其他槽保留初始哨兵。
非法尺寸使用独立合法的底层存储，不把负维度转换为分配字节数。

Destroy 的十条 Host 回归使用真实生命周期实现和模拟 Runtime，覆盖默认 stream、同步降级、
释放失败后继续回收、首错保留、未完成任务的内存保护及重试；不访问设备。加载 CANN 头文件环境后执行：

```bash
g++ -std=c++17 -fsanitize=address -fno-omit-frame-pointer -g \
    -I include -I "$ASCEND_HOME_PATH/include" \
    src/utils/aclsolver_auxiliary.cpp test/cholesky/destroy_test.cpp -o /tmp/cholesky-destroy-test
ASAN_OPTIONS=detect_leaks=1 /tmp/cholesky-destroy-test
```

CATLASS 仅作编译期头文件依赖，由 `cmake/third_party/catlass-fetch.cmake` 按仓内 FetchContent 方式管理。
构建自动下载官方固定提交 `e420328d38945dea53ac8ee56795215db20b943b` 的完整源码归档，
以整包 SHA256 `1718ad8b2a48bb7b7fbb18bbec110193ba6226a625793ad1346e57650a3df87a` 校验，
解压至 `build/third_party/catlass`。只引入 include，不配置或编译 CATLASS 根工程。
离线构建时，将同一官方归档原样保存为
`third_party/cache/catlass-e420328d38945dea53ac8ee56795215db20b943b.tar.gz`，再运行上述 build.sh；
本地归档仍执行同一 SHA256 校验，不接受重新打包的头文件快照。运行 solver 库不需要 CATLASS 源码。
