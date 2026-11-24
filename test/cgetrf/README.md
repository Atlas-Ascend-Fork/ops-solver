## Cgetrf算子实现

## 概述

Solver Cgetrf算子实现。

## 支持的产品

- Ascend 950PR / Ascend 950DT

本测试通过 `test/utils/complex_solver_test.h` 驱动 `aclsolverStatus_t` 新接口（含 `aclsolverCgetrf_bufferSize` workspace 查询），仅在 ascend950 构建注册；接口契约见 [Cgetrf](../../docs/zh/cgetrf.md)。

## 目录结构介绍

```
├── cgetrf
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   ├── cgetrf_test.cpp     // 算子调用样例
│   └── data                // 测试数据目录
│       ├── gen_data.py     // 生成测试数据脚本
│       └── verify_result.py // 验证结果脚本
```

## 算子描述

- 算子功能：
    Cgetrf算子对复数矩阵进行LU分解，对应的数学表达式为：
    $$A = P \cdot L \cdot U$$
    其中$A$为$m \times n$阶复数矩阵，$P$为置换矩阵，$L$为单位下三角矩阵，$U$为上三角矩阵。

- 算子规格：
    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">Cgetrf</td>
    </tr>
    <tr>
        <td rowspan="5" align="center">算子输入</td>
        <td align="center">name</td>
        <td align="center">shape</td>
        <td align="center">data type</td>
        <td align="left">Description</td>
        <td align="center">format</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">矩阵A的行数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">矩阵A的列数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[m, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输入矩阵A，行主序</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为N）</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">算子输出</td>
        <td align="center">A</td>
        <td align="center">[m, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输出的L和U矩阵，L和U合并存储在A中</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">ipiv</td>
        <td align="center">[min(m,n)]</td>
        <td align="center">INT32</td>
        <td align="left">置换矩阵的pivot信息</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">分解结果信息</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">cgetrf_kernel</td>
    </tr>
    </table>


- 算子约束：
  - m 的有效范围为 [0, 4096]，n 的有效范围为 [0, 8192]（与 docs/zh/cgetrf.md 及 src/cgetrf/cgetrf_host.cpp InvalidShape 现行口径一致）；空问题 m=0 或 n=0 成功返回并写 devInfo=0。
  - lda >= max(1, m)，支持合法 padding（leading-dimension 间隙保留，不要求连续布局）。
  - A、Workspace、devIpiv、devInfo 均为 Device 内存；devIpiv 可为 nullptr（不选主元），非空时返回 min(m, n) 个 1-based 行交换索引。
  - devInfo 为 0 表示成功，-i 表示除 handle 外第 i 个参数非法，正值表示首个精确零主元位置（1-based）。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量
  请根据当前环境上CANN开发套件包的安装方式，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行
  ```bash
  bash build.sh --soc=ascend950 --ops=cgetrf --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  > **精度判定口径**：`verify_result.py` 以 `error_tol = 1e-4` 为判据（LU 重构后与参考逐元素比较，相对误差 `ratio = |a−b|/|a| > eps（eps = 1e-3）` 的元素计入超差，占比 ≤ 0.01% 判通过（即 error_tol = 1e-4，万分之一）；非有限值直接拒绝）。

> **路径解析不对称说明**：`gen_data.py` 按脚本所在目录产出数据（`__file__` 相对），`verify_result.py` 按 cwd 相对读取——`build.sh --run` 从仓根执行两端均可正确定位；单步运行 `verify_result.py` 时须以仓库根目录为 cwd（与测试二进制同前提）。

执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy verification passed.
  ```

> **运行前提**：直接运行测试二进制须在**仓库根目录**下执行（二进制以 `./test/<op>/data/...` 相对路径读取数据）。

直接运行前需重新生成数据（参数与测试一致）：

```bash
python3 test/cgetrf/data/gen_data.py 32 32
./build/test/cgetrf/cgetrf_test 0 32 32
python3 test/cgetrf/data/verify_result.py 32 32
```
