## CmatinvBatched算子实现

## 概述

Solver CmatinvBatched算子实现。

## 支持的产品

- Ascend 950PR / Ascend 950DT

本测试通过 `test/utils/complex_solver_test.h` 驱动 `aclsolverStatus_t` 新接口，仅在 ascend950 构建注册；接口契约见 [CmatinvBatched](../../docs/zh/cmatinv_batched.md)。

## 目录结构介绍

```
├── cmatinv_batched
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   ├── cmatinv_batched_test.cpp       // 算子调用样例
│   └── data                // 测试数据目录
│       ├── gen_data.py     // 生成测试数据脚本
│       └── verify_result.py // 验证结果脚本
```

## 算子描述

- 算子功能：
    CmatinvBatched算子计算批量复数矩阵的逆矩阵，对应的数学表达式为：
    $$A^{-1}A = I$$
    其中$A$为$n \times n$阶非奇异复数方阵，$I$为$n$阶单位矩阵。


- 算子规格：
    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">CmatinvBatched</td>
    </tr>
    <tr>
        <td rowspan="7" align="center">算子输入</td>
        <td align="center">name</td>
        <td align="center">shape</td>
        <td align="center">data type</td>
        <td align="left">Description</td>
        <td align="center">format</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">单个矩阵A的行数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[batchSize, n, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">公式中的矩阵A，行主序</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">lda_inv</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">输出的逆矩阵同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[batchSize, 1]</td>
        <td align="center">INT32</td>
        <td align="left">每个batch矩阵的求逆结果信息</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">batchSize</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">复数矩阵求逆中的矩阵数量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">算子输出</td>
        <td align="center">Ainv</td>
        <td align="center">[batchSize, n, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输出的逆矩阵</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">cmatinv_batched_kernel</td>
    </tr>
    </table>


- 算子约束：
  - 入参 n 的有效范围为 [0, 32]，batchSize 的有效范围为 [0, 10000000]（docs/zh/cmatinv_batched.md 与 src/cmatinv_batched/cmatinv_batched_host.cpp InvalidShape 现行口径）。
  - lda >= max(1, n)、lda_inv >= max(1, n)，均支持合法 padding（不要求连续布局）。
  - A/Ainv/info 均为 Device 内存：外层指针数组与各矩阵都在 Device。
  - info 逐 batch 写入状态（非"未启用"）：0 表示成功，负值表示参数非法，正值表示该矩阵奇异。
  - n > 32 不再转发 aclsolverCgetriBatched（现行实现仅支持 n <= 32）；更大规模请使用 aclsolverCgetriBatched 先经 aclsolverCgetrf 分解后求逆。

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
  bash build.sh --soc=ascend950 --ops=cmatinv_batched --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  > **确定性**：`gen_data.py` 固定 `np.random.seed(42)`——产物逐字节可复现。

执行结果如下，说明精度对比成功。
  ```bash


[Success] Case accuracy verification passed.
  ```

> **运行前提**：直接运行测试二进制须在**仓库根目录**下执行（二进制以 `./test/<op>/data/...` 相对路径读取数据）。

> **精度判定口径**：`verify_result.py` 以 `error_tol = 1e-4` 为判据（A·A⁻¹ 与单位阵逐元素比较，错误元素占比 ≤ 0.01% 判通过（即 error_tol = 1e-4，万分之一））。

直接运行前需重新生成数据（参数与测试一致）：
```bash
python3 test/cmatinv_batched/data/gen_data.py 3 2   # 参数：n、batchSize
./build/test/cmatinv_batched/cmatinv_batched_test 0 3 2
python3 test/cmatinv_batched/data/verify_result.py 3 2
```
