# ops-solver

## 🔥Latest News

- [2026/04] ops-solver项目上线。

## 🚀概述
ops-solver 是基于昇腾NPU芯片的高级数值求解算子库，提供包括矩阵分解、矩阵求逆、特征值计算、特征向量计算、线性方程组求解等功能。

## 📌版本配套

当前仓库已验证通过的社区版 CANN Toolkit 如下：

| CANN 版本 | 时间戳 | 验证结果 | 下载链接 |
| --- | --- | --- | --- |
| `9.0.0` | `20260422000325096` | ✅ PASS | [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260422000325096/Ascend-cann-toolkit_9.0.0_linux-aarch64.run) / [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260422000325096/Ascend-cann-toolkit_9.0.0_linux-x86_64.run) |

请根据实际 CPU 架构，从上述链接目录中自行选择对应的 `.run` 安装包。

## 🛠️环境准备

[环境部署](docs/zh/install/quick_install.md)是体验本项目能力的前提，请先完成NPU驱动、CANN包安装等，确保环境正常。

## ⬇️源码下载

环境准备好后，下载与CANN版本配套的分支源码，命令如下，\$\{tag\_version\}替换为分支标签名。

> 说明：若环境中已存在配套分支源码，**可跳过本步骤**，例如CANNLab默认已提供最新商发版CANN对应的源码。

```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-solver.git
```

说明：对于CANNLab云开发环境，已默认提供最新商发CANN版本配套的源码，如需获取其他版本源码，参考上述命令获取。

## Atlas 950PR单精度实数LU与求逆

新增950PR实数Device接口及测试接入，使用上表配套CANN环境：

| 接口 | 功能 |
| --- | --- |
| [Sgetrf](docs/zh/sgetrf.md) | 单矩阵LU分解及workspace查询 |
| [Sgetri](docs/zh/sgetri.md) | 从LU和主元求逆及workspace查询 |
| [SgetriBatched](docs/zh/sgetri_batched.md) | 从批量LU和主元求逆 |
| [SmatinvBatched](docs/zh/smatinv_batched.md) | n≤32的原始小矩阵批量求逆 |

矩阵使用Device列主序，主元、INFO和workspace在Device；批量接口接收Device上的Device指针数组。
单矩阵workspace查询的Lwork在Host，以FLOAT32元素计。参数约束、调用示例和各产品验证范围见接口文档。

在仓库根目录构建库、adapter并运行四接口基础测试：

```bash
bash build.sh --soc=ascend950 --ops=sgetrf,sgetri,sgetri_batched,smatinv_batched --run
```

构建需要Python3；`scripts/generate_real_lu_fixed_vf.py`在构建目录生成950小矩阵向量代码。
各接口测试README说明adapter和基础测试入口；官方任务包全量测试工具、参考数据和原始结果随自验证材料交付。

## 📖学习教程

- [快速入门](docs/QUICKSTART.md)：从零开始快速体验项目核心基础能力，涵盖源码编译、算子调用、开发与调试等操作。

## 💬相关信息

- [目录结构](docs/zh/install/dir_structure.md)
- [接口列表](docs/api_list.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-solver/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-solver/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-solver/wiki)获取技术文章，如系列化教程、优秀实践等。
