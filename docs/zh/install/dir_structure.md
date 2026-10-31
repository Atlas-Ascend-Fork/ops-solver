# 项目目录

> 本章罗列的部分目录或文件为可选，请以实际交付件为准。另请注意：
>
> - **算子目录与源码前缀**：`src/` 下子目录名表示一类求解能力，目录内 `*_host.cpp`、`*_kernel.cpp` 等文件名前缀可能与目录名不同，以目录内实际文件为准。
> - **架构子目录**：若某算子目录下存在 `${arch_dir}`（如 `arch35`、`arch20`），表示面向特定 NPU 架构的差异化实现，由顶层 CMake 按当前 `SOC_VERSION` 选择是否编入；若缺少该目录，通常表示当前实现未拆分架构专用源码。
> - **测试工程**：`test/` 下用例需在配置工程时开启 `BUILD_TEST` 并设置 `TEST_NAMES`（分号分隔的算子测试子目录名）后才会参与构建，详见根目录 [CMakeLists.txt](../../../CMakeLists.txt) 与 [QuickStart](../../../docs/QUICKSTART.md)。
> - **构建输出目录**：`build/`、`build_out/` 等常为本地 CMake 生成目录，未必随源码一并提供，以本地构建配置为准。
> - 若需补充算子或文档，欢迎参考 [贡献指南](../../../CONTRIBUTING.md)。

项目全量目录层级介绍如下：

```text
├── cmake                                               # 工程辅助 CMake 脚本（版本、打包、第三方获取等）
│   ├── package.cmake                                   # CPack/安装包相关逻辑（ENABLE_PACKAGE 时使用）
│   ├── version.cmake
│   ├── makeself.cmake
│   └── third_party                                     # 第三方工具脚本（如 makeself-fetch）
├── src                                                 # 求解算子源码目录
│   ├── CMakeLists.txt                                  # 源文件收集、架构子目录参与规则
│   ├── utils                                           # Host/Kernel 侧公共实现与内核工具头文件等
│   └── ${op_name}                                      # 单个求解相关能力目录，${op_name} 为小写目录名
│       ├── ${op}_host.cpp                              # Host 侧：参数检查、任务下发、与 ACL 交互等
│       ├── ${op}_kernel.cpp                            # Device 侧：Ascend C 核函数入口（若存在）
│       ├── kernel                                      # 可选，核函数或子模块头文件子目录
│       │   └── ...
│       └── ${arch_dir}                                 # 可选，面向特定架构的源码，由 SOC 配置决定是否编译
├── include                                             # 对外头文件
│   ├── cann_ops_solver.h                               # aclSolver API 声明
│   └── cann_ops_solver_common.h                        # 公共类型与声明
├── docs                                                # 项目文档
│   ├── QUICKSTART.md                                   # 快速入门
│   ├── api_list.md                                     # 接口/算子列表（概要）
│   └── zh
│       ├── install
│       │   ├── quick_install.md                        # 环境部署
│       │   └── dir_structure.md                        # 本文档
│       └── ${api_doc}.md                               # 各接口中文说明（随算子扩展）
├── scripts                                             # 辅助脚本（打包、版本信息、检查等）
│   ├── package
│   └── util
├── test                                                # 算子级测试（BUILD_TEST=ON 且配置 TEST_NAMES 时参与构建）
│   ├── CMakeLists.txt                                  # 按 TEST_NAMES 批量 add_subdirectory
│   └── ${op_name}
│       ├── CMakeLists.txt
│       ├── ${test_entry}.cpp                           # 可执行测试源文件
│       ├── data                                        # 可选，生成/校验数据脚本
│       └── README.md                                   # 可选，本算子测试说明
├── CMakeLists.txt                                      # 工程入口：ops_solver 库、安装规则、可选测试与打包
├── CONTRIBUTING.md                                     # 贡献指南
├── LICENSE                                             # 开源许可证
├── README.md                                           # 项目总览
├── SECURITY.md                                         # 安全声明
├── build.sh                                            # 编译脚本（可选）
└── ...
```

当前 `src/` 下已提供的算子示例子目录包括：`cgetrf`、`cgetri`、`cgetri_batched`、`cheevj`、`cmatinv_batched`、`sgetrf`、`sgetri`（随版本迭代可能增减，以仓库实际目录为准）。
