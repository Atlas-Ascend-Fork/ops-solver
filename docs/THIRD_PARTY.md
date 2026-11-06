# 第三方组件许可声明（Third-Party Notices）

本文件是 ops-solver 仓库的第三方组件许可清单，声明随源码分发、且被编入
`libops_solver.so` 发布产物的非 CANN 许可成分及其义务边界。新增第三方成分时
必须同步更新本清单。

## 组件清单

| 组件 | 仓库内路径 | 许可证 | 许可证全文 / 声明文件 | 版权人与商用联系 |
|------|-----------|--------|----------------------|------------------|
| PACO GEQRF（I/O 高效并行 QR 分解参考实现，经适配进入本仓） | `src/cgeqrf/`（`cgeqrf_host.cpp`、`cgeqrf_kernel.cpp`、`cgeqrf_launchers.hpp` 及 `kernel/*.hpp`）；`test/cgeqrf/`（`cgeqrf_test.cpp`、`CMakeLists.txt`） | GPL-3.0-or-later / 商业双许可（默认条款 GPL-3.0-or-later） | 仓库根目录 `COPYING-PACO`（GPL 全文）、`LICENSE-PACO`（双许可声明） | Shina Guo <fs20091003@163.com>、Yuan Tang <yuantang@fudan.edu.cn> |

仓库其余源码均为 CANN Open Software License Agreement Version 2.0（见根目录
`LICENSE`）。

## 合规提示（发布产物）

根 `CMakeLists.txt` 将 `src/` 下全部源码（含上表 PACO 成分）编入同一
`libops_solver.so` 共享库。据此：

- 对外分发该库（run 包 / 安装包）时，发布产物同时包含 CANN OSL 2.0 与
  GPL-3.0-or-later 两类许可成分的编译产物，分发方需自行评估并满足 GPL-3.0
  的整体性义务（含源码提供义务）；两类许可的分发条款难以在同一二进制上
  同时满足时，应避免直接混合分发；
- 闭源 / 商用分发场景应先向版权人取得 PACO 商业许可；
- 若上述义务无法满足，可选的处置方向为：将 PACO 成分拆分为单独构建、单独
  分发的动态组件；或以本仓自有实现（complex_qr 公共路径）替换。处置方案
  的最终选择需由仓库维护者与法务共同决策（跟踪见对应 issue）。

## 维护约定

- 触碰上表所列文件时保留其原始 PACO 双许可文件头，不得替换为 CANN OSL 头；
- 新增第三方源码时：在组件清单中登记路径、许可证与声明文件，并随提交附
  许可证全文（仓库根目录或 docs/ 下）。
