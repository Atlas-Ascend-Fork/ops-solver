# 复数 QR 公共基础组件

此目录只保留四个算子实际复用的基础设施，不放置 QR 分解、显式 Q 生成、Q 应用或最小二乘算法。

| 组件 | 职责 | 使用者 |
| --- | --- | --- |
| complex_qr.h | 内部参数类型、对齐及设备核心数声明 | 四个算子 |
| complex_qr_host.h / arch35/complex_qr_host.cpp | Handle、stream、workspace 边界、错误回传；由算子提供策略回调 | 四个算子 |
| arch35/complex_qr_device_helpers.h | 复数算术、SIMT 归约 | Cgeqrf、Cunmqr、CCgels |
| arch35/complex_qr_transfer.h | DMA 读写和事件同步 | 四个算子 |
| arch35/complex_qr_finite.h | 通用矩阵有限值扫描，不决定算子参数与错误编号 | 四个算子的 validate kernel |
| arch35/complex_qr_platform.cpp | 硬件核心数查询 | Cgeqrf、Cungqr、Cunmqr、CCgels |
| arch35/complex_qr_info_kernel.cpp / complex_qr_finalize_info_kernel.cpp | 通用 Device 状态写回与归约结果转换 | 四个算子 |

算子参数检查、workspace 规划、Device 输入校验、执行入口均在各自 `src/<算子>/arch35/`。
Cgeqrf 拥有分解面板和反射器生成；Cungqr 拥有显式 Q 初始化；Cunmqr 拥有反射器应用、块更新及其补偿矩阵乘/打包；CCgels 拥有最小二乘编排与三角回代。
其他算子需要上述算法时，显式包含算法所属算子的内部头文件，不通过 utils 中转声明。
