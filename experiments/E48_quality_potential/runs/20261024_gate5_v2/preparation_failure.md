# 可选环境观察依赖缺失

修正后的构造检查已经通过，随后环境记录因本机独立conda中没有threadpoolctl而停止；尚未读取/求解真实五例。该包只用于显示数值库线程信息，不参与求解。修正为可选观察项，保留明确的环境、Torch、Clarabel和HiGHS线程设置；不为记录信息安装额外依赖。原config/source_snapshot保留，正式五例转入v3。
