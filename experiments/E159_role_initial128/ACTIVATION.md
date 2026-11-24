# E159 正式启动授权

2026-11-21 用户明确要求启动训练。采用保留MIB的E158同口径对照，三层、四头、每头32维、width128，reference1.0、完整原重叠组0.25。仅L0，36k/batch64，最多4小时/200轮，结束冻结own-best并从已存监测记录生成E158/E159 Loss比较。

本次没有执行防御性测试、预检或试训。正式任务使用隐藏python.exe进程，标准输出和错误日志写入background。准备时冻结的README/DESIGN/plan保持原件；实时状态见authorization.json与queue_state.json。
