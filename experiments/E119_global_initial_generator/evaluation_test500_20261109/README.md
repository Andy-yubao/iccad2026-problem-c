# E119 best55：一次授权的独立test500评价

2026-11-09用户明确要求修复并重启：原进程以0xC000013A被外部终止，原始现场归`recovery1_20261109/original_evidence/`。单次恢复使用pythonw无控制台监督进程及CREATE_NO_WINDOW独立进程组工作进程，逐例加载/记录进度；模型、原预算、冻结名单与评分数学不变。当前进度改看`recovery1_20261109/output/state.json`及`recovery1_20261109/supervisor_state.json`，日志在该恢复目录；仍不自动重试。原失败记录保持历史身份。

用户2026-11-09明确要求后台运行另外500例测试，使用原预算后处理。仅开启random_reference40k test2000中seed20261109随机无放回冻结的500例；其余1500例保持封存。名单在任何测试载荷读取前冻结，不按N/标签/Loss/Q选题。

复用原NPZ，不生成数据或构建训练缓存。E119 best55、原E98.depth_budget.v1、原面积/无MIB本地Q/原硬检查；不训练、不重选模型、不新增图。错误/停止不自动重试。

实时进度output/state.json；日志stdout.log、stderr.log；完成结果output/results.json及output/report.md。本次测试500例运行后记为已曝光，不用于选择权重或调预算。
