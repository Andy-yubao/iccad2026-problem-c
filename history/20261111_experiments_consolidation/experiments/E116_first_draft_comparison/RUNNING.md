# E116：后台训练与查看入口

当前已完成，两组均正常早停、无错误；original4 best dev500 Loss0.468609，new_first 0.470894。没有继续训练或新增评价。[完成复核](result_review.md) · [原始自动汇总](comparison_report.md)。下方保留启动时的运行说明，实时队列当前为complete。

用户于2026-11-08明确授权后台训练。当前派发仅包含seed0两组，顺序为 **original4 → new_first**；第二组在第一组正常完成后开始。每组最多4小时或200轮，保留原早停和数值保护；错误或用户停止时不推进、不自动重试。

两组使用冻结16k、相同单稿完整Loss，只监督最终首稿，全部后续网络移除。原结构前4层207987参数；新版4个Attention＋SwiGLU推理块、一次全局条件读出209891参数。原精度、batch32、增强、AdamW、LR和更新门禁不变。不同结构的计算量与实际更新步数、训练时间分别报告。

- [队列状态](queue_state.json)：当前组、进程和已完成阶段。
- [original4训练心跳](runs/original4_seed0/heartbeat.json) · [original4进度](runs/original4_seed0/progress.json)。
- [original4日志](launch_logs/original4.stdout.log) · [original4错误日志](launch_logs/original4.stderr.log)。
- new_first启动后使用同名`runs/new_first_seed0/`和`launch_logs/new_first.*.log`入口；未启动时这些文件不存在。
- 两组完成后，只读取训练日志生成`loss_comparison.json`与`comparison_report.md`，不增加模型推理、后处理、Q或测试评价。

best/latest分别保存，latest保留完整Adam/RNG/数据排列与游标/调度/步长保护/累计预算。实验根STOP或QUEUE_STOP会阻止后续组并请求当前训练安全保存退出。

[启动授权与源码登记](launch_registration_20261108/README.md)保留了准备阶段authorization、STOP、preflight及状态原件。准备通过的模型、Loss、数据、训练器、plan和根README原样冻结；根README中“训练关闭”是准备阶段记录，现以授权文件和队列状态为准。E114主研究基线、E101交付及其他实验旧STOP均未改变。
