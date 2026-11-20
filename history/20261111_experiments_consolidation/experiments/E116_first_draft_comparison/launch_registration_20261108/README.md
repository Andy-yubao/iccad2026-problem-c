# E116 后台训练启动登记

2026-11-08 用户明确指令：“好的。后台启动训练”。仅覆盖E116旧的“准备后不启动”边界，启动已准备的 original4→new_first 两组seed0串行训练。

冻结16k、模型、完整单稿Loss、FP32/batch32、增强、优化器/门禁、各4小时/200轮预算、best/latest及完整状态保留规则全部按原plan.json。错误或用户停止均不推进下一组，无自动重试，不恢复任何旧实验或修改旧STOP。训练完成后仅整理已保存日志中的Loss，不新增推理、后处理、Q或测试读取。

准备阶段的模型、Loss、数据、训练器、plan、provenance和实验根README均原样冻结，五例preflight直接只读复用；该根README中的训练禁用叙述属于准备记录，当前授权以authorization.json及本启动登记为准。新增控制器放在本子目录，独立记录哈希，不修改通过检查的核心源码。原authorization、STOP和准备状态副本保存在本目录。

本次只启动一次队列；queue_state.json已存在即拒绝重复派发。实验根STOP或QUEUE_STOP可阻止后续阶段并转发当前训练的安全停止。launch_logs保存每组标准输出/错误；runs/<arm>_seed0保留完整训练状态。

启动模块：`experiments.E116_first_draft_comparison.launch_registration_20261108.run_queue`。控制器保持在持续后台进程中；不创建定时自动唤醒。
