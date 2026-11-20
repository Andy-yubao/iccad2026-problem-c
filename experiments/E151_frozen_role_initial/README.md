# E151 固定 E149 初稿并重新训练后段

2026-11-18 用户授权准备完成后后台训练。L0 使用 E149 best52 固定权重，后段从头适配。整网 681141 参数，训练题目编码器与 S0/S1/S2 共 573402 参数。

沿用原 36k/batch64/4 小时训练预算和 E148 的 L4/L8/L12 监督。L4 用去重叠 Loss，L8/L12 用完整 Loss，层权重为 0.1/0.1/0.8。终末用原 dev500 逐题检查 S1 对重叠、分组、贴边及 F/P 误差的影响，并与已存 E148/E150 同题结果对照。

启动入口为 `python -u -B -m experiments.E151_frozen_role_initial.run_queue --start`。实时状态见 `queue_state.json`、`runs/e151_frozen_role_initial_seed0/heartbeat.json`。结果写入 `loss_results.json` 和 `runs/e151_frozen_role_initial_seed0/dev500_stage_audit/`。具体冻结配置见 [DESIGN.md](DESIGN.md)。
