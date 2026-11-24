# E164 两层宽 64 静态关系消息 L0

单个从零训练的 L0 候选。2026-11-23 按用户指令迁移到本地 GPU，沿用原 36k 和固定 500/500 面板。配置见 [DESIGN.md](DESIGN.md) 和 [plan.json](plan.json)。

从项目根目录运行本地训练入口。正常结束后，该入口会冻结 best 并写 E149 Loss 对照。

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E164_static_relation_l0.run_local
```

本地进度见 [heartbeat.json](runs/e164_l0_seed0/heartbeat.json)，结果见 [loss_results.json](runs/e164_l0_seed0/loss_results.json)。

## Kaggle 提交记录

私有 GPU 任务已提交为 Version 1。用 `andyyubao` 登录后，可以在 Kaggle 左侧底部的 **View Active Events** 查找 `ICCAD E164 Static Relation L0 36k`，也可以在 Notebook 编辑器右上角打开 **Version History** 查看 Version 1。[Kaggle Notebook](https://www.kaggle.com/code/andyyubao/iccad-e164-static-relation-l0-36k) · [最近一次 API 状态](kaggle/latest_remote_status.json)

从项目根目录在本机 PowerShell 中查询，提示时输入 Kaggle API token。

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\E164_static_relation_l0\kaggle\deploy.ps1 -Status
```

任务结束后用 `-Download` 下载输出。参照 `e164-results/loss_results.json`、`frozen_best.json`、`runs/e164_l0_seed0/checkpoints/`、`train.log` 和 `bootstrap-status.json`。
