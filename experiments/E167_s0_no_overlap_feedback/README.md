# E167 · S0 关闭重叠反馈

在 E166 的基础上固定 E165 L0 best67，从零训练 S0 与它的题目编码器。共享双层 S0 连续调用两次，最后读出 L4。L1–L4 保留其余几何输入，屏蔽模块的重叠统计、逃逸和位移探针、形状探针的重叠分量，以及模块对的轴向穿透量。轴向间隙保留非负部分，分组、贴边、外框、形状、连接和坐标输入仍可用。

训练只使用 E166 的 L4 出口。保留 0.10 出口权重、reference 系数 1、直接重叠倍率 0、分组接触与 MST 阻挡。L8 重叠倍率保留为 1。本阶段包含 L0–L4。固定 dev500 按实际训练 loss 选最佳已完成轮次。

完整原 no-MIB train36k，固定 train500/dev500，seed 0，batch 64（尾批 32），E166 的 FP32 CUDA Graph、尾批掩码、MST/阻挡计算、AdamW 和学习率/早停策略，最多 8 小时或 200 轮。Kaggle 运行训练与选模监测。

```bash
python -m experiments.E167_s0_no_overlap_feedback.deploy_kaggle \
  --parent /path/to/l0_pretrained_best.pt --work-dir /path/to/staging
```

Kaggle 使用 NumPy 2.2.6 和 PyTorch 2.7.1+cu128。启动器优先复用匹配的预装版本；需要安装时，将二进制 wheel 安装到任务独立的临时目录。启动日志记录环境检查、数据验证和训练进度。
