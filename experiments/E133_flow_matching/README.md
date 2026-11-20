# E133 Flow Matching

单组scratch seed0、920643参数：400→192→96公开编码＋96×6纯Attention，时间/状态条件与AdaLN，共享速度读出。固定16k、原train500/dev500、FP32/batch32；原始面积、无MIB、F/P全预测。训练纯速度MSE，独立高斯到reference直线路径。具体公式/门禁/增强与总8h预算见 `plan.json`、`pipeline_budget.json` 及run配置。

## 已完成结果

训练CUDA捕获在零正式更新边界失败，登记改为eager FP32；原模型/目标/Adam/门禁及预算起点保留。正式训练247轮/123500步正常早停，唯一FM Loss-best239，dev Loss0.415453，权重SHA256：

`1510d786df358ad3e4390596b948dd2c64289945b7f4639ef1ed9acef80d352f`

原自动评价因重复dict字段在0例失败，`queue_state.json` 保留needs_attention现场。用户后续明确请求后，以独立适配器和独立1800秒只读预算完成原dev500：Euler8、每例唯一固定噪声、原E98.depth_budget.v1；最终500/500硬合法、0回退，草稿0/500合法，混合Q1.157321，完整均时206.4ms/P95499.3ms。弱于E119历史基线，但16k对36k/单seed/目标不同不支持FM普遍无效的因果结论。4/16步仅五训练例诊断，未做Q选步数。

## 证据位置

- 训练：`runs/fm_seed0/`；正式日志：`launch_logs/20261112_seed0_prepared/`。
- 独立成功评价与完整报告：`evaluation_quality_20261113/`，逐例数据在其 `dev500/cases/`。
- 启动登记：`launch_registration_20261112/`；初次捕获失败原件：`preparation_diagnostics/startup_capture_failure1/`。

上述两处记录和独立评价于2026-11-13从根output归回本实验，字节/哈希核对见 `history/20261113_workspace_cleanup/manifest.json`。冻结源码里的旧output路径保留历史原样，不能直接用于新运行；本次未重写原监督器/失败队列。新运行需另行登记入口与授权，不自动重试、重训、追加候选或读取test。
