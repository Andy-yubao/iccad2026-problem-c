# E119 global2

当前主要研究基线为scratch seed0、36k、712275参数的global2 best55：两层初始全局交互＋12层几何主干/共享头，监督.2L4+.8L12，FP32/batch32。原面积、无MIB、全模块预测，固定规格由后处理满足。

训练57轮正常早停，原dev500末层Loss0.333128；原E98.depth_budget.v1后处理混合Q1.021568，最终500/500硬合法，草稿0/500。来源与身份见 `plan.json`、`frozen_model.json` 和 `protocol/research_baseline_v3.json`；原件在 `runs/global2_seed0/`，队列complete。

- 原dev500：`runs/global2_seed0/evaluation_no_mib_dev500/`。
- 历史三倍工作预算：`evaluation_budget500ms/`，不替换默认预算。
- 已授权独立test500：`evaluation_test500_20261109/recovery1_20261109/output/`，已完成且曝光，其余1500例继续封存。
- 独立新Loss变体：`variants/public_priority_v1/`，不是原基线权重。

浏览已存结果统一调用根目录 `python -B -m visualization --e119 --port 8779`，说明见 `visualization/README.md`。原评价只保存末层；2026-11-13用户请求后，仅对固定10普通例＋5最坏例补存真实12层，末层与原草稿逐位一致，后处理复用原坐标，数据及选择登记在 `visualization/20261113_dev15/`。没有训练、后处理/Q重跑或test读取。旧E118串行启动命令属于已完成历史，不能作为新启动授权；不自动恢复、重训、Q选模或追加测试。
