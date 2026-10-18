# ICCAD 2026 Problem C

FloorSet Challenge 赛后研究工作区。先读 [执行协议](docs/protocol.md) 与 [结构管理规范](docs/workspace_structure.md)；核心边界见 [AGENTS.md](AGENTS.md)。

## 当前起点

当前[E9：面向 E8 后处理的模型改进](experiments/E9_model_postprocess_alignment/README.md)尚未达到平均Q＜1.20。中心修正训练10轮早停，dev100由1.211556降至1.210118，但同一冻结候选在另一开发100例由1.232217退化至1.239278，均100/100硬合法。不提升该候选或追加同方向训练，继续保留旧模型＋E8；权重和全部证据保留，测试仍封存。[完整冻结复核](experiments/E9_model_postprocess_alignment/runs/20261018_center_refiner_confirm100_seed0/analysis.md)。

E9采用的后处理参照为[E8：快速、稳定的高质量后处理与模型联调](experiments/E8_fast_robust_postprocess/README.md)。锥优化＋可行种子救援已在两组共200个开发布局全部硬合法，均值与尾部均优于各自原参照，推理＋后处理加速8.66/10.14倍；额外组计入公开文件交换后加速9.62倍。[最终完整对照](experiments/E8_fast_robust_postprocess/runs/20261018_robust_confirmation100_seed0/research.md) · [调用入口](experiments/E8_fast_robust_postprocess/optimizer.py)。E7继续冻结，无新增训练或封存测试读取；更高绝对质量目标尚未全部达到，开发结果不当作最终泛化证明。

[E7：降低修复难度的草稿](experiments/repair_friendly_draft/README.md)已按用户要求终止。新占用损失改善了同10例草稿几何，但最终Q基本持平、没有平均耗时改善证据；矩形架构单独使用更差。停止采用本轮新增损失，保留原geometry目标与E6参照；未完成组合取消续训，dev100/test50不补跑。代码、权重与历史记录保留，50例目标未验证。[阶段结论](experiments/repair_friendly_draft/conclusion.md)

本阶段结束不自动恢复下文提到的任何旧训练或后处理研究；历史设计与授权仅保留背景和复核价值。

结题后用户另行授权既有attention_geometry的100例测试及10例图集：[执行入口](experiments/repair_friendly_draft/runs/20261017_attention_geometry_test100_seed0/README.md)。已准备并检查，测试尚未运行，不恢复训练。

保留一套无训练硬约束基线：[experiments/baseline](experiments/baseline/README.md)。它补齐 fixed 尺寸锁定、preplaced 位置与障碍处理，并提供构造与可行搜索。官方 SA 原件保留在 third_party，已有失败输出和问题审计归入基线目录供核对。

冻结 dev5 的四组既有输出均硬合法，search 的中性时间质量分为 1.9090；只支持五例开发观察，不是泛化结论或正式比赛成绩。[实验索引](experiments/README.md) · [基线诊断](experiments/baseline/diagnostics.md)。

官方 Beta `raw_score` 按无时间质量分作为历史参照；优秀质量目标、数值核查与隐藏集不可直接比对的边界见 [质量目标](docs/aim.md)。

[短期计划](experiments/short_term_plan.md) 已完成首轮短探针与 [E1 短预算学习比较](experiments/graph_proposal/runs/20261016_short_budget_comparison_seed0/analysis.md)：确定性提议相对同后处理启发式在 dev5 上有有限收益，flow 暂未获益；保留确定性模型参照，不启动 E4 新增训练。E2 几何兜底、E3 候选覆盖、E5 对全局控制的收益均未满足继续投入条件。未证明泛化、收敛或超过现有基线。

旧方向的模型、训练入口、Loss、checkpoint、结果和设计文档已按用户要求移除，含旧源码归档中的相关内容。后续实验从明确的新假设开始，不自动恢复已废弃路线。

## 环境与数据

独立环境 `.conda/envs/iccad2026`，Python 3.10.22、PyTorch 2.7.1+cu128；[环境定义](environment.yml)保持可重建。本地数据、环境与缓存不上传 GitHub。

固定 [split_v1.json](protocol/split_v1.json) 的 80/10/10：7,200 train、900 dev、900 local_test 分片；数据位置由 [paths.json](protocol/paths.json) 解析。本地测试默认封存，本次授权评价已读取部分测试；公开验证仅用于兼容与公开对照。见 [数据使用](docs/datasets.md)。

后续训练统一采用 [v2 工作集](protocol/workset_v2.json)：20,000 train / 2,500 dev / 2,500 默认封存测试，覆盖 100 个 worker；固定 500 例验证监测，要求学习率下降、早停与 best/latest 保存，dev5 仅作排错。确定性 v2 重训与本次测试均已按用户要求停止，进入 [拟合与布局分析](experiments/graph_proposal/runs/20261016_frozen_best_test2500_seed0/analysis.md)：完整 train/dev MSE 为 0.07487/0.07508，无明显过拟合迹象；部分测试 **957 例、39/100 worker** 全部硬合法，中性时间加权 Q=3.85918。测试覆盖不均衡，不代表完整 2,500 例结果，不与旧 dev5 直接比较；不续训、不补跑。

最新[原始提议独立评价](experiments/graph_proposal/runs/20261016_raw_proposal_train100_dev100_seed0/analysis.md)已完成train100/dev100、各100worker，GPU推理，无后处理。模型已学到粗略全局关系，但局部冲突和形状误差仍明显；固定形状下不满足微小平移可修目标，允许形变的情形尚待检验。结合[软约束诊断](experiments/graph_proposal/runs/20261017_soft_constraint_gallery_seed0/analysis.md)，当前[短期计划](experiments/short_term_plan.md)转入新实验 [E6：可修复提议与意图感知修复](experiments/intent_aware_repair/README.md)：优先检验模型架构/表示与训练目标，再从改善后草稿的轻微重叠等几何线索设计简单修复，最后交叉归因；禁止额外关系头。五组模型训练/完整评价已完成，累计训练任务76.6分钟。当前进入模型与后处理交替迭代，冻结新架构＋几何Loss，本轮全局后处理已使固定dev100加权中性时间Q由3.636240降至1.262626，100/100硬合法、97/100单例低于1.5；5%位移/形变门槛已撤销。平均核心后处理10.854秒/例（旧0.437秒），不是同耗时或正式比赛成绩。[完整结果](experiments/intent_aware_repair/runs/20261017_reweighted_shape_dev100_seed0/analysis.md) · [前后图](experiments/intent_aware_repair/runs/20261017_reweighted_shape_dev100_seed0/index.html)。禁止关系头，旧权重与封存测试保持现状。

```powershell
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -B scripts/make_splits.py
python -B scripts/verify_workspace.py
```

## 文件与版本管理

- `experiments/baseline/`：当前基线的实现、说明、运行证据和诊断。
- `datasets/`、`scoring/`：仅复用数据处理与官方输出评分；其他实验算法独立维护。
- [visualization/](visualization/README.md)：统一布局绘图，复用既有E6图集的实际生成代码。
- `docs/`：核心协议、结构规范、数据说明、[赛题导读](docs/research_guide.zh-CN.md)与 [Git 使用约定](docs/git.md)。
- `references/` 与 `third_party/`：[官方资料与论文](references/README.md)、未修改的上游代码。
- `history/`：必要的基线旧证据和整理记录，不保留已明确删除方向的源码或权重。

本地 Git 主分支 `main`，GitHub 公开仓库 [Andy-yubao/iccad2026-problem-c](https://github.com/Andy-yubao/iccad2026-problem-c)，远程名 `origin`。提交代码、文档、冻结索引和必要小型证据；数据集、环境、缓存及 checkpoint 不进入 Git。

[题目全文](references/official/problem_statement.md) · [官方 Q&A](references/official/official_qa.md) · [提交指南](references/official/submission_guidelines.md) · [官方比赛 README](third_party/FloorSet/iccad2026contest/README.md)。

官方代码 Apache-2.0，数据集 CC BY 4.0；资料保留原署名与引用要求。[比赛官网](https://www.iccad-contest.org/Problems) / [官方代码](https://github.com/IntelLabs/FloorSet) / [官方数据](https://huggingface.co/datasets/IntelLabs/FloorSet)。
