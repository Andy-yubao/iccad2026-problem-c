# ICCAD 2026 Problem C

FloorSet Challenge 赛后研究工作区。先读 [执行协议](docs/protocol.md) 与 [结构管理规范](docs/workspace_structure.md)；核心边界见 [AGENTS.md](AGENTS.md)。

## 当前起点

当前只保留一套无训练硬约束基线：[experiments/baseline](experiments/baseline/README.md)。它补齐 fixed 尺寸锁定、preplaced 位置与障碍处理，并提供构造与可行搜索。官方 SA 原件保留在 third_party，已有失败输出和问题审计归入基线目录供核对。

冻结 dev5 的四组既有输出均硬合法，search 的中性时间质量分为 1.9090；只支持五例开发观察，不是泛化结论或正式比赛成绩。[实验索引](experiments/README.md) · [基线诊断](experiments/baseline/diagnostics.md)。

旧方向的模型、训练入口、Loss、checkpoint、结果和设计文档已按用户要求移除，含旧源码归档中的相关内容。后续实验从明确的新假设开始，不自动恢复已废弃路线。

## 环境与数据

独立环境 `.conda/envs/iccad2026`，Python 3.10.22、PyTorch 2.7.1+cu128；[环境定义](environment.yml)保持可重建。本地数据、环境与缓存不上传 GitHub。

固定 [split_v1.json](protocol/split_v1.json) 的 80/10/10：7,200 train、900 dev、900 local_test 分片；数据位置由 [paths.json](protocol/paths.json) 解析。本地测试封存，公开验证仅用于兼容与公开对照。见 [数据使用](docs/datasets.md)。

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
- `docs/`：核心协议、结构规范、数据说明、[赛题导读](docs/research_guide.zh-CN.md)与[通用文献索引](docs/literature_review.zh-CN.md)。
- `references/` 与 `third_party/`：[官方资料与论文](references/README.md)、未修改的上游代码。
- `history/`：必要的基线旧证据和整理记录，不保留已明确删除方向的源码或权重。

本地 Git 主分支 `main`，GitHub 公开仓库 [Andy-yubao/iccad2026-problem-c](https://github.com/Andy-yubao/iccad2026-problem-c)，远程名 `origin`。提交代码、文档、冻结索引和必要小型证据；数据集、环境、缓存及 checkpoint 不进入 Git。

[题目全文](references/official/problem_statement.md) · [官方 Q&A](references/official/official_qa.md) · [提交指南](references/official/submission_guidelines.md) · [官方比赛 README](third_party/FloorSet/iccad2026contest/README.md)。

官方代码 Apache-2.0，数据集 CC BY 4.0；资料保留原署名与引用要求。[比赛官网](https://www.iccad-contest.org/Problems) / [官方代码](https://github.com/IntelLabs/FloorSet) / [官方数据](https://huggingface.co/datasets/IntelLabs/FloorSet)。
