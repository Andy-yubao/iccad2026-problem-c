# 资料索引、来源与验证记录

获取日期：2026-10-15（北京时间）。本轮仅归档非数据集内容。以下链接均指向工作区中的文件；[比赛官方页面](https://www.iccad-contest.org/Problems)提供原始链接。

## 官方资料

| 本地资料 | 官网标注/原文件日期 | 用途 |
|---|---|---|
| [题目 PDF](official/C_20260522.pdf) | 官网最后更新 06/08，文件名 20260522，正文 v10 | 题目定义与数学公式 |
| [最新 Q&A](official/C_QA_20260827.pdf) | 官网最后更新 08/28，文件名 20260827 | 29 组问答与最新澄清 |
| [提交指南](official/C_Submission_Guidelines_20260616.pdf) | 06/16 | 封装与环境要求 |
| [op_wrapper.py](../third_party/official/op_wrapper.py) | 官网最后更新 06/18 | 官方独立发布的 wrapper |
| [Alpha 前五成绩](official/results/C_Alpha_Top5_20260720.csv) | 官网最后更新 07/21 | 历史成绩，不是本机结果 |
| [Alpha 逐例中位耗时](official/results/C_Median_Runtime_Alpha.csv) | 07/21 | 公开验证集上历史运行时间 |
| [Beta 评测报告](official/results/C_beta_evaluation_report_hidden_final.txt) | 08/19 | 隐藏集评测说明及封装问题 |
| [Beta 排行榜](official/results/C_beta_leaderboard_20260823.csv) | 08/23 | 历史隐藏集分数 |
| [Beta 逐例中位耗时](official/results/C_median_runtimes_beta_hidden.csv) | 08/23 | 隐藏集历史运行时间 |
| [FloorSet 原始论文](papers/FloorSet_2405.05480.pdf) | arXiv:2405.05480，本轮获取版本 | 数据集背景与构造方法 |
| [Problems 页面快照](official/Problems.html) | 2026-10-15 获取 | 下载入口存档 |
| [FAQ 页面快照](official/FAQ.html) / [News 页面快照](official/News.html) | 2026-10-15 获取 | 通用规则与发布沿革 |

网页快照只保存 HTML，不是完整离线网站，部分图片/样式仍引用网络。论文来源为 [arXiv PDF](https://arxiv.org/pdf/2405.05480)；这不是题目中所有相关文献均已下载的声明。46 条原始书目见 [problem_references.md](official/problem_references.md)。

当前下载的论文仅保留 FloorSet 官方原始论文，来源和哈希纳入下述资源清单；原题书目是官方原文的一部分，不表示保留了其中论文的本地副本。

上述 13 个下载文件的来源 URL、大小和 SHA-256 均在 [resources_manifest.json](resources_manifest.json)。当前官方题目 PDF 与上游代码内 v10 副本内容一致。旧核验记录中的根目录 PDF 路径属于历史位置；当前全文入口使用 references/official/。

## 官方代码快照

来源：[IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet)。固定提交：`aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`。归档位置：[third_party/FloorSet](../third_party/FloorSet/README.md)。共 **144 个文件，20,486,614 bytes**，保留代码、notebooks、文档、许可证、示意图、测试布局图和基准指标表。

Git 的 Windows schannel 传输失败，改用 HTTPS API 枚举固定提交的文件树，再从该提交的 raw URL 获取文件。每个文件都核对 Git blob SHA-1，另记录 SHA-256。此目录**不包含 `.git`，不是 clone**；以后更新应先比较版本，不能直接运行 `git pull`。

明确排除：

- `LiteTensorDataTest/`：上游 Git 中携带的验证张量数据（200 文件）。
- `data/`：上游携带的 Prime 验证数据压缩包。
- `.ipynb_checkpoints/`：notebook 自动备份。

这些排除项记录在 [upstream_manifest.json](../third_party/upstream_manifest.json)，144 个归档文件都有固定提交的来源 URL、Git blob SHA-1、大小与 SHA-256。保留 `inteltest_layouts/` 等布局插图用于阅读，不包含求解数据张量。

比赛专用入口：[比赛 README](../third_party/FloorSet/iccad2026contest/README.md)、[评测器](../third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py)、[优化器模板](../third_party/FloorSet/iccad2026contest/optimizer_template.py)、[训练示例](../third_party/FloorSet/iccad2026contest/training_example.py)、[依赖清单](../third_party/FloorSet/iccad2026contest/requirements.txt)。

## 文档衍生品

- [题目 Markdown](official/problem_statement.md)：保留正文、公式、表格与图；[46 条参考文献](official/problem_references.md)单列。
- [Q&A Markdown](official/official_qa.md)：完整 29 组问答，恢复换行，不删减长问题。
- [提交指南 Markdown](official/submission_guidelines.md)：保留原文，附已知 CUDA 更正指引。
- [中文导读](../docs/research_guide.zh-CN.md)：编辑说明、实际接口及冲突表，与官方原文分开。

题目公式页 5、7、9 已渲染并对照核验；图 1 从原 PDF 提取后检查了完整性。Q&A PDF 存在部分损坏的非正文对象指针，`pypdf` 给出 warning；全部 14 页均可提取文字，已核对 Q1–Q29 / A1–A29 的顺序和数量，未改写原 PDF。

## 复查与重取

```powershell
# 只校验本地归档、脚本语法和工作区 Markdown 文件链接，不联网、不读数据
python .\scripts\verify_workspace.py

# 按已有清单固定的提交补齐官方代码（不会下载数据）
python .\scripts\fetch_upstream.py

# 获取其他官方资料；已有文件核对格式并重记哈希，不自动覆盖
python .\scripts\fetch_resources.py

# 再生成 Q&A/提交指南/书目/图 1；需 pypdf、pdfplumber
python .\scripts\convert_official_docs.py
```

`problem_statement.md` 的公式与正文经过人工排版，不由脚本覆盖。若官方同名 Drive 文件更新，先保存旧版本再人工更新资源，不能把已有缓存的哈希当成远端仍未变更的证据。

资料验证记录：[verification.json](verification.json)。验证范围是资料完整性、静态语法与链接；该脚本不会重新计算大数据压缩包的哈希。

后续训练集导入记录：[training_installation.json](../datasets/training_installation.json)。浏览器下载的完整压缩包已复制进工作区，源文件和副本均通过官方大小与 SHA-256 检查，已实际解压并检查 worker 与分片数量。

公开验证集记录：[validation_installation.json](../datasets/validation_installation.json)。已核对官方 SHA-256、100 个 config 目录和 100 对输入/标签，全部 200 个解压文件与压缩包内容一致。修复了下载脚本将 config 目录误按文件检查的问题，同一命令已重跑成功。

本页的获取/安装描述是当时记录，当前训练进度以实验索引为准。历史单例与环境安装证据已压缩至 [bootstrap_20261015.zip](../history/bootstrap_20261015.zip)。独立环境中的 [100 例结果](../experiments/baseline/runs/full100_clean_cu128_seed0/run/results.json)和最新 [5 例核查](../experiments/baseline/diagnostics.md)仍保留原始文件。仅为划分查看了 5 个开发分片的结构，未做训练；项目执行规则见 [protocol.md](../docs/protocol.md)。

## 2026-10-16 位置变更

原件内容与来源哈希保留，官方资料和论文已从 docs 移到 references；上游清单在 third_party，数据安装记录在 datasets。历史 verification/安装记录未改写。路径映射见 [迁移清单](../history/20261016_workspace_migration/manifest.json)。已废弃方向的旧源码快照按用户要求清除，原始资料与安装证据保留。
