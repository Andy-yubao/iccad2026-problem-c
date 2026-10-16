# 官方模板来源与版本复核

复核日期：2026-10-15。本次只核对来源、内容和提交历史，**未运行新的求解实验**。

## 结论

**在当前公开发布入口范围内，没有下载错仓库、拿错模板或混用新旧评测器。** 工作区中的模板、评测器和验证加载器与官方当前 main 文件内容一致。官方公布的模板确实没有实现全部当前硬约束。

更准确的表述是：官方公开的是需要参赛者替换算法的实现模板，并不承诺其中示例 SA 满足完整比赛约束。我此前直接称其为“官方基线”，没有强调它的约束实现范围，容易使人误以为它是已验证合法的参考求解器。

## 官网入口与文件内容

实时读取[比赛官网 Problems](https://www.iccad-contest.org/Problems)，Problem C 的 FloorSet 链接仍指向：

`https://github.com/IntelLabs/FloorSet/tree/main/iccad2026contest`

这是我们下载和运行的目录，不是第三方 fork，也不是 FloorSet 根目录中的旧通用评估程序。官方顶层 README 的 Quick Start 同样使用 `optimizer_template.py`。

| 对象 | 本地与当前官方是否一致 | 依据 |
|---|---|---|
| 仓库 main HEAD | 一致 | `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa` |
| `iccad2026contest/optimizer_template.py` | 逐字节一致 | SHA-256 `34f42cf5e38d459d44ae42cdab42efa4cd285f5aa1a81d6b1811a26db6e9131c` |
| `iccad2026contest/iccad2026_evaluate.py` | 逐字节一致 | SHA-256 `64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498` |
| `lite_dataset_test.py` | 逐字节一致 | SHA-256 `dbe1af1fdd315852594b3e3e5624f2156470ce4c448cc83ba98b0330dc6a6368` |

内容复核结果见[原始哈希记录](runs/template_version_check/content_hashes.json)。官网另附的 `op_wrapper.py` 是调用参赛者可执行文件的适配器，不包含另一个 floorplanning 求解算法。

## 是否遗漏了其他分支或发布版

GitHub API 当前返回 5 个公开分支：main 与 `udaymallappa-patch-1` 至 `patch-4`。后四个分支的末次提交都在 2024 年，均不存在 `iccad2026contest/optimizer_template.py`。当前公开 tags、releases 均为空。

因此没有找到一个应当替换 main 的新版 2026 模板，也没有发现独立的公开 release。该结论覆盖已核查的公开入口，不代表主办方内部不存在其他实现。

## 提交历史揭示了什么

以下日期均为北京时间，具体 UTC 时间与提交 SHA 另存 JSON。

| 提交日期 | 提交 | 与问题有关的实际变化 |
|---|---|---|
| 2026-04-16 | [`c1d2cd7`](https://github.com/IntelLabs/FloorSet/commit/c1d2cd77b8a0e38c0671221cca4ce00be46bb9d5) | 模板新增 `target_positions` 参数，初始化时读取固定宽高；没有增加预放置坐标锁定或固定块旋转保护 |
| 2026-04-20 | [`a01137f`](https://github.com/IntelLabs/FloorSet/commit/a01137f8cb0406fcb1eef4a76e09445d95aab863) | 更新题目 v9、README 和评测器，将 fixed/preplaced 明确归为硬约束；这次没有修改模板 |
| 2026-05-23 | [`fc28d6d`](https://github.com/IntelLabs/FloorSet/commit/fc28d6d4a7e173c572329e1c9326160ad0ce90ac) | 模板的最后一次更新只修改顶部 docstring，补充尺寸不可变的硬约束说明；算法逻辑未改变。评测器也在同一提交更新 |

这不是“我们拿到了 4 月旧模板，却配了 5 月新评测器”。我们拿到的是同一当前 main 快照，两个文件都已包含各自最新修改；问题是官方最新模板的算法逻辑仍不完整。仓库 README 的部分条目日期与 Git 提交落库日期不同，不能仅凭条目日期推断版本错配。

提交记录和分支检查见 [history_findings.json](runs/template_version_check/history_findings.json)。

## 数据版本是否错配

当前 Hugging Face HEAD 仍为 `7dc7c1bec07bcae583a274f5ed95e5be399cb8db`。训练 v2 和验证压缩包的当前大小、LFS 内容哈希与本地清单一致，没有发现上游已更换压缩包的情况。

更直接的交叉检查：对 21、40、60、90、120 模块这 5 例的 10 个输入/标签文件计算 Git blob SHA-1，**全部等于官方 GitHub main 中直接提交的验证数据文件哈希**。因此，这 5 例并非仅在 Hugging Face 上存在的另一版数据；它们与代码同一官方仓库所带验证数据完全一致。

完整记录见 [origin_and_data.json](runs/template_version_check/origin_and_data.json)。没有重新下载数据集。

## “baseline”名称的区别

- `optimizer_template.py`：官方称作 B*-tree SA baseline，但顶部同时注明不处理 fixed/preplaced 等约束，且要求替换算法实现。
- `--baseline`：评测器中的命令，功能是从数据标签导出参考指标，不是运行另一个有效求解器。
- 评测器内还保留 `RandomOptimizer`、`SimulatedAnnealingOptimizer` 等简易示例；其签名没有新增 `target_positions` 参数，也未实现固定/预放置规格保护。未发现它们是官网另行推荐的完整替代基线，本次没有运行它们。

所以，当前证据支持：**来源与版本正确，但公开模板中的示例算法不满足当前完整规则**。既有[5 例对照核查](diagnostics.md)为这个实现缺口提供了运行证据。
