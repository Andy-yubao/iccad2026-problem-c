# 2026-11-01研究工作区整理

已保留E42/E50/E74/E81/E85/E90/E95在日常区，其余21个实验同盘归档；manifest.json逐文件记录原路径、新路径、大小、SHA256，移动前后全量校验通过，约8.53GiB/5764文件。归档不释放对应磁盘空间，不删除唯一权重、数据或有效缓存。

旧文档原件在documents_before/，旧实验在experiments/。历史JSON/源码/日志不批量改写路径；当前导航使用迁移清单。恢复旧入口时先按manifest恢复引用或建立明确的读取适配，不把旧暂停/预算/授权当成当前指令。

E95首轮已停训，保留原始run、best/latest、inputs及源码快照；文档与后续修复引入的身份变化另行登记，不直接绕过旧checkpoint校验。Git初始改动清单在git_status_before.txt，未执行reset或提交。

旧目标/旧报告见documents_before/docs/aim.md及report_zhu_20261018.md；研究导读转入references/research_guide.zh-CN.md。数据、conda环境、原始冻结索引和已有历史归档未改动。
