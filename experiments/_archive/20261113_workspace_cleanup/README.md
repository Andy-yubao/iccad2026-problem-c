# 2026-11-13 工作区精简

按用户要求缩短AGENTS与长期文档，布局展示统一为E6分组色/黑框/底部图例的一套入口，支持真实已存中间层滑块。

- `documents_before/`：修改前文档原件，含AGENTS、README及docs；不作为现行授权。
- `visualization_legacy/`：旧可视化全部版本及已有图片/数据，同盘rename保留；现行入口仅 `visualization/README.md`。
- `manifest.json`：迁移文件清单/哈希和删除的根output文件清单。旧可视化首次迁移后的读取遇MAX_PATH，已用长路径补完整清单；不声称这部分有持久化的迁移前后双哈希证据。其余迁移前后逐文件哈希核对一致。
- `maintenance/`：本次一次性整理脚本和验证记录。

E133唯一成功的质量评价移回 `experiments/E133_flow_matching/evaluation_quality_20261113/`，关键启动登记/捕获失败记录也归该实验。根 `output/` 其余内容直接删除，不保留整目录副本。旧文件里的output路径是历史文字，可能已不存在。实验内部的 `output/` 未删除。

训练源码/权重、原始评价、数据、静态缓存、预算和STOP不变；本次未新增研究实验。旧可视化曾有专用评分器和图像任务，不再是现行布局展示方式，旧调用复现需先核对归档路径。
