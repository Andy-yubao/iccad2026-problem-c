# 朱老师交付报告与素材

日期：2026-10-24；修订v2。

先阅读[报告正文](report.md)。本版以`references/official/results`的Beta历史结果为主要成绩参照，方法部分解释输入、位置/形状预测、12轮信息交换及后续几何修复。布局图均为E42的“原参考—网络草稿—最终布局”三相图。

| 文件 | 用途 |
|---|---|
| [report.md](report.md) | 完整报告，6张配图；不使用内部旧版本对比 |
| [factbase.md](factbase.md) / [factbase.json](factbase.json) | 18条事实、源证据和解释边界 |
| [materials/README.md](materials/README.md) | 官方比较素材、网络图、结果分布和三相图 |
| `evidence/` / [evidence_index.json](evidence_index.json) | 26份源记录逐字副本及来源/哈希，包含官方results全部5个文件 |
| [reproduce.md](reproduce.md) | 模型、数据、结果定位及素材重建方式 |
| [build_materials_v2.py](build_materials_v2.py) | 制图与factbase生成；3例草稿有缓存，不重复生成 |
| [verify_bundle_v2.py](verify_bundle_v2.py) | 数字、链接、源文件及图片检查 |
| `qa.json` / `manifest.json` | 本材料包验收和文件哈希清单 |

本版更正了原稿的表达和比较对象，清除了旧对比图与相应汇总副本。研究原始运行记录保留在仓库原位置。

三相图的最终布局和成绩沿用已有1000例记录；只为展示补生成了3例冻结模型草稿，没有重新训练、重新最终求解或重新评分。原参考布局仍是面积变换前的形状，图中参考/草稿两栏不标正式分数。

官方Beta使用隐藏100例和另一套硬件，本地使用MIB平均面积任务的1000例，因此官方表格是历史数值参照，不能据此推算名次。证据原件保持原字节，原件中的历史状态、来源路径和相对链接应按当时记录理解；报告、factbase和素材索引中的包内链接另行验收。
