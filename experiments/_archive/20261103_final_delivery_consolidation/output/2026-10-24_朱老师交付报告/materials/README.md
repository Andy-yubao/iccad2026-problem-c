# 素材索引（v2）

日期：2026-10-24。主要成绩参照来自`references/official/results`，布局展示只使用E42自己的三个阶段。

| 文件 | 内容 |
|---|---|
| [official_results_reference.png](official_results_reference.png) / [SVG](official_results_reference.svg) | 官方Beta前十raw_score与单独列出的本地结果；不同数据、不同硬件，不排同榜 |
| [official_beta_reference.csv](official_beta_reference.csv) / [JSON](official_beta_reference.json) | 官方Beta原数据与total_runtime/100的换算 |
| [network_structure.png](network_structure.png) / [SVG](network_structure.svg) | 输入、初步猜测、12轮修正、草稿及后续几何优化；图中展开一轮具体计算 |
| [E42_test_distribution.png](E42_test_distribution.png) | E42自己的1000例Q和耗时分布，无内部版本对比 |
| [layout_typical.png](layout_typical.png) | 原参考—E42草稿—E42最终布局，最终Q排序第500例，index302 |
| [layout_worst_quality.png](layout_worst_quality.png) | 三相图，最终Q最大例，index695 |
| [layout_slowest.png](layout_slowest.png) | 三相图，最终完整耗时最长例，index746 |
| [layout_examples.json](layout_examples.json) | 选例规则、成员、最终指标及原记录位置 |
| [E42_draft_examples.json](E42_draft_examples.json) | 为三相图补生成的3例草稿坐标及冻结权重身份；只有这一份新草稿缓存 |
| [derived_metrics.json](derived_metrics.json) | 已有记录汇总出的草稿/最终硬合法、零软违反例数及时间分项 |

全部布局图统一调用仓库`visualization/layouts.py`。三栏使用相同模块编号、约束和坐标尺度。颜色表示分组，纹理表示MIB，F/P表示固定形状/预放置，L/R/T/B表示边界要求，红轮廓提示重叠。

原始1000例记录未保存草稿，本次用冻结权重仅补生成3个展示样例的草稿，缓存后不重复生成。最终求解和原成绩均未重跑。参考来自同一成员的原始标签；它的形状未按平均面积修改，不保证是变换后任务的硬合法解。参考/草稿未重新评分，图中的n/a表示没有正式分数。红框和接触诊断仅用于可视解释。

完整参考和最终坐标仍保留在原数据/运行记录中，本目录不重复保存。三例选择与成绩不因制图结果更换。更详细来源见[evidence_index.json](../evidence_index.json)。
