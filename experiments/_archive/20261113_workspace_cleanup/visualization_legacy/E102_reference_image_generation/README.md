# E102 图片生成对比

当前BF16＋batch4迁移续训的[新图集入口](20261106_semantic_v21_bf16_b4_seed0/index.html)。从累计12297步续训，完成迁移后100次更新发布预览，后续Loss-best按新精度更新。旧FP32／RGB图集及停止标记保持，不与新引擎混称同一数值结果。

新版semantic训练已于2026-11-06按用户要求终止于12297步，图集保留停止前最新Loss-best，不再自动生成更新。原RGB运行也保持停止。

新版正式运行入口：[20261106_semantic_v21_seed0](20261106_semantic_v21_seed0/index.html)。固定dev100，左reference、右全MASK单次生成；训练第100次更新后发布早期预览，其后随固定dev500双视图总Loss best更新。没有几何提取／修复或Q。旧RGB图集保留且不再更新。

2026-11-06新版模型／Loss的[单训练图拟合对照](20261106_semantic_v21_fit1/index.html)另存于此。右图是250更新后的全MASK直接预测，模块IoU82.45%；这是1张训练图的拟合，不是dev或未见题成绩。16轮迭代另存`generation_comparison.png`，效果更差，不作默认。旧正式运行已经用户暂停，以下自动更新说明只描述暂停前的流程。

2026-11-06用户明确要求后台启动E102，并在visualization下保存布局图片对比。

当前入口：[20261106_rgb512_canvas_v2_seed0/index.html](20261106_rgb512_canvas_v2_seed0/index.html)。训练完成100次更新后发布固定dev100的早期图片；之后每次固定dev500 RGB-MSE改善保存best时更新入口，并保留先前图片。训练准备期尚无生成图，入口不展示占位网络结果。

每例左侧为原始reference目标图片，右侧为网络直接生成图片。相同物理画布、模块ID颜色和512像素尺寸；可点击图片查看原图。无评分控件、不自动打开页面。不作矩形／矩阵提取、几何修复、MIB同形或Q评价。训练Loss与选择仍只看登记的dev500 MSE，显示的dev100不另选模型。

每次发布目录内manifest.json记录run、epoch／step、成员和图片误差，区分早期预览与经过正式监测的Loss best。图片及完整缓存留在本机。

本轮使用公开画布v2：原v1在正式准备的一例训练成员上超界，且没有发生训练更新；现把公开端口最远坐标与预放置最远边界共同计入画布范围。原失败现场保留。详见[授权与计划](../../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0_launch/authorization.json)。
