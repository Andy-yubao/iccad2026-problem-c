# 公共布局绘图

布局图统一调用此目录。按用户2026-10-17的要求，从实际生成其偏好图集的 `experiments/intent_aware_repair/plot_gallery.py` 提取绘图实现，未根据截图反推。来源和原源码快照哈希见 [origin.json](origin.json)。原始15例图集、坐标、配置与历史快照不改写。

保留Set3分组配色、MIB组纹理、模块编号/G/M/F/P标签、L/R/T/B状态徽记、重叠红轮廓、自身bbox虚线、三栏共享范围、底部统计和图例。每张PNG为3315×1989左右（19.5×11.7英寸、170dpi）。

```python
from visualization.layouts import plot_case, plot_evaluation_case

# 原E6图集格式：panel/case_id/constraints，variants下reference/raw/final。
# 每个阶段含positions(XYWH)，可附metrics和soft_details。
plot_case(saved_row, output_png, stages=(
    ('reference', 'Reference layout'),
    ('raw', 'Model proposal'),
    ('final', 'Current postprocessing'),
))

# E7已保存的评价格式（reference_positions、命名模型的raw_positions/final）。
plot_evaluation_case(saved_evaluation_row, 'attention_geometry', output_png)
```

调用方负责样本选择、读取既有JSON和图集索引。模块只接收坐标、约束和已存指标，不读取数据集、不推理、不修复、不产生官方分数。E7适配器复用原gallery.py的几何状态描述代码，为徽记/连通分量/形状数提供显示信息；这些描述不回写官方评分。缺失官方指标显示n/a，缺失布局明确显示不可用，不用别的布局替代。

当前接入：E6 `plot_gallery.py`、E7 `render.py`、本次test100运行入口。历史输出保留；其他历史独立脚本仅供复现，不批量重写。新布局绘图应调用本模块，模型、损失和求解算法继续按实验独立维护。
