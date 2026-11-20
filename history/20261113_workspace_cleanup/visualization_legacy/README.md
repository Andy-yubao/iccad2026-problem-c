# 公共布局绘图

`layer_capture.py`提供网络无关的只读readout钩子，调用方仍拥有模型和前向，不在展示模块中训练或执行模型。`layer_budget_viewer.py`消费保存的坐标，支持8/12等不同外层数；按外层完成比例比较时明确显示最近真实层，不插值、不声称等算力。当前E112/E113自动更新[同五例入口](critical_budget/20261107_pair/index.html)，A已有帧只读复用。

2026-11-07当前展示改为[同五例逐层对比](layer_comparisons/20261107_same_train5/index.html)，不再以100例三相图为默认入口。`layer_comparison.py`只读取已保存的逐层坐标，支持原E98/A/B/分阶段最终模型同层比较，以及分阶段4/8层best各自的真实输出。样例、ID、颜色和全模型全层范围固定；不自动打开界面，不运行模型或后处理。历史100例图片保留。

`layer_trace.py`用于只读层间诊断：读取已保存的真实head输出，固定模块ID颜色与全阶段共同坐标范围，生成中间层图、分支幅度曲线和本地交互入口；不运行网络、修复或Q。首个结果为E98 best71五个原登记训练样例，见[诊断报告](../experiments/E98_loss_weight_learning/runs/20261106_layer_diagnosis5/report.md)。

E102纯图像任务使用`raster_layouts.py`：按固定模块ID配色、公开画布和矩形亚像素覆盖渲染原始reference训练图片，并生成不同分辨率／模型输出的纯图片对照。该路径不调用几何描述、实例提取、修复或评分；没有文字、纹理、人工小模块放大或评分控件。既有几何布局绘图路径保持原样。

E102正式训练的同例reference／生成图片对比保存在[E102_reference_image_generation](E102_reference_image_generation/README.md)，由`image_gallery.py`生成不含评分控件的本地HTML及原尺寸图片。每次发布保存epoch／step与成员来源，入口指向最新已完整发布的图集，不自动打开页面。

布局图统一调用此目录。按用户2026-10-17的要求，从实际生成其偏好图集的 `experiments/intent_aware_repair/plot_gallery.py` 提取绘图实现，未根据截图反推。来源和原源码快照哈希见 [origin.json](origin.json)。原始15例图集、坐标、配置与历史快照不改写。

保留Set3分组配色、MIB组纹理、模块编号/G/M/F/P标签、L/R/T/B状态徽记、重叠红轮廓、自身bbox虚线、三栏共享范围、底部统计和图例。每张PNG为3315×1989左右（19.5×11.7英寸、170dpi）。

显式`stages`支持1～3个已保存布局；默认三栏保持原像素输出不变，单栏/双栏按实际列数安排画布和图例，避免空列。返回的`axes_bounds`可用于纯展示总览缩略图，不更改坐标或指标。

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

## 可复用草稿评分前端

2026-10-25用户更新：以后不自动启动评分界面。默认仅展示reference／当前模型draft／此前最优模型draft的同例对比；下述评分功能仅在用户明确重新要求评分时使用。历史记录保留。

只浏览已保存布局、不要求评分时，可用[saved_gallery.py](saved_gallery.py)的`write_gallery(folder, manifest)`生成只读`index.html`。支持分组、逐例切换、三列/单列、缩放及原高清图；单列裁切使用`plot_case`返回的`axes_bounds`。清单仅含图片路径和展示指标，不读取数据或运行模型。首个实例为[E50的35例对照](../experiments/E50_anytime_quality/runs/20261025_full500_review/analysis.md)。可直接打开生成的HTML，也可在本地用`python -m http.server 8766 --bind 127.0.0.1 --directory <图集目录>`浏览。既有评分前端和评分数据保持独立。

[review.py](review.py)及[review_web](review_web/)提供本地逐张评分：三相对照、单阶段放大、缩放、1/2/3快捷键、自动下一张、回看改分、未评分导航、自由备注/标签、整轮评价、断点恢复及JSON/CSV导出。评分只评价草稿：1差、2一般、3好。Q/Loss等辅助数值默认折叠，展开会记录；PNG不嵌入这些数值。反馈研究原则见[执行协议8.1](../docs/protocol.md#81-草稿模型的人类反馈研究原则2026-10-24)。

实验先生成三相图及`runs/<round>/manifest.json`，再启动：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -m visualization.review --session 'experiments/<实验>/runs/<round>' --port 8765
```

访问`http://127.0.0.1:8765/`。关闭服务不丢评分；同一目录重新启动恢复原顺序和进度。首次接入E42的抽样/补生成入口为[prepare_review.py](../experiments/E42_joint_regularization/prepare_review.py)，一键启动见[E42说明](../experiments/E42_joint_regularization/README.md)。后续实验只需产生相同展示清单，不导入E42私有算法。

最小清单字段如下。`axes_bounds`来自`plot_case`返回值，是相对整幅PNG的三个绘图区`[left,bottom,width,height]`；放大功能直接裁切既有PNG，不另画布局。

```json
{
  "schema_version": 1,
  "session_id": "unique_round",
  "title": "模型 / 草稿评分",
  "display_note": "数据角色、草稿及最终布局来源与已知限制",
  "rubric": {"version": 1, "target": "raw_draft_only", "scale": {"1": "差", "2": "一般", "3": "好"}},
  "selection": {"seed": 20261024, "source_run": "...", "data_policy": "..."},
  "cases": [{"id": "case_0001", "n": 20, "member": ["worker_0/layouts_0.th", 1], "image": "images/case_0001.png", "axes_bounds": [[0.035,0.3,0.3,0.5],[0.358,0.3,0.3,0.5],[0.681,0.3,0.3,0.5]], "auxiliary": {"后处理Q": 1.02}}]
}
```

清单顺序就是展示顺序，抽样、随机化和来源验证由实验负责。评分后清单哈希与数据库绑定，变更清单须新建评分轮次。`ratings.sqlite3`使用事务和完整同步写入，记录当前分数及UTC时间的所有变更事件，含所用标准版本与看过指标的事件；多页面版本冲突明确拒绝写入，避免覆盖。自由标签支持增加观察维度，不能把旧分数自动解释为新标签。JSON导出包含不可变清单、当前状态及全部历史；CSV保留逐例分数、备注、标签，文本公式仅在CSV中加转义，JSON原话保持不变。

保存备注不会自动打分；离开当前图片前会保存备注，关闭页面时对未保存文字提示。评分保存成功后才自动下一张；图片失败则禁用评分。全100例都有分数才可标记本轮完成，未完成期间不显示分数分布。整轮评价可留空并回到聊天叙述。所有反馈文件留本机；工具不调用模型、数据集、后处理或评分器，不把分数接入训练。服务仅监听127.0.0.1，仅开放清单中图片和固定资源；不开放整个工作区。

2026-10-24扩展：清单可用`view_stages`指定两栏A/B或三栏名称；`rubric.target_label/prompt/instructions/scale`设置该轮评分含义。`rubric.questions`可添加可选维度，每项提供`prompt`及1/2/3的`scale`，持久保存为`ratings[id].dimensions`，未知维度拒绝写入。原单例草稿评分默认含义保持不变。E42的整齐对照使用独立会话和rubric v2，主分1=A更整齐/2=差不多/3=B更整齐，另可选可修复性/留白偏好；不能与旧“1差/2一般/3好”混为同一变量。可选维度先选择，按主分时一并保存；同一可选项再点一次可清除为未回答。CSV追加dimensions列，JSON保留原始字典及历史。
