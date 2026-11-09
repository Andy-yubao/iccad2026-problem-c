# 统一调参和100例对照入口

以后只编辑本目录的[tuning.json](tuning.json)，无需改源码、方案脚本或输出路径。每次启动只运行文件里这一组设置，不会自动扫描其他参数，也不会训练网络或读取测试。

在项目根目录运行：

```powershell
& ./experiments/E99_attraction_repulsion/run_tuning.ps1
```

也可以从其他目录使用脚本的完整路径；脚本会自动切换到项目并使用独立conda环境。等价Python入口：

```powershell
& .conda/envs/iccad2026/python.exe -u -B -m experiments.E99_attraction_repulsion.tune
```

## 常用参数

百分数用小数填写，例如`0.05`表示5%。移动的尺度始终使用**原草稿模块短边**，不会随着形变偷偷变化。

| tuning.json位置 | 当前值 | 含义 |
|---|---:|---|
| `label` | relaxed20_move40_shape50 | 显示名称，不影响数学身份；只改名字不会重复求解 |
| `relaxation.iterations` | 20 | 最多迭代轮数；无位移且无形变时可提前退出 |
| `relaxation.step_fraction` | 0.05 | 单轮中心位移上限：原短边5% |
| `relaxation.total_fraction` | 0.4 | 累计中心路程及相对原稿位移上限：原短边40% |
| `relaxation.shape_step_fraction` | 0.03 | 单轮宽高变化因子最多1.03；缩小使用其倒数 |
| `relaxation.shape_total_fraction` | 0.5 | 累计对数形变路程≤log(1.5)，宽高在原来的1/1.5～1.5倍内 |
| `relaxation.step_size` | 0.5 | 中心受力响应强度；仍受上面的移动预算限制 |
| `relaxation.shape_step_size` | 0.5 | 形变受力响应强度；仍受上面的形变预算限制 |
| `relaxation.attraction_range` | 0.5 | 吸引范围，相对于两模块较小短边 |
| `relaxation.attraction_strength` | 0.25 | 吸引强度 |
| `relaxation.repulsion_strength` | 1.0 | 排斥强度 |
| `relaxation.shape_enabled` | true | 是否启用形变；false可检查相同预算只平移 |
| `planner` | 原完整后处理设置 | 后续几何规划器的设置，与前置参数保存在同一文件 |
| `comparison.baseline_mode` | auto | 优先复用严格匹配的原后处理结果；fresh强制建立一次同期原后处理对照 |
| `comparison.previous_best_run` | auto | 图中第二列：在本次运行开始前完成、同草稿/同面板/同后处理的旧AR版本中，选择最终加权Q最低者；也可填兼容旧运行路径固定参考 |

这些是上限和响应参数，不代表每轮必定移动或变形到上限。面积质量、固定形状锁定、预放置锁定和保面积规则保持；前置步骤没有引入group/MIB/外框/贴边/线长目标。后半部分仍按`planner`设置处理这些约束。

数值参数只检查正数/有限性等必要条件，不再受旧实验10轮、3%、20%、15%的源码上限限制。原V3算法函数本身不变；通过参数子类使用新预算。相同V3参数的坐标已验证逐位一致。

## 每次运行会得到什么

- 固定同100例：`v5 dev_monitor500`前100例，原C3草稿、原始split和公开面积变换保持不变。
- “原后处理”与“吸引排斥＋相同后处理”的官方中性时间Q、硬检查、软约束、线长/外框统计，以及逐例改善/持平/恶化。
- 草稿的精确交面积、真实空白率、覆盖<10%模块比例和最坏模块，中心位移/形变/面积/固定几何核查。
- 前置步骤与CPU草稿到最终硬检查的均时/P95。NN、输入构造和传输不在这个计时范围内，不代表完整请求延迟。
- 同100例固定四列主图：**原始草稿／此前最优AR处理后的草稿／当前AR处理后的草稿／当前AR＋全部后处理最终布局**。统一ID与尺度，不显示评分控件，不自动打开界面。第二列全100例来自同一个旧版本，不能逐例拼接；目前自动参考是v3（10轮、累计位移19.99%、尺寸因子1.15）。
- 独立配置快照、源码快照、成员和输入哈希、逐例坐标与轨迹。此前最优仅作为第二列显示参考，当前数值参数始终取配置快照。

入口：[`runs/tuning/LATEST.json`](runs/tuning/LATEST.json)指向最近运行，[运行记录表](runs/tuning/index.md)列出已完成配置。每个新参数身份会生成独立目录，名字包含时间和身份摘要，旧结果不会覆盖。运行中的`tuning.json`修改不会改变已冻结的当前运行。

相同数值配置、输入和几何/求解/评分源码再次执行，会校验已有坐标与图片后直接返回已有报告，不重复100次求解。仅更改显示名称、JSON排版、出图或报告代码也会复用计算结果；新图片有独立版本和来源哈希，不覆盖旧图。后处理设置改变或源码不再匹配旧控制时，`auto`会建立新的对应控制；`fresh`也会运行一次同期控制，两臂交替顺序、单线程串行。

只想用已有坐标重新生成这四列图，不运行任何几何处理，可执行：

```powershell
& .conda/envs/iccad2026/python.exe -B -m experiments.E99_attraction_repulsion.tuning_gallery
```

默认读取最近完成的统一入口运行；`--run "运行目录"`可指定旧运行。主图入口为该运行的`gallery/index.html`，其`PRIMARY.json`记录当前四列图位置和旧参考版本。原有其他图仍保留。

复用历史控制时，其时间只标为“历史repair时间”，不能和新流程的当前耗时当作配对提速证据。需要同期耗时比较可将`baseline_mode`改成`fresh`后启动一次。

## 停止与恢复

在当前运行目录创建`STOP`文件，或在本实验`runs/STOP`创建全局停止文件。程序会在案例/处理臂边界保存退出，不启动后续成员；已经完成的处理臂和图片也会保存。

移除相应STOP后，保持参数不变，明确恢复：

```powershell
& ./experiments/E99_attraction_repulsion/run_tuning.ps1 -Resume "E:/project/ICCAD/2026/problemC/experiments/E99_attraction_repulsion/runs/tuning/实际运行目录"
```

如果实时`tuning.json`已经改过，用原运行的快照：

```powershell
& ./experiments/E99_attraction_repulsion/run_tuning.ps1 -Config "实际运行目录/settings.json" -Resume "实际运行目录"
```

恢复会检查数值设置、源码和输入身份；已完成案例不会再次求解，图片从保存坐标生成。错误不会自动重试。进程锁由操作系统释放，异常退出不需要手动删除锁文件。固定封存测试不会通过这个入口开放。

这100例已被反复观察，手动调整参数仍属于自适应开发，不能把结果称为独立泛化或正式比赛成绩。
