# E54：不参考reference／只参考reference

2026-10-26，**用户取消本实验，不再继续训练或调整。** no_reference由用户启动后失稳，停止于epoch24/step12080，best仍为epoch0；reference_only尚未启动。已设置global STOP，保留[失稳诊断](runs/20261026_no_reference_seed0/convergence_analysis.md)和唯一运行证据；下述设计与命令仅作历史记录，不代表待执行计划。旧E53自主任务与定时唤醒仍暂停。

## 实验问题与解释范围

从同一份[E53 geometry best62](../E53_loss_geometry/runs/20261026_geometry_seed0/analysis.md)分出两条支线，只保留两类Loss中的一类。模型已经接受过reference监督，因此第一组的准确含义是“这段后续训练完全不使用reference”，不能称为从未受监督训练，也不能据此证明从头无监督训练的能力。

不新增惩罚、不重新调权重、不重归一化、不设置容差。将E53实际启用的固定Loss按是否读取自由模块reference标签分解，两个极端之和还原原E53目标。几何约束的形式不代表它不依赖reference：参考接触关系与参考外框比例都归参考组。

| 支线 | 固定保留项及权重 | 关闭项 |
|---|---|---|
| `no_reference` | 穿透重叠.75；公开边界要求.5；MIB形状一致.5；逐模块重叠负担.75；bbox/公开总模块面积的紧凑性.05 | 中心、形状、相对位移、参考接触、参考bbox、全部其他reference派生目标 |
| `reference_only` | 参考中心.35；参考形状.1；参考近邻相对位移.125；参考接触1；参考bbox比例.15 | 独立重叠、公开边界/MIB惩罚、逐模块重叠与公开紧凑性目标 |

精确定义见[loss.py](loss.py)和[loss_config.py](loss_config.py)。no_reference进入函数后先使用公开输入白名单，计算路径不访问target/reference；标签仍可存在于冻结缓存中，但不参与该组前向、Loss或梯度。reference_only的目标均有reference提供的监督目标，不独立惩罚所有预测冲突。

no_reference保留公开group模型输入，但**不新造无监督分组损失**：原contact依赖reference边接触，故完整移至reference_only。这组是原Loss中不依赖reference部分的消融，不代表已经覆盖全部官方约束的完整无监督优化器。两组模型解码仍有面积守恒、非负坐标、公开固定/预放置信息等原架构边界，reference_only不移除这些固有行为。

## 同一起点与控制变量

- 起点唯一：`experiments/E53_loss_geometry/runs/20261026_geometry_seed0/checkpoints/best.pt`，SHA256 `1cc0e5ad5dad483e1bc6f35a9d26333a5ee2aa591de6f42f79128e254261d3ef`。只加载模型权重，严格校验同架构和输入张量语义，不复制/覆盖原checkpoint。
- 两组都重置AdamW动量/步数与调度器/早停历史，因为目标已改变；这是从同一权重分叉微调，不是恢复E53已结束的优化器状态。新组的epoch/预算从0开始；各组自身中断恢复后累计进度和预算不清零。
- 架构固定96×12、rectangle、FFN dropout .05，1,471,539参数；FP32、确定性CUDA、batch32、clip1、AdamW decay .0003。两组seed0、相同新样本顺序与随机状态初始化。
- 沿用原学习率1e-3→2.5e-4→1e-4→5e-5，第15轮起按原平台规则下降/早停；这是完整新阶段的统一规则，没有为任一支线单独改学习率。初期可能扰动已训练权重，需看完整曲线，不把单个batch变化当成结论。
- v5 train16,000、dev2,000、固定train_monitor500/dev_monitor500；训练反传和监测共用各组固定Loss，best/LR/早停只用该Loss在dev500上的均值。两组总Loss尺度不同，不能直接比较其数值或best epoch大小。
- 各组累计训练上限2小时或200轮，串行GPU训练；复用E52快速FP32引擎。来源与完整冻结源见[provenance.json](provenance.json)、[plan.json](plan.json)。代码独立复制，不跨实验导入私有算法。

## 训练命令

两组依次训练；同一个命令也能恢复未完成的断点，并跳过已完成训练：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E54_reference_extremes\launch.ps1' -Stage train -Arm all -Resume
```

只运行某一支线时将`-Arm all`替换为`-Arm no_reference`或`-Arm reference_only`。入口自己定位工作区与独立conda Python，无需激活环境。不得同时开两个终端运行这两组；串行锁避免共享GPU竞争。

输出分别是：

```text
runs/20261026_no_reference_seed0/
runs/20261026_reference_only_seed0/
```

首次运行不存在checkpoint时从共同E53权重初始化；只有本组latest存在时，`-Resume`才恢复本组模型、优化器、调度器、随机状态、数据游标和累计预算。`-Resume`同时显式移除本实验global/对应分支STOP；不触碰E53的STOP、状态或自动任务。若第一组被用户停止，不继续启动第二组。

安全停止：在本目录`runs/`新建`STOP`文件，或按Ctrl+C等待保存。之后用上面的命令恢复。best和latest均保留，原E53/E42不改动。

## 训练后的比较

用户训练完成后按请求验证；也可手动执行两组各一次冻结best开发评价与绘图：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E54_reference_extremes\launch.ps1' -Stage final -Arm all
```

每组先以该组同一固定Loss评价完整dev2000，再用原flash在同dev500做唯一一次后处理；两进程×两线程。图集复用同一随机开发100例，保存reference／当前分支draft／冻结E42 draft，无评分控件，不自动启动页面。测试保持封存。

比较重点是参考几何的保持、重叠及深冲突、留白与线长、接触/MIB/边界、后处理移动与改形、救援和极端失败；固定flash加权Q/尾部作为共同辅助对照。共同起点E53与冻结E42已有结果直接复用，不重跑控制组。不能按最终Q换checkpoint，也不能用两种总Loss横向排名。本轮只建立这两个对照，不继续原来的自适应优化队列。

## 已完成检查

同5个train成员[50,36,22,62,45]重复成batch32，未读测试；[证据](runs/20261026_check5/results.json)：

- 两极端总Loss之和与E53相等（本批差0），对模型输出的梯度最大差1.86e-9。
- no_reference删除标签、将标签替换为NaN，Loss和输出梯度完全不变；reference_only确实响应参考标签变化，并不使用独立边界/MIB损失。
- epoch/include_quality参数不改变目标；两组训练模式CUDA图捕获与有限性检查通过。
- 两组实际训练入口均只做一步保存→恢复到第二步，优化器、来源身份和累计预算保留；这些implementation_check checkpoint是重复五例的排错产物，禁止转为正式模型。

短检查约36秒，不是两组正式训练的启动或结论。新Loss不同，旧E53的开发Loss0.27406只记录为历史目标，不作为这两组的新选模标准。
