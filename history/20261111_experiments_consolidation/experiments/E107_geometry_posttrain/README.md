# E107：E105局部几何后训练，reference10%

**2026-11-05已停止：接受22次更新，第23批触发相对单例Loss门禁；STOP保留，未完成一轮或终末dev500后处理。** [停止报告](runs/20261105_geometry_ref10_seed0/abort_report.md) · [退出核验](runs/20261105_geometry_ref10_seed0/stop_verification.json) · [只读诊断](runs/20261105_geometry_ref10_seed0/diagnosis_and_revision_proposal.md)。

诊断末尾提出的相对父稿固定2%总位移/.02总形变限制，后来已被用户否定；[后续公开能量提案](../../output/reference_free_optimization_proposal_20261105.md)尚未实施，不是续跑授权。下文仅为本次失败运行的固定设计/启动记录，不自动重试或改阈值。

2026-11-05用户授权：从E105基线后训练，以总重叠、重叠超标模块和真实空白为主；原参考位置/形状权重降为10%（.1/.025），严格检查Loss爆炸，一旦出现立刻终止；准备好后直接后台启动。只训练一个候选，不使用E106代理，不恢复旧E98或其他实验。E105源码、best58和已完成评价均保护。来源复制记录见provenance.json，资产assets/parent_best58.pt固定sha256，不进Git。

## 固定科学方案

保持E105 128×16 gated pre-norm/SwiGLU结构，3,490,819参数；原v6 train前32000、固定train/dev500、公开MIB平均面积任务、原split_v1保持。模型/特征/组树/面积变换/decoder/端口压缩逐字节复制；网络只用公开输入，原始reference只用于剩余10%监督。原50%完整轴交换、FFN Dropout.10、残差Dropout.05、batch32/FP32、deterministic/TF32关闭、CPU4线程与AdamW decay.0003/clip1/模块联合步长上限.005保持。non_loss_alignment.json登记相同项和明确变更。

新阶段仅继承E105 Loss-best58模型张量，fresh AdamW/RNG/排列/游标/调度与阶段预算；新Loss不复用原优化器或best标准。LR1e-4→5e-5，无保护期，epoch0建立基准，连续3轮无明显改善降档，最低档至少完整2轮且连续3轮停滞正常早停；min_delta=max(1e-5,基准×.001)，best门槛1e-5，无硬时间/轮数上限。阶段内恢复必须保留全部状态，数值错误不自动恢复。

三个主项：

1. 重叠总量=(模块总面积−矩形占用并集面积)/模块总面积。它按额外重复覆盖计数，避免很多模块叠成一堆而“一次计数的重叠区域”反而变小；另单列真实覆盖层数≥2的区域占比。
2. 令r_i为模块i被其他模块覆盖的精确并集比例、e_i=max(r_i−.10,0)，训练tail=mean[e_i/(e_i+.05)+e_i]，兼顾平滑超标模块比例与严重程度；另直接报告真实N10及N10/N。
3. 空白=(外框面积−占用并集面积)/外框面积，避免用模块面积总和冒充并集而低估空洞。

采用按矩形边界排序的事件网格计算并集、重叠区域和逐模块覆盖：不做图像栅格化，面积/比例在拓扑固定区间内可微，边界重合处采用框架子梯度。三项各按冻结E105原train_monitor500的均值归一化、下限1e-4、等权1，不做总和再归一化。归一化只一次、无更新、无dev/test，成员和哈希在normalization.json冻结。数据侧标定是登记的训练面板处理，不作实施短检结论。

辅助项：.1×position+.025×shape；保留原2B×软约束贡献和.5B×wirelength（B=.3336832108335229）。对边界/分组/MIB原raw项超过max(同题冻结父模型项,容差.01/.01/.0025)的比例超额平方，权重2；线长超过父模型1.05倍的比例超额平方，权重1（父分母下限1e-4）。另以.05×原penetration提供完全覆盖平台区的局部分离梯度，不代替三个主指标。

局部探索：冻结E105 eval预测为参照；中心位移超过.1×sqrt(公开模块面积)、log宽高比差超过.1才罚归一化超额平方，按可移动/可变形模块各自平均、权重1。它是软约束，不承诺每模块严格不越界。学生漂移项专门用关闭Dropout的eval前向（仍反传），其余新Loss用训练Dropout前向；避免把相同权重的正常Dropout波动当成真实漂移。监测时学生全部eval、同一Loss函数，无增强；冻结父模型不反传，不输出到部署求解器。

## 每批立即终止门禁

每批执行前向/反向后、AdamW与projection之前检查，监测每批同样检查：

- 任一输出/任一Loss项/裁剪前梯度norm含NaN/Inf，拒绝更新。
- 保留原绝对保护：批均值或单例Loss≥1000、裁剪前grad norm≥10000，拒绝更新。
- 新相对保护：批均值≥max(3×同批同题冻结父模型新Loss均值,父均值+5)，或任一单例≥max(10×同题父模型新Loss,父单例+20)，拒绝更新。父模型基准是当前公开输入的固定函数，非不断抬高的滑动阈值。

任何一项触发即抛出LossExplosion，写该run的STOP，保存loss_explosion.pt/failure.pt、完整latest与best及诊断后退出；控制器不启动评价。不跳批，不重试，不自动调低LR/权重或放宽门禁。所有真实异常均显式报告，准备短检不伪造正式失败。参数步长保护不能替代Loss保护，小LR不能单独保证输出局部性。

## 最小检查、后台和终末评价

最多同5个训练题检查新事件网格与独立CPU并集/覆盖算法一致、x/y轴交换后的Loss一致、一次真实GPU更新、冻结父模型无梯度；对实际pre-Adam入口注入拒绝标志验证不更新Adam/参数，检查NaN和相对Loss尖峰被拒绝。不预先测试恢复；同一训练题做一次完整求解评分接口检查。默认不增加其他实施题。

正式训练保留best/latest的模型、Adam、RNG、数据排列/游标/seen、调度、projection和累计阶段时间；epoch0和每轮固定train/dev500记录同新Loss和精确几何指标。best/LR/早停共用新Loss，原E105 Loss作为解释指标，不与新Loss的数值直接比较。FP32图包含冻结父模型、学生clean分支和学生Dropout分支，objective和mask无近似降级；资源和吞吐实测记录，不用GPU占用替代每轮时间。

只有正常早停才冻结唯一新Loss-best，自动一次同固定dev500、E98.depth_budget.v1后处理，三Q、草稿/最终硬合法、重叠/空白/N10、完整均值/P95/最大延迟及250ms超限分别报告；与父模型已完成500只读对照。如best恰好与父权重完全相同，则复用已完成结果和原计时并注明，不重复求解。原参考/E105父稿/E107草稿/最终后处理可从保存坐标绘图，统一ID尺度、无评分控件，不自动打开页面。Q不回头选checkpoint、改Loss或继续训练；测试封存，E106与其他实验不修改。

启动：powershell -File experiments/E107_geometry_posttrain/launch.ps1。先运行calibrate冻结train500尺度，再shortcheck；目前启动已由本次用户明确授权。运行runs/20261105_geometry_ref10_seed0，stdout.log/stderr.log为日志，heartbeat.json/progress.json/pipeline_state.json为状态。STOP在本实验或该run中均有效，安全保存后阻止下一阶段。错误STOP不得自动移除。

实施记录：首次五题数学检查通过，排序反传在FP32确定性CUDA图捕获失败。仅把坐标排序梯度改为detach stable argsort确定排列、one-hot bmm传播相同几乎处处梯度，不改Loss/精度/门禁或归一化尺度。失败源码和证据保留于runs/implementation_capture_fix；正式训练未受影响。
