# E98泛化增强基线：单个从头训练候选

用户2026-11-04明确授权先准备并后台训练此基线，训练期间再准备其他改动。旧多轮运行保持STOP，不恢复。新基线与原baseline、pilot及停止运行分开保存，不覆盖原源码或证据。

96×12、1,471,539参数、seed0、batch32、FP32；FFN Dropout=.10，两个残差分支Dropout=.05。AdamW decay=.0003、clip1及每模块联合更新上限.005保持。所有参数随机初始化，不继承任何旧权重。

原v5 train16k、固定train/dev各500、原MIB平均面积派生任务、原十项固定Loss与冻结分片保持。每训练成员每epoch按独立固定种子以50%概率交换x/y轴，完整同步公开特征、端口、预放置、边界位、形状与监督；不从自由reference导出网络输入变换参数。增强在训练CUDA图中执行；固定train/dev监测均使用未增强原任务、eval模式及同一Loss。采样顺序沿用seed+epoch，增强随机流独立于采样和Dropout，按seed/epoch/成员确定，不依赖隐含游标。

沿用原从头训练LR：1e-3→2.5e-4→1e-4→5e-5；前14轮保护，第15轮建立平台基准，3轮无明显改善降档，最低档完整至少2轮才允许早停。明显改善=max(1e-5,基准Loss×.001)，best门槛1e-5；dev500算术均值Loss统一负责best/LR/早停。原4小时/200轮上限保留，不把预算结束称为收敛。best/latest分别保存模型、Adam、RNG、排列/游标、调度、保护计数与累计耗时。

运行目录：`experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0`。状态heartbeat.json、progress.json；启动日志为本源码目录stdout.log/stderr.log。本次不调用后处理、不计算train/dev Q、不读取测试，不自动进入调权；训练结束只出Loss报告。其他准备工作不得修改本运行冻结源码身份。

五例必要短检检查轴交换重建/反变换与Loss对称性、新Dropout和真实更新；不预先测试恢复。短检入口为本包preflight，后台入口为launch.ps1。恢复仅用于未结束且身份一致的运行，使用本包train --resume；不自动恢复error。创建新run的STOP安全停止；旧动态运行STOP与本运行互不干扰。
