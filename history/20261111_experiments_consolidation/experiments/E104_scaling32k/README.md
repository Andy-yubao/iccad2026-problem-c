# E104：32k、128×16 泛化增强版扩容实验

**当前：已正常早停82轮，修复阶段唯一Loss-best81；固定dev500评价完成，流水线结束，无下一阶段。** [终末报告](runs/20261104_scaling128x16_seed0/evaluation_depth_dev500/report.md) · [完成状态](runs/20261104_scaling128x16_seed0/pipeline_state.json)。混合Q1.03744678、500/500最终合法；step20350的local-prenorm迁移使本结果不是纯缩放对照。下文是既定设计与迁移记录，启动命令不应再次执行。

用户2026-11-04明确授权新实验、从头训练、后台启动，完成后自动固定dev500评价；已明确确认不是封存test。唯一候选从E98增强版设计扩大：16k→32k，96×12→128×16，1,471,539→3,463,747参数（2.354倍）。seed0、batch32、FP32、AdamW decay=.0003、clip1、每模块联合步长上限.005、固定原十项Loss、FFN Dropout=.10、残差Dropout=.05、训练50%完整轴交换保持。不继承任何权重或E103系数。

成员为原reference_workset_v6训练前32000，原v5前16000不变、模块数分布精确加倍；原split_v1和固定train/dev500保持。32k已在E58/E74/E101等历史研究中曝光，不称全新数据；公开MIB平均面积派生任务保持。网络仅公开信息，端口保留每模块总权重与加权中心，完整连接仅用于Loss/评价。test载荷禁止读取。

原从头LR四档1e-3→2.5e-4→1e-4→5e-5；前14轮保护、第15轮建立平台，3轮无明显改善降档，最低档完成至少2轮且3轮停滞早停。明显改善max(1e-5,基准×.001)，best门槛1e-5。无硬时间/轮数上限，正常dev Loss早停不等于train已完全拟合。反传/固定train500/dev500/best/LR/早停同一固定Loss；监测不增强且eval。保存best/latest与Adam、RNG、排列/游标、调度和累计耗时。所有私有算法独立复制，来源哈希见provenance.json，不跨实验导入。

准备短检最多五个训练实例，覆盖新增32k成员、原增强语义、新结构真实更新和同一训练实例的完整评价链。正式训练在预检和源码身份门禁通过后启动。CPU只读缓存、pinned batch传输和FP32 CUDA图，不整库进显存，保留数值/梯度/步长门禁。性能记录每100步和完整epoch时间，不以GPU占用代替吞吐。

正常早停后流水线在独立进程冻结唯一dev Loss-best及源码/权重/面板哈希，复制的E98.depth_budget.v1对固定dev500单次评价，不按Q重选、不继续训练。基线只读复用原增强best71已完成500结果。分别报告固定Loss曲线、绝对/相对泛化间隙、三Q、草稿/最终硬合法与几何指标、完整逐例均值/P95/最大值/250ms超限，冷启动/磁盘/评分/绘图另列，同100例统一ID尺度图；不自动打开页面。完成报告后停止。

这是单seed、数据和模型共同扩大的试验，不足以拟合scaling指数或区分数据/宽/深的单项贡献。dev面板已反复开发使用，不称独立未见泛化或正式比赛成绩。E101/E102/E103与已停止E98不改动、不恢复。

运行：`powershell -File experiments/E104_scaling32k/launch.ps1`；入口`.conda/envs/iccad2026/python.exe -m experiments.E104_scaling32k.pipeline`。状态位于runs/20261104_scaling128x16_seed0/pipeline_state.json、heartbeat.json、progress.json。停止：在本实验目录或run创建STOP；训练在批次边界保存latest，控制器不启动评价。error不自动重试；恢复需核实身份后明确执行train --resume，并单独处理后续阶段。


2026-11-05用户授权修复并续训：原step20350于第21轮触发单例Loss门禁。失败题自由输入无明显异常、reference Loss约.078，原路径隐藏状态逐层放大到约996，eval Loss约2752；保存失败后RNG重放原路径最大单例Loss约1914。局部消息原先直接聚合未归一化h，现改用现有norm1(h)后聚合，参数仍3,463,747，原十项Loss/增强/Dropout/门禁不变；这是数值前向路径迁移，非位级续训或纯扩容证据。原before/中保护全部源码、best18/latest/failure、曲线和日志。修复短检使用同一已失败批次（32成员，为定位实际error的必要检查，不增新面板）通过一次真实更新，该更新丢弃：max Loss2.90、grad norm11.77。正式从原latest20350继续，保留模型/Adam/RNG/排列/游标/seen/当前LR/累计时间；由于前向函数改变，新best/监测历史和平台基准重新建立，LR档位和drop/epoch计数保留、无新保护期。最终报告单列此限制；正常早停后仍自动冻结唯一修复阶段best进行固定dev500。迁移登记及启动证据见runs/20261104_scaling128x16_seed0/repair_20261105。
