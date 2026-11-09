# E105：方案一，完整前置归一化＋残差门控＋SwiGLU

用户2026-11-05明确授权实施方案一，准备好后直接后台训练，其他方面与E104一致。只启动一个新结构候选，不自动补跑控制或方案二/三。独立复制E104基础设施，不跨实验导入私有模型、Loss或算法；来源见provenance.json。E104所有源码/权重/证据保持不变。

模型128宽×16层，3,490,819参数，相对E104的3,463,747增加0.7816%。注意力、三关系局部消息、FFN、几何反馈分别独立LayerNorm。四条残差路径各有每层独立sigmoid标量门，初值.1，范围(0,1)，不扫描。共享几何反馈MLP为3→128→128（SiLU），随后每层专用LayerNorm/门；共享三维几何head前增独立LayerNorm。FFN换为SwiGLU，mid256=2w、乘性门控后Dropout.10。attention/local/FFN输出分别残差Dropout.05，几何反馈不额外加Dropout；保持原概率，但结构改变会改变Dropout抽样数量/顺序，不能声称跨结构逐位一致。原19维geometry bias、head其余部分与public decoder接口、4 heads、public输入与端口完全压缩规则保持。仍不保证草稿合法或数学上的绝对稳定。

其他条件逐项冻结为E104：v6 train前32000、同train/dev500、原split_v1，公开MIB平均面积派生任务；每epoch/成员独立确定50%完整x/y轴交换，SeedSequence([20261104,seed,epoch])，训练sampling seed+epoch，原监测无增强/eval。同16个基础设施文件逐字节一致，成员顺序和方案字段核对见non_architecture_alignment.json。原端口逐模块总权重和加权中心为网络输入，完整端口连接仅供Loss/评分。

从头随机seed0，不加载任何E104权重；batch32、FP32、deterministic、TF32关闭、CPU4线程、原AdamW decay.0003、梯度clip1、每可训练模块联合更新上限.005。原十项固定Loss用于反传、train/dev监测、best/LR/早停，绝不使用E103拟合系数。LR1e-3→2.5e-4→1e-4→5e-5，前14轮保护，第15轮建立平台，patience3/min_delta=max(1e-5,baseline×.001)、best阈值1e-5、最低档至少完整2轮，正常早停；无硬时间/轮数上限。不改变原数值门禁、跳批或无限重试。best/latest完整保存模型、Adam、RNG、顺序/游标/seen、调度、projection及累计时间。

最小实施短检只用5个训练实例，覆盖原E104失败题worker_75/layouts_4032.th第97例和扩展成员、固定/预放置、分组/MIB；检查轴交换语义、Loss一致性、新结构参数/独立归一化/门初值与梯度、一次真实更新及同一实例的完整评价接口。不预先测试恢复，不新增面板；短检不证明稳定泛化。FP32 CUDA图/AdamW/projection引擎保持，CPU只读缓存和pinned batch，不整库驻留GPU。训练每100步记录Loss/grad/max_case/output_max/门值范围及样本吞吐，完整epoch单独报告。

后台流水线：训练正常早停→冻结唯一dev Loss-best及源码/成员/权重哈希→自动一次同固定dev500、同E98.depth_budget.v1单线程完整评价→三Q、草稿/最终硬合法、重叠/外框和完整均值/P95/最大耗时/250ms超限→同100例图/报告→停止。评分、绘图、磁盘、初始化与批量吞吐分开，不自动打开页面。测试载荷封存，不按Q重选或继续训练。

对照只读复用E104已完成best81/dev500。E104在step20350改变过forward并重建了Loss基准，E105从头固定方案一，因此历史结果对照不能严格分离结构效应；没有获得额外控制训练的明确授权，本次不自动新增控制候选。成功只能先归于方案一整体，不孤立归因为门控或SwiGLU。

启动：powershell -File experiments/E105_gated_prenorm_placer/launch.ps1。运行目录runs/20261105_gated128x16_seed0，heartbeat.json/progress.json/pipeline_state.json实时状态，stdout.log/stderr.log日志；正常结束报告在evaluation_depth_dev500。STOP写在本实验或该run目录，批次边界保存后退出并阻止后续评价。error不自动恢复；身份相同且未结束的显式恢复使用train --resume。旧E98 STOP和其他实验不受影响。
