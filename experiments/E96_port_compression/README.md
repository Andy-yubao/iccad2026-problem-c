# E96：完全压缩的独立全局修正器

本轮已完成并核验：9872步、19个完整epoch，训练19.69分钟后触及步数上限；唯一best为epoch19。dev2000单次交面积1.0854%、平均N5=8.342、B/A=1.10790、原始硬合法0/2000；flash500全部合法，加权Q1.026523，未超过冻结E90。未收敛结论待定，不自动续训。[完整结果检查](runs/20261101_centroid_bbox05_seed0/inspection.md) · [同100例四阶段图](runs/20261101_centroid_bbox05_seed0/evaluation/gallery/index.html)。

用户于2026-11-01确认 E95 失败，并授权采用已接受的权重方案启动后台训练。当前只执行 centroid；regional32 已停用，不排队新候选。[实时状态](state.json) · [固定计划](plan.json) · [权重依据](weight_design_proposal.md) · [授权与旧源码](runs/20261101_bbox05_registration/authorization.json)。

固定目标：`overlap + 0.5*B/A + 0.05*wirelength + 0.2*soft + 0.1*intent`。各项定义与归一化保持原样，反传、固定train/dev各500、best、LR和早停共用同一函数；dev500算术均值选择唯一best。相对位置来自当前L草稿。外框0.5按原五例有限变化代价推导，保守余量是设计选择，并非已验证最优值。

F只保留原17维模块特征里的端口总权重和加权中心，没有独立端口token；完整原始端口/连接仍供实际Loss、评分和后处理。F为192宽/12层/4头，全局交互、一次并行修正；冻结L输入不变。无内部搜索、逐模块自回归或reference自由位置输入。

从同一冻结E90 best12兼容编码器和零输出头初始化，fresh AdamW/seed0，不继承E95修正器或优化器。新运行`runs/20261101_centroid_bbox05_seed0`，不覆盖旧证据。冻结v5 train16k/dev2000，经校验v6前缀缓存读取；train/dev固定500监测、旧L草稿按成员/来源/解码哈希只读复用，测试封存。

FP32/batch32，无AMP/TF32；efficient SDPA及32×120×480公开输入CUDA图，F实际只有120个模块token槽。AdamW lr初始0.001、weight_decay0.0003、裁剪1；四档LR 0.001/0.00025/0.0001/0.00005。epoch0基准、无保护；调度min_delta=max(1e-5,abs(best)*0.001)，3轮停滞下降；最低档至少等待2轮且early_bad>=3后早停。任何dev均值更小仍保存best。

训练最多7200秒（含训练进程初始化、监测和存盘）、9872更新、200轮，任一先到。预检费用另记，不复用E95预算。终末评价另最多3600秒。best/latest分存；latest保留Adam、RNG、数据顺序/游标、seen、调度和累计费用。STOP在批次边界保存并阻止后续评价，数值失败不自动重试。

新源码先通过`runs/preflight_bbox05_cuda_centroid/results.json`：原五train成员，包含固定扩张/收缩与局部重叠梯度几何探针、真实Dropout/Adam更新、同padding eager/CUDA图比较、重建恢复下一步逐位一致与非有限值拒绝。探针只检查Loss行为，不作为F机制或dev扫系数；实施更新不属于正式训练。旧preflight结果原位保留。

终末只评价唯一best：dev2000原始草稿、dev500一次flash、同100例reference/L/F×1/最终后处理图，统一ID与尺度；辅助2/4/8次不选模。报告重叠尾部、外框/空白、线长、公开约束/意图、合法性及时间，Q作辅助。无评分控件、自动开图、自动唤醒或后续候选。

首次启动（后台已经运行时不要重复）：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E96_port_compression.pipeline --arm centroid --execute
```

正常中断在确认停止后使用同命令追加`--resume`，保留累计费用；错误需诊断，不能当作普通恢复。
