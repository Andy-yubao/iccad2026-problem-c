# 工作区结构

更新：2026-11-04。experiments保留E101最终C5求解器、用户另行授权的E102条件图像兴趣实验，以及用户明确从history恢复的E98 Loss动态调权研究；各实验源码／数据任务／运行独立。其余旧实验、旧输出及辅助脚本归history/20261103_final_delivery_consolidation，旧记录原路径由该目录manifest解释。E98的后续迁回映射及逐文件哈希另见[E98迁回清单](../history/20261103_final_delivery_consolidation/E98_restore_20261104.json)，原归档manifest保持不变。运行中的最终源码、缓存、baseline和best/latest保护。执行边界见[协议](protocol.md)，工程路径见[GPU训练](gpu_training.md)。

| 位置 | 唯一职责 |
|---|---|
| README.md | 当前方向、保留基线、状态与入口 |
| AGENTS.md | 简短现行边界，不保存过期授权 |
| docs/ | 当前可复用协议、数据、结构、Git与训练工程 |
| experiments/En_xxx/ | 该实验的模型、Loss、算法、方案、入口 |
| experiments/En_xxx/runs/ | 命令、身份、源码快照、日志、权重、坐标和报告 |
| datasets/、scoring/、visualization/ | 共享数据处理、官方评分、纯展示绘图 |
| protocol/ | 冻结索引与当前路径配置，不随清理重写 |
| references/、third_party/ | 官方资料、阅读材料、固定上游源码 |
| history/ | 旧实验和文档原件、迁移清单，不恢复旧计划 |
| output/ | 面向交付的报告；历史报告明确日期 |
| .conda/、.cache/ | 独立环境及可重建缓存 |

E98当前新增三个互不覆盖的源码入口：generalization_baseline（当前后台从头训练，源身份不随其他准备工作变化）、depth_evaluation（新计算深度后处理，原qualification/E101不变）、dynamic_weight_learning_v2（冻结父模型、双候选15轮、1.2扰动、单次100/1000反馈，仅准备）。原dynamic_mixedq运行保持STOP。所有候选best/latest独立保留，不自动转入下一训练阶段。

模型/Loss/搜索/合法化按实验独立维护，允许复制并记录来源，不跨实验导入私有算法；数据和评分模块不反向依赖实验。布局图统一调用visualization。

同一研究问题的修复、控制、种子与运行使用同一实验目录。结构/表示根本改变再开新En目录，不预建空目录。新运行用日期/变量/seed；原运行和checkpoint不覆盖。

运行config记录源码提交/未提交快照、数据与面板哈希、来源、Loss和引擎、种子及预算；results记录实际停止原因、质量/合法性/时间与失败。完整坐标保留一个主来源，图从保存坐标生成，避免重复整套结果。

E101新历史链路位于`runs/20261103_replay`：顶层保存计划、控制脚本和证据；`s42`/`s74`/`sf`为隔离源码与输出，使用不复制数据字节的文件硬链接，保留原数据路径校验。三个生成运行根不入Git。旧64k运行及旧delivery保护，新模型在该bundle的`sf/experiments/E101_final_c5_solver/delivery`独立导出；完整自动链路见其README。

用户2026-11-04另授权`runs/20261104_chain64k_seed0`，三阶段均用同一冻结64k成员，随后dev500及原2000例重复测试。其控制脚本/计划/短检证据独立保留，生成的`s42`/`s74`/`sf`同样不入Git；仅此新目录内的数据入口/教师扩展/I/O实现适配，不修改前一运行的冻结源码和权重。

清理顺序：核实活跃进程→列出保留/引用→保存文档和源码原件→同盘归档→文件清单/大小/哈希校验→更新当前导航。移动或递归删除前确认绝对路径在工作区目标内。旧JSON、日志、命令、哈希保持原样，由manifest解释新位置；不批量替换历史路径。

保留best/latest及唯一导出、有效数据缓存、原始数据和环境。只删已确认可重建且无运行占用的缓存、空临时物和真实重复副本，不因目录名含old就删除。Git已有改动不重置；归档源码也需显式检查后入库，权重/数据/环境不入库。

E98调权研究已停止并放弃，所有STOP与权重证据保留，不自动恢复。新增E103_offline_loss_alignment独立研究区，assets保存五个完成训练模型的best副本，sources保存对应推理源码，evidence保存原训练结果。当前仅模型整理与汇报，不启动1000×5数据库或训练。

E103用户已确认五模型并授权自主研究、后台长任务及4路并行：同1000训练题×5份原始草稿，E98固定计算深度后处理，600/200/200题目隔离的权重拟合与40题局部下降检查；状态见experiments/E103_offline_loss_alignment/runs/20261104_bank1000x5_seed0/state.json。E98不恢复，测试封存。

用户最新明确授权E103使用第一轮拟合的新Loss，从E98数据增强best71续训一个候选：LR从1e-3开始、无下降保护、取消15轮且不设轮数/时间上限，保留正常早停、原16k/固定train-dev500、增强与Dropout。LR四档1e-3→2.5e-4→1e-4→5e-5，epoch0基准，连续3轮停滞降档、最低档3轮停滞早停；best只依据新验证Loss。新run为experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0，训练入口finetune_learned_v1；原E98 STOP、父模型和离线证据保护，不自动后处理Q或读测试。

E103最新授权：仅使用E98增强best71和当前新Loss best29进行一次权重拟合。原1000训练题保持不变，增强版结果复用；当前版只有已完成dev500，故补齐同1000训练题后处理，不把验证Q拿来拟合。四路后台补齐、固定lambda1仅fit600一次拟合，200/200只报告，结束保存提案即停。入口experiments/E103_offline_loss_alignment/pair_weight_refit，状态runs/20261104_pair_enhanced_current_v1/state.json。


用户2026-11-04另行授权[E104扩容实验](../experiments/E104_scaling32k/README.md)：E98泛化增强设计从随机初始化扩大为冻结train32k、128宽×16层、3,463,747参数，原固定Loss/增强/Dropout与正常LR早停保持，无硬时间/轮数上限。准备完成后后台训练，正常结束冻结唯一Loss-best，自动同固定dev500与E98.depth_budget.v1评价并报告，用户已确认不是封存test；不按Q重选或追加候选，不恢复旧E98搜索，不改变E101/E102/E103。状态入口experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0。


E104用户已在准备完成后明确要求“启动”：解除原准备暂停，后台从头训练32k/128×16唯一候选，正常早停后冻结唯一Loss-best并自动同固定dev500评价。原Loss、增强、调度与无硬预算配置不变，不读取封存test，不追加候选。

E103最新研究目标已纠正为跨题目全局Loss–Q相关性：旧排序数据不用于新分析；增强best71/当前best29各负责新1000道不同训练题，共2000独立观察，排除旧1000题，所有草稿/Q重新计算。1400拟合/600检查，样本等权全局非负回归、相关性和图，保留问题间差异，不进行同题排序。入口experiments/E103_offline_loss_alignment/global_loss_q，输出runs/20261104_global_independent2000_seed0；测试封存、不自动训练。

用户新增并明确授权E103/reference075_posttrain小实验：E98原增强best71，position=0.75、shape=0.1875、其他8项原值，不归一化；LR从倒数第二档1e-4至5e-5，无15轮或时间上限，按验证Loss正常早停；后台训练→冻结唯一best→自动E98固定计算深度dev500→报告即停。已有父模型500结果及已通过精确复现桥复用，测试封存。run为experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0，总体状态pipeline_state.json，日志reference075_posttrain/stdout.log。


E104于2026-11-05获用户明确授权修复数值错误并续训：step20350原未归一化局部消息路径触发单例Loss门禁，登记local-prenorm.v2迁移后继续；模型参数/Adam/顺序/游标/RNG/LR/累计时间保留，固定Loss和门禁不变，新路径重建best与平台基准，旧证据归同run的repair_20261105/before。此后结果包含路径修复，不称纯缩放对照；正常早停后仍单次固定dev500，不读测试。


2026-11-05用户明确授权[E105方案一](../experiments/E105_gated_prenorm_placer/README.md)：128×16完整分支pre-norm、四路残差门控与SwiGLU，从头随机seed0；其他数据/增强/Dropout概率/Loss/batch/精度/优化器/LR/早停及完整固定dev500评价与E104保持一致。准备与五例最小短检通过后直接后台训练，正常早停冻结唯一best自动评价并报告即停；仅一个候选，不自动补控制或方案二/三、不读测试。成员/源码/方案对齐见non_architecture_alignment.json，运行runs/20261105_gated128x16_seed0。

新增E106_postprocessor_aware_loss，源码、计划、模型推理副本与runs独立归属。现行v2：5000题×5=25000草稿、约210万参数，只有实现和数据准备，无训练。被用户替换的16000份准备稿保存在本实验revisions/20261105_prepared16000_superseded，不作为现行方案；没有覆盖旧实验或恢复旧搜索。

E106最新已获“数据生成完后自动启动训练”明确授权；依赖衔接控制代码与独立状态放在runs/20261105_bank25000_seed0，不改运行中的数据/科学源码身份。仅自动启动一个代理训练，布局模型不训练。


2026-11-05用户授权[E107几何后训练](../experiments/E107_geometry_posttrain/README.md)：继承E105 best58，参考位置/形状降为10%（.1/.025），三个精确并集几何主项＋软约束/线长及局部漂移保护，归一化仅原train500；原32k/增强/Dropout/结构保持，新阶段fresh AdamW，LR1e-4→5e-5、epoch0、无保护/硬预算、正常早停。每批绝对/同题父模型相对Loss爆炸门禁，触发即Adam前拒绝、写STOP并保存退出，不重试或衔接评价。五题短检通过后直接后台启动一个候选，仅正常早停自动固定dev500评价报告即停，测试封存，不影响E106或旧STOP。状态runs/20261105_geometry_ref10_seed0。


E107首轮已因严格相对单例Loss门禁停止：22次接受更新后第23批在Adam前拒绝，单例49.70237超过同题父模型3.766752对应门槛37.667519，数值仍有限。STOP保持、latest/best及失败现场保存，进程退出；未完成一轮、未启动终末dev500，不自动恢复或放宽阈值。见experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/stop_verification.json。
