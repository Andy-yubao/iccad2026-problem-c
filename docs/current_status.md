# 当前状态快照

2026-11-08 E114大预算评价及最差10例展示已完成：原固定dev500、同best48、500份草稿逐位一致。工作上限乘3后加权Q1.047151→1.021820、平均Q1.028120→1.021457、混合Q1.037636→1.021638，500/500硬合法、零行回退；逐例143改善/318基本相同/39退化。完整均时176.0→341.8ms，P95 1027.3ms、最大3654.0ms、82例超过500ms，500ms为平均目标而非逐例硬截止。本次最差10按新最终Q降序、index稳定破同分，indices为358/481/351/174/34/165/60/460/215/4。[三栏逐层展示](../visualization/E114_supervision/20261108_budget3x_worst10/index.html)只读复用全部最终坐标，5例旧逐层缓存＋5例补推理，无新增后处理/Q/test；三栏/全层同尺度同ID、G组配色/无组灰色、G/F/P及L/R/T/B状态、底部图例，无MIB。120状态通过离线DOM/canvas交互检查，已检查第一例静态预览，未声称实际浏览器交互检查。[全量预算对照](../experiments/E114_supervision_placement/evaluation_budget500ms/output/comparison_report.md)。未更换默认后处理或研究基线。下方“后台运行”为启动快照。

2026-11-08当前后台评价：E114 staged 4+12 best48、原固定dev500增大后处理预算。预登记3倍工作上限，以平均完整约500ms/例为目标，不强制截断；原模型与原500例评价保持，独立原规格硬检查/无MIB本地Q。PID28516，启动时已载入500例、正在初始化；后续以[实时进度](../experiments/E114_supervision_placement/evaluation_budget500ms/output/state.json)为准。[stdout](../experiments/E114_supervision_placement/evaluation_budget500ms/stdout.log) · [stderr](../experiments/E114_supervision_placement/evaluation_budget500ms/stderr.log)。无新训练/test/预算扫描，完成后保存与原结果的全量成对比较。

2026-11-08 E116已完成：两组均正常早停、无错误，original4完成37轮/18500步、best34，train/dev500 Loss 0.444695/0.468609；new_first完成51轮/25500步、best51，0.446312/0.470894。新版dev Loss高0.488%，记录每步42.57→28.56ms；参考/紧凑度Loss改善但重叠惩罚更高。新版best在末轮、两组均未观察到train平台，不能称充分拟合；不自动续训或改门禁。只读训练日志复核，无新增推理/Q/test；E114基线、E101交付不变。[完成复核](../experiments/E116_first_draft_comparison/result_review.md)。下方启动/准备记录为历史快照。

2026-11-08 20:59（北京时间）E116启动核验：original4已完成3轮/1500次真实更新，latest包含82组Adam状态、完整RNG/游标/调度与相同步数的步长保护；模型参数已变化，准备通过的核心源码身份保持一致，错误日志为空。当前已选best epoch3、固定dev500 Loss0.755625仅为早期监测，不是终末比较。new_first尚未启动，等待第一组正常完成。[核验记录](../experiments/E116_first_draft_comparison/launch_verification.json)；后续实时状态仍以队列/run为准。

2026-11-08 E116最新启动：[原四层→新版首稿](../experiments/E116_first_draft_comparison/RUNNING.md)已获明确后台训练授权并派发，original4先运行，new_first等待其正常完成；固定16k，各4h/200轮，原模型/Loss/精度/批量/门禁不变。启动时处于第一组数据准备阶段，后续实际步数以[队列](../experiments/E116_first_draft_comparison/queue_state.json)及run heartbeat/progress为准。错误/用户停止不推进，无Q/后处理/test；结束仅汇总已存Loss。下方“准备完成、训练关闭”是此前快照，原件已保存在启动登记。

2026-11-08最新：[E116独立首稿对照](../experiments/E116_first_draft_comparison/README.md)准备完成。原E114结构前4层207987参数，对照新版4个Attention＋SwiGLU隐藏推理块、一次全局条件读出209891参数；两组都删去全部后续网络，只监督最终首稿，使用冻结random16k与共同完整Loss。五个训练成员、batch32/FP32前后向/CUDA图检查通过；44个公共初始张量及全局CPU RNG一致，原模型输出复现，新版头仅一次读出。零Adam更新，训练开关关闭、E116 STOP保留，没有训练run/checkpoint/队列，不读dev/test载荷、不做Q。首次独立入口缺失CuBLAS确定性设置的失败已留档并针对性修复；通过记录见[E116 preflight](../experiments/E116_first_draft_comparison/preflight.json)。这不是训练或质量结论，E114基线及E101交付不变。

2026-11-08：E114三组与E115已完成，用户选择4+12为后续主要基线（[v2](../protocol/research_baseline_v2.json)）。加权Q/尾部坏例、四模型训练用时及单例入口干预见[最新诊断](../output/supervision_q_analysis_20261108/report.md)；[4+12固定5例＋最差5例三相图](../visualization/E114_supervision/20261108_staged_three_phase10/index.html)。全500结果与旧实验证据保持，不重训、不读test。

2026-11-08最新：用户选择E114 staged（4+12、best48）作为后续主要研究基线，替代E109 A的主要地位；旧身份保留，E101交付不变，未来默认16k不改，不启动新训练。见 `protocol/research_baseline_v2.json`。当前展示改为左reference／中间滑块切换1–12层真实draft／右固定后处理结果，固定5个训练例＋已完成dev评价最终Q最差5例，同色同尺度，不自动读test或重评整套；见 `protocol/three_phase_display_v2.json`。E114/E115已实际使用轴交换P=.5、Dropout .1/.05及AdamW decay .0003，不能记录为“未做增强”。


## 2026-11-07：4+12恢复→E115九层1+9已准备并获后台启动授权

用户明确两组均用原36k、原train/dev各500，与之前terminal/dense一致。4+12直接恢复E114 staged完整latest，已消耗513.41秒不清零；E115独立大首头64→256→256→256→3，9图块/64维，.2L1+.8L9，609682参数。共用初始化172张量及全局RNG精确一致；五个原训练成员一次受保护更新通过，未预演恢复、未读dev/test。新队列仅这两组，错误/停止不推进、无自动Q。已完成两模型及原旧三组队列/QUEUE_STOP保留；未来默认16k不变。[计划](../experiments/E115_first_head_supervision/plan.json) · [实时队列](../experiments/E115_first_head_supervision/queue_state.json)。后台派发状态以launch.json/queue_state为准，不将准备写成实际参数更新。


## 2026-11-07：E114训练已暂停，验证两个已完成模型

terminal正常早停61轮/68625步、own-Loss best59；dense正常早停53轮/59625步、best52。staged在数据准备阶段step0、epoch0保存完整latest并退出，累计513.37秒保留，Adam尚为空、未更新，未清零预算。队列和训练进程均退出，E114 STOP/QUEUE_STOP保持；根训练授权关闭，不自动恢复。用户仅授权前两组固定dev500验证，测试封存。

两个best在Q前冻结，评价用原random40k固定dev500，原始面积/无MIB/全模块真实草稿；同复制E98.depth_budget.v1后处理，输入副本仅MIB列置零、原面积和硬规格保留。五个原E114训练成员的真实13次共享头读出已保存，CPU FP32 hooks前后输出逐位一致，权重未变，统一ID/颜色/跨层模型尺度。[暂停核验](../experiments/E114_supervision_placement/pause_verification_20261107.json) · [验证计划](../experiments/E114_supervision_placement/evaluation/plan.json) · [五例逐层查看](../visualization/E114_supervision/20261107_two_models/index.html)。两组固定dev500验证已完成：末层/密集加权Q1.053574440/1.045991903，平均Q1.030172463/1.028682065，混合Q1.041873451/1.037336984；各500/500最终硬合法、草稿0/500、零回退。共同末层Loss.350673211/.341550118，草稿重复覆盖6.2146%/5.0130%；完整均时/P95为196.827/471.696与176.364/406.724ms。密集训练耗时约1.93倍，最差Q3.2281略差于末层3.1641；逐例Q胜/平/负186/147/167。单seed开发结果，旧数据/任务不直接排名；第4/12层尚0更新，不能判断稀疏监督优劣。[完整报告](../experiments/E114_supervision_placement/evaluation/report.md)。

## 2026-11-07：E114之后固定16k训练子集入口已准备

用户要求在第4/12层监督实验之后的新实验用16000训练成员，从现有36000训练池挑出。已按seed2026110716000均匀无放回冻结一次子集，后续复用；训练监测500从子集内独立固定抽取，原dev2000/dev500和封存test2000不变。原NPZ/PNG只读复用，不重渲染、不改变面积/规格/无MIB/全模块预测规则。当前E114三组继续原36k，未更改其训练源码、配置或队列。

处理入口为 `python -B -m datasets.random_reference16k prepare` / `check --payloads 5`；已建立索引，prepare再次执行会拒绝覆盖。16k唯一成员/父集合归属/确定性抽样/面板及父元数据校验通过，5个train载荷校验通过，dev/test载荷读取均为0，test reader拒绝。E114实施门禁对应源码哈希仍一致；未启动任何新模型训练。[规则](../protocol/research_training16k_v1.json) · [冻结索引](../protocol/random_reference16k_v1.json) · [处理检查](../protocol/random_reference16k_preparation.json) · [使用说明](datasets.md#后续16k训练子集)。

## 2026-11-07：E114已获后台启动授权

用户明确要求后台启动。terminal→dense→staged固定串行，原每组4h/200轮。仅启用已准备配置，不改数值路径，不自动Q或测试；错误/停止阻断后续。旧准备原件保存在launch_registration_20261107；实时进度见[队列](../experiments/E114_supervision_placement/queue_state.json)和各run heartbeat，启动核验另存startup_verification.json。E112/E113仍暂停。


## 2026-11-07：E114：仅准备小模型监督位置对照，训练禁用

用户要求准备入口，不启动训练。三组共用64维/12图块/共享head，删除MIB通道后612307参数：terminal=L12；dense=.8L12+.2mean(L1..11)；staged=.8L12+.2L4。相同seed0初始化、batch32/FP32、原增强/优化器/门禁/调度；各4小时/200轮上限只是保护，不称等训练量。

按当前新实验规则接入已完成random40k的train36000和固定train/dev各500监测，原始面积、无MIB、全模块预测，固定/预放置规格只作条件；这不是与历史A/E105/E74同任务对照。只读五个train成员实施检查通过：三组同初始权重和输出、标签隔离、全模块预测/监督、首末层梯度、CUDA计算图捕获重放；零Adam更新，没有dev/test载荷、恢复演练、checkpoint或训练run目录。

authorization.training_enabled=false，实验STOP保留，无自动队列/评价。三arm入口实测均在数据读取和GPU初始化前拒绝；旧E112/E113 STOP不变，不恢复。后续启动须按用户新授权登记；终末Q和五例展示未运行。[说明与命令](../experiments/E114_supervision_placement/README.md) · [检查](../experiments/E114_supervision_placement/preflight.json) · [准备核验](../experiments/E114_supervision_placement/preparation_verification.json)。


## 2026-11-07 12:20：用户暂停E112/E113队列

已先设置E112 QUEUE_STOP阻断后续，再设置候选STOP安全停止。E112于17519步、已完成35轮／新一轮cursor608保存完整latest并退出；best仍为epoch34。模型、Adam、Python/NumPy/Torch/CUDA RNG、16000排列/游标、调度、参数更新保护和累计14364.2秒均保留，原4小时预算只剩约35.8秒，未清零。训练error为null。

训练器2036与队列20748已核实退出，GPU计算进程为空；E113未启动，dev500与五例提取也未启动。E112 STOP/QUEUE_STOP、E113 STOP保持，不自动恢复。队列原始needs_attention记录是其拒绝用户停止状态后阻止后续任务的退出记录，不是数值训练失败，原件保留。[暂停核验](../experiments/E112_deep_geometry_budget/pause_verification_20261107.json)。

## 2026-11-07：E112/E113参数预算重分配准备

用户明确授权两个顺序运行的实验，以A为既有对照，不重训。两组8个外层图块、96维主干，将参数重新分配给更深关键MLP：几何19→168→168→168→4共1466915参数；共享头96→472→472→472→3共1469107，A为1471539。两组参数相差A不到0.4%，不称等算力或单因素深度消融；基础任务沿A原16k/MIB/固定通道，40k不参与。

准备范围包括同五例一次真实更新短检、独立源码、原固定dev500后处理、自动五例逐层展示；不生成100三相图。实际启动/阶段见[队列](../experiments/E112_deep_geometry_budget/queue_state.json)，[展示入口](../visualization/critical_budget/20261107_pair/index.html)在训练和提取完成后更新。只读通用钩子与变层数展示已放在visualization，不自动打开界面。

后台已启动并核实E112完成初始train/dev各500监测、进入真实参数更新；两组五例短检均通过，参数数目及共同初始化已检查，新增残差子网和首/末图块均实际更新。错误日志为空，A权重/结果与旧冻结索引哈希不变；E113及各自验证/展示已排队。[启动核验](../experiments/E112_deep_geometry_budget/startup_verification.json)。展示目前只有已存在的A帧，新模型完成后加入，不将准备状态当生成结果。

## 2026-11-07：A选为重要研究基线；关键子网络深度仅提案

用户认为前层学习问题已有效改善，选择E109共享head逐层监督A为后续重要基线（best epoch61、sha前缀5123c81c），不替换E101交付。[身份](../protocol/research_baseline_v1.json)。已存dev500的A原Loss从第8层.380948降到第12层.341899，末四层仍有约10.3%收益；五个训练例重复覆盖从6.763%降到4.392%。后层位移小不等于没有贡献，也未证明深度不足。

提出优先单独加深19→24→4几何关系偏置网络，保留A逐层监督和外部结构；独立加深坐标头、关系向量消息及较低几何刷新频率为备选。[提案](../output/critical_network_depth_proposal_20261107.md)。未实施或启动新训练/评价，不自动新增候选，不因基线选择追改任何旧结果。

## 2026-11-07：三组完成；展示改为同五例逐层对比

E109、E110、E111三组训练与各固定dev500验证均已完成；不再运行后续训练或评价。用户要求当前入口改为同五例逐层输出，取代100例三相图。新入口为[五例逐层对比](../visualization/layer_comparisons/20261107_same_train5/index.html)：原E98、A、B、E111最终12层模型可同层并排，也可切换单模型逐层播放；另外提供E111第4/8层阶段各自best。五例沿用先前N=21/45/70/95/120的训练成员，没有按结果挑题；统一ID、颜色和跨模型/跨层尺度。

本次仅CPU FP32 eval/no_grad读取保存best，模型前向未改写；hooks前后末层输出逐位一致，权重/checkpoint未变，无重新训练、后处理或Q评价。原E98同身份五例结果直接复用，100例历史图集保留。两个早期阶段checkpoint与最终模型前缀明确区分；五个训练例不支持总体泛化结论。

## 2026-11-07：E109／E110／E111准备与后台队列

用户授权A共享head逐层监督、B独立head逐层监督，以已完成E98泛化增强版为历史对照且不重训。新40k搁置；本次保留原16k/MIB/固定规格通道条件，A仅改层间Loss，B另有用户确认的独立head。两组初始权重/末层前向与原E98精确一致，各完成五例一次真实更新短检。E111追加为4→8→12层，只训练新增四层与共享head，旧层权重冻结，只用当前末层Loss；三个阶段在同五例上各一次更新验证了冻结参数不变、新层及head更新。没有预演断点续训。

已准备A→B→E111三阶段→三组固定dev500终末验证的有界串行队列。中间阶段仅正常早停后推进；失败/用户停止不自动继续。当前真实运行状态以[队列](../experiments/E109_deep_supervision_shared/queue_state.json)和各run heartbeat为准，不将“准备”写成完成训练。全部旧STOP保留，无测试读取。[A](../experiments/E109_deep_supervision_shared/README.md) · [B](../experiments/E110_deep_supervision_heads/README.md) · [分阶段组](../experiments/E111_progressive_depth/README.md)。

后台启动已核实：A完成初始train/dev各500监测并进入真实参数更新，错误日志为空；B、E111及其后各500例验证已排队。三个短检来源门禁均与冻结源码一致，E98原best/结果及旧split/成员哈希未变。[启动核验](../experiments/E109_deep_supervision_shared/startup_verification.json)。

## 2026-11-07：E98五例逐层诊断完成，E108数据准备完成

用户明确要求先观察中间布局再讨论结构。E98增强版best71在五个原登记训练样例（N=21/45/70/95/120）上取出初始及12层共13次共享head读出；仅CPU FP32 eval，未训练、未读取测试、未做后处理/Q。带/不带观察hooks的最终输出逐位一致，权重未变。图示第6～7层附近明显展开，后四层仍显著减少剩余重复覆盖；不是12层多数冗余或表达能力不足的因果证明。[报告](../experiments/E98_loss_weight_learning/runs/20261106_layer_diagnosis5/report.md) · [逐层查看](../visualization/E98_layer_diagnosis/20261106_best71_train5/index.html)。历史MIB登记输入忠实保留，此结果不是新无MIB模型。

上次E108后台任务已于2026-11-06 23:24完成40000张：train36000/dev2000/test2000，538.297秒；文件数量、来源/输出校验及旧索引不变检查通过。test仅制作，仍封存，没有训练或模型评价。[完成报告](../experiments/E108_random_reference40k/runs/20261106_random40k_v1/report.json)。

## 2026-11-06：E108独立随机40k数据准备

用户明确更新后续实验：保留原始面积与规格，忽略MIB且不计MIB软罚分；新数据为36000训练/2000验证/2000测试，重新随机完整分片及抽样，不沿用旧名单/筛选。已冻结新索引，五例训练样本原始NPZ及512 RGB图片实施检查通过；单例评分适配检查确认MIB罚分归零、原规格与硬判定未变。仅授权后台数据准备，未授权训练或测试评价。测试制作完成后仍封存；全部旧STOP保持。[E108说明](../experiments/E108_random_reference40k/README.md) · [实时状态](../experiments/E108_random_reference40k/runs/20261106_random40k_v1/state.json) · [授权](../experiments/E108_random_reference40k/runs/20261106_random40k_v1/authorization.json)。

已核实后台实际生成：启动检查时完成3550/40000（train3180/dev196/test174），错误日志为空，旧冻结索引哈希未变；这是运行中快照，非完成报告。[启动核验](../experiments/E108_random_reference40k/runs/20261106_random40k_v1/startup_verification.json)。最终数量和完整校验由完成后的report.json记录。

## 2026-11-06 22:25：E102用户停止续训

最新用户要求停止，已在25614步／seen77860（约4.866轮）安全保存完整latest；Adam、RNG／遮罩RNG、排列游标、调度和累计24922.651秒均保留，best dev500 Loss为1.550133。训练器、控制器及曲线伴随进程均已退出，GPU计算进程为空，错误日志为空；新旧STOP全部保留，不自动恢复。[停止核验](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_resume5_seed0/pause_verification_20261106.json)。下文为此前授权及运行历史。

## 2026-11-06 21:02：E102明确授权有界续训

用户在曲线评估后要求“续训吧”。新目录从暂停的20193步／seen56176完整latest恢复，原暂停run及全部STOP保留；BF16、batch4、Loss、Adam／RNG／顺序／调度和原累计8小时预算保持。约5轮完整监测后安全停止，正常早停／预算／用户停止优先，不自动追加阶段。[本次登记](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_resume5_seed0_launch/README.md) · [实时状态](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_resume5_seed0_launch/state.json)。沿用原图片入口并恢复曲线伴随更新；无computer-use、test或几何处理。下文为此前暂停和启动历史。

## 2026-11-06：E102单次后台启动

**最新：用户要求先暂停BF16＋batch4训练，已于20193步／seen56176／第4轮cursor8176保存完整latest，控制器正常退出、GPU释放。** 原Adam、RNG／遮罩RNG、排列游标、调度和累计19936.632秒保留，run和launch的STOP保持，不自动恢复。[暂停核验](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_seed0/pause_verification_20261106.json)。没有使用computer-use；下文迁移和启动记录仅追溯此前顺序。

**BF16迁移续训已核实实际更新并保存新checkpoint，固定100例迁移后预览已生成，错误日志为空。** [核验与实测吞吐](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_seed0_launch/startup_verification.json)。近期更新吞吐约4.1张/秒，不能将其当作完整轮次加速。

**当前：用户已明确授权BF16＋batch4迁移续训。** [迁移方案与四例短检](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_seed0_launch/README.md) · [实时阶段](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_seed0_launch/state.json)。独立新run从完整12297步latest继承Adam／RNG／成员顺序／cursor8594及原累计11778.767秒；原总8小时不重置，旧RGB与FP32目录STOP均保留。网络BF16，Loss／主参数／Adam FP32，逻辑batch4；模型Loss数据不变，下一监测按seen28000保留每约4000成员节奏。迁移基线对原固定best重算BF16 dev500及继承latest train/dev500，校准数值标量，不推进调度／停滞计数。四例短检一次更新通过，Loss差约0.094%，峰值约13.4GiB；稳态吞吐以新run日志为准。以下停止与早期启动描述只是此前顺序，不覆盖本次迁移授权。

**最新：用户明确终止新版semantic训练，已在12297步／seen24594／第2轮cursor8594保存完整状态，控制器正常退出、GPU释放，run与launch的STOP保留，不自动恢复。** [终止核验](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_seed0/stop_verification_20261106.json)。旧RGB也仍停止；以下授权与启动记录仅追溯发生顺序。batch2是两套布局题一次Adam更新，双视图不增加独立样本数；实测峰值allocated约7959.9MiB，未实测更大batch的吞吐／优化对照，不能称最优批量，也不能将效果不足单独归因于batch。

**新版启动已核实：初始固定train/dev500监测已完成，超过100次真实更新，固定100例早期图集完整生成，错误日志为空。** [启动核验](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_seed0_launch/startup_verification.json)。尚无训练后固定dev500监测，不能据此判断泛化。

**当前新增：用户已明确要求启动新版以观察泛化。** 唯一新运行[20261106_semantic_v21_seed0](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_seed0_launch/authorization.json)，[实时后台阶段](../experiments/E102_reference_image_generation/runs/20261106_semantic_v21_seed0_launch/state.json)。16k原始成员、512／batch2／FP32、从头seed0，双视图总Loss保持v2.1；初始及每2000更新／每轮结束固定train/dev500，同总Loss选best／调LR／早停，上限100轮或累计8小时。图片用全MASK单次生成，固定dev100，第100步及每次Loss-best发布于[新图集](../visualization/E102_reference_image_generation/20261106_semantic_v21_seed0/index.html)。测试仍封存；dev属于历史开发面板，不称独立测试泛化。新旧权重、缓存、运行不混用，旧RGB STOP保持。下文暂停和单图检查记录描述此前发生顺序，不覆盖这次新版明确授权。

**最新：用户要求暂停，已在4369步／第三轮cursor2952安全保存并核实退出，GPU释放，STOP保留，不自动恢复。** [暂停核验](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0/pause_verification_20261106.json) · [Loss及架构初步判断](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0/pause_review_20261106.md)。以下启动／修复信息是发生顺序，不能当成仍在训练。

后续按用户要求已重新设计模型／Loss，实施代码在[E102 semantic](../experiments/E102_reference_image_generation/semantic/README.md)。同一张训练图的250步拟合：v2首版IoU0.551未通过；v2.1修改为每步全MASK／随机MASK双视图监督、边界直接耦合模块概率后，直接输出IoU0.8245、前景准确率91.78%。16轮采样仅0.5777，保留但不作默认。这是训练侧有界学习能力诊断，不是正式训练／未见泛化。旧STOP不移除，新版正式长训练未启动。

后续故障修复：1192次更新后因Windows心跳JSON替换WinError 5退出；按用户修复请求，从完整1100步latest保留全部训练状态实际恢复，92步重跑，原未保存耗时20.304秒补入预算。仅修改文件I/O，原错误保留；当前日志见launch目录`train_resume_iofix.*.log`及run内`updates.jsonl`，最新阶段仍看launch/state.json。[修复、验证和迁移](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0/engineering/20261106_iofix/report.md)。

用户已明确授权E102后台启动及visualization图片对比。唯一运行登记为[20261106_rgb512_canvas_v2_seed0_launch](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0_launch/authorization.json)，实时阶段看[启动状态](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0_launch/state.json)，实际更新看[训练heartbeat](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0/heartbeat.json)。数据准备期间尚无训练heartbeat，不把prepare当成模型更新。

2026-11-06 10:01左右已核实训练完成100次以上真实GPU更新，固定dev100早期预览100例及全部成对PNG/HTML已写出；见[启动核验](../experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0_launch/startup_verification.json)。目前仍是首轮内早期训练，尚未有完整dev500选出的best。

512×512／batch8／FP32／seed0，v5原始成员与原始面积，固定train/dev500图片MSE，100轮或累计8小时上限。第一次正式准备在一个训练例上发现v1画布超界，未发生训练；失败目录保留，v2仅将公开端口范围纳入画布，不依reference缩放。没有增加候选、重启其他实验或读取test。

图片入口：[visualization/E102_reference_image_generation/20261106_rgb512_canvas_v2_seed0/index.html](../visualization/E102_reference_image_generation/20261106_rgb512_canvas_v2_seed0/index.html)。第100次更新发布固定dev100早期预览，每次固定dev500 Loss-best改善后保留并更新同例对照。仅图片相似程度和边界观察，不作MIB同形、几何提取、修复、后处理或Q。

以下2026-11-05表格是原核对截点，E102“未启动”记录已被这次单运行授权和实时状态更新，不扩展其他完成／停止任务的权限。

核对：2026-11-05，北京时间。进程快照为15:58:51；当时python/pythonw/conda进程数0。CIM命令行查询在本机不可用，实际使用Get-Process名称核查；没有把状态文件的旧PID当活进程，也没有停止其他应用。此页不是实时监控；新执行前重新查状态/进程。

[进程原件](../history/20261105_workspace_handoff/process_snapshot.json) · [18份运行状态/停止证据快照与哈希](../history/20261105_workspace_handoff/run_state_snapshot.json)。原run仍是最终事实来源；静态计划/授权说明允许做什么，不证明正在做。

| 工作 | 核实状态 | 决定性原件 |
|---|---|---|
| E101首次64k暖启动 | complete，训练和已登记2000完成 | [state](../experiments/E101_final_c5_solver/runs/20261103_final64k_seed0/state.json) |
| E101历史链路16k→32k→16k | complete，训练/dev500/重复2000完成 | [state](../experiments/E101_final_c5_solver/runs/20261103_replay/state.json) · [报告](../experiments/E101_final_c5_solver/runs/20261103_replay/report.md) |
| E101三阶段均64k | user_stop_verified，E42 step63561，后续未启动 | [停止核验](../experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/stop_verification_20261104.json) |
| 原C5补test2000 | complete，固定028ea3…权重，不重训/不Q选模 | [state](../experiments/E101_final_c5_solver/runs/20261104_original_C5_test2000/state.json) · [报告](../experiments/E101_final_c5_solver/runs/20261104_original_C5_test2000/report.md) |
| E98旧调权 | 用户停止并放弃，STOP不解除 | [train500停止核验](../experiments/E98_loss_weight_learning/runs/20261104_dynamic_train500_v3_seed0/stop_verification_20261104.json) |
| E102图片兴趣 | 五例分辨率/CPU更新短检，正式长训练未启动 | [README](../experiments/E102_reference_image_generation/README.md) · [五例结果](../experiments/E102_reference_image_generation/runs/20261103_cpu_update5/result.json) |
| E103 1000×5 | complete，5000稿/离线拟合/40题方向检查完成 | [报告](../experiments/E103_offline_loss_alignment/runs/20261104_bank1000x5_seed0/report.md) |
| E103 learned_v1 | 32轮早停best29，后续固定dev500完成 | [训练state](../experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/state.json) · [评价state](../experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500/state.json) |
| E103 pair refit | complete，仅保存提案，后被全局统计目标替代 | [state](../experiments/E103_offline_loss_alignment/runs/20261104_pair_enhanced_current_v1/state.json) |
| E103全局2000 | complete，1400拟合/600检查及图完成，系数未训练 | [state](../experiments/E103_offline_loss_alignment/runs/20261104_global_independent2000_seed0/state.json) · [报告](../experiments/E103_offline_loss_alignment/runs/20261104_global_independent2000_seed0/report.md) |
| E103 reference .75 | 11轮早停best8，固定dev500完成 | [pipeline](../experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0/pipeline_state.json) |
| E104 | 82轮早停best81，local-prenorm迁移后固定dev500完成 | [训练](../experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0/results.json) · [pipeline](../experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0/pipeline_state.json) |
| E105 | 61轮早停best58，固定dev500完成 | [训练](../experiments/E105_gated_prenorm_placer/runs/20261105_gated128x16_seed0/results.json) · [pipeline](../experiments/E105_gated_prenorm_placer/runs/20261105_gated128x16_seed0/pipeline_state.json) |
| E106 | 25000标签完成，26轮/13000更新早停best7，check1000/诊断完成 | [整体衔接](../experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/autostart_state.json) · [代理结果](../experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/critic/results.json) |
| E107 | 22次更新，第23批相对单例门禁停止，未终末评价 | [停止核验](../experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/stop_verification.json) · [报告](../experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/abort_report.md) |

## 容易误读的状态来源

- E106数据生成`state.json`的`data_ready_training_disabled`是纯数据入口完成态；它不涵盖后来独立autostart控制器。原preparation_state/plan中的training=false是准备期快照；后续授权绑定同一科学计划，再训练，整体完成须看autostart_state和critic/results。没有改写旧JSON来掩盖先后关系。
- E104/E105训练results里的“next_stage由pipeline管理”是训练器的静态说明；pipeline_state已complete，固定评价已有报告，不代表还有任务排队。
- E103 learned_v1的训练state写“no_next_stage”，后续用户另授权评价，评价目录也已完成。不能只看训练state推断Q未评价。
- E101“64k”可能指两个不同运行：首次L/F暖启动已完成，三阶段均64k已用户停止。不要把其中一份完成态套到另一份。
- E107原诊断末尾的小范围总位移建议后来被否定；[最新提案](../output/reference_free_optimization_proposal_20261105.md)只是设计，未实施/训练，不构成解除STOP。

旧完成/停止标记和全部best/latest/失败现场均保留。新实验或恢复须依据新的明确任务和登记；本次整理不重开封存测试、不恢复自动唤醒、不追加候选。
