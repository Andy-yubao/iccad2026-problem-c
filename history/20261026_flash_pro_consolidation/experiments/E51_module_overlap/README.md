# E51：逐模块相对重叠、面积重要性与连续个数因子

2026-10-25训练完成：52轮早停，唯一best52，开发统一Loss0.377453166。按用户本轮明确要求准备并后台执行[固定开发100例flash评价](runs/20261025_flash100_seed20261025/)：从dev_monitor500以seed20261025均匀抽100例，2进程各2线程，复用原flash/steady；不启动完整dev2000/终末500，不读封存测试。`progress.json`为实时状态，`summary.json`仅在评价和100张三相图全部完成后生成；`evaluation_summary.json`先保存数值结果。恢复入口为该目录`run_flash100.py`，只补未完成成员。

运行中的[学习率数值与下降倍率审计](runs/20261025_lr_scale_review/analysis.md)：读取E39/E41/E42/E43与当前E51，现有证据不支持直接提高1e-3初始值；下一轮指定候选为1e-3→2e-4→5e-5→1e-5，尚待对照验证。当前任务不改源码、Loss或调度，以下执行规则仍是当前已登记的unified_v2规则。

2026-10-25按用户最新要求改为 **unified_v2：训练、开发监测与未来获授权测试共用唯一固定Loss**。旧v1训练结果及权重已直接舍弃，仅保留[作废说明](runs/20261025_discarded_v1/note.json)，不得恢复旧断点。新版从头初始化seed0，主命令只训练main一组，未由助手启动。新版检查见[runs/20261025_preparation_unified_v2/results.json](runs/20261025_preparation_unified_v2/results.json)。旧准备记录仅说明历史v1检查，不适用于新版。

## 研究问题与取舍

用户观察到不只是小模块，一些中模块也承担了严重重叠；目标是使draft在保留相对方向与邻接意图的同时，经少量迭代、轻微平移和改形即可处理。拟检验：全局总量与平均尺度穿透项可能掩盖逐模块的相对冲突负担；给大模块适度重要性，并额外处罚同时卷入多个冲突的模块，是否改善草稿。

用户原意：对每个模块计算“与其他模块交叠的累计面积/自身面积”，乘个数因子和面积因子；个数因子需要连续。暂不新增独立最差局部项，不做受限修复训练，不改关系监督。参考坐标监督后期略降的想法留待独立实验，本轮不在第一次LR平台时更换目标。

保留原穿透项：低面积覆盖率并不充分保证轻松修复，细长交叉、预放置障碍和冲突链仍可能很难。多邻居更难也是待检验假设；同样总负荷下，多个浅冲突未必比单个深冲突更难。本轮只保留完整main目标，不提供另一套可选训练Loss。

## 首轮固定定义

设公开输入模块面积为a_i，解码矩形为R_i，I_ij为两矩形交集面积。固定/预放置模块沿用公开尺寸对应面积，与原batch解码一致。以下均按单个布局计算，然后对mini-batch各布局等权平均。

1. 相对交叠r_ij = I_ij/a_i；累计负荷l_i = sum_j r_ij。**保留重复覆盖**：同一区域同时被多个模块覆盖会逐对累计，l_i可以超过1，不是并集覆盖比例。5%仅作观察阈值，不是达标后不再处罚的死区。
2. 面积权重w_i = sqrt(a_i)。相同覆盖比例下，面积大四倍、重要性大两倍；不直接乘面积，避免抵消自身面积归一化。仅使用输入面积，模型不能通过改形改变权重。
3. 连续个数c_i = sum_j r_ij/(r_ij+0.05)。零重叠贡献0，5%相对交叠贡献半个，深重叠接近一个。这里的5%是计数过渡尺度，不表示每对5%就足够容易修复。
4. 个数因子f_i = 1 + c_i/(1+c_i)，在[1,2)内连续递增。保留梯度，不detach。饱和限制额外处罚最多接近翻倍，避免初始化密集团使其他目标失去作用；不把总负荷一起截断，严重交叠仍继续增罚。
5. L_module = 0.5 × sum_i(w_i l_i f_i)/sum_i(w_i)。分母只含不可学习的有效节点面积权重，不将个数因子放进分母，避免冲突数增加反而稀释已有处罚。

0.5是固定量级约定：若面积指数设为1且去掉个数因子，恰好退化为E43无序对交集面积/总模块面积。本轮面积指数固定0.5，不开放系数搜索。

包含自由—自由、自由—预放置碰撞，以及这些碰撞对两端模块的负荷；排除自身、padding和不可变—不可变对。仅位置不可变的预放置块不产生位置梯度，自由端仍有梯度。几何边接触不加正间隙，不发生硬计数跳变；min/max/clamp在接触等位置仍有分段导数折点，这不是数值不连续。

实现：[module_loss.py](module_loss.py)、[loss.py](loss.py)。

## 训练目标、选模与对照

唯一总Loss为`原E42 geometry + 0.25 × L_module`。从第一个batch起，所有项采用固定完整权重；**没有渐入、后期权重变化、或额外occupancy选模项，也没有叠加E43原0.25总重叠面积项**。原center1、shape .25、relative .25、penetration .5、contact .5、boundary .25、MIB .1、bbox .1全部保留。新模块项的系数0.25、计数过渡0.05及饱和上界在正式训练前固定；轻量检查只核对有限性和量级，不根据开发结果调系数。

仅有main一组：完整面积权重＋连续饱和个数因子，从头初始化seed0。E42/E43只作历史参考，不重训、不重新选择checkpoint；它们的旧代理数值与E51新总Loss不直接比较。可比较原有同定义分项和固定flash的后处理指标。

训练反向传播、固定train_monitor500/dev_monitor500、完整dev2000，以及未来明确授权的测试都调用[loss.py](loss.py)同一个函数，使用相同项、权重、归一化和每布局聚合规则。`include_quality`、epoch等兼容参数不改变返回的total。每个布局先计算Loss，mini-batch或完整面板按布局算术平均；尾批按实际样本数加权，不受batch划分影响。

**best、LR下降与早停全部使用固定dev_monitor500的total均值。** 日志明确显示train_loss和dev_loss；旧结构字段selection_proxy/best_proxy仅为兼容JSON格式，其值就是total，不代表另一套代理。分项与Q只作解释，不作为另一套选择标准。monitor保持eval/no_grad，训练Dropout的模式差异不改变Loss定义。测试数据仍默认封存，不因统一Loss而自动读取，不参与选模。

## 数据、资源与恢复

- 冻结v5：train16000/dev2000，固定train_monitor500/dev_monitor500；保持原split_v1分片归属、MIB平均面积规则、参考监督和public-only模型输入。所有测试继续封存。
- 沿用E42模型96×12、FFN Dropout .05、AdamW decay 3e-4、无增强、batch32、clip1。训练CUDA单GPU、CPU四线程；不自动改设备或batch。
- 每轮监测一次。LR固定1e-3→2.5e-4→1e-4→5e-5；第15轮起连续3次未达到相对0.1%或绝对1e-5的明显改善降档，完成三次下降且最低档至少两轮后才允许早停。best更新绝对阈值1e-5，规则继承控制。
- 每臂累计最多4小时或200轮。训练墙钟含缓存加载和监测，暂停前累计不重置。E42/E43在本机约110分钟；本轮预计约2～4小时，新增项尚未做正式速度测量，终末评价/绘图另计。
- 只读复用E25已准备的train/dev张量缓存，校验成员、数据及特征/batch语义；不跨实验导入私有算法，不复制数据或权重。所有本实验算法均为独立副本，来源见[provenance.json](provenance.json)。
- 保存best和完整latest；latest含模型、优化器、调度器、随机状态、已见成员、顺序/游标和累计时间。启动时保存实际命令、Git提交和脏状态、输入路径/冻结索引哈希及完整源码快照。恢复核对源码/设备/数据/预算身份，完成后不追加训练。

## 用户运行命令

PowerShell任意目录执行，**只训练main，完成即退出**：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\launch.ps1' -Stage train -Resume
```

首次没有latest则从头；中断后同命令恢复；已完成则核对后退出。主输出`runs/20261025_rectangle_w96d12_unified_v2_main_seed0/`，日志`runs/20261025_rectangle_w96d12_unified_v2_main_seed0_pipeline/console.log`。`progress.json`保存固定监测曲线，`checkpoints/best.pt`与`latest.pt`分工保存。

先预览命令、不训练、不评价：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\launch.ps1' -Stage train -Resume -DryRun
```

训练完成后，由用户另行执行一次冻结best评价：完整dev2000同一总Loss检查→固定dev500原flash/steady→配对报告及曲线→随机100张三相图准备。不会调用pro，也不会追加训练：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\launch.ps1' -Stage final -Resume
```

预计终末与绘图合计约5～15分钟，未经本轮测量。CPU后处理2进程各2线程，绘图4进程各1线程。完整求解时间为特征＋CPU推理＋解码＋全部修复/救援，初始化、JIT、评分另计；与E42旧控制同口径。Q是本地中性时间开发指标，不是正式比赛分数。恢复只补未完成成员，不重评完整面板，不按终末Q更换checkpoint。

## 当前默认看图方式

2026-10-25用户明确要求以后不启动评分界面。当前默认看[reference／E51 best52 draft／E42 best51 draft对比](runs/20261025_reference_E51_E42_draft100/index.html)，复用flash100的相同成员和顺序。E51草稿复用实际保存记录，E42草稿由冻结best51补生成；无后处理、无评分控件。看图页支持前后切换、选择样本、放大和打开原图。旧评分图集与历史评分文件保留，但不再自动启动；只有用户明确重新要求评分才使用评分服务。

本轮看图服务地址为`http://127.0.0.1:8768/`。重新启动只读看图服务（不做推理）可以运行：

```powershell
& 'E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026\python.exe' -m http.server 8768 --bind 127.0.0.1 --directory 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\runs\20261025_reference_E51_E42_draft100'
```

## 停止与判读

训练窗口Ctrl+C会在安全边界保存，或创建本实验STOP（不影响E42/E43）：

```powershell
New-Item -ItemType File -Force -Path 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\runs\STOP' | Out-Null
```

确认任务退出且决定恢复后，移除自己创建的STOP，再使用同一命令：

```powershell
Remove-Item -LiteralPath 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\runs\STOP' -ErrorAction SilentlyContinue
& 'E:\project\ICCAD\2026\problemC\experiments\E51_module_overlap\launch.ps1' -Stage train -Resume
```

恢复检查失败时保留证据，不覆盖身份或追加新预算。出现非有限值、数据/源码身份不符立即停止。

首要参考是用户同例草稿观察/评分；统一总Loss及其分项为主要数值参考；flash后处理Q及耗时为辅助。记录累计负荷均值/P95/最大值、>5%与>50%模块比例、软/硬冲突邻居数、每布局按面积排序的等人数小/中/大三分组（并列按原索引），并同时检查bbox、contact/boundary/MIB及三相图留白，防止靠摊大或破坏邻接降碰撞。旧E42缺少的新字段不能以缺失当作零；新的逐模块指标改善需要同例同口径基线，不能只看训练新项下降。

至少要求：负荷改善不以明显摊大/破坏接触换取；观察到中模块局部冲突改善；固定flash的救援、改形或修复负担有一致支持。没有事前证据把5%设成修复容易的充分条件，不凭单种子均值升级基线。若统一Loss下降却结构明显恶化，不继续仅靠增加系数追求低重叠；保留反例再解释机制。训练过程不调用后处理。

准备检查入口为[check.py](check.py)：合成机制＋3个train/2个dev样本，零优化器更新、零求解器调用；检查结果只证明实现行为，不证明训练收益。
