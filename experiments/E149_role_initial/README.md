# E149：只训练带结构角色编码的初稿生成器

2026-11-17 用户授权：尝试结构角色编码方案，但只训练初始布局层，直接监督初稿输出、无重叠项；准备通过后后台启动，只关心Loss。

模型仅包含初稿网络：原16维输入编码、16维结构角色编码、完整引用端口token编码、两个width64/4-head静态关系attention块、初稿读出头。共107739参数，全部scratch seed0训练。没有S0/S1/S2几何单元、形变反馈、后段反传或历史权重。原初稿105539参数，本方案增加2200（约2.1%）。

模块query读取所有模块和静态端口key/value；端口token不作为query、层间不更新。关系bias包含直接边权、是否直接相连、同组、精确最短hop、可达性、自身标记。局部通路保留公开连接/分组聚合。端口坐标按公开总面积平方根归一；同位置端口合并并累加权重，不丢弃引用的端口。P模块作为已知位置锚点，F尺寸标记不当作已知位置。不存在reference坐标、reference外框或MIB输入。

16维角色包括度、邻居度分布、连通分量比例、2/4-hop可达比例、平均hop、孤立标记、到P/端口的最近hop及可达标记、端口覆盖、全局密度和模块数。最短路基于正权B2B/P2B连接，无人为补边或距离封顶；不可达用单独标记，不等同于很远。分组是独立关系，不冒充电气连接。

仅监督初稿一次、权重1：原reference、wire/compact、grouping、boundary及公开F/P尺寸和P位置项系数保留，完全去掉直接overlap组（penetration、mean load、max excess）。原完整Loss和碰撞项仅作诊断。全部模块预测全部坐标/形状；不覆盖F/P输出。反传、train500/dev500、best、LR、早停使用同一个初稿去重叠目标。

本轮延续原36k及原train500/dev500。兼容基础CPU缓存逐字节复用；只新建带版本、源码hash、成员身份的公开role sidecar缓存，不复制reference或旧模型状态。batch64/尾批32、确定性FP32/TF32关闭、原增强/AdamW/门禁/.5%每模块更新界限、LR/早停不变，整个候选累计4h/200epoch上限。初稿和角色编码均从头训练，准备短检模型不复用。

准备检查覆盖独立BFS、P/F锚点区别、不可达与无端口、完整端口分布、节点/端口排列、标签隔离、轴交换、初稿去重叠Loss及梯度、CUDA64/尾批32、保存恢复。通过后后台仅训练和冻结own-Loss-best，写[LOSS]摘要；无后处理、无Q、无test、无自动追加候选或重试。

入口：`python -u -B -m experiments.E149_role_initial.run_queue --start`。日志`background/queue.stdout.log`，实时状态`queue_state.json`和`runs/e149_role_initial_seed0/heartbeat.json`，完成摘要`background/loss.log`及`loss_results.json`。

E148初稿并未按本次独立目标训练，不将历史初稿Loss当匹配对照。此轮先检查可训练性及初稿Loss，不能仅凭一次结果把效果归因于某个新增特征。
