# E108：独立随机40k原始reference数据及图片

2026-11-06用户明确要求修改后续实验MIB约定，重新随机划分数据并在后台准备；数量澄清为 **40000 = 36000训练 + 2000验证 + 2000测试**。此次不训练模型，也不恢复E102或其他停止运行。

原始面积、所有题目规格和reference保持；后续新实验忽略MIB，无平均面积、同形协调或MIB软罚分，其他约束保留。[研究协议](../../protocol/research_no_mib_v1.json) · [数据说明](../../docs/datasets.md)。官方原件、旧split/成员、历史权重和旧评分不改写。

从paths.json解析的全量原始9000分片出发，使用新生成并冻结的128位seed，随机完整分片划分为8100/450/450池，然后分别均匀无放回抽36000/2000/2000样本。不读旧名单来选题，不筛MIB相容性、不做模块数配平、不按质量筛选。分片池及成员互斥；仍可能与历史模型见过的成员重合，不能作为旧模型未见测试的保证。

生成512×512原始reference RGB PNG，公共visualization负责固定ID配色、亚像素覆盖；公开面积/端口/预放置决定方形画布，不拟合自由reference、不裁剪或修复。每例NPZ保留原面积、连接、约束、遮罩规格及分离的reference/存储指标，MIB字段仅作来源留档。PNG/NPZ及来源分片逐一SHA256。异常即停，无自动重试、跳题或替换；完成即停。

测试部分仅本次制作读取，禁止模型评价/调参，制作完成继续封存；新通用reader拒绝test。固定train/dev各500监测子集登记在新索引中，生成任务不计算模型Loss或Q。

- [冻结成员](../../protocol/random_reference40k_v1.json)
- [本次授权及原索引校验](runs/20261106_random40k_v1/authorization.json)
- [五例实施短检](runs/20261106_random40k_v1/check5.json)
- [实时状态](runs/20261106_random40k_v1/state.json)
- 完成报告：`runs/20261106_random40k_v1/report.json`，仅完成后存在。
- 输出：paths.json中的`random_reference40k_root`，本地大文件不入Git。

准备入口：`python -m experiments.E108_random_reference40k.prepare prepare`、`check`、`run`。前两步在后台启动前完成；run为一次性生成，不隐式覆盖或恢复已有输出。后台4个CPU进程、不使用GPU；同目录STOP可停止后续提交，保留已写文件和状态。
