# E161 无MIB的四阶段收尾架构

2026-11-22用户改为亲自启动全新scratch训练。旧L0运行、权重、日志、旧状态、源码和预算证据完整归档至[`history/e161_before_user_scratch_20261122T160405Z`](../../history/e161_before_user_scratch_20261122T160405Z/archive_manifest.json)；旧状态虽为running，但归档期间已持有原队列与worker的OS锁，未把旧运行改写为正常退出。当前队列为`prepared`，本轮尚未启动，原预算独立保留，新一轮预算从零计。训练设计和自动原dev2000评价保持；不执行防御性测试、试训或computer-use。

回到原`random_reference40k.v1`：原36k训练、原train500/dev500监测、完整dev2000终末评价。面积和reference WHXY不做MIB修正，不使用新MIB兼容筛选成员。原共享CPU缓存与E149公开role sidecar保持身份兼容并复用，不复制载荷、不重新生成标签。test封存。

L0三个width128/四头attention块；S0/S1/S2各两个独立块、各调用一次、各出口解码一次。保留完整端口、16维结构角色、多跳、静态pair6与秩8接收者相关静态关系value。后段仅入口编码布局，两个块之间只传隐藏状态，不计算任何动态几何反馈，不额外放题目attention编码器。共九个attention块、四次真实布局读出。

L0/S0/S1/S2全部从头顺序训练；每段只更新自身，冻结前段own-Loss-best、eval/no-grad，新建AdamW。E160是MIB监督权重，不作为本轮L0初始化。没有MIB输入通道、同MIB关系bias/value、MIB聚合或MIB Loss；公开约束MIB列不参与模型条件与后处理，评分分子分母均排除MIB。

每阶段末端监督为原公开任务/F/P损失，加reference、原重叠组和新增MST阻挡项：reference系数1/.75/.5/.5，重叠组倍率.25/.25/1/1，阻挡系数相对原grouping为.25。没有额外瓶颈项或新移动惩罚。反传、train/dev监测、best、LR和早停用相同阶段总Loss。详见[设计](DESIGN.md)。

原36k、batch64/尾32、seed0、FP32、轴互换、AdamW weight_decay .0003、梯度裁剪1、每模块更新投影.005、四档LR/早停、每阶段4h/200轮（总16h上限）。训练模型/Loss使用eager执行，初始化不做前向/反向探测或丢弃更新；不宣称已测吞吐或显存。正常训练的更新前数值门禁保留，错误和STOP阻断后续，不自动重试。

用户启动入口（项目根目录PowerShell）：`powershell -ExecutionPolicy Bypass -File .\experiments\E161_final_no_mib\launch_user_scratch.ps1`。脚本使用项目`.conda/envs/iccad2026/python.exe`，设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`并将控制台输出保存到`background/queue.stdout.log`。原Python模块入口保持。authorization中的训练和自动dev2000均已启用。各阶段best冻结至`frozen_e161_*.json`。终末评价准备了与E151相同原后处理预算的独立副本，使用[无MIB dev2000](../../protocol/final_evaluation_no_mib_dev2000_v1.json)，不会运行旧MIB面板。

`preparation.json`仅保存源码/配置身份，不代表运行验证。实际队列状态见`queue_state.json`；后台日志见`background/queue.stdout.log`（合并stdout/stderr）。历史E151/E160源码、权重、STOP和结果均不改写。
