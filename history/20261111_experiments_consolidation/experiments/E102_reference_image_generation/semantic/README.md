# E102新版：模块身份图生成

**最新：用户已暂停BF16＋batch4续训，停于20193步，完整状态保存、后台正常退出、GPU释放，STOP保留，不自动恢复。** [暂停核验](../runs/20261106_semantic_v21_bf16_b4_seed0/pause_verification_20261106.json)。下文迁移／启动记录仅描述此前发生顺序。

**当前：已明确授权从12297步以BF16＋batch4迁移续训。** [新方案／继承边界](../runs/20261106_semantic_v21_bf16_b4_seed0_launch/README.md) · [后台状态](../runs/20261106_semantic_v21_bf16_b4_seed0_launch/state.json) · [新对比图](../../../visualization/E102_reference_image_generation/20261106_semantic_v21_bf16_b4_seed0/index.html)。旧STOP保持，原8小时累计预算不重置；Loss、主参数和Adam FP32。旧停止段落仅描述迁移前状态。当前日志在`runs/20261106_semantic_v21_bf16_b4_seed0/`，启动标准输出在对应`_launch/train.stdout.log`。

**最新：用户已明确终止本轮正式训练。** 已在12297步（累计24594样本次、完成1轮＋第二轮8594/16000）安全保存模型／Adam／RNG／遮罩RNG／排列游标／调度与累计预算，后台退出、GPU释放。run及launch的STOP保留，不自动恢复。[终止核验](../runs/20261106_semantic_v21_seed0/stop_verification_20261106.json)。下文原授权／启动描述保留发生顺序，不表示仍运行。

**2026-11-06用户已明确授权正式训练观察泛化。** 新运行[授权与计划](../runs/20261106_semantic_v21_seed0_launch/authorization.json) · [后台状态](../runs/20261106_semantic_v21_seed0_launch/state.json) · [固定dev100对比图](../../../visualization/E102_reference_image_generation/20261106_semantic_v21_seed0/index.html)。新版从头初始化，不载入单图拟合权重；原16k／固定train/dev500、512、batch2、FP32、seed0，上限100轮或累计8小时。初始及每2000更新／每轮结束做固定双视图Loss监测，唯一dev500总Loss best。正式计划以新launch目录plan.json为准，本目录plan.json仍保留准备／检查阶段的状态。

当前日志：`runs/20261106_semantic_v21_seed0/updates.jsonl`每50步追加，`heartbeat.json`最近发布心跳，`progress.jsonl`固定监测，launch目录`train.stdout.log`／`train.stderr.log`为运行输出。初始监测完成前没有实际更新；初始监测行steps=0不能当成已训练泛化。旧RGB STOP保留。

[设计与固定Loss](DESIGN.md) · [配置](plan.json) · [单图拟合结果](../runs/20261106_semantic_v21_fit1/result.json) · [reference／生成图片](../../../visualization/E102_reference_image_generation/20261106_semantic_v21_fit1/index.html)

用户在2026-11-06要求重新设计模型和Loss，并允许采用diffusion。本目录已实施约1281万参数的多尺度条件U-Net、模块ID像素分类、直接耦合边界的图片Loss，以及MaskGIT式迭代掩码生成。没有坐标输出、矩形／矩阵提取、MIB同形、几何修复或Q。

每次更新同时学习从全MASK生成整张图，以及从随机遮罩图去噪；两视图各0.5。每个视图的固定Loss为：前景逐模块平均交叉熵＋模块平均Dice＋0.2边界概率BCE，背景交叉熵另乘0.25。正式监测／best／LR／早停如实施也必须使用这个双视图总Loss和固定遮罩种子，不得只改成RGB-MSE或以采样图另选模型。

已对原五例面板中同一张21模块训练图做有界学习能力检查：

| 版本 | 更新 | 全MASK模块平均IoU | 16轮迭代IoU | 结论 |
|---|---:|---:|---:|---|
| 首版v2 | 250 | 0.5512 | 0.3773 | 基本拟合未通过，原证据保留 |
| 修订v2.1 | 250 | 0.8245 | 0.5777 | 基本拟合通过，前景准确率91.78%，边界仍弯曲 |

v2.1的同一全MASK评估状态，Loss由5.2704降至0.4591；这些Loss只适用于v2.1自身，不能与旧RGB模型0.106的Loss作大小比较。两版仅检查同一个训练成员，未使用dev或test；新方案经过训练侧诊断修订，不能冒称预先固定的独立泛化实验。

当前主输出选用全MASK直接预测，再按固定模块颜色显示；16轮迭代在这次检查上更差，不强行将它作为默认，也不因此断言所有diffusion都无效。它只借鉴离散掩码生成，没有实现完整DDPM／D3PM。尚未证明未见题能生成相似图片，仍不能保证目标已达成。

旧正式RGB运行仍停在4369步，STOP保持。上述小图结果属于正式授权之前的检查；后续新版正式启动以页面顶部新run为准。新数据、目标和架构与旧RGB不同，已登记独立新运行、从头初始化，不覆盖旧checkpoint／optimizer证据，没有自动唤醒或后续候选。

源码：`model.py`为模块条件与U-Net，`loss.py`为固定图片Loss，`generation.py`为遮罩／采样，`data.py`为冻结原始成员与公开输入，`fit_check.py`为有界单训练图检查。标签生成和图片展示统一调用visualization，不从RGB反推几何。
