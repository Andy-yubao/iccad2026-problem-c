# 单次有界续训

2026-11-06用户在曲线评估建议后明确要求“续训吧”。从暂停的20193步／seen56176完整latest恢复；保留Adam、全部RNG、数据排列／游标、LR调度和累计19936.632秒。原总8小时预算不重置。BF16、batch4、Loss、train/dev500及固定dev100图片规则不变，不作新的精度校准。

原暂停run及launch、STOP、best/latest和日志原位保留。新run复制完整恢复资产及历史曲线，以原冻结trainer的`--resume`运行，启动前核对源码／缓存身份、复制后哈希。没有实施断点测试或额外学习检查。新run的identity是继承的来源身份，复制与续训来源另见continuation.json。

控制器只增加本次授权窗口：完成约5轮（epoch5固定监测已保存）后写新run的STOP，下一批边界保存退出，可能多出少量更新；原8小时预算／正常早停／用户STOP仍优先。该窗口在控制器boundary_stop.json中记录；冻结训练器会将STOP编码为user_stopped，须结合该原因区分。失败不重试、不自动接新阶段。

继续使用原图片入口，曲线伴随进程读取新run（包含此前BF16曲线）并在终止后退出。固定监测点按seen4000／轮末更新；图片仍按Loss-best发布。

- 新run：`../20261106_semantic_v21_bf16_b4_resume5_seed0/`
- 当前控制器状态：state.json
- 训练输出／错误：train.stdout.log、train.stderr.log
- 用户停止：在本launch或新run创建STOP。
- 不读取test、不作Q／几何处理、不使用computer-use。
