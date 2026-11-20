# E104 local-prenorm修复与续训

用户2026-11-05明确要求修复并续训。原失败在step20350、第21轮cursor11200，尚未接受该批次更新；旧best18与latest、failure、源码、监测、日志等在before完整保留。

同一保存失败批次（仅训练数据，32成员，为实际error定位所需）在保存的失败后RNG下复现最大单例Loss1913.685，梯度4678.25且均为有限数；异常题worker_75/layouts_4032.th第97例。该题reference Loss .078；旧eval输出Loss2751.71，隐藏幅度从8.99逐层到996.51。不是NaN，亦没有证据表明原始题目损坏。

唯一修复：local消息聚合使用现有norm1(h)，不增参数，其他前向通路、固定十项Loss、batch32/FP32、增强/Dropout、AdamW与数值/步长门禁保持。旧checkpoint省略normalize_local时仍表示旧forward。修复短检同批次max Loss2.902、grad11.77、输出max1.83，一次真实更新通过且丢弃该诊断更新，不向正式权重写入。

迁移从原latest接受步20350直接保留全部模型/Adam、数据顺序/游标/seen、保存RNG、LR1e-3、降档/epoch与projection计数及累计时间；不回载best或跳批。保存的RNG在拒绝之后，原拒绝前的Dropout抽样不可完全重建，因此不声称bit-exact重放。前向路径已改变，旧best18/history保持历史证据；新阶段重新监测固定train/dev500、建立best与平台参考，无新保护期，不增加候选。原错误阈值未放宽。

migration.json记录旧新完整身份，repair_shortcheck.json为实际更新证据，implementation_gate.json为源码差异门禁。controller.py后台执行train --resume，正常早停后自动固定唯一新best做dev500；错误/STOP不自动重试，不读封存test。stdout.log/stderr.log为本次日志。此后结果同时含局部归一化修复，不称纯scaling对照。
