# E107首轮启动后的严格门禁终止

已从E105 best58按reference10%启动。完成22次接受更新后，第23批在Adam前被拒绝。worker_40/layouts_9296.th第69例的新Loss49.70237，冻结父模型同题3.766752，单例相对门槛37.667519。批均值5.30584未超过9.61678；梯度范数482.23175低于10000，数值均有限。因此这是预设相对单例尖峰保护触发，不是已证实NaN/Inf或全局梯度发散。

STOP在触发后写入，optimizer/projection/progress接受步均22；best保持epoch0父模型，新阶段没有完成一轮。latest、best、failure.pt、loss_explosion.pt与abort_diagnosis.json均保存，训练与控制器退出、GPU释放。未启动dev500后处理、未读取test；不自动重试、降档、调权、放宽阈值或清除STOP。
