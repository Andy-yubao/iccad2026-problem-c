# 2026-11-03 最终交付归档

用户选定C5＋原完整后处理并授权后台64k重训。确认真实Adam更新后，同盘归档所有旧实验、旧输出及辅助脚本。首次E100目录整体改名遇Windows目录占用；按已登记清单逐文件迁移，逐项核对size/mtime，源码、小证据及唯一权重另核对SHA256。被占用原件若存在则保护保留，详见manifest.json；没有删除唯一证据或强行终止其他应用。

manifest.json解释旧记录到新位置的映射。旧JSON、历史命令和权重不改写。大缓存沿用生产者哈希，全部size/mtime核验，不声称重新哈希所有大型数值载荷。Git已有改动保留，未批量提交或push。

日常唯一求解器：experiments/E101_final_c5_solver。运行有效缓存独立准备，E42旧缓存仍由归档映射供必要复用。

2026-11-04用户明确要求继续E98 Loss动态调权研究，E98已整体迁回[experiments/E98_loss_weight_learning](../../experiments/E98_loss_weight_learning/RESTORED_20261104.md)。原归档manifest记录历史归档事件，保持不变；当前E98位置由[E98_restore_20261104.json](E98_restore_20261104.json)补充解释，逐文件核对SHA256、大小和mtime。本次不启动训练或评价，其余归档保持原位。
