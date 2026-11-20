# E132 关系输入96×6

E131停止后的独立scratch对照。full/pure共用400→192→96编码、96×6/4头/SwiGLU192及192→96→3标准读出；参数786195/674979。固定16k、原train500/dev500、FP32/batch32、原增强/LR/数值门禁；原始面积、无MIB、F/P全预测不覆盖。共享完整CPU缓存，未加载E131权重/Adam。

队列已complete，两组均正常dev早停：

| arm | 完成轮数 | best epoch | own dev Loss |
|---|---:|---:|---:|
| full | 153 | 145 | 0.403183 |
| pure | 144 | 136 | 0.775958 |

原件在 `runs/full_seed0/results.json` 与 `runs/pure_seed0/results.json`，最新状态见 `queue_state.json`。两组均未观察到train平台，不称充分拟合。只有训练授权，未做后处理/Q/test；Loss差异不等于最终质量差异。错误/用户停止不自动重试，不追加seed或恢复E131。
