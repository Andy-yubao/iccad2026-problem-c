# ICCAD 2026 Problem C

研究已于 2026 年 11 月 26 日收尾。最终求解器为 **E169 Ref080 best33 + 强化后处理 + FP32 CUDA Graph**。

- [求解器用法与公开样例](solver/README.md)
- [E1–E170 实验综述](deliverables/实验流程记录_E1-E170.md)
- [交付报告（独立 HTML）](deliverables/ICCAD2026_ProblemC_E169_求解器报告.html)
- [实验目录与恢复说明](experiments/README.md)

原 no-MIB dev2000 的硬合法为 **2,000/2,000**，加权 Q 为 **1.005851**，混合 Q 为 **1.006563**，完整单例均时 **217.6 ms**。评测使用本地 RTX 5080 Laptop GPU 和八个 CPU 后处理进程。 [原始结果](experiments/E169_s2_progressive_unfreeze/evaluation/gpu_graph_dev2000_v1/results.json)
