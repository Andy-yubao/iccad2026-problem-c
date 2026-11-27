# E169 FloorSet 求解器

交付版本为 **E169 Ref080 best33 + 强化软约束后处理 + FP32 CUDA Graph**。求解采用原目标面积与 no-MIB 规则。输入限于公开面积、连接、端口与约束，输出为各模块的 `[x, y, width, height]`，并通过独立硬合法检查。

## 在本机运行

从仓库根目录运行内置的 41 模块公开样例。新环境可用根目录的 [environment.yml](../environment.yml) 创建。公开仓库同步源码、版本清单和评测摘要，冻结权重保存在研究工作区。另行取得权重后，将其放在 `solver/model.pt`，程序会按 [provenance.json](provenance.json) 核验 SHA-256。

默认使用 CUDA。用 `--input case.json` 换用同格式题目，用 `--device cpu` 选择 CPU 前向。多题直接复用一个 `Solver` 实例，可复用 GPU 权重和计算图。

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m solver
```

```python
from solver import Solver

solver = Solver(device='cuda')
positions = solver.solve((n, areas, b2b, p2b, pins, constraints, specs))
```

`b2b` 和 `p2b` 各行是两个零基端点和权重。`constraints` 各行为 `[F, P, MIB, G, boundary]`，MIB 在本求解器中屏蔽。`specs` 提供公开矩形规格，未知分量用 `-1`。完整样例见 [example.json](example.json)。

冻结权重及导出文件哈希见 [provenance.json](provenance.json)。原 no-MIB dev2000 实测硬合法 2,000/2,000，混合 Q 1.006563、加权 Q 1.005851。该评价使用一份常驻 GPU 模型及八个 CPU 后处理工作进程，完整单例均时 217.6 ms、P95 556.3 ms。逐例 CLI 使用单个 CPU 进程。 [归档结果](../experiments/E169_s2_progressive_unfreeze/evaluation/gpu_graph_dev2000_v1/results.json)
