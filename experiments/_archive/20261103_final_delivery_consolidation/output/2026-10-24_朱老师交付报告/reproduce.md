# 交付对象和复现说明（v2）

日期：2026-10-24。仓库根目录为`E:\project\ICCAD\2026\problemC`。

## 原件定位

| 对象 | 仓库相对位置 |
|---|---|
| 模型、输入处理、推理与几何优化源码 | `experiments/E42_joint_regularization/` |
| 冻结训练运行 | `experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/` |
| 最终模型 | 上述目录`checkpoints/best.pt`，第51轮 |
| 恢复权重 | 同目录`checkpoints/latest.pt`；训练已结束，不自动续训 |
| 已完成1000例运行 | `experiments/E42_joint_regularization/runs/20261024_final_test1000_seed20261024/` |
| 评价入口 | `experiments/E42_joint_regularization/final_test.py` |
| 完整最终坐标 | 1000例运行内`cases/0000.json`至`cases/0999.json` |
| 原评价源码快照 | 同目录`source_snapshot.zip`；源码身份见`config.json` |
| 官方历史结果 | `references/official/results/`，5个原件均已复制至本材料的`evidence/` |
| 数据路径及冻结成员 | `protocol/paths.json`、`protocol/split_v1.json`、`protocol/reference_workset_v5.json` |
| 独立运行环境 | `.conda/envs/iccad2026/` |

最终权重SHA-256：

```text
12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6
```

原1000例使用Windows、Python3.10.22、Torch2.7.1+cu128、NumPy1.26.4、SciPy1.15.3、Clarabel0.11.1、HiGHS1.15.1、Numba0.61.2、Shapely2.1.2。CPU4进程、每进程1数值线程，最终评价没有使用CUDA推理。精确记录见[E42_environment.json](evidence/E42_environment.json)。当时有未提交源码，应结合运行源码快照和哈希恢复，不能只凭Git提交号还原全部工作区。

## 重建素材与factbase

在仓库根目录执行：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B `
  output/2026-10-24_朱老师交付报告/build_materials_v2.py
```

脚本读取已有E42最终结果和官方results，不重复训练、最终求解或官方评分。三相图的3例草稿已生成并保存在`materials/E42_draft_examples.json`，之后直接复用。若该缓存丢失，脚本才会用冻结权重为这3个既定成员各前向计算一次；不会改变原最终布局、计时或评分。

参考列读取同一成员原标签；草稿列只使用公开输入生成；最终列读取原运行。三列来源和图示重建事实写在`materials/layout_examples.json`。参考/草稿没有新评分，最终指标沿用已完成评价。

更新材料验收与文件哈希：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B `
  output/2026-10-24_朱老师交付报告/verify_bundle_v2.py
```

网络函数`inference.predict(model, public_inputs)`生成草稿，`solver.optimize(draft, public_inputs, **STEADY)`完成几何优化。原参考标签和分母只供监督/评分，不是这两个函数的输入。本次1000例已完成，不为修改报告重新评价。

本目录提供报告、事实与素材，模型权重、完整数据和最终布局仍通过上表定位到仓库唯一原件。正式比赛隐藏集成绩和官方提交包验收没有在本材料中取得证据。剩余900例未用于本次写作或制图。
