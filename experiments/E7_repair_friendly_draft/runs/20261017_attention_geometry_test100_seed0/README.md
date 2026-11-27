# attention_geometry：100例测试与10例布局图

状态：2026-10-17入口检查完成，尚未执行测试推理，尚无100例Q结果。该任务是E7结题后用户单独授权的冻结模型评价，不恢复训练或新增损失方向。

绘图现统一调用根目录 `visualization/layouts.py`，直接复用用户指定E6图集的生成代码。仅更新准备阶段的绘图源码指纹，原元数据计划备份为 `plan_before_shared_renderer.json`；100例成员、10例图例、checkpoint和后处理参数完全一致，已检查。

使用 `20261017_factorial_seed0/attention_geometry/checkpoints/best.pt`，step48125、第77轮best，SHA256 `47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28`。不是E6同名模型。后处理固定120秒预算、24轮形状调整、reweighted开启、MIP关闭；CUDA推理、2个CPU线程。主指标为100例Q算术平均，失败计Q=10，另报硬合法数、加权Q、分位数与实际耗时。此Q采用官方中性时间口径。

精确成员见[plan.json](plan.json)。从冻结workset_v2 local_test内保守排除已暴露及可能在途的worker0..39，在worker40..99每个取1例，再在元数据哈希选定的40个worker各取第2例，共100例；覆盖60个worker，包含原来仅预选未读的test50。样本排序仅依赖路径和行号，不按成绩选择。10个画图样本预先固定、来自10个不同worker。train分片不交叉；不额外声称生成家族独立。

## 执行

预计约16分钟（按旧dev10均值外推），但不同样本可能明显变慢，不能可靠排除超过20分钟，因此按AGENTS.md由用户执行。100例核心后处理预算合计约3.3小时，另有加载/评分/绘图开销，非总墙钟硬上限。预计显存低于2GB、输出磁盘低于100MB，实际以运行记录为准；不重复保存checkpoint。

```powershell
Set-Location E:\project\ICCAD\2026\problemC
& .\.conda\envs\iccad2026\python.exe -B experiments/repair_friendly_draft/runs/20261017_attention_geometry_test100_seed0/run.py
```

直接调用项目独立环境，无须激活base。若希望先激活：`conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026`。

按Ctrl+C会在当前案例完成后安全停止，已完成案例逐例保存。原命令末尾添加 `--resume` 可继续；不重算已保存案例，不更换checkpoint或参数。也可在本目录创建STOP文件，在案例间停止；恢复前仅移除本目录STOP，不触碰训练目录STOP。若机器强制中断，仅恢复末尾未写完整的JSONL行，该案例重新计算。请勿并发运行同一目录。

## 输出

- `cases.jsonl`：逐例Q、合法性、耗时及参考/draft/final坐标。
- `results.json`：持续更新完成数量及汇总Q，只有100例完成才作为完整面板结论。
- `analysis.md`：完整面板结果报告。
- `figures/`：10张参考/draft/final三栏同尺度PNG。
- `index.html`、`gallery.json`：图集入口与精确成员。
- `config.json`、`source_snapshot.zip`、`invocations.jsonl`：版本和运行证据。

绘图失败后可用原命令加 `--report-only` 从保存坐标重建报告/图集，不重读测试、不重跑后处理。`--prepare`仅冻结元数据，`--check`仅检查checkpoint元数据和已有dev绘图；两项已完成，见[checks.json](checks.json)。格式预览 `checks/gallery_format.png` 来自既有dev案例，不是本次测试结果。

测试张量以分片为容器加载，仅对选定100成员解码和评价；访问范围以逐例source记录披露。后续若根据这些图和分数修改模型，应披露测试反馈使用。
