# E134：架构×监督间隔，四组对照

2026-11-13用户确认：E119/E120两种架构×每三层/每两层监督，全部采用已定义的新F/P Loss；仅准备，给用户命令自行启动。历史E119 Q≈1.02不是可靠标杆，本轮按四组互相比对。单seed不作稳定架构排名。

| 顺序/arm | 架构 | 辅助监督，总权重0.2 | 末层 |
|---|---|---|---|
| e119_stride3 | E119，712275参数 | L3/L6/L9，各0.2/3 | 0.8×L12 |
| e119_stride2 | E119，712275参数 | L2/L4/L6/L8/L10，各0.2/5 | 0.8×L12 |
| e120_stride3 | E120，725619参数 | L3/L6/L9，各0.2/3 | 0.8×L12 |
| e120_stride2 | E120，725619参数 | L2/L4/L6/L8/L10，各0.2/5 | 0.8×L12 |

共同scratch seed0、36k原训练池和原固定train500/dev500（本轮延续已讨论36k对照）、FP32/batch32、原增强/AdamW/LR/早停/数值门禁，每组训练累计4h或200轮；本轮不加入历史4/12组。新Loss沿用public_priority_v1：原项不变，F/P固定尺寸和P中心项分别按相关子集平均、系数各1；所有模块继续预测位置/形状，草稿不覆盖规格。末层公共新/旧Loss仅诊断，各组自身固定聚合Loss统一用于反传、监测、best、LR和早停。

## 启动与暂停

在项目根目录的PowerShell中运行：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E134_supervision_interval.run_queue --start
```

这是前台命令，可直接看到日志。准备完成不会自动执行它；明确执行`--start`才激活本队列四组。数据优先读取完整兼容CPU缓存，缺失时报错，禁止静态预处理重建。

按一次 **Ctrl+C** 后等待提示`Paused safely`。控制器要求工作进程暂停，当前受保护更新或当前后处理例安全完成后存盘，不推进后续阶段；不要直接关闭窗口或强杀进程。完成加载/图捕获/存盘或单例修复可能需要等待。暂停后继续：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E134_supervision_interval.run_queue --resume
```

续训读取latest，恢复模型、Adam、Python/NumPy/CPU与CUDA RNG、样本排列/游标/seen、LR/早停、投影状态和累计预算；暂停期间不计时，重新加载/捕获耗时继续计入，不清零4h。预算计时按每次实际运行累计。CUDA图会重新捕获并恢复保存的随机状态；等价性以准备诊断记录为准。

`--resume`只接受安全暂停的队列，不自动重试error或崩溃现场。仅本队列的`PAUSE_REQUEST`在显式恢复时归档；STOP/QUEUE_STOP和历史STOP不自动删除。

## 自动后处理与记录

每组成功训练结束后，冻结唯一own-Loss-best，再自动运行原dev500、原E98.depth_budget.v1后处理、原始面积/无MIB本地Q及独立硬检查。完成该组评价后再启动下一组。Ctrl+C暂停评价时，当前例完整写入后停止；恢复校验权重/源码/成员，复用连续已提交记录，从下一例继续，不重评已完成例。

错误阻断后续，不自动重试；没有test、新seed、预算扫描、历史实验恢复或新增布局图。最终按四组的合法性、草稿几何、公共末层Loss、Q及均值/P95/最大耗时比较，不能直接比较不同聚合定义的own总Loss。

- 总状态：`queue_state.json`；启动开关：`authorization.json`；配置：`plan.json`。
- 每组训练：`runs/<arm>_seed0/`，`heartbeat.json`、`progress.json`、`checkpoints/latest.pt`与`best.pt`。
- 后处理：各run的`evaluation_no_mib_dev500/`；提前冻结身份：`evaluation/frozen_<arm>.json`。
- 每次启动/恢复新建日志：`launch_logs/<session>/`；最终汇总：`comparison.json`、`comparison.md`。
- 准备检查：`preflight.json`与`preparation_diagnostics/`。只使用原五训练成员，诊断权重独立可丢弃；不是正式训练，不读取dev/test张量、不调用后处理/Q。

独立模型/Loss/后处理副本和来源在`provenance.json`；原E119/E120源码、权重、预算与STOP保持。
