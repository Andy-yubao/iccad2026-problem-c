from pathlib import Path
p=Path('experiments/E104_scaling32k')
(p/'README.md').write_text('''# E104：32k、128×16 泛化增强版扩容实验

用户2026-11-04明确授权新实验、从头训练、后台启动，完成后自动固定dev500评价；已明确确认不是封存test。唯一候选从E98增强版设计扩大：16k→32k，96×12→128×16，1,471,539→3,463,747参数（2.354倍）。seed0、batch32、FP32、AdamW decay=.0003、clip1、每模块联合步长上限.005、固定原十项Loss、FFN Dropout=.10、残差Dropout=.05、训练50%完整轴交换保持。不继承任何权重或E103系数。

成员为原reference_workset_v6训练前32000，原v5前16000不变、模块数分布精确加倍；原split_v1和固定train/dev500保持。32k已在E58/E74/E101等历史研究中曝光，不称全新数据；公开MIB平均面积派生任务保持。网络仅公开信息，端口保留每模块总权重与加权中心，完整连接仅用于Loss/评价。test载荷禁止读取。

原从头LR四档1e-3→2.5e-4→1e-4→5e-5；前14轮保护、第15轮建立平台，3轮无明显改善降档，最低档完成至少2轮且3轮停滞早停。明显改善max(1e-5,基准×.001)，best门槛1e-5。无硬时间/轮数上限，正常dev Loss早停不等于train已完全拟合。反传/固定train500/dev500/best/LR/早停同一固定Loss；监测不增强且eval。保存best/latest与Adam、RNG、排列/游标、调度和累计耗时。所有私有算法独立复制，来源哈希见provenance.json，不跨实验导入。

准备短检最多五个训练实例，覆盖新增32k成员、原增强语义、新结构真实更新和同一训练实例的完整评价链。正式训练在预检和源码身份门禁通过后启动。CPU只读缓存、pinned batch传输和FP32 CUDA图，不整库进显存，保留数值/梯度/步长门禁。性能记录每100步和完整epoch时间，不以GPU占用代替吞吐。

正常早停后流水线在独立进程冻结唯一dev Loss-best及源码/权重/面板哈希，复制的E98.depth_budget.v1对固定dev500单次评价，不按Q重选、不继续训练。基线只读复用原增强best71已完成500结果。分别报告固定Loss曲线、绝对/相对泛化间隙、三Q、草稿/最终硬合法与几何指标、完整逐例均值/P95/最大值/250ms超限，冷启动/磁盘/评分/绘图另列，同100例统一ID尺度图；不自动打开页面。完成报告后停止。

这是单seed、数据和模型共同扩大的试验，不足以拟合scaling指数或区分数据/宽/深的单项贡献。dev面板已反复开发使用，不称独立未见泛化或正式比赛成绩。E101/E102/E103与已停止E98不改动、不恢复。

运行：`powershell -File experiments/E104_scaling32k/launch.ps1`；入口`.conda/envs/iccad2026/python.exe -m experiments.E104_scaling32k.pipeline`。状态位于runs/20261104_scaling128x16_seed0/pipeline_state.json、heartbeat.json、progress.json。停止：在本实验目录或run创建STOP；训练在批次边界保存latest，控制器不启动评价。error不自动重试；恢复需核实身份后明确执行train --resume，并单独处理后续阶段。
''',encoding='utf-8')
(p/'launch.ps1').write_text('''$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/../..").Path
$python = Join-Path $root ".conda/envs/iccad2026/python.exe"
$run = Join-Path $PSScriptRoot "runs/20261104_scaling128x16_seed0"
if ((Test-Path "$PSScriptRoot/STOP") -or (Test-Path "$run/STOP")) { throw "STOP present" }
if (Test-Path "$run") { throw "Existing run; do not overwrite or restart automatically" }
$proc = Start-Process -FilePath $python -ArgumentList @("-u","-m","experiments.E104_scaling32k.pipeline") -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot/stdout.log" -RedirectStandardError "$PSScriptRoot/stderr.log" -PassThru
@{pid=$proc.Id; started_utc=[DateTime]::UtcNow.ToString("o"); authorization="Explicit background train and automatic dev500"} | ConvertTo-Json | Set-Content -Encoding utf8 "$PSScriptRoot/launch.json"
Write-Output "Started E104 controller PID $($proc.Id)"
''',encoding='utf-8')
# Remove unused upstream standalone preflight, which binds its original experiment's plan.
(p/'depth_evaluation/preflight.py').unlink()
