from pathlib import Path
import json,hashlib
import numpy as np
from datasets.reference_layouts import registry,REGISTRY
root=Path.cwd();owner=root/'experiments/E103_offline_loss_alignment'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
records=registry()['sets']['train']['records'];assert len(records)==16000
ids=np.random.default_rng(2026110431).choice(16000,1000,replace=False).tolist()
permutation=np.random.default_rng(2026110432).permutation(1000).tolist()
split=dict(fit=permutation[:600],tune=permutation[600:800],check=permutation[800:])
implementation=[]
for target in (21,60,90,110,120):
 selected=min((i for i in range(1000) if i not in implementation),key=lambda i:(abs(records[ids[i]]['n']-target),i))
 implementation.append(selected)
panel=dict(revision='E103.train1000x5.v1',sampling_seed=2026110431,split_seed=2026110432,
 train_indices=ids,members=[records[i]['member'] for i in ids],n=[records[i]['n'] for i in ids],
 analysis_splits=split,direction_indices=split['tune'][:40],implementation_indices=implementation,
 registry_sha256=hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),original_population=16000,
 selection='Uniform without replacement; sampled before model inference/Loss/Q; original train only',test_read=False)
if (owner/'panel.json').exists():assert read(owner/'panel.json')==panel
else:save(owner/'panel.json',panel)
plan=read(owner/'plan.json')
if not (owner/'evidence/selection_plan_original.json').exists():save(owner/'evidence/selection_plan_original.json',plan)
plan.update(status='authorized_preparing_offline_background_pipeline',authorization='User accepted five-model selection and authorized autonomous research continuation; long tasks background, four parallel independent tasks. User explicitly requires E98 postprocessor.',
 next_stage='Launch database1000x5 -> offline fits -> local descent checks -> report; inspect evidence to register subsequent finite neural training comparison. Do not start unsupported long training.',
 numerical_path='New registered CPU FP32 deterministic inference for all five frozen models; four single-thread workers with original source public-input and decoder semantics. No historical draft/Q reuse.',
 inference_reproduction='CPU and dynamic exact-n padding may differ numerically from original GPU training/evaluation; original weights and mathematical graph preserved, not claiming historical bitwise draft reproduction.',
 parallelism=dict(worker_processes=4,threads_per_worker=1,backend='Windows spawn ProcessPoolExecutor',unit='one case with five frozen source models; independent cases run concurrently',offline_fits='four regularization fits concurrently; five source holdouts at most4 concurrently'),
 analysis_registration=dict(case_splits=dict(fit=600,tune=200,check=200),terms=10,objective='case-weighted within-case logistic ranking with nonnegative simplex scaled weights; fit-only75th percentile absolute pair-difference scales; original coefficient prior; same total physical coefficient sum',regularization_grid=[.0001,.01,1.,10.],minimum_Q_pair_gap=.0001,selection='tune200 mixed regret only, tie by ranking accuracy then stronger regularization; freeze proposal before finalcheck200 and source-holdout reports',source_holdout='5 leave-one-source-out fits using the already selected regularization; source-involving pairs on check200 descriptive only',case_weights='half equal-case plus half official exp(n/12), normalized on each analysis panel',network_training_case_aggregation='not yet registered; must be fixed before any new neural training'),
 direction_registration=dict(cases=40,source_split='first40 internally randomized tune cases, chosen before Q',model='e98_generalization',losses=['original','learned'],steps=[.002,.01],additional_postprocessing_evaluations=160,step_rule='RMS-normalized gradient in per-module sqrt(area) center units and log-aspect; mask fixed channels, clamp normalized gradient components to[-3,3]; original area/spec decoder',gate='both step sizes improve mixedQ >.0001 and outperform original-Loss local steps; layout-space evidence only, not neural training proof'),
 postprocessor_registration=dict(source='experiments/E98_loss_weight_learning/depth_evaluation',copy='experiments/E103_offline_loss_alignment/postprocessor',version='E98 fixed-depth v1 unmodified, main shape-call caps4/4/7/8, all original hard guards',Q='official neutral-runtime cost; mixedQ=(official size-weightedQ+meanQ)/2',runtime='concurrent CPU timings are throughput diagnostics, not deployment latency or contest score'),
 stop='Create current RUN/STOP or experiment STOP. Workers finish the active fixed-depth solve and stop before following draft/case; no later analysis/stages.',
 terminal='Offline stage reports numerical evidence and finite next-step recommendation. No dev postprocessingQ or test payloads; no automatic retry on numerical/error cases.')
plan['database']['sampling_started']=True
plan['database']['population']='Frozen panel.json uniformly sampled1000 from original train16000'
plan['scope'].update(current_stage='authorized database, offline mathematical analysis and local direction research',database_generation_started=False,loss_fitting_started=False,formal_training_started=False)
save(owner/'plan.json',plan)
(owner/'launch.ps1').write_text('''$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\\..')).Path
$taskRun=Join-Path $PSScriptRoot 'runs\\20261104_bank1000x5_seed0'
if(Test-Path -LiteralPath (Join-Path $taskRun 'config.json')){throw 'Run already launched; refusing duplicate'}
if((Get-Content -LiteralPath (Join-Path $PSScriptRoot 'preflight.json') -Raw|ConvertFrom-Json).status -ne 'passed'){throw 'Preflight did not pass'}
$taskPython=Join-Path $taskRoot '.conda\\envs\\iccad2026\\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User approved E103 continuation, background long tasks, four parallel; E98 depth postprocessor';workers=4;test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started E103 coordinator PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('E103 exited '+$taskProc.ExitCode)}
''',encoding='utf-8')
p=owner/'README.md';s=p.read_text(encoding='utf-8')
s=s.replace('当前只完成模型选择和独立副本整理，先向用户汇报；未读训练/验证/测试载荷，未生成草稿、后处理Q或拟合权重，不启动新训练。','用户已认可五个模型并授权自主推进、后台运行长任务和4路并行。当前新登记为数据库1000×5 → 离线权重拟合 → 局部下降方向检查 → 报告。神经网络长训练须根据该证据先登记固定目标与有限对照，当前流水线不盲目自动训练。')
s=s.replace('此阶段不抽样、不生成。','同1000题已按seed2026110431冻结在panel.json，600拟合/200调参/200内部最终检查；检查按题目隔离。')
s=s.replace('所有草稿在同一公开MIB平均面积任务、ID与尺度下比较；','本次冻结CPU FP32单线程原模型推理和E98当前固定计算深度版后处理（源码与预算未改）。四个进程按题目并行，每个题目顺序执行五个模型，最多4个计算任务；数值路径与原GPU运行不同，原数学输入/解码及权重保持，不冒称历史草稿逐位复现。所有草稿在同一公开MIB平均面积任务、ID与尺度下比较；')
s=s.replace('本阶段到模型汇报为止，无自动后台任务。','后台入口launch.ps1，状态runs/20261104_bank1000x5_seed0/state.json，日志stdout.log/stderr.log。该run/STOP或实验STOP可停止后续任务，已完成逐例证据保留，不自动重试。原E98所有STOP保留。')
s+='''

## 已登记的离线分析与方向检查

600题拟合十项非负权重，量纲按拟合题内差异的75%分位数固定；4个正则强度(.0001/.01/1/10)并行，200题只用于调参，选定后冻结提案，另外200题仅终末检查。原系数作为先验与对照。按一半题目等权、一半官方exp(n/12)规模权重拟合同题排序。五种留出来源检查不参与重选超参或系数。

十项值在各模型原始网络state上用同一函数计算，实际解码草稿决定Q；保存原state及草稿，明确解码截断可能使不同state对应相同几何。随后在调参区预先固定40题的E98泛化草稿上，新/旧Loss各作.002/.01两个步长的RMS归一化梯度更新，共160次额外后处理。固定位置/形状通道不变，原解码保持面积。若两个步长均改善且优于原Loss，才建议登记下一步神经训练对照；否则先诊断表示不足或梯度错位。

所有1000题都在原训练分片及源模型的训练历史中，内部检查200只对系数拟合未见，不是网络未见验证集。四路并发耗时不作为部署逐例延迟。五例实施短检生成的25份正式面板草稿直接保留，不重复求解。无验证后处理Q调权，无测试访问，无额外模型。
'''
p.write_text(s,encoding='utf-8')
for name in ('AGENTS.md','README.md','docs/protocol.md','docs/workspace_structure.md','experiments/README.md'):
 p=root/name;s=p.read_text(encoding='utf-8')
 if name=='AGENTS.md':
  a=s.index('另授权E103离线Loss对齐研究');b=s.index('除此不得擅自新增候选',a)
  s=s[:a]+'另授权E103离线Loss对齐研究，已认可五个完成训练模型并允许自主推进和长任务后台4路并行；数据库为原train16k同1000题×5模型共5000原始草稿，使用E98固定计算深度后处理；按题目隔离600拟合/200调参/200检查，完成离线权重与40题局部方向检查，根据证据登记后续固定目标及有限神经训练。'+s[b:]
  s=s.replace('E103当前阶段仅模型整理与汇报，不自动启动数据库、拟合或长训练。','E103已授权后台数据库、离线分析与局部方向检查，四个单线程CPU任务并行；后续真实训练必须先根据证据登记固定目标和有限对照，不盲目扩展。')
 else:
  s+='\nE103用户已确认五模型并授权自主研究、后台长任务及4路并行：同1000训练题×5份原始草稿，E98固定计算深度后处理，600/200/200题目隔离的权重拟合与40题局部下降检查；状态见experiments/E103_offline_loss_alignment/runs/20261104_bank1000x5_seed0/state.json。E98不恢复，测试封存。\n'
 p.write_text(s,encoding='utf-8')
print('Frozen1000 panel and finite offline research plan; E98 postprocessor unmodified;4 processes.')
