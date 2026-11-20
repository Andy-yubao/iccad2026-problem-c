from pathlib import Path
import json,hashlib
import numpy as np
import torch
from datasets.reference_layouts import registry,REGISTRY
root=Path.cwd();exp=root/'experiments/E103_offline_loss_alignment';owner=exp/'global_loss_q'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
records=registry()['sets']['train']['records'];old=read(exp/'panel.json')
allowed=np.array(sorted(set(range(16000))-set(old['train_indices'])))
chosen=np.random.default_rng(2026110461).choice(allowed,2000,replace=False)
assignment=np.random.default_rng(2026110462).permutation(2000)
model_names=np.empty(2000,dtype=object);split=np.empty(2000,dtype=object)
for mid,positions in [('enhanced_best71',assignment[:1000]),('current_best29',assignment[1000:])]:
 model_names[positions]=mid
 positions=np.random.default_rng(np.random.SeedSequence([2026110463,0 if mid=='enhanced_best71' else 1])).permutation(positions)
 split[positions[:700]]='fit';split[positions[700:]]='check'
samples=[dict(index=i,train_index=int(idx),member=records[int(idx)]['member'],n=records[int(idx)]['n'],model_id=model_names[i],split=split[i]) for i,idx in enumerate(chosen)]
implementation=[]
for target in (21,60,90,110,120):
 i=min((j for j in range(2000) if j not in implementation),key=lambda j:(abs(samples[j]['n']-target),j));implementation.append(i)
if len({samples[i]['model_id'] for i in implementation})<2:
 replacement=next(i for i in range(2000) if samples[i]['model_id']!=samples[implementation[0]]['model_id']);implementation[-1]=replacement
panel=dict(samples=samples,implementation_indices=implementation,sampling_seed=2026110461,model_assignment_seed=2026110462,
 split_seed=2026110463,excluded_old_train_indices=old['train_indices'],new_distinct_problems=2000,
 old_Loss_Q_observations_used=0,registry_sha256=sha(REGISTRY),test_read=False)
(owner/'panel.json').write_text(json.dumps(panel,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
parent=exp/'assets/e98_generalization/best.pt';current=exp/'runs/20261104_learned_v1_lr1e3_seed0/checkpoints/best.pt'
assert sha(parent)=='3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea'
assert sha(current)=='85d21d2dee2a6266894a9bf0036210c3a0ff7c2290ca0a6ec0b4e32784c84325'
current_coefficients=torch.load(current,map_location='cpu',weights_only=False)['coefficients']
plan=dict(revision='E103.global_independent_LossQ.v1',authorization='User explicitly corrected objective to pooled cross-problem Loss-Q statistics, old data not used, two current models only, requested analysis and Loss comparison figures.',
 models=[dict(id='enhanced_best71',checkpoint=parent.relative_to(root).as_posix(),sha256=sha(parent)),dict(id='current_best29',checkpoint=current.relative_to(root).as_posix(),sha256=sha(current))],
 current_training_coefficients=current_coefficients,
 sampling='2000 fresh distinct original train problems uniformly without replacement after excluding old1000; random balanced model assignment1000 each; one draft/problem; no pairing or matching by Q/Loss.',
 fit=1400,check=600,split='Balanced700/300 per source before any Q observed. Global fit-only centering/scaling; no within-problem centering, differences or ranks.',
 observations='Generate all2000 drafts/Q anew, same E98 fixed-depth postprocessor; four CPU1 workers. Raw dataset tensor cache allowed; no old draft/Loss/Q labels or prior fitted weights used.',
 statistics='Unweighted global Pearson/Spearman; primary nonnegative least squares with free intercept on standardized raw Q/ten terms. No hyperparameter search or prior penalty. Convert coefficients to raw units, normalize physical coefficient total to original E98.',
 retained_data='All Q values including equal1 and high tails, all memberships and source IDs retained. No clipping/winsorizing for primary correlations.',
 secondary_diagnostics='Per-source correlations, partial Pearson controlling linear module countN, fit-only partial correlations controlling other9 terms, signed standardized regression diagnostic; primary remains pooled cross-problem association.',
 bootstrap='128 source-worker cluster resamples for coefficient stability and heldout Pearson-gain interval; do not select bootstrap winners.',
 figures='Standalone PNG/PDF: original/current/new Loss vs Q on600 check observations, full range plus explicitly labeled body zoom; feature correlations and joint standardized coefficients.',
 neural_training_started=False,test_read=False,maximum_new_observations=2000,
 caveat='Problem difficulty and size confounding intentionally retained for primary goal, association not causal training usefulness. All observations are from source-network training pool.',
 stop='Newrun STOP or package STOP; accepted case boundary stop; no extra sources, model training, or automatic retries.')
(owner/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(owner/'launch.ps1').write_text('''$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..')).Path
$taskRun=Join-Path $taskRoot 'experiments\\E103_offline_loss_alignment\\runs\\20261104_global_independent2000_seed0'
if(Test-Path -LiteralPath (Join-Path $taskRun 'config.json')){throw 'Global analysis already started'}
if((Get-Content -LiteralPath (Join-Path $PSScriptRoot 'preflight.json') -Raw|ConvertFrom-Json).status -ne 'passed'){throw 'Preflight did not pass'}
$taskPython=Join-Path $taskRoot '.conda\\envs\\iccad2026\\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.global_loss_q.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');workers=4;authorization='User requested fresh pooled global Loss-Q analysis, two models, show comparison plots';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started global independent Loss-Q analysis PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Global analysis exited '+$taskProc.ExitCode)}
''',encoding='utf-8')
p=root/'AGENTS.md';s=p.read_text(encoding='utf-8');needle='除此不得擅自新增候选、恢复旧实验或自动唤醒。'
s=s.replace(needle,'用户最新纠正为跨题目全局Loss–Q统计：旧草稿/Q及排序结果搁置，仅增强best71和当前best29，排除旧1000题后新抽2000不同训练题、每题一草稿、各模型1000；四路后台重算，1400拟合/600检查，样本等权非负全局回归和相关性，不同题内排序，输出对比图，不自动训练，dev/test不参与。'+needle,1);p.write_text(s,encoding='utf-8')
entry='\nE103最新研究目标已纠正为跨题目全局Loss–Q相关性：旧排序数据不用于新分析；增强best71/当前best29各负责新1000道不同训练题，共2000独立观察，排除旧1000题，所有草稿/Q重新计算。1400拟合/600检查，样本等权全局非负回归、相关性和图，保留问题间差异，不进行同题排序。入口experiments/E103_offline_loss_alignment/global_loss_q，输出runs/20261104_global_independent2000_seed0；测试封存、不自动训练。\n'
for name in ('README.md','docs/protocol.md','docs/workspace_structure.md','experiments/README.md'):
 p=root/name;s=p.read_text(encoding='utf-8');p.write_text(s+entry,encoding='utf-8')
p=exp/'README.md';s=p.read_text(encoding='utf-8');p.write_text(s+'\n## 当前全局统计研究\n\n用户确认的主目标是跨题目全局Loss–Q关联，旧同题排序路线与数据仅保留历史。[新全局统计](global_loss_q/README.md)使用排除旧1000题后的新2000不同训练题，仅增强/当前两个模型，每题一个观察；1400拟合/600检查，输出Loss–Q对比图和全局权重，不自动训练。\n',encoding='utf-8')
print('Frozen fresh2000 unpaired problems,1000 per model,1400 fit/600 check; old observations excluded.')
