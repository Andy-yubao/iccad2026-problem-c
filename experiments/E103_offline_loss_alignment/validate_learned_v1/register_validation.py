from pathlib import Path
import ast
owner=Path('experiments/E103_offline_loss_alignment/validate_learned_v1')
for path in owner.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
(owner/'launch.ps1').write_text('''$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..')).Path
$taskRun=Join-Path $taskRoot 'experiments\\E103_offline_loss_alignment\\runs\\20261104_learned_v1_lr1e3_seed0\\evaluation_depth_dev500'
if(Test-Path -LiteralPath (Join-Path $taskRun 'frozen.json')){throw 'Evaluation already registered; inspect before restart'}
$taskPython=Join-Path $taskRoot '.conda\\envs\\iccad2026\\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.validate_learned_v1.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User: training completed, please validate';panel='fixed dev500';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started frozen-best29 dev500 evaluation PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Validation exited '+$taskProc.ExitCode)}
''',encoding='utf-8')
root=Path.cwd()
for name in ('AGENTS.md','docs/protocol.md','README.md'):
 p=root/name;s=p.read_text(encoding='utf-8')
 if name=='AGENTS.md':
  s=s.replace('不新增其他候选或自动Q评价。','不新增其他候选。用户已在训练完成后明确要求验证：冻结best29，用E98固定计算深度版同dev500评价，基线五题复现后只读复用父模型已完成500，报告三Q/硬合法/草稿/耗时，不按Q重选，不读测试。')
 else:s+='\n用户在E103续训32轮正常早停后明确要求验证：唯一新Loss best29已冻结，按原dev500及E98固定计算深度后处理评价；基线best71五题复现后复用已完成500结果，不按Q重选权重，测试封存。结果位于experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500。\n'
 p.write_text(s,encoding='utf-8')
print('Frozen best29, same-dev500 validation registered; terminal evaluation only.')
