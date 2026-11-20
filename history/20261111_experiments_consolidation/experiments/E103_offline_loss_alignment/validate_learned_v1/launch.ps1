$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun=Join-Path $taskRoot 'experiments\E103_offline_loss_alignment\runs\20261104_learned_v1_lr1e3_seed0\evaluation_depth_dev500'
if(Test-Path -LiteralPath (Join-Path $taskRun 'frozen.json')){throw 'Evaluation already registered; inspect before restart'}
$taskPython=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.validate_learned_v1.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User: training completed, please validate';panel='fixed dev500';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started frozen-best29 dev500 evaluation PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Validation exited '+$taskProc.ExitCode)}
