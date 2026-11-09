$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun=Join-Path $taskRoot 'experiments\E103_offline_loss_alignment\runs\20261104_global_independent2000_seed0'
if(Test-Path -LiteralPath (Join-Path $taskRun 'config.json')){throw 'Global analysis already started'}
if((Get-Content -LiteralPath (Join-Path $PSScriptRoot 'preflight.json') -Raw|ConvertFrom-Json).status -ne 'passed'){throw 'Preflight did not pass'}
$taskPython=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.global_loss_q.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');workers=4;authorization='User requested fresh pooled global Loss-Q analysis, two models, show comparison plots';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started global independent Loss-Q analysis PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Global analysis exited '+$taskProc.ExitCode)}
