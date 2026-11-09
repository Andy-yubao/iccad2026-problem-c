$ErrorActionPreference = 'Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun=Join-Path $taskRoot 'experiments\E98_loss_weight_learning\runs\20261104_weaken_reference120_seed0'
if(Test-Path -LiteralPath (Join-Path $taskRun 'config.json')){throw 'Run already started; no duplicate.'}
if((Get-Content -LiteralPath (Join-Path $PSScriptRoot 'preflight.json') -Raw|ConvertFrom-Json).status -ne 'passed'){throw 'Implementation gate not passed.'}
$taskPython=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E98_loss_weight_learning.single_reference_step.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User explicitly requested continuing only one weakening of position/shape; previous search results excluded';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Pipeline exited '+$taskProc.ExitCode)}
