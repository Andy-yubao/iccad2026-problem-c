$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskPython = Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskRun = Join-Path $taskRoot 'experiments\E98_loss_weight_learning\runs\20261104_dynamic_mixedq_seed0'
if (Test-Path -LiteralPath (Join-Path $taskRun 'config.json')) { throw 'Run already started; refusing duplicate launch.' }
if (Test-Path -LiteralPath (Join-Path $taskRun 'STOP')) { throw 'STOP exists.' }
$taskGate = Get-Content -LiteralPath (Join-Path $taskRun 'preflight.json') -Raw | ConvertFrom-Json
if ($taskGate.status -ne 'passed') { throw 'Implementation short check has not passed.' }
$taskProc = Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E98_loss_weight_learning.dynamic_weight_learning.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'stdout.log') -RedirectStandardError (Join-Path $taskRun 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User explicitly requested five designed directions followed by seeded multi-round search, background launch; train feedback100/1000, validation Loss early stop, test sealed';test_read=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRun 'launch.json') -Encoding utf8
Write-Output ('Started PID '+$taskProc.Id)
