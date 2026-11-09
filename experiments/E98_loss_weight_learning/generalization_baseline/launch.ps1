$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskPython = Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskRun = Join-Path $taskRoot 'experiments\E98_loss_weight_learning\runs\20261104_generalization96_seed0'
if (Test-Path -LiteralPath $taskRun) { throw 'Formal run already exists; do not duplicate.' }
$taskGate = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'preflight.json') -Raw | ConvertFrom-Json
if ($taskGate.status -ne 'passed') { throw 'Five-case implementation check not passed.' }
$taskOut = Join-Path $PSScriptRoot 'stdout.log'
$taskErr = Join-Path $PSScriptRoot 'stderr.log'
$taskProc = Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E98_loss_weight_learning.generalization_baseline.train') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User: prepare and start generalization-enhanced baseline in background first; then prepare other work';test_read=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started PID '+$taskProc.Id)
$taskProc.WaitForExit()
if ($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0) { throw ('Training exited '+$taskProc.ExitCode) }
