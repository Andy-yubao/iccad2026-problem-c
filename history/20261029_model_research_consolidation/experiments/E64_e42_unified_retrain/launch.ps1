$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskPython = Join-Path $taskRoot '.conda/envs/iccad2026/python.exe'
$taskRuns = Join-Path $PSScriptRoot 'runs'
New-Item -ItemType Directory -Path $taskRuns -Force | Out-Null
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E64_e42_unified_retrain.pipeline','--resume') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRuns 'controller.log') -RedirectStandardError (Join-Path $taskRuns 'controller.stderr.log') -PassThru
[ordered]@{controller_pid=$taskProcess.Id; launched_utc=[DateTime]::UtcNow.ToString('o'); command="$taskPython -u -B -m experiments.E64_e42_unified_retrain.pipeline --resume"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRuns 'launch.json') -Encoding utf8
Write-Output "E64 background controller PID=$($taskProcess.Id)"
