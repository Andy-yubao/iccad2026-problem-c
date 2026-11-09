$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskRuns = Join-Path $PSScriptRoot 'runs'
New-Item -ItemType Directory -Path $taskRuns -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $taskRuns '20261103_relation_guided_seed0')) { throw 'Formal run already exists; do not duplicate or overwrite.' }
$taskProc = Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E100_relation_guided_placer.train') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuns 'train.stdout.log') -RedirectStandardError (Join-Path $taskRuns 'train.stderr.log')
@{ pid=$taskProc.Id; started_utc=[DateTime]::UtcNow.ToString('o'); authorization='User explicitly requested implementation and background training of the relation-guided placer; one arm, four-hour cap'; test_read=$false } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRuns 'launch.json') -Encoding utf8
Write-Output ('Started PID ' + $taskProc.Id)
