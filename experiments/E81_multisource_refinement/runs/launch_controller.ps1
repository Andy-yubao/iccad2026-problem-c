$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$taskExperiment = Join-Path $taskRoot 'experiments/E81_multisource_refinement'
$taskOutput = Join-Path $taskExperiment 'runs/20261029_e42_multisource_seed0'
if (Test-Path -LiteralPath $taskOutput) { throw 'Formal output already exists; inspect state before an explicit resume.' }
if (Test-Path -LiteralPath (Join-Path $taskExperiment 'runs/STOP')) { throw 'STOP exists; do not override it.' }
$taskLock = [System.IO.File]::Open((Join-Path $taskExperiment 'runs/pipeline.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$taskLock.Dispose()
$taskStamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$taskStdout = Join-Path $taskExperiment "runs/controller_$taskStamp.stdout.log"
$taskStderr = Join-Path $taskExperiment "runs/controller_$taskStamp.stderr.log"
$taskPython = Join-Path $taskRoot '.conda/envs/iccad2026/python.exe'
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E81_multisource_refinement.pipeline') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
$taskRecord = @{controller_pid=$taskProcess.Id;started_utc=[DateTime]::UtcNow.ToString('o');stdout=$taskStdout;stderr=$taskStderr;command='python -u -B -m experiments.E81_multisource_refinement.pipeline';training_authorized=$true}
$taskRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskExperiment 'runs/controller_launch.json') -Encoding utf8
$taskRecord | ConvertTo-Json
