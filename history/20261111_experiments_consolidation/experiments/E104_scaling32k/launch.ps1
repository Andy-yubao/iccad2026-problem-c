$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/../..").Path
$python = Join-Path $root ".conda/envs/iccad2026/python.exe"
if (Test-Path "$PSScriptRoot/TRAINING_HOLD") { throw "User hold: preparation only; do not start" }
$run = Join-Path $PSScriptRoot "runs/20261104_scaling128x16_seed0"
if ((Test-Path "$PSScriptRoot/STOP") -or (Test-Path "$run/STOP")) { throw "STOP present" }
if (Test-Path "$run") { throw "Existing run; do not overwrite or restart automatically" }
$proc = Start-Process -FilePath $python -ArgumentList @("-u","-m","experiments.E104_scaling32k.pipeline") -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot/stdout.log" -RedirectStandardError "$PSScriptRoot/stderr.log" -PassThru
@{pid=$proc.Id; started_utc=[DateTime]::UtcNow.ToString("o"); authorization="Explicit background train and automatic dev500"} | ConvertTo-Json | Set-Content -Encoding utf8 "$PSScriptRoot/launch.json"
Write-Output "Started E104 controller PID $($proc.Id)"
