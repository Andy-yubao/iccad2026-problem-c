param([double]$MaxSeconds = 172800, [int]$MaxAttempts = 60000)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
& (Join-Path $taskRoot '.conda/envs/iccad2026/python.exe') -B -m experiments.E14_stable_teacher.archive_bank generate --max-seconds $MaxSeconds --max-attempts $MaxAttempts
if ($LASTEXITCODE -ne 0) { throw 'Layout archive generation failed; inspect teacher_archive25000_v1/last_error.json.' }
