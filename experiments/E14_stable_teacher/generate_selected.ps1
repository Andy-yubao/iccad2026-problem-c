param([double]$MaxSeconds = 172800, [int]$MaxSolves = 60000)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
& (Join-Path $taskRoot '.conda/envs/iccad2026/python.exe') -B -m experiments.E14_stable_teacher.selective_bank generate --max-seconds $MaxSeconds --max-solves $MaxSolves
if ($LASTEXITCODE -ne 0) { throw 'Selected teacher generation failed; inspect last_error.json before retrying.' }
