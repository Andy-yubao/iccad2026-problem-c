param([double]$MaxSeconds = 172800, [int]$MaxSolves = 60000)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
& (Join-Path $taskRoot '.conda/envs/iccad2026/python.exe') -B -m experiments.E14_stable_teacher.parallel_bank generate --workers 4 --draft-device cuda --batch-size 8 --max-seconds $MaxSeconds --max-solves $MaxSolves
if ($LASTEXITCODE -ne 0) { throw 'Parallel teacher generation failed; inspect the parallel bank last_error.json.' }
