$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\project\ICCAD\2026\problemC'
Set-Location -LiteralPath $taskRoot
$py = Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$runRoot = Join-Path $taskRoot 'experiments\E97_gpu_hybrid_solver\runs'
if (Test-Path -LiteralPath (Join-Path $runRoot 'candidate7_launch.json')) { throw 'Candidate7 already launched; inspect logs and locks, do not duplicate' }
$c1 = Get-Content -LiteralPath (Join-Path $runRoot '20261102_candidate6_deep_geometry_seed0\evaluation\results.json') -Raw | ConvertFrom-Json
if (-not $c1.finished) { throw 'Previous evaluation not complete' }
$check = Get-Content -LiteralPath (Join-Path $runRoot '20261102_candidate7_preflight\results.json') -Raw | ConvertFrom-Json
if ($check.status -ne 'passed') { throw 'Preflight not passed' }
$env:PYTHONUTF8 = '1'
$worker = Start-Process -FilePath $py -ArgumentList @('-u','-B','-m','experiments.E97_gpu_hybrid_solver.candidate7_pipeline') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot 'candidate7.stdout.log') -RedirectStandardError (Join-Path $runRoot 'candidate7.stderr.log') -PassThru
[ordered]@{ process_id=$worker.Id; started_at=(Get-Date).ToString('o'); command=@($py,'-u','-B','-m','experiments.E97_gpu_hybrid_solver.candidate7_pipeline'); candidate=7; training_cap_seconds=3600; global_deadline_utc='2026-11-02T02:58:12Z'; automatic_next_candidate=$false; test_read=$false } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runRoot 'candidate7_launch.json') -Encoding utf8
Write-Output ('Started candidate7 pipeline PID '+$worker.Id)
