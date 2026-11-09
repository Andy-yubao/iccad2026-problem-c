$ErrorActionPreference='Stop'
$taskRoot='E:\project\ICCAD\2026\problemC'
Set-Location -LiteralPath $taskRoot
$runRoot=Join-Path $taskRoot 'experiments\E97_gpu_hybrid_solver\runs'
if (Test-Path -LiteralPath (Join-Path $runRoot 'round2_candidate1_launch.json')) { throw 'Round2 candidate1 already launched; inspect existing logs/locks' }
$control=Get-Content -LiteralPath (Join-Path $runRoot '20261102_round2_current_control\results.json') -Raw | ConvertFrom-Json
$check=Get-Content -LiteralPath (Join-Path $runRoot '20261102_round2_candidate1_preflight\results.json') -Raw | ConvertFrom-Json
if (-not $control.finished -or $check.status -ne 'passed') { throw 'Control or preflight incomplete' }
if ([DateTimeOffset]::UtcNow -ge [DateTimeOffset]::Parse('2026-11-02T09:04:02Z')) { throw 'Round2 deadline reached' }
$py=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$env:PYTHONUTF8='1'
$worker=Start-Process -FilePath $py -ArgumentList @('-u','-B','-m','experiments.E97_gpu_hybrid_solver.round2_candidate1_pipeline') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot 'round2_candidate1.stdout.log') -RedirectStandardError (Join-Path $runRoot 'round2_candidate1.stderr.log') -PassThru
[ordered]@{process_id=$worker.Id;started_at=(Get-Date).ToString('o');command=@($py,'-u','-B','-m','experiments.E97_gpu_hybrid_solver.round2_candidate1_pipeline');training_cap_seconds=3600;deadline_utc='2026-11-02T09:04:02Z';cpu_threads=1;test_read=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'round2_candidate1_launch.json') -Encoding utf8
Write-Output ('Started Round2 candidate1 pipeline '+$worker.Id)
