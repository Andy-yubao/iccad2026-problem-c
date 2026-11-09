$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun=Join-Path $PSScriptRoot 'runs\20261104_mechanism_v1'
if(Test-Path -LiteralPath (Join-Path $taskRun 'config.json')){throw 'Diagnosis already launched'}
$taskPython=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.mechanism_study.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');workers=4;authorization='Q sanity passed; user requested advancing E103';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started four-way mechanism diagnosis PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Mechanism diagnosis exited '+$taskProc.ExitCode)}
