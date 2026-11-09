$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskRun=Join-Path $PSScriptRoot 'runs\20261105_bank25000_seed0'
if(Test-Path -LiteralPath (Join-Path $taskRun 'bank_config.json')){throw 'Data run exists; no duplicate launch'}
$taskPython=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E106_postprocessor_aware_loss.build_bank') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'bank_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'bank_stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');data_only=$true;training=$false;workers=4}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'bank_launch.json') -Encoding utf8
Write-Output ('Started DATA ONLY pipeline PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Data pipeline exited '+$taskProc.ExitCode)}
