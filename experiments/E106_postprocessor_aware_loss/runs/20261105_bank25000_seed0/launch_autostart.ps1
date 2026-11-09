$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
if(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'autostart_launch.json')){throw 'Autostart already launched; no duplicate'}
$taskPython=Join-Path $taskRoot '.conda\envs\iccad2026\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B',(Join-Path $PSScriptRoot 'autostart_after_data.py')) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'autostart_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'autostart_stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');authorization='User requested automatic critic training after data completes';layout_training=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'autostart_launch.json') -Encoding utf8
Write-Output ('Started E106 automatic data-to-critic handoff PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('E106 autostart exited '+$taskProc.ExitCode)}
