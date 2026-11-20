param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$chainBundle = $PSScriptRoot
$chainWorkspace = (Get-Item -LiteralPath $chainBundle).Parent.Parent.Parent.Parent.FullName
$chainPython = Join-Path $chainWorkspace '.conda\envs\iccad2026\python.exe'
$chainArguments = @('-u','-B',"$chainBundle\run.py")
if ($Resume) { $chainArguments += '--resume' }
$chainStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$chainProcess = Start-Process -FilePath $chainPython -ArgumentList $chainArguments -WorkingDirectory $chainWorkspace -WindowStyle Hidden -RedirectStandardOutput "$chainBundle\pipeline_$chainStamp.log" -RedirectStandardError "$chainBundle\pipeline_${chainStamp}_stderr.log" -PassThru
[pscustomobject]@{pipeline_pid=$chainProcess.Id;started_utc=[DateTime]::UtcNow.ToString('o');resume=[bool]$Resume;scope='all three64k stages -> freeze -> dev500 -> repeated registered test2000';serial_after='20261103_replay evaluation complete';stdout="pipeline_$chainStamp.log";stderr="pipeline_${chainStamp}_stderr.log"} | ConvertTo-Json | Set-Content -LiteralPath "$chainBundle\launch_$chainStamp.json" -Encoding utf8
Get-Content -LiteralPath "$chainBundle\launch_$chainStamp.json"
