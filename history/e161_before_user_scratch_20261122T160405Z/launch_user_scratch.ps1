# Formal training only. The user invokes this script; it does not prepare or test models.
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path $taskRoot '.conda/envs/iccad2026/python.exe'
$taskBackground = Join-Path $PSScriptRoot 'background'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$taskEnvironment = Split-Path -Parent $taskPython
$env:PATH = "$taskEnvironment;$taskEnvironment\Scripts;$taskEnvironment\Library\bin;" + $env:PATH
New-Item -ItemType Directory -Force -Path $taskBackground | Out-Null
$taskStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$taskLog = Join-Path $taskBackground "queue.$taskStamp.log"
$taskLaunch = [ordered]@{
    created_utc=[DateTime]::UtcNow.ToString('o')
    launcher_pid=$PID
    launch_owner='user'
    python=$taskPython
    command='-u -B -m experiments.E161_final_no_mib.run_queue --start'
    combined_log=$taskLog
    training_enabled=$true
    auto_dev2000_enabled=$true
    defensive_tests_run=$false
}
$taskEncoding = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText((Join-Path $taskBackground "launch.$taskStamp.json"), ($taskLaunch | ConvertTo-Json) + "`n", $taskEncoding)
$taskLaunchCode = 1
Push-Location $taskRoot
try {
    # Continue permits native stderr to reach Tee-Object on Windows PowerShell 5.1.
    # The native exit code is captured and returned after the formal queue exits.
    $ErrorActionPreference = 'Continue'
    & $taskPython -u -B -m experiments.E161_final_no_mib.run_queue --start 2>&1 |
        Tee-Object -FilePath $taskLog |
        Tee-Object -FilePath (Join-Path $taskBackground 'queue.stdout.log') -Append
    $taskLaunchCode = $LASTEXITCODE
} finally {
    Pop-Location
    $taskExit = [ordered]@{exit_code=$taskLaunchCode; finished_utc=[DateTime]::UtcNow.ToString('o'); combined_log=$taskLog}
    [System.IO.File]::WriteAllText((Join-Path $taskBackground "exit.$taskStamp.json"), ($taskExit | ConvertTo-Json) + "`n", $taskEncoding)
}
exit $taskLaunchCode
