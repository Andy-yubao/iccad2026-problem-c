$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\project\ICCAD\2026\problemC'
$taskFolder = Join-Path $taskWorkspace 'experiments\E137_standard_transformer256'
$taskShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskName = 'Codex_ICCAD_E137_20261114'
if (Test-Path -LiteralPath (Join-Path $taskFolder 'background\launch.json')) { throw 'Already dispatched; do not retry.' }
$taskGate = Get-Content -LiteralPath (Join-Path $taskFolder 'preflight.json') -Raw | ConvertFrom-Json
$taskState = Get-Content -LiteralPath (Join-Path $taskFolder 'queue_state.json') -Raw | ConvertFrom-Json
if ($taskGate.status -ne 'passed' -or $taskState.status -ne 'prepared') { throw 'Preparation gate or queue state is not ready.' }
$taskService = New-Object -ComObject 'Schedule.Service'
$taskService.Connect()
$taskRoot = $taskService.GetFolder('\')
$existingTask = $null
try { $existingTask = $taskRoot.GetTask($taskName) } catch { }
if ($null -ne $existingTask) { throw 'Task already exists; will not replace it.' }
$taskDefinition = $taskService.NewTask(0)
$taskDefinition.RegistrationInfo.Description = 'User-authorized E137 standard256 scratch36k training plus frozen dev500 postprocessing and address diagnostic. One manual dispatch; no trigger or retry.'
$taskDefinition.Principal.UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskDefinition.Principal.LogonType = 3
$taskDefinition.Principal.RunLevel = 0
$taskDefinition.Settings.Enabled = $true
$taskDefinition.Settings.Hidden = $true
$taskDefinition.Settings.DisallowStartIfOnBatteries = $false
$taskDefinition.Settings.StopIfGoingOnBatteries = $false
$taskDefinition.Settings.ExecutionTimeLimit = 'PT24H'
$taskDefinition.Settings.MultipleInstances = 2
$taskAction = $taskDefinition.Actions.Create(0)
$taskAction.Path = $taskShell
$taskAction.Arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $taskFolder 'run_hidden.ps1') + '"'
$taskAction.WorkingDirectory = $taskWorkspace
$registeredTask = $taskRoot.RegisterTaskDefinition($taskName, $taskDefinition, 2, $null, $null, 3, $null)
$runningTask = $registeredTask.Run($null)
$taskBackground = Join-Path $taskFolder 'background'
[void](New-Item -ItemType Directory -Force -Path $taskBackground)
[ordered]@{
    authorized = $true
    task_name = $taskName
    method = 'independent hidden Windows task; manual one-shot dispatch; no trigger/retry'
    started_at = [DateTime]::UtcNow.ToString('o')
    scope = 'E137 standard256 seed0 scratch36k, then frozen original dev500 and address diagnostic, safe pause/resume'
    automatic_retry = $false
    test_read = $false
    launch_script_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskFolder 'launch.ps1') -Algorithm SHA256).Hash.ToLower()
    hidden_wrapper_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskFolder 'run_hidden.ps1') -Algorithm SHA256).Hash.ToLower()
    state = [int]$runningTask.State
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBackground 'launch.json') -Encoding utf8
Write-Output 'Independent hidden E137 standard256 queue dispatched.'

