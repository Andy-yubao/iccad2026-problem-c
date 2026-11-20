$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\project\ICCAD\2026\problemC'
$taskFolder = Join-Path $taskWorkspace 'experiments\E119_global_initial_generator\evaluation_test500_20261109'
$taskPython = Join-Path $taskWorkspace '.conda\envs\iccad2026\python.exe'
$taskRunner = Join-Path $taskFolder 'run.py'
$taskName = 'Codex_ICCAD_E119_test500_20261109_1'
if (Test-Path -LiteralPath (Join-Path $taskFolder 'launch.json')) { throw 'Already dispatched; no automatic retry.' }
$taskService = New-Object -ComObject 'Schedule.Service'
$taskService.Connect()
$taskRoot = $taskService.GetFolder('\')
$existingTask = $null
try { $existingTask = $taskRoot.GetTask($taskName) } catch { }
if ($null -ne $existingTask) { throw 'Task name already exists; will not replace it.' }
$taskDefinition = $taskService.NewTask(0)
$taskDefinition.RegistrationInfo.Description = 'User-authorized one-time E119 best55 evaluation on frozen random test500, original depth budget; no training/retry.'
$taskDefinition.Principal.UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskDefinition.Principal.LogonType = 3
$taskDefinition.Principal.RunLevel = 0
$taskDefinition.Settings.Enabled = $true
$taskDefinition.Settings.Hidden = $true
$taskDefinition.Settings.DisallowStartIfOnBatteries = $false
$taskDefinition.Settings.StopIfGoingOnBatteries = $false
$taskDefinition.Settings.ExecutionTimeLimit = 'PT2H'
$taskDefinition.Settings.MultipleInstances = 2
$taskAction = $taskDefinition.Actions.Create(0)
$taskAction.Path = $taskPython
$taskAction.Arguments = '-B -u "' + $taskRunner + '"'
$taskAction.WorkingDirectory = $taskWorkspace
$registeredTask = $taskRoot.RegisterTaskDefinition($taskName, $taskDefinition, 2, $null, $null, 3, $null)
$runningTask = $registeredTask.Run($null)
[ordered]@{
    authorized = $true
    task_name = $taskName
    method = 'independent hidden Windows task; one-time manual dispatch, no trigger/retry'
    started_at = [DateTime]::UtcNow.ToString('o')
    scope = 'E119 best55; frozen test500; original E98.depth_budget.v1; no training or checkpoint selection'
    automatic_retry = $false
    test_read = $true
    selected_test_count = 500
    remaining_test1500_sealed = $true
    training = $false
    state = [int]$runningTask.State
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskFolder 'launch.json') -Encoding utf8
Write-Output 'Independent hidden test500 evaluation dispatched.'
