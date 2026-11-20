$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\project\ICCAD\2026\problemC'
$taskFolder = Join-Path $taskWorkspace 'experiments\E119_global_initial_generator\evaluation_test500_20261109\recovery1_20261109'
$taskName = 'Codex_ICCAD_E119_test500_20261109_recovery1'
if ((Test-Path -LiteralPath (Join-Path $taskFolder 'launch.json')) -or (Test-Path -LiteralPath (Join-Path $taskFolder 'supervisor_state.json'))) { throw 'Recovery already dispatched; no retry.' }
$taskService = New-Object -ComObject 'Schedule.Service'
$taskService.Connect()
$taskRoot = $taskService.GetFolder('\')
$existingTask = $null
try { $existingTask = $taskRoot.GetTask($taskName) } catch { }
if ($null -ne $existingTask) { throw 'Task exists; no replacement.' }
$taskDefinition = $taskService.NewTask(0)
$taskDefinition.RegistrationInfo.Description = 'Explicit user-requested single recovery of E119 best55 original-budget frozen test500; console-free supervisor, no auto retry.'
$taskDefinition.Principal.UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskDefinition.Principal.LogonType = 3
$taskDefinition.Principal.RunLevel = 0
$taskDefinition.Settings.Hidden = $true
$taskDefinition.Settings.DisallowStartIfOnBatteries = $false
$taskDefinition.Settings.StopIfGoingOnBatteries = $false
$taskDefinition.Settings.RunOnlyIfIdle = $false
$taskDefinition.Settings.IdleSettings.StopOnIdleEnd = $false
$taskDefinition.Settings.RestartCount = 0
$taskDefinition.Settings.ExecutionTimeLimit = 'PT2H'
$taskDefinition.Settings.MultipleInstances = 2
$taskAction = $taskDefinition.Actions.Create(0)
$taskAction.Path = Join-Path $taskWorkspace '.conda\envs\iccad2026\pythonw.exe'
$taskAction.Arguments = '-B "' + (Join-Path $taskFolder 'supervisor.py') + '"'
$taskAction.WorkingDirectory = $taskWorkspace
$registeredTask = $taskRoot.RegisterTaskDefinition($taskName, $taskDefinition, 2, $null, $null, 3, $null)
$runningTask = $registeredTask.Run($null)
[ordered]@{ authorized_recovery = $true; attempt = 1; task_name = $taskName; method = 'pythonw supervisor; worker CREATE_NO_WINDOW and new process group'; started_at = [DateTime]::UtcNow.ToString('o'); automatic_retry = $false; selected_test_count = 500; original_budget_unchanged = $true; state = [int]$runningTask.State } | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $taskFolder 'launch.json')
Write-Output 'Single console-free background recovery dispatched.'
