$ErrorActionPreference = 'Stop'
$viewerWorkspace = 'E:\project\ICCAD\2026\problemC'
$viewerRuns = Join-Path $viewerWorkspace 'experiments\E135_initial_supervision\runs'
$viewerName = 'Codex_ICCAD_E135_viewers_20261114'
$viewerService = New-Object -ComObject 'Schedule.Service'
$viewerService.Connect()
$viewerRoot = $viewerService.GetFolder('\')
$viewerExisting = $null
try { $viewerExisting = $viewerRoot.GetTask($viewerName) } catch { }
if ($null -ne $viewerExisting) { throw 'Viewer task already exists; will not replace or retry.' }
$viewerDefinition = $viewerService.NewTask(0)
$viewerDefinition.RegistrationInfo.Description = 'User-requested E135 A/B/C read-only visualizations; one shared viewer implementation, localhost only, no training or scoring.'
$viewerDefinition.Principal.UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$viewerDefinition.Principal.LogonType = 3
$viewerDefinition.Principal.RunLevel = 0
$viewerDefinition.Settings.Hidden = $true
$viewerDefinition.Settings.Enabled = $true
$viewerDefinition.Settings.DisallowStartIfOnBatteries = $false
$viewerDefinition.Settings.StopIfGoingOnBatteries = $false
$viewerDefinition.Settings.ExecutionTimeLimit = 'PT0S'
$viewerDefinition.Settings.MultipleInstances = 2
$viewerAction = $viewerDefinition.Actions.Create(0)
$viewerAction.Path = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$viewerAction.Arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $viewerRuns 'run_viewers_hidden.ps1') + '"'
$viewerAction.WorkingDirectory = $viewerWorkspace
$viewerRegistered = $viewerRoot.RegisterTaskDefinition($viewerName,$viewerDefinition,2,$null,$null,3,$null)
$viewerRunning = $viewerRegistered.Run($null)
[ordered]@{task=$viewerName; state=[int]$viewerRunning.State; ports=@(8784,8785,8786); read_only=$true; restart=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $viewerRuns 'viewer_launch.json') -Encoding utf8
Write-Output 'Hidden E135 saved-gallery services dispatched.'
