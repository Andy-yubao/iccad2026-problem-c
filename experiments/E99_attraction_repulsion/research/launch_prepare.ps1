param([switch]$Resume)
$ErrorActionPreference = "Stop"
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "../../..")).Path
$taskPython = Join-Path $taskRoot ".conda/envs/iccad2026/python.exe"
$planFile = Join-Path $PSScriptRoot "prepare_plan.json"
$planData = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
$studyFile = Join-Path $PSScriptRoot "study.json"
$studyData = Get-Content -LiteralPath $studyFile -Raw | ConvertFrom-Json
if ($studyData.PSObject.Properties.Name -contains "active_pid") {
    if ($studyData.active_pid -and (Get-Process -Id $studyData.active_pid -ErrorAction SilentlyContinue)) {
        throw "A recorded research process is still present; inspect it before launching again."
    }
}
$stamp = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
$logFolder = Join-Path $taskRoot ("experiments/E99_attraction_repulsion/runs/research/launches/prepare_" + $stamp)
New-Item -ItemType Directory -Path $logFolder | Out-Null
$taskArgs = @("-u", "-B", "-m", "experiments.E99_attraction_repulsion.research.prepare")
if ($Resume) { $taskArgs += "--resume" }
$child = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logFolder "stdout.log") -RedirectStandardError (Join-Path $logFolder "stderr.log") -PassThru
$studyData.status = "preparing_frozen_C3_training_control_background"
$studyData.current_run = $planData.output
$studyData | Add-Member -NotePropertyName active_pid -NotePropertyValue $child.Id -Force
$studyData | Add-Member -NotePropertyName launch_logs -NotePropertyValue $logFolder -Force
$studyData | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $studyFile -Encoding utf8
@{ pid = $child.Id; executable = $taskPython; arguments = $taskArgs; logs = $logFolder; output = $planData.output; resumed = [bool]$Resume } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $logFolder "launch.json") -Encoding utf8
Write-Output ("Started frozen C3 training-panel control in background. PID=" + $child.Id)
Write-Output $logFolder
