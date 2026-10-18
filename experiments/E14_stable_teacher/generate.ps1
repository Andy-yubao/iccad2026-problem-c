param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.conda/envs/iccad2026/python.exe'
$taskStages = @(
    @{ Split = 'train'; Output = 'experiments/E14_stable_teacher/runs/teacher_train16000_v1'; Budget = 86400 },
    @{ Split = 'dev_monitor'; Output = 'experiments/E14_stable_teacher/runs/teacher_devmonitor500_v1'; Budget = 14400 }
)
foreach ($taskStage in $taskStages) {
    $taskArgs = @('-B', '-m', 'experiments.E14_stable_teacher.bank', '--split', $taskStage.Split,
                  '--output', $taskStage.Output, '--max-seconds', [string]$taskStage.Budget)
    if ($Resume -and (Test-Path -LiteralPath $taskStage.Output)) { $taskArgs += '--resume' }
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) { throw "Teacher stage failed: $($taskStage.Split)" }
    $taskState = Get-Content -Raw -LiteralPath (Join-Path $taskStage.Output 'progress.json') | ConvertFrom-Json
    if ($taskState.status -ne 'completed') { break }
}
