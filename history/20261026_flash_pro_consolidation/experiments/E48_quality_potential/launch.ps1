param(
    [ValidateSet('gate','screen')][string]$Stage = 'screen',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$PotentialRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$PotentialPython = Join-Path $PotentialRoot '.conda/envs/iccad2026/python.exe'
$PotentialRun = if ($Stage -eq 'gate') { '20261024_gate5_v4' } else { '20261024_screen100_frozen_v4' }
$PotentialExecution = Join-Path $PSScriptRoot "runs/$PotentialRun/execution.json"
if ((Test-Path -LiteralPath $PotentialExecution) -and (Get-Content -LiteralPath $PotentialExecution -Raw | ConvertFrom-Json).complete) {
    Write-Host "该轮已完成，不重复推理或求解：$PotentialRun"
    Write-Host (Join-Path $PSScriptRoot "runs/$PotentialRun/summary.json")
    return
}
$PotentialArgs = @('-m','experiments.E48_quality_potential.run','--stage',$Stage,'--output',(Join-Path $PSScriptRoot "runs/$PotentialRun"),'--gate',(Join-Path $PSScriptRoot 'runs/20261024_gate5_v4'))
if ($Resume) { $PotentialArgs += '--resume' }
Push-Location $PotentialRoot
try {
    & $PotentialPython @PotentialArgs
    if ($LASTEXITCODE -ne 0) { throw '评估停止；已保存的逐例记录保留，请查看错误原因后再恢复。' }
} finally { Pop-Location }
