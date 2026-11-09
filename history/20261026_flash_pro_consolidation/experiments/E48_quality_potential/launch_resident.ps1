param(
    [ValidateSet('gate','screen')][string]$Stage = 'screen',
    [switch]$Resume,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$PotentialRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$PotentialPython = Join-Path $PotentialRoot '.conda/envs/iccad2026/python.exe'
$PotentialArgs = @('-m','experiments.E48_quality_potential.resident_run','--stage',$Stage)
if ($Resume) { $PotentialArgs += '--resume' }
if ($DryRun) { $PotentialArgs += '--dry-run' }
Push-Location $PotentialRoot
try {
    & $PotentialPython @PotentialArgs
    if ($LASTEXITCODE -ne 0) { throw '评估停止，逐例结果已保留；请核对错误后恢复。' }
} finally { Pop-Location }
