param(
    [ValidateSet('train','final','all')][string]$Stage='train',
    [ValidateSet('all','no_reference','reference_only')][string]$Arm='all',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonPath = Join-Path $workspaceRoot '.conda\envs\iccad2026\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Missing project Python: $pythonPath" }
$pipelineArgs = @('-u','-B','-m','experiments.E54_reference_extremes.pipeline','--stage',$Stage,'--arm',$Arm)
if ($Resume) { $pipelineArgs += '--resume' }
Push-Location $workspaceRoot
try { & $pythonPath @pipelineArgs; if ($LASTEXITCODE -ne 0) { throw "E54 exited with code $LASTEXITCODE" } }
finally { Pop-Location }
