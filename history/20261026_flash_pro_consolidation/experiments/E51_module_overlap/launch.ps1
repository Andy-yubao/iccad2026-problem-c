param(
    [ValidateSet('all','train','final','review')][string]$Stage='train',
    [ValidateSet('main')][string]$Arm='main',
    [switch]$Resume,
    [switch]$DryRun
)
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonPath = Join-Path $WorkspaceRoot '.conda\envs\iccad2026\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Missing iccad2026 Python environment" }
$Arguments = @('-u','-B','-m','experiments.E51_module_overlap.pipeline','--stage',$Stage,'--arm',$Arm)
if ($Resume) { $Arguments += '--resume' }
if ($DryRun) { $Arguments += '--dry-run' }
$PriorErrorPreference = $ErrorActionPreference
Push-Location $WorkspaceRoot
try {
    $ErrorActionPreference = 'Continue'
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "E51 stopped with exit code $LASTEXITCODE" }
} finally {
    $ErrorActionPreference = $PriorErrorPreference
    Pop-Location
}
