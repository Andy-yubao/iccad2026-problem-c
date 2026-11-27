param(
    [ValidateSet('pair','geometry_control','full_tokens','pooled_tokens','parameter_control')][string]$Arm='pair',
    [ValidateSet('all','train','final')][string]$Stage='all',
    [ValidateSet(96,160)][int]$Width=96,
    [ValidateSet(4,8,12)][int]$Depth=4,
    [string]$Tag='20261023_tokens',
    [int]$Seed=0,
    [double]$MaxHours=4,
    [int]$MaxEpochs=200,
    [switch]$Resume
)
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonPath = Join-Path $WorkspaceRoot '.conda\envs\iccad2026\python.exe'
$PriorWidth = $env:E40_WIDTH
$PriorDepth = $env:E40_DEPTH
$PriorErrorPreference = $ErrorActionPreference
$env:E40_WIDTH = "$Width"
$env:E40_DEPTH = "$Depth"
$Arguments = @('-u','-B','-m','experiments.E40_endpoint_constraint_tokens.pipeline',
    '--arm',$Arm,'--stage',$Stage,'--tag',$Tag,'--seed',"$Seed",'--max-hours',
    $MaxHours.ToString([System.Globalization.CultureInfo]::InvariantCulture),'--max-epochs',"$MaxEpochs")
if ($Resume) { $Arguments += '--resume' }
Push-Location $WorkspaceRoot
try {
    $ErrorActionPreference = 'Continue'
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "E40 stopped with exit code $LASTEXITCODE" }
} finally {
    $ErrorActionPreference = $PriorErrorPreference
    $env:E40_WIDTH = $PriorWidth
    $env:E40_DEPTH = $PriorDepth
    Pop-Location
}
