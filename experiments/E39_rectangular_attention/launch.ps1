param(
    [ValidateSet('pair','geometry_control','rectangle','rectangle_no_gap')][string]$Arm='pair',
    [ValidateSet('all','train','final')][string]$Stage='all',
    [ValidateSet(96,160)][int]$Width=96,
    [ValidateSet(4,8,12)][int]$Depth=4,
    [string]$Tag='20261022_rectangle',
    [int]$Seed=0,
    [double]$MaxHours=4,
    [int]$MaxEpochs=200,
    [switch]$Resume
)
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonPath = Join-Path $WorkspaceRoot '.conda\envs\iccad2026\python.exe'
$PriorWidth = $env:E39_WIDTH
$PriorDepth = $env:E39_DEPTH
$PriorErrorPreference = $ErrorActionPreference
$env:E39_WIDTH = "$Width"
$env:E39_DEPTH = "$Depth"
$Arguments = @('-u','-B','-m','experiments.E39_rectangular_attention.pipeline',
    '--arm',$Arm,'--stage',$Stage,'--tag',$Tag,'--seed',"$Seed",'--max-hours',
    $MaxHours.ToString([System.Globalization.CultureInfo]::InvariantCulture),'--max-epochs',"$MaxEpochs")
if ($Resume) { $Arguments += '--resume' }
Push-Location $WorkspaceRoot
try {
    $ErrorActionPreference = 'Continue'
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "E39 stopped with exit code $LASTEXITCODE" }
} finally {
    $ErrorActionPreference = $PriorErrorPreference
    $env:E39_WIDTH = $PriorWidth
    $env:E39_DEPTH = $PriorDepth
    Pop-Location
}
