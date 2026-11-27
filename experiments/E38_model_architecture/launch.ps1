param(
    [ValidateSet('all','base96x4','deep96x8','deep96x12','wide160x4')][string]$Arm='all',
    [ValidateSet('all','train','final')][string]$Stage='all',
    [string]$Tag='20261022_capacity',
    [int]$Seed=0,
    [double]$MaxHours=4,
    [int]$MaxEpochs=200,
    [switch]$Resume
)
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonPath = Join-Path $WorkspaceRoot '.conda\envs\iccad2026\python.exe'
$Arguments = @('-u','-B','-m','experiments.E38_model_architecture.pipeline',
    '--arm',$Arm,'--stage',$Stage,'--tag',$Tag,'--seed',"$Seed",'--max-hours',
    $MaxHours.ToString([System.Globalization.CultureInfo]::InvariantCulture),'--max-epochs',"$MaxEpochs")
if ($Resume) { $Arguments += '--resume' }
$PriorErrorPreference = $ErrorActionPreference
Push-Location $WorkspaceRoot
try {
    $ErrorActionPreference = 'Continue'
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "E38 stopped with exit code $LASTEXITCODE" }
} finally {
    $ErrorActionPreference = $PriorErrorPreference
    Pop-Location
}
