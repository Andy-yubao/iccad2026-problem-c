param(
    [ValidateSet('Prepare','Train','Final')][string]$Stage = 'Prepare',
    [ValidateSet('All','supervised_small','curriculum_small','curriculum_large')][string]$Arm = 'All',
    [string]$Device = 'cuda',
    [string]$RunPrefix = '20261019_reference_v5',
    [ValidateRange(0.01,1000)][double]$TotalTrainingHours = 8,
    [ValidateRange(1,8)][int]$Workers = 2,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
if ($RunPrefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunPrefix must contain letters, digits, underscores or hyphens' }
if ($env:E16_REGISTRY) { throw 'Remove E16_REGISTRY pilot override before using the formal launcher' }
$arms = if ($Arm -eq 'All') { @('supervised_small','curriculum_small','curriculum_large') } else { @($Arm) }
$armHours = ($TotalTrainingHours / 3).ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
Push-Location $repo
try {
    if ($Stage -eq 'Prepare') {
        & $python -B -m experiments.E16_reference_curriculum.prepare --workers 4
        if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed; training was not started' }
        $dataState = Join-Path $PSScriptRoot 'runs\20261019_balanced_v5_seed20261019\progress.json'
        if ((Test-Path -LiteralPath $dataState) -and ((Get-Content -LiteralPath $dataState -Raw | ConvertFrom-Json).status -ne 'complete')) {
            Write-Host 'Dataset preparation paused. Remove STOP if present, then rerun the same command.'
            return
        }
        & $python -B -m experiments.E16_reference_curriculum.cache --prepare --workers 4
        if ($LASTEXITCODE -ne 0) { throw 'Shared reference cache preparation failed' }
        foreach ($role in @('train','dev')) {
            if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "runs\prepared_v5_cache_v1\$role\manifest.json"))) {
                Write-Host 'Cache preparation paused. Remove cache STOP if present, then rerun Prepare.'
                return
            }
        }
        Write-Host 'Dataset members and shared cache are ready. No training was started.'
        return
    }
    foreach ($item in $arms) {
        $output = Join-Path $PSScriptRoot "runs\${RunPrefix}_${item}_seed0"
        if ($Stage -eq 'Final') {
            $arguments = @('-B','-m','experiments.E16_reference_curriculum.final_evaluate','--run',$output,
                           '--device',$Device,'--workers',"$Workers")
            if ($Resume) { $arguments += '--resume' }
            & $python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Final evaluation $item failed" }
            if (-not (Test-Path -LiteralPath (Join-Path $output 'final_evaluation\results.json'))) { break }
            continue
        }
        foreach ($role in @('train','dev')) {
            if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "runs\prepared_v5_cache_v1\$role\manifest.json"))) {
                throw 'Run -Stage Prepare to completion before training'
            }
        }
        $arguments = @('-B','-m','experiments.E16_reference_curriculum.train','--arm',$item,
                       '--output',$output,'--device',$Device,'--seed','0',
                       '--max-hours',$armHours,'--max-epochs','200','--batch-size','16')
        if ($Resume -and (Test-Path -LiteralPath (Join-Path $output 'checkpoints\latest.pt'))) { $arguments += '--resume' }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Training $item failed; subsequent arms were not started" }
        $progress = Get-Content -LiteralPath (Join-Path $output 'progress.json') -Raw | ConvertFrom-Json
        if (-not $progress.finished) {
            Write-Host "Training $item stopped recoverably; subsequent arms were not started."
            break
        }
    }
}
finally { Pop-Location }
