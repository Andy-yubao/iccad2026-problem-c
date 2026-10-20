param(
    [ValidateSet('All','Remaining','supervised_small','curriculum_small','curriculum_large')]
    [string]$Arm = 'All',
    [string]$Device = 'cuda',
    [string]$RunPrefix = '20261019_cached',
    [ValidateSet('Train','Final')]
    [string]$Stage = 'Train',
    [ValidateRange(1,8)]
    [int]$Workers = 2,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
if ($RunPrefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunPrefix must contain letters, digits, underscores or hyphens' }
$arms = if ($Arm -eq 'All') { @('supervised_small','curriculum_small','curriculum_large') } elseif ($Arm -eq 'Remaining') { @('curriculum_small','curriculum_large') } else { @($Arm) }
Push-Location $repo
try {
    if ($Stage -eq 'Train') {
        & $python -B -m experiments.E15_quality_curriculum.cache --prepare
        if ($LASTEXITCODE -ne 0) { throw 'Shared cache preparation failed; no training was started' }
    }
    foreach ($item in $arms) {
        $output = Join-Path $PSScriptRoot "runs\${RunPrefix}_${item}_seed0"
        if ($Stage -eq 'Final') {
            $finalArgs = @('-B','-m','experiments.E15_quality_curriculum.final_evaluate','--run',$output,'--device',$Device,'--workers',"$Workers")
            if ($Resume) { $finalArgs += '--resume' }
            & $python @finalArgs
            if ($LASTEXITCODE -ne 0) { throw "Final evaluation $item failed" }
            if (-not (Test-Path -LiteralPath (Join-Path $output 'final_evaluation\results.json'))) { break }
            continue
        }
        if (Test-Path -LiteralPath (Join-Path $output 'training_closed.json')) {
            Write-Host "Training $item has been finalized by the user; skipping."
            continue
        }
        $arguments = @('-B','-m','experiments.E15_quality_curriculum.train','--arm',$item,
                       '--output',$output,'--device',$Device,'--seed','0',
                       '--max-hours','4','--max-epochs','200','--batch-size','16')
        if ($Resume -and (Test-Path -LiteralPath (Join-Path $output 'checkpoints\latest.pt'))) {
            $arguments += '--resume'
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Training $item exited with code $LASTEXITCODE; subsequent arms were not started" }
        $progress = Get-Content -LiteralPath (Join-Path $output 'progress.json') -Raw | ConvertFrom-Json
        if (-not $progress.finished) {
            Write-Host "Training $item stopped recoverably; subsequent arms were not started."
            break
        }
    }
}
finally { Pop-Location }
