param(
    [ValidateRange(0.01,1000)][double]$HoursPerArm = 4,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
if ($env:E16_REGISTRY) { throw 'Remove E16_REGISTRY pilot override before formal training' }
$queue = Join-Path $PSScriptRoot 'runs\20261020_delayed_queue_seed0'
New-Item -ItemType Directory -Path $queue -Force | Out-Null
$hours = $HoursPerArm.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
Push-Location $repo
try {
    foreach ($start in @(16,21)) {
        if (Test-Path -LiteralPath (Join-Path $queue 'STOP')) {
            Write-Output 'Queue stopped; no further training started.'
            return
        }
        $output = Join-Path $PSScriptRoot "runs\20261020_reference_v5_start${start}_small_seed0"
        $arguments = @('-u','-B','-m','experiments.E16_reference_curriculum.train',
            '--arm','curriculum_small','--output',$output,'--device','cuda','--seed','0',
            '--max-hours',$hours,'--max-epochs','200','--batch-size','16',
            '--curriculum-start-epoch',"$start",'--scheduler-start-epoch',"$($start+9)")
        if ($Resume -and (Test-Path -LiteralPath (Join-Path $output 'checkpoints\latest.pt'))) {
            $arguments += '--resume'
        }
        Write-Output "Starting start${start}: $output"
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Training start${start} failed; queue stopped" }
        $progress = Get-Content -LiteralPath (Join-Path $output 'progress.json') -Raw | ConvertFrom-Json
        if (-not $progress.finished) {
            Write-Output "Training start${start} paused recoverably; queue stopped."
            return
        }
    }
    Write-Output 'Both training arms finished. No solver postprocessing or test evaluation was launched.'
}
finally { Pop-Location }
