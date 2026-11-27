param(
    [ValidateSet('screen','confirm','stress','full')][string]$Stage = 'screen',
    [switch]$Resume,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$QualityRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$QualityPython = Join-Path $QualityRoot '.conda/envs/iccad2026/python.exe'
if (!(Test-Path -LiteralPath $QualityPython)) { throw '未找到项目独立Python环境。' }

function Invoke-QualityRun([string]$RunName, [string]$CaseIds, [bool]$UseScreen, [string]$Profile, [bool]$UseFull = $false) {
    $QualityOutput = Join-Path $PSScriptRoot "runs/$RunName"
    $QualityExecution = Join-Path $QualityOutput 'execution.json'
    if ((Test-Path -LiteralPath $QualityExecution) -and (Get-Content -LiteralPath $QualityExecution -Raw -Encoding UTF8 | ConvertFrom-Json).complete) {
        Write-Host "已有完整结果，不重复推理/求解：$RunName"
        Write-Host (Join-Path $QualityOutput 'summary.json')
        return
    }
    $QualityArgs = @('-m','experiments.E50_anytime_quality.run','--output',$QualityOutput,'--profile',$Profile)
    if ($UseFull) { $QualityArgs += '--full' } elseif ($UseScreen) { $QualityArgs += '--screen' } else { $QualityArgs += @('--indices',$CaseIds) }
    if ($Resume) { $QualityArgs += '--resume' }
    if ($DryRun) {
        Write-Host "仅核对计划，不加载模型或运行求解：$RunName"
        Write-Host "$QualityPython $($QualityArgs -join ' ')"
        return
    }
    & $QualityPython @QualityArgs
    if ($LASTEXITCODE -ne 0) { throw '评估中断，已有逐例结果保留；核对原因后加 -Resume 恢复。' }
}

Push-Location $QualityRoot
try {
    switch ($Stage) {
        'screen' { Invoke-QualityRun '20261025_frozen100_parallel_v3' '' $true 'parallel_v3_adaptive' }
        'full' { Invoke-QualityRun '20261025_full500_parallel_v3' '' $false 'parallel_v3_full500' $true }
        'confirm' {
            $QualityManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runs/20261025_confirm100_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            Invoke-QualityRun '20261025_confirm100_parallel_v3' ($QualityManifest.indices -join ',') $false 'parallel_v3_confirmation'
        }
        'stress' {
            foreach ($QualityRepeat in 1..3) {
                Invoke-QualityRun "20261025_stability_v3_r$QualityRepeat" '164,443,246,21,259' $false 'parallel_v3_stability'
            }
        }
    }
} finally { Pop-Location }
