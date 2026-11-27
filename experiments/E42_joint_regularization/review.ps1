param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$ReviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ReviewPython = Join-Path $ReviewRoot '.conda/envs/iccad2026/python.exe'
$ReviewSession = Join-Path $PSScriptRoot 'runs/20261024_visual_review100_seed20261024'
Push-Location $ReviewRoot
try {
    & $ReviewPython -m experiments.E42_joint_regularization.prepare_review
    if ($LASTEXITCODE -ne 0) { throw '评分图片准备失败；已完成内容保留，下次可继续。' }
    Write-Host "评分入口：http://127.0.0.1:$Port/（关闭窗口或 Ctrl+C 停止，评分会保留）"
    & $ReviewPython -m visualization.review --session $ReviewSession --port $Port
    if ($LASTEXITCODE -ne 0) { throw '评分服务未能启动。' }
} finally {
    Pop-Location
}
