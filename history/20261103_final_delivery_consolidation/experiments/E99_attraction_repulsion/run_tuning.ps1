param(
    [string]$Config = "",
    [string]$Resume = ""
)
$ErrorActionPreference = "Stop"
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "../..")).Path
$taskPython = Join-Path $taskRoot ".conda/envs/iccad2026/python.exe"
if (-not $Config) { $Config = Join-Path $PSScriptRoot "tuning.json" }
$taskArgs = @("-u", "-B", "-m", "experiments.E99_attraction_repulsion.tune", "--config", $Config)
if ($Resume) { $taskArgs += @("--resume", $Resume) }
Push-Location -LiteralPath $taskRoot
try {
    & $taskPython @taskArgs
    $taskExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExit
