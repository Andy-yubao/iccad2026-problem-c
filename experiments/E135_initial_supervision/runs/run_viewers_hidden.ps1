$ErrorActionPreference = 'Stop'
$viewerWorkspace = 'E:\project\ICCAD\2026\problemC'
Set-Location -LiteralPath $viewerWorkspace
& (Join-Path $viewerWorkspace '.conda\envs\iccad2026\python.exe') -B -m experiments.E135_initial_supervision.runs.start_viewers
exit $LASTEXITCODE
