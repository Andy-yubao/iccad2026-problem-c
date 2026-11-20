$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\project\ICCAD\2026\problemC'
Set-Location -LiteralPath $taskWorkspace
# Inherit the hidden console so the queue and all workers retain real log streams.
& (Join-Path $taskWorkspace '.conda\envs\iccad2026\python.exe') -u -B -m experiments.E136_frontend_F_linear.background_launch --start
exit $LASTEXITCODE
