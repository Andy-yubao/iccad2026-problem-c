param(
    [ValidateSet('Preview','CheckCPU','CheckGPU','Run')][string]$Action = 'Preview',
    [ValidateSet('centroid','both')][string]$Arm = 'centroid',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $workspace '.conda\envs\iccad2026\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python environment not found' }
if ($Action -in @('Preview','Run') -and $Arm -eq 'both') { throw 'Only one training arm can be selected; centroid is first' }
Set-Location -LiteralPath $workspace
if ($Action -eq 'CheckCPU') {
    & $python -u -B -m experiments.E96_port_compression.preflight --device cpu --arm $Arm
} elseif ($Action -eq 'CheckGPU') {
    & $python -u -B -m experiments.E96_port_compression.preflight --device cuda --arm $Arm
} else {
    $invocation = @('-u','-B','-m','experiments.E96_port_compression.pipeline','--arm',$Arm)
    if ($Action -eq 'Run') { $invocation += '--execute' } else { $invocation += '--dry-run' }
    if ($Resume) { $invocation += '--resume' }
    & $python @invocation
}
if ($LASTEXITCODE -ne 0) { throw "Command exited with code $LASTEXITCODE" }
