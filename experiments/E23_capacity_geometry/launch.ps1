param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
Push-Location $repo
try {
    $arguments = @('-u','-B','-m','experiments.E23_capacity_geometry.pipeline')
    if ($Resume) { $arguments += '--resume' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'E17 pipeline failed; later stages were not started' }
}
finally { Pop-Location }
