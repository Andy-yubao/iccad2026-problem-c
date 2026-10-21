param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
$logs = Join-Path $PSScriptRoot 'runs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$consoleLog = Join-Path $logs 'training_console.log'
Push-Location $repo
try {
    $arguments = @('-u','-B','-m','experiments.E27_overlap_weight.pipeline')
    if ($Resume) { $arguments += '--resume' }
    & $python @arguments 2>&1 | Tee-Object -FilePath $consoleLog -Append
    if ($LASTEXITCODE -ne 0) { throw 'E27 pipeline failed; later stages were not started' }
}
finally { Pop-Location }
