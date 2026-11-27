$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $root '.conda\envs\iccad2026\python.exe'
$background = Join-Path $PSScriptRoot 'background'
$statePath = Join-Path $PSScriptRoot 'queue_state.json'
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.status -ne 'prepared') {
    throw 'The E170 queue is not in the prepared state.'
}
if (Test-Path -LiteralPath (Join-Path $background 'queue.log')) {
    throw 'The E170 queue log already exists.'
}
New-Item -ItemType Directory -Path $background -Force | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$process = Start-Process -FilePath $python -ArgumentList @(
    '-B', '-u', '-m', 'experiments.E170_three_block_stages.run_queue', '--start'
) -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $background 'queue.log') `
    -RedirectStandardError (Join-Path $background 'queue_stderr.log')
[ordered]@{
    pid = $process.Id
    stdout = (Join-Path $background 'queue.log')
    stderr = (Join-Path $background 'queue_stderr.log')
    launched_utc = [DateTimeOffset]::UtcNow.ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $background 'launch.json') -Encoding utf8
$process.Id
