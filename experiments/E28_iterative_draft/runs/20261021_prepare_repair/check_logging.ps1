$ErrorActionPreference = 'Stop'
$experiment = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repo = (Resolve-Path (Join-Path $experiment '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
. (Join-Path $experiment 'native_process.ps1')
$okLog = Join-Path $PSScriptRoot 'native_ok.log'
$failLog = Join-Path $PSScriptRoot 'native_fail.log'
$ok = Invoke-E28LoggedProcess -Executable $python -Arguments @('-u','-B',(Join-Path $PSScriptRoot 'native_ok.py')) -LogPath $okLog
if ($ok -ne 0) { throw 'Harmless native stderr caused failure' }
$text = Get-Content -LiteralPath $okLog -Raw
if ($text -notmatch 'E28 stdout before stderr' -or $text -notmatch 'E28 harmless stderr' -or $text -notmatch 'E28 stdout after stderr') {
    throw 'Native output was truncated'
}
$bad = Invoke-E28LoggedProcess -Executable $python -Arguments @('-u','-B',(Join-Path $PSScriptRoot 'native_fail.py')) -LogPath $failLog
if ($bad -eq 0) { throw 'Failing child returned success' }
$text = Get-Content -LiteralPath $failLog -Raw
if ($text -notmatch 'Traceback' -or $text -notmatch 'RuntimeError: E28 full traceback sentinel') { throw 'Traceback tail was lost' }
if ($ErrorActionPreference -ne 'Stop') { throw 'Caller error preference changed' }
@{
    status = 'passed'
    powershell_version = $PSVersionTable.PSVersion.ToString()
    stderr_warning_exit = $ok
    traceback_exit = $bad
    complete_traceback_preserved = $true
    caller_error_preference_preserved = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'logging_results.json') -Encoding UTF8
Write-Output 'Native stderr and complete traceback checks passed'
