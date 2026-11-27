function Invoke-E28LoggedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $nativeState = @{ ExitCode = $null }
    # Windows PowerShell 5.1 wraps redirected native stderr in ErrorRecord.
    # Continue only inside the native invocation; logging failures still stop.
    & {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1
        $nativeState.ExitCode = $LASTEXITCODE
    } | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $LogPath -Append | Out-Host
    if ($null -eq $nativeState.ExitCode) {
        throw 'E28 native process did not return an exit code'
    }
    return [int]$nativeState.ExitCode
}
