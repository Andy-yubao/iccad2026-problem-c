param([string]$Username)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskCli=Join-Path $taskRoot '.cache/kaggle-cli-py311plus/Scripts/kaggle.exe'
$taskOldToken=$env:KAGGLE_API_TOKEN
try {
    if (-not $Username) { $Username=Read-Host 'Kaggle username' }
    if (-not $env:KAGGLE_API_TOKEN) {
        $taskSecure=Read-Host 'Kaggle API token (hidden; not saved)' -AsSecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        try { $env:KAGGLE_API_TOKEN=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer) }
    }
    $taskKernel="$Username/iccad-e161-s1-resume-20261123"
    & $taskCli kernels status $taskKernel
    if ($LASTEXITCODE -ne 0) { throw 'Could not read remote status' }
    $taskDestination=Join-Path $PSScriptRoot ('downloaded/'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
    & $taskCli kernels output $taskKernel -p $taskDestination
    if ($LASTEXITCODE -ne 0) { throw 'Outputs are not available or download failed' }
    Write-Output "Saved remote outputs to $taskDestination. Local E161 paused run was not overwritten."
} finally { $env:KAGGLE_API_TOKEN=$taskOldToken }
