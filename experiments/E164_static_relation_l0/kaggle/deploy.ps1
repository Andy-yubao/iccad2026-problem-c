param(
    [string]$Username = 'andyyubao',
    [switch]$Status,
    [switch]$Download
)
$ErrorActionPreference = 'Stop'
if ($Status -and $Download) { throw 'Choose either -Status or -Download.' }
if ($Username -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid Kaggle username.' }
$taskRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$taskPython = Join-Path $taskRoot '.cache/kaggle-cli-py311plus/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'The existing dedicated Kaggle CLI Python environment is missing.' }
$taskOldToken = $env:KAGGLE_API_TOKEN
try {
    $taskConfig = if ($env:KAGGLE_CONFIG_DIR) { $env:KAGGLE_CONFIG_DIR } else { Join-Path $env:USERPROFILE '.kaggle' }
    if (-not ($env:KAGGLE_API_TOKEN -or $env:KAGGLE_KEY -or
              (Test-Path (Join-Path $taskConfig 'kaggle.json')) -or (Test-Path (Join-Path $taskConfig 'access_token')))) {
        $taskSecureToken = Read-Host 'Kaggle API token (hidden, not saved)' -AsSecureString
        $taskPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecureToken)
        try { $env:KAGGLE_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer) }
    }
    $taskAction = if ($Status) { 'status' } elseif ($Download) { 'download' } else { 'submit' }
    & $taskPython -u -B (Join-Path $PSScriptRoot 'submit.py') $Username $taskAction
    if ($LASTEXITCODE -ne 0) { throw "Kaggle $taskAction stopped. See the API diagnostic above." }
} finally {
    $env:KAGGLE_API_TOKEN = $taskOldToken
}
