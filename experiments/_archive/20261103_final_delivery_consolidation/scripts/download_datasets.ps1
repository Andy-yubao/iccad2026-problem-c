[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet('Training', 'Validation', 'All')]
    [string]$Set = 'Validation',
    [switch]$Extract
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $workspaceRoot 'datasets\datasets_manifest.json') -Raw | ConvertFrom-Json
$archiveRoot = Join-Path $workspaceRoot 'datasets\archives'
$repositoryRoot = Join-Path $workspaceRoot 'datasets'
$selected = @($manifest.files | Where-Object { $Set -eq 'All' -or $_.set -eq $Set })

foreach ($item in $selected) {
    $archive = Join-Path $archiveRoot $item.filename
    $partial = $archive + '.part'
    Write-Host ($item.filename + ': ' + $item.bytes + ' bytes')
    Write-Host ('Source: ' + $item.url)
    if (-not $PSCmdlet.ShouldProcess($archive, 'Download/resume and verify SHA-256')) { continue }
    New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null
    if (-not (Test-Path -LiteralPath $archive)) {
        & curl.exe --fail --location --retry 5 --retry-delay 5 --continue-at - --output $partial $item.url
        if ($LASTEXITCODE -ne 0) { throw 'Download failed. Re-run the same command to resume the .part file.' }
        $actualSize = (Get-Item -LiteralPath $partial).Length
        if ($actualSize -ne [long]$item.bytes) { throw ('Unexpected byte count: ' + $partial) }
        $actualHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $item.sha256) { throw ('SHA-256 mismatch. The .part file is retained for inspection: ' + $partial) }
        Move-Item -LiteralPath $partial -Destination $archive
    } else {
        $actualSize = (Get-Item -LiteralPath $archive).Length
        $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSize -ne [long]$item.bytes -or $actualHash -ne $item.sha256) {
            throw ('Existing archive failed verification; preserved without overwriting: ' + $archive)
        }
    }
    Write-Host ('Verified: ' + $archive)
    if ($Extract -and $PSCmdlet.ShouldProcess($repositoryRoot, 'Extract the verified dataset archive')) {
        New-Item -ItemType Directory -Force -Path $repositoryRoot | Out-Null
        & tar.exe -xzf $archive -C $repositoryRoot
        if ($LASTEXITCODE -ne 0) { throw ('Extraction failed: ' + $archive) }
        if ($item.set -eq 'Training') {
            $workers = @(Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'floorset_lite') -Directory -Filter 'worker*')
            if ($workers.Count -lt 100) { throw 'Expected at least 100 floorset_lite/worker* directories; check archive layout.' }
        } else {
            $validationRoot = Join-Path $repositoryRoot 'LiteTensorDataTest'
            $configs = @(Get-ChildItem -LiteralPath $validationRoot -Directory -Filter 'config_*')
            if ($configs.Count -lt 100) { throw 'Expected at least 100 LiteTensorDataTest/config_* directories; check archive layout.' }
            foreach ($blockCount in 21..120) {
                $configRoot = Join-Path $validationRoot ('config_' + $blockCount)
                foreach ($fileName in @('litedata_1.pth', 'litelabel_1.pth')) {
                    $dataFile = Join-Path $configRoot $fileName
                    if (-not (Test-Path -LiteralPath $dataFile -PathType Leaf)) {
                        throw ('Missing validation input/label file: ' + $dataFile)
                    }
                    if ((Get-Item -LiteralPath $dataFile).Length -eq 0) {
                        throw ('Empty validation input/label file: ' + $dataFile)
                    }
                }
            }
            Write-Host 'Validated config_21 through config_120, each with a non-empty input/label pair.'
        }
        Write-Host 'Extracted; archive retained.'
    }
}
