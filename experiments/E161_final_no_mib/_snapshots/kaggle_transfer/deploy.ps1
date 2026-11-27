param([string]$Username, [string]$CliPython, [switch]$UseExistingDataset)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# The training interpreter is Python3.10, which resolves kaggle1.7.4.5 and
# cannot authenticate KGAT tokens. Use a separate Python3.11+ CLI environment.
$taskCliEnv = Join-Path $taskRoot '.cache/kaggle-cli-py311plus'
$taskCliPython = Join-Path $taskCliEnv 'Scripts/python.exe'
$taskCli = Join-Path $taskCliEnv 'Scripts/kaggle.exe'
$taskOldToken = $env:KAGGLE_API_TOKEN
$taskEncoding = [System.Text.UTF8Encoding]::new($false)
try {
    if (-not $Username) { $Username = Read-Host 'Kaggle username (profile URL name)' }
    if ($Username -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid Kaggle username' }
    if (-not (Test-Path -LiteralPath $taskCliPython)) {
        if (-not $CliPython) { $CliPython = (Get-Command python.exe -ErrorAction Stop).Source }
        & $CliPython -c "import sys; print('Kaggle CLI bootstrap:', sys.executable, sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 11) else 1)"
        if ($LASTEXITCODE -ne 0) { throw 'Kaggle Token authentication requires Python3.11+. Pass -CliPython with a suitable python.exe path.' }
        & $CliPython -m venv $taskCliEnv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated Kaggle CLI environment' }
    }
    & $taskCliPython -c "import sys; print('Kaggle CLI Python:', sys.executable, sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'The dedicated Kaggle CLI environment must use Python3.11+.' }
    # kaggle2.2.5 imports six but omits it from Requires-Dist. Require it explicitly.
    # Reuse a compatible installation instead of querying PyPI on every submission.
    & $taskCliPython -c "import importlib.metadata as m,re,sys; versions={d.metadata['Name'].lower():d.version for d in m.distributions()}; release=lambda name: tuple(map(int,re.findall(r'\d+',versions.get(name,''))[:3])); sys.exit(0 if release('kaggle') >= (1,8,4) and release('six') >= (1,17,0) else 1)"
    $taskClientReady = ($LASTEXITCODE -eq 0)
    if (-not $taskClientReady) {
        & $taskCliPython -m pip install --disable-pip-version-check --retries 0 --upgrade 'kaggle>=1.8.4' 'six>=1.17.0,<2'
        if ($LASTEXITCODE -ne 0) { throw 'Could not install Kaggle CLI dependencies; remote task was not submitted' }
    }
    # Read distribution metadata without importing kaggle (which authenticates on import).
    & $taskCliPython -c "from importlib.metadata import version; from packaging.version import Version; v=version('kaggle'); print('Kaggle CLI version:', v); raise SystemExit(0 if Version(v) >= Version('1.8.4') else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'Installed Kaggle CLI is too old for KGAT tokens. No upload attempted.' }
    & $taskCli --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Kaggle CLI could not load its dependencies. No upload attempted.' }
    if (-not $env:KAGGLE_API_TOKEN) {
        $taskSecureToken = Read-Host 'Kaggle API token (hidden; not saved)' -AsSecureString
        $taskPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecureToken)
        try { $env:KAGGLE_API_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer) }
    }
    $taskSubmission = Join-Path $PSScriptRoot 'submission'
    $taskDatasetFolder = Join-Path $taskSubmission 'dataset'
    $taskKernelFolder = Join-Path $taskSubmission 'kernel'
    New-Item -ItemType Directory -Force -Path $taskDatasetFolder,$taskKernelFolder | Out-Null
    $taskDatasetSlug = 'iccad-e161-private-transfer-20261123'
    $taskKernelSlug = 'iccad-e161-s1-resume-20261123'
    $taskDatasetId = "$Username/$taskDatasetSlug"
    $taskKernelId = "$Username/$taskKernelSlug"
    $taskDataset = [ordered]@{title='ICCAD E161 Private Transfer 20261123';id=$taskDatasetId;licenses=@(@{name='other'})}
    [System.IO.File]::WriteAllText((Join-Path $taskDatasetFolder 'dataset-metadata.json'),($taskDataset | ConvertTo-Json -Depth 6),$taskEncoding)
    foreach ($taskName in @('e161-transfer.tar.gz.bin','transfer-manifest.json')) {
        # Same-volume hard links avoid duplicating the large package. No credentials are packaged.
        $taskLink = Join-Path $taskDatasetFolder $taskName
        if (-not (Test-Path -LiteralPath $taskLink)) {
            New-Item -ItemType HardLink -Path $taskLink -Target (Join-Path $PSScriptRoot "payload/$taskName") | Out-Null
        }
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'kaggle_entry.py') -Destination (Join-Path $taskKernelFolder 'kaggle_entry.py')
    $taskKernel=[ordered]@{id=$taskKernelId;title='ICCAD E161 S1 Resume 20261123';code_file='kaggle_entry.py';language='python';kernel_type='script';is_private=$true;enable_gpu=$true;machine_shape='NvidiaTeslaT4';enable_internet=$true;dataset_sources=@($taskDatasetId);competition_sources=@();kernel_sources=@();model_sources=@()}
    [System.IO.File]::WriteAllText((Join-Path $taskKernelFolder 'kernel-metadata.json'),($taskKernel | ConvertTo-Json -Depth 6),$taskEncoding)
    # Use structured API results: CLI stderr must not terminate a readiness poll,
    # and a successful process exit alone does not prove a kernel was accepted.
    $taskApiArguments=@((Join-Path $PSScriptRoot 'deploy_api.py'),'--username',$Username,'--dataset-folder',$taskDatasetFolder,'--kernel-folder',$taskKernelFolder)
    if ($UseExistingDataset) { $taskApiArguments += '--use-existing-dataset' }
    & $taskCliPython -u -B @taskApiArguments
    if ($LASTEXITCODE -ne 0) { throw 'Kaggle deployment stopped. The API diagnostic is above; an accepted dataset upload is preserved and will not be repeated.' }
} finally {
    $env:KAGGLE_API_TOKEN = $taskOldToken
}
