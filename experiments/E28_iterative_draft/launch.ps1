param(
    [ValidateSet('recurrent_full','single_pooled','single_full','recurrent_pooled','all')]
    [string]$Arm = 'recurrent_full',
    [ValidateSet('prepare','train','final','all')][string]$Stage = 'all',
    [string]$Tag = '20261021',
    [double]$MaxHours = 4,
    [int]$MaxEpochs = 200,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'native_process.ps1')
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.conda\envs\iccad2026\python.exe'
if ($Tag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Use a simple run tag' }
$logs = Join-Path $PSScriptRoot 'runs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$consoleLog = Join-Path $logs ($Tag + '_' + $Arm + '_console.log')
Push-Location $repo
try {
    $arguments = @('-u','-B','-m','experiments.E28_iterative_draft.pipeline','--arm',$Arm,'--stage',$Stage,
        '--tag',$Tag,'--max-hours',[string]$MaxHours,'--max-epochs',[string]$MaxEpochs)
    if ($Resume) { $arguments += '--resume' }
    $exitCode = Invoke-E28LoggedProcess -Executable $python -Arguments $arguments -LogPath $consoleLog
    if ($exitCode -ne 0) { throw "E28 stage failed (exit $exitCode); see full traceback in $consoleLog; later stages were not started" }
}
finally { Pop-Location }
