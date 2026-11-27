param(
    [ValidateSet('All','Teachers','Train','Evaluate')][string]$Stage='All',
    [ValidateSet('Both','geometry','structure')][string]$Arm='Both',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName='distill_v1_seed0',
    [string]$Device='cuda',
    [switch]$DryRun
)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
$taskPython=Join-Path $taskRoot '.conda/envs/iccad2026/python.exe'
$taskExperiment='experiments/E13_solver_distillation'
$taskTrainBank='experiments/E12_distillation_readiness/runs/teacher_train20000_v1'
$taskDevBank='experiments/E12_distillation_readiness/runs/teacher_devmonitor500_v1'
function Invoke-ResearchPython([string[]]$TaskArguments) {
    if ($DryRun) { Write-Output ("$taskPython " + ($TaskArguments -join ' ')); return }
    & $taskPython @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw "Research process failed (exit $LASTEXITCODE). Saved outputs remain in place." }
}
function Assert-CompleteBank([string]$TaskPath) {
    if ($DryRun) { return }
    $taskState=Get-Content -LiteralPath (Join-Path $TaskPath 'results.json') -Raw | ConvertFrom-Json
    if ($taskState.status -ne 'completed' -or $taskState.error_count -ne 0) { throw "Teacher bank incomplete: $TaskPath. Resume the same command; no partial-data training." }
}
if ($Stage -in @('All','Teachers')) {
    foreach ($taskBankSpec in @(@('train',$taskTrainBank,'43200'),@('dev_monitor',$taskDevBank,'3600'))) {
        $taskArguments=@('-B','-u','-m','experiments.E12_distillation_readiness.bank','--split',$taskBankSpec[0],'--max-seconds',$taskBankSpec[2],'--output',$taskBankSpec[1])
        if (Test-Path -LiteralPath (Join-Path $taskBankSpec[1] 'config.json')) { $taskArguments+='--resume' }
        Invoke-ResearchPython $taskArguments
        Assert-CompleteBank $taskBankSpec[1]
    }
}
$taskArms=if ($Arm -eq 'Both') { @('geometry','structure') } else { @($Arm) }
foreach ($taskArm in $taskArms) {
    $taskRun=Join-Path "$taskExperiment/runs/$RunName" $taskArm
    if ($Stage -in @('All','Train')) {
        Assert-CompleteBank $taskTrainBank;Assert-CompleteBank $taskDevBank
        $taskArguments=@('-B','-u','-m','experiments.E13_solver_distillation.train','--arm',$taskArm,'--train-bank',$taskTrainBank,'--dev-bank',$taskDevBank,'--device',$Device,'--max-seconds','14400','--output',$taskRun)
        if (Test-Path -LiteralPath $taskRun) { $taskArguments+='--resume' }
        Invoke-ResearchPython $taskArguments
        if (-not $DryRun) {
            $taskState=Get-Content -LiteralPath (Join-Path $taskRun 'progress.json') -Raw | ConvertFrom-Json
            if (-not $taskState.finished) { throw "Training paused: $taskRun. Remove STOP when ready and rerun the same command." }
        }
    }
    if ($Stage -in @('All','Evaluate')) {
        foreach ($taskPhase in @('supervised','repair')) {
            $taskEvaluation=Join-Path $taskRun "eval_$taskPhase"
            $taskArguments=@('-B','-u','-m','experiments.E13_solver_distillation.evaluate','--checkpoint',"$taskRun/checkpoints/best.pt",'--phase',$taskPhase,'--device',$Device,'--output',$taskEvaluation)
            if ($taskPhase -eq 'repair') { $taskArguments+=@('--teacher-bank',$taskDevBank) }
            if (Test-Path -LiteralPath $taskEvaluation) { $taskArguments+='--resume' }
            Invoke-ResearchPython $taskArguments
            if (-not $DryRun) {
                $taskState=Get-Content -LiteralPath (Join-Path $taskEvaluation 'results.json') -Raw | ConvertFrom-Json
                if ($taskState.status -ne 'completed') { throw "Evaluation paused: $taskEvaluation" }
            }
        }
    }
}
