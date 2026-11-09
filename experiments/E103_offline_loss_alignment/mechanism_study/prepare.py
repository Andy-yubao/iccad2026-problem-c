from pathlib import Path
import json,hashlib,ast
root=Path.cwd();owner=root/'experiments/E103_offline_loss_alignment';here=owner/'mechanism_study';bank=owner/'runs/20261104_bank1000x5_seed0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
assert read(bank/'audits/q_sanity.json')['status']=='passed'
plan=dict(revision='E103.mechanism_diagnosis.v1',authorization='User requested Q sanity check then autonomous research continuation; previous authorization permits four parallel background tasks.',
 prerequisite='5000 Q labels sanity checked, 5 unique problems/25 saved geometries independently rescored exactly; all5000 hard legal',
 reason='V1 improves heldout ranking modestly but failed registered two-step descent gate; diagnose before long neural training.',
 loss_keys=['position','shape','penetration','module_load_mean','module_max_excess10','wirelength','compactness','boundary_contribution','grouping_contribution','mib_contribution'],
 case_boundary='Only original fit600/tune200 for all new fits and diagnostic comparisons; original check200 already exposed, never used for new tuning. No dev/test data.',
 workers=4,tasks=dict(
 capacity='Signed linear and quadratic within-case centered ridge, fixed penalty.01; source-only and always-E42 controls. Diagnostic only; no sweep or neural objective adoption.',
 stability='32 fixed-seed case bootstraps of registered nonnegative fit; original selected lambda1 fixed; report quantiles, do not pick a bootstrap winner.',
 canonical_view='Recompute position/shape supervision from decoded geometry; eight public terms unchanged; fit exactly one canonical-view positive model with fixed lambda1. Compare tune only.',
 descent='Reconstruct existing40-case/160-step raw Loss changes, compare with saved Q/geometry; zero additional postprocessor calls.'),
 candidate_followup='Only if canonical-view tune ranking gains at least.02 and mixed regret reduces>.0005 versus V1 does it warrant a separately registered local direction check. No automatic long training.',
 frozen_input_hashes={p.relative_to(root).as_posix():sha(p) for p in [owner/'panel.json',owner/'model_manifest.json',bank/'analysis/proposal.json',bank/'analysis/results.json',bank/'direction/results.json',bank/'audits/q_sanity.json']},
 outputs='Per-task JSON plus summary report; original database, first-stage results and E98 solver immutable.',
 formal_training=False,new_postprocessor_evaluations=0,test_read=False,automatic_wakeups=False)
(here/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for p in here.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
(here/'launch.ps1').write_text('''$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..')).Path
$taskRun=Join-Path $PSScriptRoot 'runs\\20261104_mechanism_v1'
if(Test-Path -LiteralPath (Join-Path $taskRun 'config.json')){throw 'Diagnosis already launched'}
$taskPython=Join-Path $taskRoot '.conda\\envs\\iccad2026\\python.exe'
$taskProc=Start-Process -FilePath $taskPython -ArgumentList @('-u','-B','-m','experiments.E103_offline_loss_alignment.mechanism_study.run') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'stderr.log')
@{pid=$taskProc.Id;started_utc=[DateTime]::UtcNow.ToString('o');workers=4;authorization='Q sanity passed; user requested advancing E103';test_read=$false}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
Write-Output ('Started four-way mechanism diagnosis PID '+$taskProc.Id)
$taskProc.WaitForExit()
if($null -ne $taskProc.ExitCode -and $taskProc.ExitCode -ne 0){throw ('Mechanism diagnosis exited '+$taskProc.ExitCode)}
''',encoding='utf-8')
print('Registered four bounded diagnostics on fit/tune only; no new solver calls or neural training.')
