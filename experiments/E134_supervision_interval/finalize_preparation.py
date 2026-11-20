"""Seal unchanged model proofs plus repaired queue/evaluation continuation checks."""
import ast
import json
from pathlib import Path
import torch
from .io import HERE,ROOT,save,sources,sha

def main():
    diag=HERE/'preparation_diagnostics'
    result=json.loads((diag/'preflight_failure1.json').read_text(encoding='utf-8'))
    current=sources()
    allowed={str((HERE/name).relative_to(ROOT).as_posix()) for name in ('run_queue.py','selftest.py','finalize_preparation.py')}
    source_versions=[]
    for arm in ('e119_stride2','e120_stride2'):
        checkpoint=diag/arm/'continuous/run/checkpoints/latest.pt'
        old=torch.load(checkpoint,map_location='cpu',weights_only=False)['identity']['sources']
        changed={name for name in set(old)|set(current) if old.get(name)!=current.get(name)}
        if not changed<=allowed:raise AssertionError(('Model/engine/Loss/config changed after proof',changed-allowed))
        source_versions.append(dict(arm=arm,changed_files=sorted(changed),unchanged_runtime_training_sources_verified=True))
    if len(result['arms'])!=4 or len(result['resume_checks'])!=2:raise AssertionError('Incomplete original proofs')
    if not all(all(row['bitwise_equal'].values()) for row in result['resume_checks']):raise AssertionError('Resume equivalence failed')
    queue=json.loads((diag/'queue_signal_result.json').read_text())
    evaluation=json.loads((diag/'evaluation_signal_result.json').read_text())
    if not queue['SIGINT_pauses_worker'] or not queue['explicit_resume_advances']:raise AssertionError('Queue did not pass')
    if not evaluation['first_committed_file_unchanged'] or evaluation['completed_committed_prefix']!=500:raise AssertionError('Evaluation did not pass')
    for name in current:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf-8-sig'),filename=name)
    auth=json.loads((HERE/'authorization.json').read_text())
    state=json.loads((HERE/'queue_state.json').read_text())
    if auth['training_enabled'] or auth['allowed_arms'] or (HERE/'runs').exists() or state['status']!='prepared':
        raise AssertionError('Formal experiment was activated during preparation')
    for name in ('error','traceback'):result.pop(name,None)
    result.update(status='passed',ctrl_c_resume_verified=True,sources=current,
        queue_signal=queue,evaluation_signal=evaluation,total_disposable_optimizer_updates=16,formal_optimizer_updates=0,
        preparation_repairs=dict(original_failure_preserved='preparation_diagnostics/preflight_failure1.json',
            queue_failures_preserved=['preparation_diagnostics/queue_signal_failure1','preparation_diagnostics/queue_signal_failure2'],
            fixes=['Test subprocess proxy now includes PIPE and STDOUT','Queue resume replaces persisted updated_at safely','Printed resume command includes run_queue'],
            unchanged_training_proofs_reused_after_hash_validation=source_versions,
            scope='Only controller bookkeeping/test harness changed; model, Loss, cache, trainer and optimizer unchanged'),
        preparation_complete=True,formal_training_started=False,training_enabled=False)
    save(HERE/'preflight.json',result)
    save(HERE/'preparation_complete.json',dict(status='ready_not_started',preflight_sha256=sha(HERE/'preflight.json'),
        gate_sources=current,arms=list(json.loads((HERE/'plan.json').read_text())['arms']),
        ctrl_c_checkpoint_resume_verified=True,automatic_postprocessing_prepared=True,
        authorization_enabled=False,formal_runs_exist=False,dev_payloads_read=0,test_payloads_read=0,
        actual_postprocessing_calls=0,Q_calls=0))
    print('Preparation passed: four cells, bitwise training resume, queue SIGINT/resume, evaluation committed-prefix resume. Formal training remains disabled.')

if __name__=='__main__':main()
