"""Finish only the corrected synthetic evaluator check; preserve passed model proofs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ast
import json
import subprocess
import sys
import torch
from .io import HERE, ROOT, sha, save, sources


def main():
    old=HERE/'preparation_diagnostics/attempt2_controller_name'
    result=json.loads((old/'preflight.json').read_text())
    if result['status']!='failed' or result['error']!="RuntimeError('Controller short check failed: evaluation')":
        raise ValueError('Not the reviewed test-harness name mismatch')
    required=('input_checks','raw_to_cached_public_input_exact','relative_update_bound_verified',
              'capture_preserves_parameters','all_branches_have_gradients','real_inference_path_checked_on_train_member')
    assert all(result[key] for key in required)
    assert all(result['resume_checks'][0]['bitwise_equal'].values())
    assert result['queue_signal']['no_next_stage_after_pause'] and result['queue_signal']['explicit_resume_advances']
    assert not (HERE/'runs').exists()
    auth=json.loads((HERE/'authorization.json').read_text());assert not auth['training_enabled'] and not auth['allowed_arms']
    checkpoint=HERE/'preparation_diagnostics/standard256/continuous/run/checkpoints/latest.pt'
    proven=torch.load(checkpoint,map_location='cpu',weights_only=False)['identity']['sources']
    current=sources()
    changed={name for name in set(proven)|set(current) if proven.get(name)!=current.get(name)}
    expected={str(HERE.relative_to(ROOT).as_posix())+'/'+name for name in ('selftest.py','finish_preflight.py')}
    if changed!=expected:raise ValueError('Runtime source changed beyond the reviewed test harness: '+str(changed))
    def training_functions(path):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
            if isinstance(node,ast.FunctionDef) and node.name in ('worker','equal','invoke','compare_resume')}
    assert training_functions(old/'selftest_before.py')==training_functions(HERE/'selftest.py')
    completed=subprocess.run([sys.executable,'-u','-B','-m','experiments.E137_standard_transformer256.selftest',
        '--evaluation-controller-test'],cwd=ROOT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),capture_output=True,text=True)
    (HERE/'preparation_diagnostics/evaluation_controller.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
    if completed.returncode:raise RuntimeError('Corrected synthetic evaluator check failed')
    report=json.loads((HERE/'preparation_diagnostics/evaluation_signal_result.json').read_text())
    assert report['first_committed_file_unchanged'] and report['completed_committed_prefix']==500
    result.pop('error');result.pop('traceback')
    result.update(status='passed',ctrl_c_resume_verified=True,evaluation_signal=report,sources=current,
        training_model_not_reused=True,disposable_updates_in_resume_checks=6,
        preparation_corrections=[dict(issue='GPU capture host constant allocation',resolution='same boundary bits built with device operators; full model/capture/update check then passed'),
            dict(issue='Synthetic evaluator stub used old arm name',resolution='stub arm renamed; unchanged actual training/resume proof verified by complete runtime hashes and training-helper AST equality; corrected evaluator commit/resume check passed')])
    save(HERE/'preflight.json',result)
    save(HERE/'preparation_complete.json',dict(status='ready_to_start',arm='standard256',formal_updates=0,
        unique_train_cases=5,dev_payloads=0,test_payloads=0,automatic_postprocessing_prepared=True))
    print(json.dumps(dict(status='passed',parameters=result['parameters'],all_resume_values_bitwise_equal=True,
        evaluation_committed_prefix=500,formal_updates=0,dev_payloads=0,test_payloads=0)),flush=True)


if __name__=='__main__':main()
