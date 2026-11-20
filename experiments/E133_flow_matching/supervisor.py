"""Hidden single-run supervisor with an aggregate8h wall-clock budget; no retry."""
import faulthandler
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.E133_flow_matching.control import OWNER,read,sha,sources,assert_training_authorized,stopped,elapsed_pipeline,budget_exhausted
from experiments.E133_flow_matching.io import save,exclusive
PYTHON=ROOT/'.conda/envs/iccad2026/python.exe'
MODULE='experiments.E133_flow_matching'
def now():return datetime.now(timezone.utc).isoformat()

def verify():
    assert_training_authorized()
    registration=read(ROOT/'output/flow_matching_20261112/registration.json')
    assert registration['frozen_sources']==sources() and registration['preflight_sha256']==sha(OWNER/'preflight.json')

def freeze():
    import torch
    run=OWNER/'runs/fm_seed0';result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result.get('error') or result['steps']<=0:
        raise RuntimeError('No successful training; terminal evaluation blocked')
    path=run/'checkpoints/best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['identity']['sources']==sources() and state['progress']['best']==result['best_fixed_loss']
    save(OWNER/'evaluation/frozen_model.json',dict(checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),
        best_epoch=result['best_epoch'],own_selection_loss=result['best_fixed_loss'],parameters=read(OWNER/'plan.json')['parameters_by_arm']['fm'],
        training_seconds=result['elapsed_seconds'],training_epochs=result['epochs'],training_steps=result['steps'],
        normalized_state=read(run/'normalization.json'),frozen_before_Q=True,frozen_at=now(),sources=sources(),test_read=False))

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONUNBUFFERED']='1'
    logs=OWNER/'launch_logs/20261112_seed0_prepared';logs.mkdir(parents=True,exist_ok=False)
    sys.stdout=(logs/'controller.stdout.log').open('x',encoding='utf-8',buffering=1)
    sys.stderr=(logs/'controller.stderr.log').open('x',encoding='utf-8',buffering=1)
    faulthandler.enable(file=sys.stderr)
    with exclusive(OWNER/'queue.lock'):
        if (OWNER/'queue_state.json').exists() or (OWNER/'runs').exists():raise FileExistsError('Already dispatched; no training retry')
        verify();started=time.time()
        if (OWNER/'pipeline_budget.json').exists():
            # Only the recorded, zero-update preparation correction may reuse
            # the original deadline. Never reset an existing experiment budget.
            assert (OWNER/'queue_state.preparation_failed1.json').is_file()
            assert read(ROOT/'output/flow_matching_20261112/startup_capture_failure1/diagnosis.json')['accepted_optimizer_updates']==0
            assert read(OWNER/'pipeline_budget.json')['maximum_seconds']==28800
        else:
            save(OWNER/'pipeline_budget.json',dict(started_unix=started,started_at=now(),maximum_seconds=28800,training_deadline_seconds=27000,
                deadline_at=datetime.fromtimestamp(started+28800,timezone.utc).isoformat(),policy='safe batch/case boundary;checkpoint finalization tail separately reported'))
        state=dict(status='running',controller_pid=os.getpid(),started_at=now(),active=None,completed=[],
            order=['disposable_startup_update','scratch_fm_train','freeze_unique_FM_Loss_best','Euler8_dev500_and_train5_step_diagnostics'],
            automatic_retry=False,total_budget_seconds=28800,test_read=False)
        child=None
        def beat(**kw):
            state.update(kw);save(OWNER/'queue_state.json',dict(**state,pipeline_elapsed_seconds=elapsed_pipeline(),updated_at=now()))
        def check():
            if stopped():raise InterruptedError('STOP; all subsequent stages blocked')
            if budget_exhausted():raise TimeoutError('Aggregate8h budget; no subsequent stage')
        def launch(stage,module):
            nonlocal child
            check();verify()
            with (logs/f'{stage}.stdout.log').open('x',encoding='utf-8') as out,(logs/f'{stage}.stderr.log').open('x',encoding='utf-8') as err:
                child=subprocess.Popen([str(PYTHON),'-u','-B','-m',MODULE+'.'+module],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
                beat(active=dict(stage=stage,pid=child.pid,started_at=now()))
                while child.poll() is None:
                    # Workers independently check the shared STOP and total deadline.
                    beat();time.sleep(5)
            if stopped():raise InterruptedError('User STOP; next stage blocked')
            if child.returncode!=0:
                if budget_exhausted():raise TimeoutError('Total8h budget expired in worker')
                raise RuntimeError(f'{stage} exited{child.returncode}; no retry')
            state['completed'].append(dict(stage=stage,completed_at=now()));child=None;beat(active=None)
        try:
            beat();launch('startup_check','startup_check')
            gate=read(OWNER/'startup_check.json');assert gate['status']=='passed' and gate['optimizer_updates']==1 and gate['sources']==sources()
            launch('train','train');check();freeze();state['completed'].append(dict(stage='freeze_unique_FM_Loss_best',completed_at=now()));beat()
            launch('evaluation','evaluation.run')
            result=read(OWNER/'runs/fm_seed0/evaluation_no_mib_dev500/results.json')
            assert result['status']=='complete' and result['count']==500
            beat(status='complete',active=None,completed_at=now(),finalization_over_budget_seconds=max(0.,elapsed_pipeline()-28800))
        except BaseException as exc:
            if child is not None and child.poll() is None:
                (OWNER/'STOP').write_text('Controller failure: save accepted state and do not continue.\n')
            beat(status='user_stop' if isinstance(exc,InterruptedError) else 'budget' if isinstance(exc,TimeoutError) else 'needs_attention',
                error=repr(exc),traceback=traceback.format_exc(),finalization_over_budget_seconds=max(0.,elapsed_pipeline()-28800))
            traceback.print_exc();raise
if __name__=='__main__':main()
