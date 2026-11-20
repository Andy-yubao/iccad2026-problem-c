"""Bounded checkpoint/SIGINT diagnostics on the five registered training cases."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
MODULE='experiments.E137_standard_transformer256'

def worker(arm,role):
    from argparse import Namespace
    from . import train
    from .io import save
    # This entry can only write beneath preparation_diagnostics, never formal runs.
    base=HERE/'preparation_diagnostics'/arm/('continuous' if role=='control' else 'interrupted')
    train.ARM=arm
    train.MODEL=dict(architecture=arm)
    train.RUN=base/'run'
    if not train.RUN.resolve().is_relative_to((HERE/'preparation_diagnostics').resolve()):raise ValueError('Unsafe diagnostic path')
    train.run(Namespace(resume=role=='resume',diagnostic=role))

def equal(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def invoke(*args):
    result=subprocess.run([sys.executable,'-u','-B','-m',MODULE+'.selftest',*args],cwd=HERE.parents[1],
                          env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),capture_output=True,text=True)
    if result.returncode:
        print(result.stdout);print(result.stderr);raise RuntimeError('Disposable resume check failed')

def compare_resume(arm):
    import torch
    invoke('--worker',arm,'control')
    invoke('--worker',arm,'pause')
    interrupted=HERE/'preparation_diagnostics'/arm/'interrupted/run'
    pause=torch.load(interrupted/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    if pause['status']!='user_stop' or pause['finished'] or pause['progress']['step']!=2 or pause['progress']['cursor']!=64:
        raise ValueError('SIGINT did not stop at the accepted second batch')
    paused_elapsed=pause['elapsed_seconds']
    invoke('--worker',arm,'resume')
    continuous=torch.load(HERE/'preparation_diagnostics'/arm/'continuous/run/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(interrupted/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    keys=('model','optimizer','scheduler','rng','projection')
    results={key:equal(continuous[key],resumed[key]) for key in keys}
    for key in ('order','cursor','step','seen','epoch','pending_monitor','best','best_epoch'):
        results['progress_'+key]=equal(continuous['progress'][key],resumed['progress'][key])
    if not all(results.values()):raise AssertionError(results)
    if resumed['progress']['step']!=3 or resumed['progress']['cursor']!=96 or resumed['elapsed_seconds']<=paused_elapsed:
        raise AssertionError('Resume cursor/budget reset')
    return dict(arm=arm,pause_after_steps=2,resume_to_steps=3,paused_cursor=64,resumed_cursor=96,
                bitwise_equal=results,paused_seconds=paused_elapsed,resumed_seconds=resumed['elapsed_seconds'],
                signal='signal.raise_signal(SIGINT) through the actual training handler',unique_train_cases=5,
                formal_updates=0,disposable_updates=6,dev_payloads=0,test_payloads=0)

def queue_probe(root,phase):
    from .io import save
    root=Path(root);run=root/'runs/e119_A_seed0';run.mkdir(parents=True,exist_ok=True)
    if (root/'probe_mode').read_text()=='pause':
        (root/'worker_ready').write_text('ready')
        deadline=time.monotonic()+20
        while not (root/'PAUSE_REQUEST').exists():
            if time.monotonic()>deadline:raise TimeoutError('Controller did not request worker pause')
            time.sleep(.05)
        save(run/'results.json',dict(status='user_stop',diagnostic_only=True))
    elif phase=='train':save(run/'results.json',dict(status='max_epochs',diagnostic_only=True))
    elif phase=='diagnostic':save(run/'address_diagnostic.json',dict(status='complete',diagnostic_only=True))
    else:save(run/'evaluation_no_mib_dev500/state.json',dict(status='complete',diagnostic_only=True))

def queue_check():
    from . import run_queue as queue
    from .io import save
    root=HERE/'preparation_diagnostics/queue_signal'
    root.mkdir(parents=True,exist_ok=False)
    for name,value in [('queue_state.json',dict(status='prepared')),('preflight.json',dict(diagnostic_only=True)),
                       ('authorization.json',dict(training_enabled=False))]:save(root/name,value)
    (root/'probe_mode').write_text('pause')
    queue.OWNER=root;queue.ARMS=('e119_A',);queue.verify=lambda:None
    queue.freeze=lambda arm:(root/'freeze_called').write_text(arm)
    queue.write_comparison=lambda:(root/'comparison_called').write_text('diagnostic only; no real evaluation')
    original_popen=subprocess.Popen
    def popen(command,**kwargs):
        phase='evaluation' if command[4].endswith('.evaluation.run') else ('diagnostic' if command[4].endswith('.diagnostics') else 'train')
        return original_popen([sys.executable,'-u','-B','-m',MODULE+'.selftest','--queue-probe',str(root),phase],**kwargs)
    # Replace only the queue's child launch; the child has no model/data/evaluation work.
    queue.subprocess=type('Proxy',(),{name:getattr(subprocess,name) for name in ('CREATE_NO_WINDOW','CREATE_NEW_PROCESS_GROUP','PIPE','STDOUT')})()
    queue.subprocess.Popen=popen
    def interrupt_when_ready():
        deadline=time.monotonic()+15
        while not (root/'worker_ready').exists():
            if time.monotonic()>deadline:return
            time.sleep(.05)
        signal.raise_signal(signal.SIGINT)
    thread=threading.Thread(target=interrupt_when_ready,daemon=True);thread.start()
    sys.argv=['run_queue','--start'];queue.main();thread.join(timeout=1)
    state=json.loads((root/'queue_state.json').read_text())
    if state['status']!='paused' or state['completed'] or (root/'freeze_called').exists():
        raise AssertionError('Ctrl+C advanced the queue after a pause')
    (root/'probe_mode').write_text('complete')
    sys.argv=['run_queue','--resume'];queue.main()
    state=json.loads((root/'queue_state.json').read_text())
    if state['status']!='complete' or len(state['completed'])!=4 or not (root/'comparison_called').exists():
        raise AssertionError('Queue resume did not continue its stages')
    return dict(SIGINT_pauses_worker=True,no_next_stage_after_pause=True,explicit_resume_advances=True,
                simulated_workers_only=True,real_postprocessing_calls=0)

def evaluation_check():
    """Run the real evaluator's pause/resume loop with synthetic stubbed engines."""
    import hashlib
    import numpy as np
    from .evaluation import run as evaluation
    from . import run_queue
    from .io import save,sha
    root=HERE/'preparation_diagnostics/evaluation_signal'
    root.mkdir(parents=True,exist_ok=False)
    owner=root/'evaluation';owner.mkdir()
    arm='e119_A';out=root/'runs'/f'{arm}_seed0/evaluation_no_mib_dev500'
    checkpoint=root/'runs'/f'{arm}_seed0/checkpoints/best.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'SYNTHETIC CONTROL-FLOW TEST ONLY')
    members=[['synthetic_only',i] for i in range(500)]
    (root/'payload.bin').write_bytes(b'synthetic; not a dataset payload')
    ledger=dict(records=[dict(id=str(i),payload_sha256=sha(root/'payload.bin')) for i in range(500)])
    (root/'checksums.jsonl').write_text(json.dumps(ledger)+'\n')
    save(owner/'plan.json',dict(members=members,payload_checksums_sha256=sha(root/'checksums.jsonl')))
    save(root/'authorization.json',dict(training_enabled=True,allowed_arms=[arm],diagnostic_only=True))
    save(owner/f'frozen_{arm}.json',dict(checkpoint_sha256=sha(checkpoint),best_epoch=0,parameters=0,
         common_final_dev_loss=0.,own_new_final_dev_loss=0.,own_selection_loss=0.,training_seconds=0.,training_epochs=0))
    class Dataset:
        def __init__(self,role):
            assert role=='dev_monitor500'
            self.indices=list(range(500));self.folder=root
            self.records=[dict(member=m,payload='payload.bin',id=str(i)) for i,m in enumerate(members)]
        def __getitem__(self,i):
            inputs=(1,np.ones(1),np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.zeros((1,5),dtype=int),np.full((1,4),-1.))
            return dict(member=members[i],solver_inputs=inputs)
    class Predictor:
        def __init__(self,*args):pass
        def predict(self,inputs):
            self.raw_state=np.zeros((1,3),dtype=np.float32)
            return np.array([[0.,0.,1.,1.]])
    class Scorer:
        def compute_total_score(self,values,counts):return float(np.mean(values))
    calls=[0];mode=['pause']
    def optimize(draft,inputs):
        calls[0]+=1
        if mode[0]=='pause':signal.raise_signal(signal.SIGINT)
        return draft,dict(seconds=0.,row_retained=False,trace=[],work={})
    evaluation.HERE=root;evaluation.OWNER=owner
    evaluation.RandomReference=Dataset;evaluation.Predictor=Predictor
    evaluation.case_from_item=lambda item:None
    evaluation.loss_components=lambda *args:({}, {})
    evaluation.warmup=lambda:None;evaluation.optimize=optimize
    evaluation.public_hard_reference=lambda *args:None
    evaluation.independent_hard_checks=lambda *args:dict(is_feasible=True)
    evaluation.evaluate_case=lambda *args:dict(metrics=dict(cost=1.,is_feasible=True))
    evaluation.official_evaluator=lambda:Scorer()
    evaluation.torch.cuda.synchronize=lambda:None
    evaluation.torch.set_num_interop_threads=lambda count:None
    run_queue.verify=lambda:None
    sys.argv=['evaluation','--arm',arm];evaluation.main()
    state=json.loads((out/'state.json').read_text())
    if state['status']!='paused' or state['completed']!=1 or calls[0]!=1:raise AssertionError('Evaluation did not commit then pause')
    first=sha(out/'cases/0000.json');mode[0]='complete'
    sys.argv=['evaluation','--arm',arm,'--resume'];evaluation.main()
    result=json.loads((out/'results.json').read_text())
    if result['status']!='complete' or result['count']!=500 or calls[0]!=500 or first!=sha(out/'cases/0000.json'):
        raise AssertionError('Evaluation resume duplicated or modified a committed case')
    return dict(real_evaluator_control_flow=True,SIGINT_after_first_commit=True,resumed_from_case1=True,
                completed_committed_prefix=500,first_committed_file_unchanged=True,
                engine_and_scores_synthetic=True,real_postprocessing_calls=0,dev_payloads=0,test_payloads=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=2)
    parser.add_argument('--queue-probe',nargs=2)
    parser.add_argument('--queue-controller-test',action='store_true')
    parser.add_argument('--evaluation-controller-test',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(*args.worker)
    elif args.queue_probe:queue_probe(*args.queue_probe)
    elif args.queue_controller_test:
        from .io import save
        save(HERE/'preparation_diagnostics/queue_signal_result.json',queue_check())
    elif args.evaluation_controller_test:
        from .io import save
        save(HERE/'preparation_diagnostics/evaluation_signal_result.json',evaluation_check())
    else:raise ValueError('A bounded diagnostic mode is required')

if __name__=='__main__':main()
