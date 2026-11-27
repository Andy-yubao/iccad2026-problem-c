"""Continue the paused, unchanged E161 trainer on one Kaggle GPU, then S2/dev2000."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE = ROOT/'experiments/E161_final_no_mib'
OUTPUT = Path('/kaggle/working/e161-results')
OUTPUT.mkdir(exist_ok=True)

def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def preserve_runtime_files():
    target = OUTPUT/'identity'
    target.mkdir(exist_ok=True)
    for path in HERE.glob('*.json'): shutil.copy2(path,target/path.name)
    if (HERE/'evaluation/frozen.json').exists():
        shutil.copy2(HERE/'evaluation/frozen.json',target/'evaluation_frozen.json')

def main():
    # Version/GPU discovery below is part of actual checkpoint migration, not a trial run.
    import numpy as np
    import torch
    from experiments.E161_final_no_mib import io
    from experiments.E161_final_no_mib.run_queue import freeze
    original_runs = HERE/'runs'
    destination = OUTPUT/'runs'
    shutil.move(str(original_runs),str(destination))
    original_runs.symlink_to(destination,target_is_directory=True)
    # The sealed legacy source directories are not required for frozen random train/dev data.
    for name in ('datasets/floorset_lite','datasets/LiteTensorDataTest'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    latest = HERE/'runs/e161_s1_seed0/checkpoints/latest.pt'
    originals = OUTPUT/'original-local-checkpoints'
    originals.mkdir(exist_ok=True)
    shutil.copy2(latest,originals/'s1-latest.pt')
    shutil.copy2(latest.with_name('best.pt'),originals/'s1-best.pt')
    state = torch.load(latest,map_location='cpu',weights_only=False)
    old_identity = copy.deepcopy(state['identity'])
    if state['progress']['step'] != 7568 or state['finished'] or state['status'] != 'user_stop':
        raise ValueError('Unexpected source checkpoint; migration is for paused S1 step7568')
    if (old_identity['torch'] != str(torch.__version__) or old_identity['numpy'] != np.__version__
            or old_identity['cuda'] != torch.version.cuda or old_identity['sources'] != io.sources()):
        raise ValueError('Pinned software/source identity differs; stop without rebuilding caches or optimizer')
    if len(state['rng']['cuda']) != 1:
        raise ValueError('Expected the original single-device CUDA RNG state')
    gpu = torch.cuda.get_device_name(0)
    state['identity']['gpu'] = gpu
    io.checkpoint(latest,state)
    migration = dict(source_checkpoint_sha256=io.sha(originals/'s1-latest.pt'),
        migrated_checkpoint_sha256=io.sha(latest),old_gpu=old_identity['gpu'],new_gpu=gpu,
        old_identity=old_identity,new_identity=state['identity'],
        reason='User explicitly authorized remote continuation on Kaggle free GPU.',
        preserved=['model','optimizer','scheduler','rng','progress','projection','elapsed_seconds'],
        elapsed_seconds=state['elapsed_seconds'],step=7568,cursor=state['progress']['cursor'],
        bit_identical_to_local_continuation=False,model_probes=0,discarded_updates=0,
        original_pause_flags='Retained locally; not placed in the explicitly authorized remote runtime copy.')
    save(OUTPUT/'migration.json',migration)
    queue=dict(status='running',active='e161_s1',completed=['e161_l0','e161_s0'],
               platform='kaggle',run_id=old_identity['plan']['run_id'],migration='migration.json')
    def queue_save():
        save(HERE/'queue_state.json',queue)
        save(OUTPUT/'queue_state.json',queue)
    queue_save()
    session_started=float(os.environ['E161_SESSION_STARTED'])
    def request_pause():
        for folder in (HERE,HERE/'evaluation'):
            (folder/'PAUSE_REQUEST').write_text('Kaggle session deadline; save state and stop.\n')
        save(OUTPUT/'session_deadline_pause.json',dict(requested=True,session_seconds=time.monotonic()-session_started))
    timer=threading.Timer(max(0.,30000-(time.monotonic()-session_started)),request_pause)
    timer.daemon=True
    timer.start()
    def execute(command):
        with (OUTPUT/'queue.stdout.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,env=dict(os.environ,PYTHONPATH=str(ROOT)))
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            return process.wait()
    try:
        for arm,resume in [('e161_s1',True),('e161_s2',False)]:
            if (HERE/'PAUSE_REQUEST').exists():
                queue.update(status='paused',active=None);queue_save();return
            queue.update(active=arm);queue_save()
            command=[sys.executable,'-u','-B','-m','experiments.E161_final_no_mib.train','--arm',arm]
            if resume: command.append('--resume')
            code=execute(command)
            path=HERE/'runs'/f'{arm}_seed0/results.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            if code:
                queue.update(status='training_error',active=None);queue_save()
                raise RuntimeError('Formal stage failed; no retry: '+arm)
            if result.get('status')=='user_stop':
                queue.update(status='paused',active=None);queue_save();return
            frozen=freeze(arm)
            queue['completed'].append(arm);queue_save();preserve_runtime_files()
        io.save(HERE/'evaluation/frozen.json',frozen)
        # The original evaluator must create a new output directory itself.
        # Creating an empty target beforehand would be mistaken for a resumable audit.
        evaluation=HERE/'evaluation/output'
        queue.update(active='dev2000');queue_save()
        code=execute([sys.executable,'-u','-B','-m','experiments.E161_final_no_mib.evaluation.run','--arm','e161_s2'])
        if code:
            queue.update(status='evaluation_error',active=None);queue_save()
            raise RuntimeError('Terminal evaluation failed; no retry')
        evaluation_state=json.loads((evaluation/'state.json').read_text())
        queue.update(status='complete' if evaluation_state['status']=='complete' else 'paused',active=None)
        queue_save()
    finally:
        timer.cancel()
        preserve_runtime_files()
        evaluation=HERE/'evaluation/output'
        if evaluation.exists():
            shutil.copytree(evaluation,OUTPUT/'evaluation',dirs_exist_ok=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        (OUTPUT/'error.txt').write_text(traceback.format_exc())
        raise
