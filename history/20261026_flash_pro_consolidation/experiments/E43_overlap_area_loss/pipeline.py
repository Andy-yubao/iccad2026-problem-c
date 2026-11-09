"""User-started single overlap-area experiment on the fixed E42 AB model."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from .io import HERE, ROOT, save, sources, exclusive, sha
from .area_loss import AREA_WEIGHT

OUTPUT=HERE/'runs/20261023_rectangle_w96d12_area025_seed0'
COORDINATOR=HERE/'runs/20261023_rectangle_w96d12_area025_pipeline_seed0'
CONTROL=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'

def commands(resume=False):
    train=[sys.executable,'-u','-B','-m','experiments.E43_overlap_area_loss.train',
        '--arm','rectangle_area','--output',str(OUTPUT),'--device','cuda','--seed','0',
        '--max-hours','4','--max-epochs','200','--batch-size','32','--scheduler-start-epoch','15']
    if resume and (OUTPUT/'checkpoints/latest.pt').exists():train.append('--resume')
    final=[sys.executable,'-u','-B','-m','experiments.E43_overlap_area_loss.final','--output',str(OUTPUT)]
    report=[sys.executable,'-u','-B','-m','experiments.E43_overlap_area_loss.report','--output',str(OUTPUT)]
    return {'train':train,'final':final,'report':report}

def control_identity():
    frozen=json.loads((CONTROL/'frozen.json').read_text(encoding='utf-8'))
    assert sha(CONTROL/'checkpoints/best.pt')==frozen['checkpoint_sha256'],'E42 control checkpoint changed'
    # Reject drift in the old algorithm snapshot while allowing README/analysis updates.
    for name,value in frozen['sources'].items():
        assert sha(ROOT/name)==value,'E42 frozen control source changed: '+name
    result={name:sha(CONTROL/name) for name in ['frozen.json','full_dev.json','terminal/results.json','checkpoints/best.pt']}
    result['audit_features_sha256']=sha(ROOT/'experiments/E39_rectangular_attention/runs/20261023_failure_mode_audit/case_features_enriched.csv')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['all','train','final'],default='all')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    command=commands(args.resume)
    if args.dry_run:
        print(json.dumps({'dry_run':True,'stage':args.stage,'commands':command,'output':str(OUTPUT),
            'training_started':False,'evaluation_started':False,'automatic_other_training':False},indent=2));return
    identity={'sources':sources(),'control':control_identity(),'model_config':{'width':96,'layers':12,'rectangle':True,'ffn_dropout':.05},
        'seed':0,'max_seconds':14400,'max_epochs':200,'batch_size':32,'cpu_threads':4,'device':'cuda',
        'weight_decay':3e-4,'overlap_area_weight':AREA_WEIGHT,'data_augmentation':False,
        'mib_mean_area':'unchanged v5','loss_weights':'E42 geometry plus .25 overlap_area; original selection proxy; MIB .1','test_read':False}
    with exclusive(COORDINATOR.with_suffix('.lock')):
        COORDINATOR.mkdir(parents=True,exist_ok=True)
        cfg=COORDINATOR/'config.json'
        if cfg.exists():
            if not args.resume or json.loads(cfg.read_text(encoding='utf-8'))!=identity:
                raise ValueError('Existing coordinator needs -Resume with identical source/data/control/budget')
        else:save(cfg,identity)
        flags=[HERE/'runs/STOP',COORDINATOR/'STOP',OUTPUT/'STOP'];requested=[False]
        def stopped():return requested[0] or any(p.exists() for p in flags)
        if stopped():print('STOP present; preserve it until explicit user resume.');return
        old=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        history_path=COORDINATOR/'progress.json'
        history=json.loads(history_path.read_text(encoding='utf-8'))['stages'] if history_path.exists() else []
        def stage(kind):
            if stopped():return False
            argv=commands(args.resume)[kind];started=time.perf_counter()
            env=dict(os.environ,E43_THREADS='4')
            with (COORDINATOR/'console.log').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'kind':kind,'argv':argv,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})+'\n');log.flush()
                process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                save(COORDINATOR/'active.json',{'kind':kind,'pid':process.pid,'argv':argv})
                def drain():
                    for line in process.stdout:sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
                reader=threading.Thread(target=drain);reader.start()
                while process.poll() is None:
                    if stopped():(HERE/'runs/STOP').touch()
                    try:process.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
                reader.join()
            history.append({'kind':kind,'argv':argv,'returncode':process.returncode,'seconds':time.perf_counter()-started})
            save(history_path,{'stages':history,'test_read':False});(COORDINATOR/'active.json').unlink(missing_ok=True)
            if process.returncode:raise RuntimeError(f'{kind} failed; preserve evidence and see {COORDINATOR / "console.log"}')
            return not stopped()
        try:
            if args.stage in ('all','train'):
                if not stage('train'):return
                result=json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                if result['status'] not in ('early_stop','budget','max_epochs'):return
            if args.stage in ('all','final'):
                if not stage('final'):return
            if not stage('report'):return
            save(COORDINATOR/'results.json',{'status':'complete','stage':args.stage,'stages':history,
                'training_arms':['rectangle96x12 joint_AB area025 seed0'],'test_read':False,'further_training_jobs':[]})
        finally:signal.signal(signal.SIGINT,old)

if __name__=='__main__':main()
