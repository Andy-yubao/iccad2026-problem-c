"""Preview by default. Explicit execution runs one requested arm, never queues two."""
import argparse,json,subprocess,sys
from .runtime import ARMS,run_dir,verify_preflight,state
from .io import HERE,ROOT,exclusive
from .train import stopped

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,default='centroid');p.add_argument('--execute',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    prefix=[sys.executable,'-u','-B','-m'];module='experiments.E96_port_compression'
    check=[*prefix,module+'.preflight','--arm',a.arm,'--device','cuda']
    train=[*prefix,module+'.train','--arm',a.arm,'--execute']
    if a.resume:train.append('--resume')
    evaluate=[*prefix,module+'.evaluate','--arm',a.arm]
    if a.dry_run or not a.execute:
        print(json.dumps(dict(prepared_only=True,default_first='centroid',selected_arm=a.arm,preflight=check,train=train,evaluate=evaluate,automatic_next_arm=False,training_started=False),indent=2));return
    output=run_dir(a.arm)
    with exclusive(HERE/'runs/pipeline.lock'):
        if stopped(output):print('STOP present');return
        try:verify_preflight(a.arm)
        except ValueError:subprocess.run(check,cwd=ROOT,check=True)
        verify_preflight(a.arm)
        result=output/'results.json'
        if not result.exists() or not json.loads(result.read_text(encoding='utf-8')).get('finished'):
            state(a.arm,dict(status='training'));subprocess.run(train,cwd=ROOT,check=True)
        if stopped(output) or not result.exists() or not json.loads(result.read_text(encoding='utf-8')).get('finished'):return
        if not (output/'checkpoints/best.pt').exists():
            state(a.arm,dict(status='no_monitored_best',run=output.relative_to(HERE).as_posix()));return
        state(a.arm,dict(status='evaluation'));subprocess.run(evaluate,cwd=ROOT,check=True)
        complete=(output/'evaluation/gallery/index.html').exists()
        state(a.arm,dict(status='complete' if complete else 'evaluation_incomplete',run=output.relative_to(HERE).as_posix()))

if __name__=='__main__':main()
