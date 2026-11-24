"""Run the unchanged E161 evaluator on the returned frozen best in an isolated directory."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OWNER=HERE/'local_dev2000_v1'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',action='store_true',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    auth=json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
    if not auth['evaluation_enabled'] or auth['training_enabled']:
        raise PermissionError('This entry authorizes only the fixed local evaluation')
    plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    frozen=json.loads((OWNER/'frozen.json').read_text(encoding='utf-8'))
    if frozen['best_epoch']!=21 or frozen['checkpoint_sha256']!='496cab3556d54ed2001ab9ff98617cbd3aa61241474898de6205d0d450ec29b7':
        raise ValueError('Unexpected final own-Loss-best')
    if sha(ROOT/frozen['checkpoint'])!=frozen['checkpoint_sha256']:
        raise ValueError('Returned weight changed')
    for name,value in plan['sources'].items():
        if sha(ROOT/name)!=value: raise ValueError('Frozen evaluation source changed: '+name)
    if sha(OWNER/'budget.json')!=sha(ROOT/'experiments/E161_final_no_mib/evaluation/budget.json'):
        raise ValueError('Original postprocessing budget differs')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(ROOT))
    from experiments.E161_final_no_mib.io import save
    from experiments.E161_final_no_mib.evaluation import run as evaluator
    evaluator.OWNER=OWNER
    launch=dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=frozen['checkpoint'],checkpoint_sha256=frozen['checkpoint_sha256'],
        best_epoch=21,total=2000,model_source_unchanged=True,original_budget=True,
        original_training_pause_flags_retained=True,test_read=False,resume=args.resume)
    save(OWNER/'background/launch.json',launch)
    print(json.dumps(dict(stage='formal_local_dev2000',**launch)),flush=True)
    sys.argv=[str(__file__),'--arm','e161_s2']+(['--resume'] if args.resume else [])
    code=0
    try:
        evaluator.main()
    except BaseException as error:
        code=1
        save(OWNER/'background/entry_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        state_path=OWNER/'output/state.json'
        state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        save(OWNER/'background/exit.json',dict(pid=os.getpid(),exit_code=code,
            ended_utc=datetime.now(timezone.utc).isoformat(),status=state.get('status'),completed=state.get('completed',0)))

if __name__=='__main__': main()
