"""E94 unique F-best fixed500 evaluation; no full-dev or training."""
import argparse,json,multiprocessing as mp,os,signal,time,zipfile
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
from ..io import HERE,ROOT,save,sha,exclusive
from ..data import load_rows,register_resident
from ..resident import ResidentRows
from ..model import GeometryModel
from ..batches import inputs_only,decode
from ..loss import per_case
from ..train import monitor as training_monitor
from ..dataset_view import ReferenceLayouts
from .common import ARM,RUN,OUTPUT,REGISTRATION,registration,ready,freeze,auxiliary_seconds,charge_auxiliary,seed_all
from .diagnostics import geometry_checks,distribution
from .metrics import module_overlap,union_area
from .worker import STEADY,summarize,initialize,evaluate

class EvaluationStop(Exception):pass

def diagnostic(payload):
    index,row,predicted,losses=payload
    p,checks,values=geometry_checks(predicted,row['case'].solver_inputs)
    overlap=module_overlap(p);bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    return {'index':index,'member':list(row['member']),'n':len(p),'positions':p.tolist(),
            'losses':losses,'raw_hard_legal':bool(checks['is_feasible']),**values,**overlap,
            'empty_fraction_of_bbox':max(0.,1.-union_area(p)/bbox)}

def validate_terminal(records,metadata):
    indices=[r['index'] for r in records]
    if len(set(indices))!=len(indices) or any(not 0<=i<500 for i in indices):raise ValueError('Terminal index drift')
    if any(r['member']!=list(metadata[r['index']]['member']) for r in records):raise ValueError('Terminal member drift')

def terminal(identity,halt,beat):
    folder=OUTPUT/'terminal';folder.mkdir(exist_ok=True)
    metadata=ReferenceLayouts('dev_monitor500').records
    config={**identity,'steady':STEADY,'timing':'same E90 CPU feature+inference+decode+flash; initialization/JIT/scoring separate'}
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text(encoding='utf-8'))!=config:
        raise ValueError('Frozen terminal configuration changed')
    save(folder/'config.json',config)
    target=folder/'results.json'
    if target.exists():
        value=json.loads(target.read_text(encoding='utf-8'));validate_terminal(value['records'],metadata)
        if len(value['records'])!=500 or value['checkpoint_sha256']!=identity['checkpoint_sha256']:
            raise ValueError('Existing terminal is incomplete or changed')
        if not (folder/'summary.json').exists():
            save(folder/'summary.json',{'checkpoint_sha256':identity['checkpoint_sha256'],**value['summary'],
                'workers':4,'threads_per_worker':2,'neutral_runtime_development_only':True,'test_read':False})
        return
    partial=folder/'partial.json'
    records=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else []
    validate_terminal(records,metadata);done={r['index'] for r in records}
    todo=iter(i for i in range(500) if i not in done);pending={}
    previous=os.environ.get('E94_EVAL_THREADS');os.environ['E94_EVAL_THREADS']='2'
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,
                                 initargs=(str(OUTPUT/'candidate_model.pt'),)) as pool:
            def submit():
                if halt():return
                index=next(todo,None)
                if index is not None:pending[pool.submit(evaluate,index)]=index
            for _ in range(4):submit()
            while pending:
                finished,_=wait(pending,timeout=1.,return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future);records.append(future.result());save(partial,records);submit()
                beat('flash500',completed=len(records),count=500)
    finally:
        if previous is None:os.environ.pop('E94_EVAL_THREADS',None)
        else:os.environ['E94_EVAL_THREADS']=previous
    if len(records)!=500:raise EvaluationStop()
    records.sort(key=lambda r:r['index']);summary=summarize(records)
    save(target,{'checkpoint_sha256':identity['checkpoint_sha256'],'summary':summary,'records':records,'test_read':False})
    save(folder/'summary.json',{'checkpoint_sha256':identity['checkpoint_sha256'],**summary,
                              'workers':4,'threads_per_worker':2,'neutral_runtime_development_only':True,'test_read':False})
    if partial.exists():partial.unlink()

def run():
    begin=time.perf_counter();prior=auxiliary_seconds();limit=registration()['evaluation_budget_seconds'];last=[0.]
    def halt():return (OUTPUT/'STOP').exists() or (HERE/'runs/STOP').exists() or prior+time.perf_counter()-begin>=limit
    def beat(activity,**fields):
        now=time.perf_counter()-begin
        if now-last[0]>=5. or activity!=getattr(beat,'activity',None):
            save(OUTPUT/'heartbeat.json',{'activity':activity,**fields,'elapsed_this_invocation':now,'remaining_seconds':max(0.,limit-prior-now),'test_read':False})
            beat.activity=activity;last[0]=now
    try:
        with exclusive(HERE/'runs/training.lock'),exclusive(OUTPUT.with_suffix('.lock')):
            state,identity=freeze();snap=OUTPUT/'source_snapshot.zip'
            if not snap.exists():
                with zipfile.ZipFile(snap,'w',zipfile.ZIP_DEFLATED) as archive:
                    for name in identity['sources']:archive.write(ROOT/name,name)
            if halt():raise EvaluationStop()
            monitor500(state,identity,halt,beat);del state;torch.cuda.empty_cache()
            terminal(identity,halt,beat)
            if halt():raise EvaluationStop()
            from .render import run as render
            beat('same100_gallery');render(OUTPUT,halt)
            from .compare import assess
            beat('saved_result_comparison');print(json.dumps(assess(identity),ensure_ascii=False,indent=2),flush=True)
            save(OUTPUT/'status.json',{'status':'completed','test_read':False});beat('completed')
    except EvaluationStop:save(OUTPUT/'status.json',{'status':'stopped','test_read':False})
    except BaseException as error:
        save(OUTPUT/'status.json',{'status':'error','error':repr(error),'test_read':False});raise
    finally:charge_auxiliary(time.perf_counter()-begin,'frozen E94 once500')

def monitor500(state,identity,halt,beat):
    target=OUTPUT/'dev500_loss.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Loss model drift')
        return
    seed_all(0,4);model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    rows=load_rows('dev_monitor500');pack=ResidentRows(rows,'cuda',max_nodes=120);register_resident(pack)
    beat('dev500_own_loss');value=training_monitor(model,rows,'F','cuda',halt)
    if value is None:raise EvaluationStop()
    if abs(value['total']-identity['selected_dev500_loss'])>1e-5:raise ValueError('Best monitor mismatch')
    save(target,{'checkpoint_sha256':identity['checkpoint_sha256'],'losses':value,'selected_training_monitor':identity['selected_dev500_loss'],'test_read':False})

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old=signal.signal(signal.SIGINT,lambda *_:(OUTPUT/'STOP').touch())
    try:run()
    finally:signal.signal(signal.SIGINT,old)
