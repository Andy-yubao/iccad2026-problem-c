"""Each new frozen intervention gets one resumable dev/raw/flash evaluation."""
import json,multiprocessing as mp,os,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch
from scoring.evaluation import evaluate_case
from .io import HERE,ROOT,sha,save
from .prepare import read
from .data import load_rows,collate
from .batches import inputs_only
from .model import GeometryModel,decode
from .loss import source_terms
from .diagnostics import geometry_checks
from .inference import predict
from .runtime import seed_all

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,
    plateau=True,robust_seeds=True,certified=True,quick_project=True,
    quadratic_project=True,incumbent_project=True,snap=1e-4)

def stopped(folder):return (HERE/'runs/STOP').exists() or (folder/'STOP').exists()
def validate(identity,state):
    from .io import sources
    if sha(ROOT/identity['checkpoint'])!=identity['checkpoint_sha256'] or state['sources']!=sources():raise ValueError('Frozen composition changed')

def full_dev(folder,state,identity):
    target=folder/'full_dev.json'
    if target.exists():
        if read(target)['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Completed raw identity changed')
        return True
    seed_all(0,4);model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'],strict=True)
    rows=load_rows('dev');path=folder/'full_dev_partial.json';records=read(path) if path.exists() else []
    if [r['member'] for r in records]!=[list(r['member']) for r in rows[:len(records)]]:raise ValueError('Raw cursor/member drift')
    with torch.no_grad():
        for first in range(len(records),len(rows),32):
            if stopped(folder):return False
            selected=rows[first:first+32];g,_=collate(selected,'cuda');state_out=model(inputs_only(g));parts=source_terms(state_out,g)
            decoded=(decode(state_out,g)*g['scale'][:,None,None]).cpu().double().numpy()
            for j,row in enumerate(selected):
                n=row['graph']['n'];_,hard,diag=geometry_checks(decoded[j,:n],row['case'].solver_inputs)
                records.append({'member':list(row['member']),'n':n,'raw_hard_legal':bool(hard['is_feasible']),**diag,
                    **{'loss_'+k:float(v[j].cpu()) for k,v in parts.items()}})
            save(path,records)
            if (first//32)%10==0:print(folder.name,'raw',len(records),'/2000',flush=True)
    save(target,{'checkpoint_sha256':identity['checkpoint_sha256'],'records':records,'count':len(records),
        'raw_hard_legal':sum(r['raw_hard_legal'] for r in records),'mean_overlap':float(np.mean([r['overlap_area_fraction'] for r in records])),
        'mean_fixed_source_loss':float(np.mean([r['loss_total'] for r in records])),'training_updates':0,'test_read':False})
    path.unlink();del model;torch.cuda.empty_cache();return True

def initialize(path):
    global worker_model,worker_rows,worker_initialization
    start=time.perf_counter();seed_all(0,2)
    from .canonical_native import warmup
    warmup();state=torch.load(path,map_location='cpu',weights_only=False)
    from .io import sources
    if state['sources']!=sources():raise ValueError('Worker sources drift')
    worker_model=GeometryModel(**state['model_config']).eval();worker_model.load_state_dict(state['model'],strict=True)
    worker_rows=load_rows('dev_monitor500');worker_initialization=time.perf_counter()-start

def evaluate(index):
    from .solver import optimize
    seed_all(index,2);row=worker_rows[index];case=row['case'];start=time.perf_counter()
    record={'index':index,'member':list(row['member']),'n':case.block_count}
    try:
        draft,inference_seconds=predict(worker_model,case.solver_inputs)
        positions,stats=optimize(draft,case.solver_inputs,**STEADY);elapsed=time.perf_counter()-start
        final_score=evaluate_case(case,positions.tolist(),elapsed);raw_score=evaluate_case(case,draft.tolist(),inference_seconds)
        _,hard,diag=geometry_checks(draft,case.solver_inputs)
        local=np.sqrt(np.asarray(case.solver_inputs[1],dtype=float))
        motion=np.linalg.norm((positions[:,:2]+positions[:,2:]/2)-(draft[:,:2]+draft[:,2:]/2),axis=1)/local
        trace=stats.get('trace',[])
        record.update(seconds=elapsed,metrics=final_score['metrics'],raw_metrics=raw_score['metrics'],raw_hard_legal=bool(hard['is_feasible']),
          draft_diagnostics=diag,draft_positions=draft.tolist(),positions=positions.tolist(),trace=trace,
          row_fallback=bool(stats['row_retained']),rescue_used=any(any(s in str(t.get('phase','')) for s in ('rescue','packed')) for t in trace),
          inference_seconds=inference_seconds,postprocess_seconds=stats['seconds'],mean_center_move_local=float(motion.mean()),max_center_move_local=float(motion.max()),
          shape_candidate_count=sum(t.get('phase')=='shape' for t in trace),failed_shape_candidate_count=sum(t.get('phase')=='shape' and not t.get('legal',False) for t in trace),
          worker_pid=os.getpid(),resident_initialization_seconds=worker_initialization)
    except Exception as error:record.update(seconds=time.perf_counter()-start,metrics={'cost':10.,'is_feasible':False},error=repr(error))
    return record

def summary(records):
    q=np.array([r['metrics']['cost'] for r in records]);n=np.array([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    result={'count':len(records),'hard_legal':sum(r['metrics']['is_feasible'] for r in records),'weighted_Q':float(np.average(q,weights=w)),
      'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),'max_Q':float(q.max()),'worst5pct_mean_Q':float(np.sort(q)[-25:].mean()),
      'seconds_mean':float(np.mean([r['seconds'] for r in records])),'rescue_cases':sum(r.get('rescue_used',False) for r in records),'errors':sum('error' in r for r in records)}
    for key in ('hpwl_gap','area_gap','grouping_violations','boundary_violations','mib_violations'):
        vals=[r['metrics'][key] for r in records if key in r['metrics']];result[key+'_mean']=float(np.mean(vals)) if vals else None
    return result

def terminal(folder,identity):
    target=folder/'terminal';target.mkdir(exist_ok=True)
    if (target/'results.json').exists():
        if read(target/'config.json')['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Completed terminal changed')
        return True
    path=target/'partial.json';records=read(path) if path.exists() else [];done={r['index'] for r in records}
    from .dataset_view import ReferenceLayouts
    metadata=ReferenceLayouts('dev_monitor500').records
    if len(done)!=len(records) or any(i not in range(500) for i in done):raise ValueError('Invalid terminal cursor')
    if any(r['member']!=list(metadata[r['index']]['member']) for r in records):raise ValueError('Terminal member drift')
    save(target/'config.json',{**identity,'steady':STEADY,'timing':'CPU input features/inference/decode and full repair; initialization/scoring excluded'})
    old=os.environ.get('E83_THREADS');os.environ['E83_THREADS']='2'
    try:
        todo=iter(i for i in range(500) if i not in done);pending={}
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(ROOT/identity['checkpoint'],)) as pool:
            def submit():
                if stopped(folder):return
                index=next(todo,None)
                if index is not None:pending[pool.submit(evaluate,index)]=index
            for _ in range(2):submit()
            while pending:
                completed,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future);records.append(future.result());save(path,records)
                    if len(records)%25==0:print(folder.name,'terminal',len(records),'/500',flush=True)
                    submit()
    finally:
        if old is None:os.environ.pop('E83_THREADS',None)
        else:os.environ['E83_THREADS']=old
    if len(records)!=500:return False
    records.sort(key=lambda r:r['index']);save(target/'results.json',{'checkpoint_sha256':identity['checkpoint_sha256'],'records':records,'summary':summary(records),'test_read':False});path.unlink()
    print(folder.name,json.dumps(summary(records)),flush=True);return True

def run(folder,state,identity):
    validate(identity,state)
    return full_dev(folder,state,identity) and terminal(folder,identity)
