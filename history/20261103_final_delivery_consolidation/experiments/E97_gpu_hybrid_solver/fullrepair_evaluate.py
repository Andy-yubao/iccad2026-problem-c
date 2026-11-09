"""New authorized evaluation only: old full E85 repair, one CPU math thread."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,datetime,gc,hashlib,importlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from .latency import HERE,ROOT,IDS,Captured,registry,public_member_inputs,sha
from .latency_fixed import padded,returned
from .nn85.base_model import decode
from .repair85_one_thread.solver import optimize
from .repair85_one_thread.canonical_native import warmup
from .solver_relations_one_thread import optimize as optimize_relations
from .relation_model import relations
from .train_io.io import save,exclusive
from datasets.expanded_reference import ExpandedReferenceLayouts
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from visualization.layouts import plot_case,geometry_details
import clarabel

PLAN_PATH=HERE/'fullrepair_singlethread_plan.json'
PLAN=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
OUTPUT=HERE/PLAN['output'];PREFLIGHT=HERE/'runs/20261102_fullrepair_singlethread_preflight'
CLARABEL_CALLS=0
_original_solver=clarabel.DefaultSolver
def checked_solver(*args,**kwargs):
    global CLARABEL_CALLS
    settings=args[-1] if args else kwargs['settings']
    assert settings.max_threads==1,'Clarabel must be single-threaded'
    CLARABEL_CALLS+=1
    return _original_solver(*args,**kwargs)
clarabel.DefaultSolver=checked_solver

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def halt():return (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists()
def setup():
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numba
    numba.set_num_threads(1)

def thread_check():
    pools=threadpool_info()
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    assert all(p['num_threads']<=1 for p in pools),pools
    return dict(torch_intra=1,torch_inter=1,pools=pools,clarabel_calls_checked=CLARABEL_CALLS,process_workers=1,concurrent_requests=1)

def sources():
    paths=[Path(__file__),PLAN_PATH,HERE/'latency.py',HERE/'latency_fixed.py',HERE/'student.py',HERE/'deep_geometry.py',HERE/'normalized_geometry.py',HERE/'relation_model.py',HERE/'solver_relations_one_thread.py',ROOT/'visualization/layouts.py']
    paths+=list((HERE/'repair85_one_thread').rglob('*.py'))+list((HERE/'nn85').glob('*.py'))+list((ROOT/'scoring').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}

def verify_copy():
    for item in PLAN['copy_manifest']:
        assert sha(ROOT/item['original'])==item['original_sha256']
        assert sha(ROOT/item['copy'])==item['copy_sha256']

def model_for(arm):
    path=ROOT/arm['checkpoint'];assert sha(path)==arm['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    cls=importlib.import_module('.'+arm['model_kind'],package=__package__).DeploymentModel
    model=cls(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    del saved
    return model

def solve_request(model,cap,inputs,arm):
    torch.cuda.synchronize();start=time.perf_counter()
    host=padded(inputs,False)
    for k,v in host.items():cap.g[k].copy_(v)
    state=cap.replay();draft=returned(state,cap.g,inputs,decode)
    draft_seconds=time.perf_counter()-start
    edges=relations(state[0,:int(inputs[0])].cpu().double().numpy(),int(inputs[0])) if arm['fixed_relations'] else None
    if edges is None:final,trace=optimize(draft,inputs,**PLAN['full_repair'])
    else:final,trace=optimize_relations(draft,inputs,**PLAN['full_repair'],relations_override=edges)
    hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
    assert hard['is_feasible'] and np.isfinite(final).all() and (final[:,:2]>=0).all()
    elapsed=time.perf_counter()-start
    return draft,final,trace,hard,elapsed,draft_seconds,edges

def summary(records):
    q=[r['metrics']['cost'] for r in records];counts=[r['n'] for r in records];wall=np.array([r['complete_seconds'] for r in records])
    return dict(count=len(records),legal=sum(r['metrics']['is_feasible'] and r['independent_hard_checks']['is_feasible'] for r in records),weighted_Q=official_evaluator().compute_total_score(q,counts),mean_Q=float(np.mean(q)),max_Q=max(q),mean_complete_seconds=float(wall.mean()),p95_complete_seconds=float(np.percentile(wall,95)),max_complete_seconds=float(wall.max()),cases_over_025s=int((wall>.25).sum()),fallbacks=sum(r['fallback'] for r in records),draft_mean_seconds=float(np.mean([r['draft_seconds'] for r in records])),repair_mean_seconds=float(np.mean([r['repair_seconds'] for r in records])),shape_calls_mean=float(np.mean([sum(t.get('phase')=='shape' for t in r['trace']) for r in records])))

@torch.inference_mode()
def preflight():
    PREFLIGHT.mkdir(exist_ok=False);setup();verify_copy();start=time.perf_counter()
    with threadpool_limits(limits=1):
        arm=PLAN['arms'][0];model=model_for(arm);warmup();records=[];entries=registry()['sets']['train']['records']
        inputs0=public_member_inputs(entries[IDS[0]],'train');g={k:v.cuda() for k,v in padded(inputs0,False).items()};cap=Captured(model,g)
        cold=time.perf_counter()-start
        for index in IDS:
            if halt():raise RuntimeError('STOP during preflight')
            inputs=public_member_inputs(entries[index],'train')
            draft,final,trace,hard,elapsed,draft_seconds,_=solve_request(model,cap,inputs,arm)
            eager=returned(model(cap.g),cap.g,inputs,decode)
            error=float(np.max(np.abs(eager-draft))/float(cap.g['scale'][0]));assert error<2e-4
            records.append(dict(train_id=index,n=int(inputs[0]),hard_legal=True,fallback=trace['row_retained'],seconds=elapsed,draft_seconds=draft_seconds,graph_eager_normalized_max_abs=error,shape_calls=sum(t.get('phase')=='shape' for t in trace['trace'])))
        result=dict(status='passed',records=records,threads=thread_check(),source_sha256=sources(),cold_initialization_seconds=cold,elapsed_seconds=time.perf_counter()-start,no_training=True,test_read=False)
        save(PREFLIGHT/'results.json',result);print(json.dumps(result['records'],indent=2),flush=True)

@torch.inference_mode()
def evaluate():
    setup();verify_copy();check=read(PREFLIGHT/'results.json');assert check['status']=='passed' and check['source_sha256']==sources()
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/'error.json').exists():raise RuntimeError('Previous error requires diagnosis; no automatic retry')
    with exclusive(HERE/'runs/fullrepair_singlethread.lock'),threadpool_limits(limits=1):
        begin=time.perf_counter();data=ExpandedReferenceLayouts('dev_monitor500',train_size=16000);rows=[data[i] for i in range(500)]
        for row in rows:
            c=row['case'];c.reference_positions=public_hard_reference(c.solver_inputs[1],c.solver_inputs[5],c.solver_inputs[6])
        binding=dict(plan=PLAN,source_sha256=sources(),members=[list(r['member']) for r in rows],test_read=False)
        frozen=OUTPUT/'frozen.json'
        if frozen.exists():assert read(frozen)==binding
        else:
            save(frozen,binding)
            with zipfile.ZipFile(OUTPUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
                for name in binding['source_sha256']:z.write(ROOT/name,name)
        warmup();summaries={}
        for arm in PLAN['arms']:
            if halt():save(OUTPUT/'state.json',dict(status='stopped',arm=arm['name'],test_read=False));return
            folder=OUTPUT/arm['name'];result=folder/'results.json'
            if result.exists() and read(result).get('finished'):
                summaries[arm['name']]=read(result)['terminal'];continue
            cold=time.perf_counter();model=model_for(arm);g={k:v.cuda() for k,v in padded(rows[0]['case'].solver_inputs,False).items()};cap=Captured(model,g);cold=time.perf_counter()-cold
            save(folder/'initialization.json',dict(cold_model_graph_seconds=cold,threads=thread_check(),checkpoint_sha256=arm['sha256'],test_read=False))
            records=[]
            for i,row in enumerate(rows):
                if halt():save(OUTPUT/'state.json',dict(status='stopped',arm=arm['name'],completed=i,test_read=False));return
                path=folder/'terminal'/f'{i:04d}.json'
                if path.exists():record=read(path);assert record['member']==list(row['member'])
                else:
                    draft,final,trace,hard,elapsed,draft_seconds,edges=solve_request(model,cap,row['case'].solver_inputs,arm)
                    scored=evaluate_case(row['case'],final,elapsed)
                    assert scored['metrics']['is_feasible'] and scored['independent_hard_checks']['is_feasible']
                    record=dict(index=i,member=list(row['member']),n=int(row['case'].solver_inputs[0]),complete_seconds=elapsed,draft_seconds=draft_seconds,repair_seconds=trace['seconds'],metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],fallback=trace['row_retained'],trace=trace['trace'],positions=final.tolist(),draft_positions=draft.tolist())
                    if edges is not None:record['relation_edges']=edges.tolist()
                    save(path,record)
                records.append(record)
                if (i+1)%25==0:
                    save(OUTPUT/'state.json',dict(status='evaluating',arm=arm['name'],completed=i+1,completed_arms=summaries,elapsed_seconds=time.perf_counter()-begin,test_read=False))
                    print(arm['name'],i+1,'/500',flush=True)
            terminal=summary(records);summaries[arm['name']]=terminal
            save(result,dict(status='terminal_complete_gallery_pending',finished=True,terminal=terminal,threads=thread_check(),checkpoint_sha256=arm['sha256'],test_read=False))
            print('RESULT',arm['name'],json.dumps(terminal),flush=True)
            del cap,model;gc.collect();torch.cuda.empty_cache()
        # Draw only after all formal timed requests; CPU rendering never overlaps timing.
        for arm in PLAN['arms']:
            folder=OUTPUT/arm['name'];gallery=folder/'gallery';gallery.mkdir(exist_ok=True);manifest=[]
            for i,row in enumerate(rows[:100]):
                if halt():return
                r=read(folder/'terminal'/f'{i:04d}.json');base=read(HERE/f'runs/20261102_candidate1_distill_seed0/evaluation/terminal/{i:04d}.json')['L_positions']
                c=np.asarray(row['case'].solver_inputs[5]).tolist();variants={k:dict(positions=v) for k,v in [('reference',row['positions'].tolist()),('placer',base),('corrected',r['draft_positions']),('final',r['positions'])]}
                for value in variants.values():value['soft_details']=geometry_details(value['positions'],c)
                path=gallery/f'{i:03d}.png'
                if not path.exists():plot_case(dict(case_id=f'dev500:{i}',panel=f'E97 full E85 repair / 1 thread / {arm["name"]}',constraints=c,variants=variants),path,stages=(('reference','Reference'),('placer','Frozen E74'),('corrected',arm['name']+' draft'),('final','Full E85 repair / one thread')),show_metrics=False)
                manifest.append(dict(index=i,member=list(row['member']),image=path.name,sha256=sha(path)))
            save(gallery/'manifest.json',dict(count=100,cases=manifest,shared_IDs_and_scale=True,score_controls=False,test_read=False))
            html='<!doctype html><meta charset="utf-8"><title>E97 full repair one thread</title><style>img{width:100%}body{font:16px system-ui}</style><h1>'+arm['name']+' / full E85 repair / one thread</h1>'
            html+=''.join(f'<p>{r["index"]}</p><img loading="lazy" src="{r["image"]}">' for r in manifest)
            (gallery/'index.html').write_text(html,encoding='utf-8')
            result=read(folder/'results.json');result.update(status='complete',gallery_count=100);save(folder/'results.json',result)
            save(OUTPUT/'state.json',dict(status='rendering',last_gallery=arm['name'],completed_arms=summaries,test_read=False))
            print('GALLERY',arm['name'],'100/100',flush=True)
        save(OUTPUT/'results.json',dict(status='complete',finished=True,arms=summaries,gallery_count_per_arm=100,no_training=True,test_read=False,threads=thread_check(),elapsed_seconds=time.perf_counter()-begin))
        save(OUTPUT/'state.json',dict(status='complete',completed_arms=summaries,test_read=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else evaluate()
    except BaseException as error:
        save((PREFLIGHT if args.preflight else OUTPUT)/'error.json',dict(error=repr(error),no_automatic_retry=True,test_read=False));raise
