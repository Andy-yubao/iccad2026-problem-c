"""One frozen E98 best, full copied E85 repair, sequential CPU1/GPU batch1."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from ..io import ROOT,HERE,save,sha,exclusive
from ..model import GeometryModel
from ..features import graph_features
from ..batches import decode
from ..cache import load_rows
from datasets.reference_layouts import registry,public_hard_reference
from datasets.public_member_inputs import public_member_inputs
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .repair.solver import optimize
from .repair.canonical_native import warmup
from .metrics import module_overlap

OWNER=Path(__file__).resolve().parent
PLAN_PATH=OWNER/'plan.json'
PLAN=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
OUTPUT=ROOT/PLAN['output']
PREFLIGHT=OWNER/'preflight.json'

def sources():
    paths=list(OWNER.rglob('*.py'))+[PLAN_PATH]
    paths += [HERE/name for name in ('model.py','features.py','batches.py','cache.py','io.py')]
    paths += list((ROOT/'scoring').glob('*.py'))+[ROOT/'visualization/layouts.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def verify():
    assert sha(ROOT/PLAN['checkpoint'])==PLAN['checkpoint_sha256']
    for record in PLAN['copy_manifest']:
        assert sha(ROOT/record['source'])==record['sha256']
        assert sha(ROOT/record['copy'])==record['sha256']

def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0);np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    import numba
    numba.set_num_threads(1)

def thread_check():
    pools=threadpool_info()
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    assert all(p['num_threads']<=1 for p in pools)
    return dict(torch_intra=1,torch_inter=1,pools=pools,workers=1,concurrent_requests=1)

def public_graph(inputs,pad=120):
    g=graph_features(inputs);n=int(inputs[0]);scale=g['scale']
    c=torch.as_tensor(inputs[5]).long();spec=g['specs'];locked=g['locked']
    area=torch.where(locked,spec[:,2:].prod(-1),g['areas'])/scale.square()
    known=torch.cat(((spec[:,:2]+spec[:,2:]/2)/scale,
                     (spec[:,2].clamp_min(1e-8)/spec[:,3].clamp_min(1e-8)).log()[:,None]),1)*(1-g['mask'])
    data=dict(features=g['features'],adjacency=g['adjacency'],mib=g['relations'][0],group=g['relations'][1],
              mask=g['mask'],area=area,known=known,specs=torch.where(locked[:,None],spec/scale,torch.zeros_like(spec)),
              constraints=c)
    out={}
    for key,value in data.items():
        shape=(pad,pad) if key in ('adjacency','mib','group') else (pad,*value.shape[1:])
        out[key]=torch.zeros((1,*shape),dtype=value.dtype)
        if key in ('adjacency','mib','group'):out[key][0,:n,:n]=value
        else:out[key][0,:n]=value
    out['valid']=torch.arange(pad)[None,:]<n
    out['scale']=scale[None]
    return out

class Predictor:
    def __init__(self,inputs):
        saved=torch.load(ROOT/PLAN['checkpoint'],map_location='cpu',weights_only=False)
        self.model=GeometryModel(**saved['model_config']).cuda().eval()
        self.model.load_state_dict(saved['model'],strict=True)
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):decode(self.model(self.g),self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=decode(self.model(self.g),self.g)
        torch.cuda.synchronize()

    def predict(self,inputs):
        host=public_graph(inputs)
        for key,value in host.items():self.g[key].copy_(value)
        self.graph.replay()
        p=(self.output[0,:int(inputs[0])]*self.g['scale'][0]).cpu().numpy().astype(float)
        c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1);pre=c[:,1]!=0
        p[locked,2:]=s[locked,2:];p[pre]=s[pre]
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite draft')
        return p

def request(predictor,inputs):
    torch.cuda.synchronize();start=time.perf_counter()
    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-start
    final,stats=optimize(draft,inputs,**PLAN['full_repair'])
    hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),
                                 public_hard_reference(inputs[1],inputs[5],inputs[6]))
    if not hard['is_feasible'] or not np.isfinite(final).all() or (final[:,:2]<0).any():
        raise RuntimeError('Final hard legality failed')
    return draft,final,stats,hard,time.perf_counter()-start,draft_seconds

def halt():return (OUTPUT/'STOP').exists() or (OWNER/'STOP').exists()

def summary(records):
    q=np.asarray([r['metrics']['cost'] for r in records]);wall=np.asarray([r['complete_seconds'] for r in records])
    return dict(count=len(records),hard_legal=sum(r['metrics']['is_feasible'] for r in records),
         weighted_Q=official_evaluator().compute_total_score(q.tolist(),[r['n'] for r in records]),
         mean_Q=float(q.mean()),p95_Q=float(np.percentile(q,95)),max_Q=float(q.max()),
         mean_complete_seconds=float(wall.mean()),p95_complete_seconds=float(np.percentile(wall,95)),
         fallbacks=sum(r['fallback'] for r in records),
         raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),
         draft_module_pass10_fraction=float(np.mean([r['raw_overlap']['module_pass10_fraction'] for r in records])),
         draft_worst_module_coverage_mean=float(np.mean([r['raw_overlap']['max_covered_fraction'] for r in records])))

@torch.inference_mode()
def preflight():
    setup();verify();started=time.perf_counter()
    records=registry()['sets']['train']['records'];ids=PLAN['preflight_train_indices']
    inputs=[public_member_inputs(records[i],'train') for i in ids]
    with threadpool_limits(limits=1):
        warmup();predictor=Predictor(inputs[0]);checks=[]
        for index,inp in zip(ids,inputs):
            draft,final,stats,hard,seconds,draft_seconds=request(predictor,inp)
            natural={k:v.cuda() for k,v in public_graph(inp,pad=int(inp[0])).items()}
            eager=(decode(predictor.model(natural),natural)[0]*natural['scale'][0]).cpu().numpy().astype(float)
            c,s=np.asarray(inp[5]),np.asarray(inp[6]);locked=c[:,:2].any(1);pre=c[:,1]!=0
            eager[locked,2:]=s[locked,2:];eager[pre]=s[pre]
            error=float(np.max(np.abs(eager-draft))/float(natural['scale'][0]));assert error<2e-4
            checks.append(dict(train_index=index,hard_legal=hard['is_feasible'],normalized_capture_error=error,seconds=seconds))
        save(PREFLIGHT,dict(status='passed',sources=sources(),checks=checks,threads=thread_check(),
             elapsed_seconds=time.perf_counter()-started,test_read=False,parameter_updates=0))
        print('PREFLIGHT PASSED',json.dumps(checks),flush=True)

@torch.inference_mode()
def evaluate():
    setup();verify();check=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    assert check['status']=='passed' and check['sources']==sources()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'error.json').exists():raise RuntimeError('Prior error requires diagnosis; no automatic retry')
    with exclusive(OWNER/'evaluation.lock'),threadpool_limits(limits=1):
        started=time.perf_counter()
        save(OUTPUT/'state.json',dict(status='loading',completed=0,pid=os.getpid(),test_read=False))
        rows=load_rows('dev_monitor500')
        expected=registry()['sets']['dev_monitor500']['records']
        assert [list(r['member']) for r in rows]==[r['member'] for r in expected]
        binding=dict(plan=PLAN,sources=sources(),members=[list(r['member']) for r in rows],test_read=False)
        if (OUTPUT/'frozen.json').exists():assert json.loads((OUTPUT/'frozen.json').read_text())==binding
        else:
            save(OUTPUT/'frozen.json',binding)
            with zipfile.ZipFile(OUTPUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in binding['sources']:archive.write(ROOT/name,name)
        for row in rows:
            case=row['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
        cold=time.perf_counter();warmup();predictor=Predictor(rows[0]['case'].solver_inputs)
        save(OUTPUT/'initialization.json',dict(seconds=time.perf_counter()-cold,threads=thread_check(),test_read=False))
        records=[]
        for i,row in enumerate(rows):
            if halt() or time.perf_counter()-started>3600:
                save(OUTPUT/'state.json',dict(status='stopped',completed=i,test_read=False));return
            path=OUTPUT/f'terminal/{i:04d}.json'
            if path.exists():
                record=json.loads(path.read_text());assert record['member']==list(row['member'])
            else:
                draft,final,stats,hard,seconds,draft_seconds=request(predictor,row['case'].solver_inputs)
                score=evaluate_case(row['case'],final,seconds);raw=evaluate_case(row['case'],draft,draft_seconds)
                assert score['metrics']['is_feasible'] and score['independent_hard_checks']['is_feasible']
                record=dict(index=i,member=list(row['member']),n=int(row['case'].block_count),
                     complete_seconds=seconds,draft_seconds=draft_seconds,repair_seconds=stats['seconds'],
                     metrics=score['metrics'],independent_hard_checks=hard,raw_metrics=raw['metrics'],
                     raw_overlap=module_overlap(draft),draft_positions=draft.tolist(),positions=final.tolist(),
                     fallback=stats['row_retained'],trace=stats['trace'])
                save(path,record)
            records.append(record)
            if (i+1)%10==0:
                save(OUTPUT/'state.json',dict(status='evaluating',completed=i+1,total=500,pid=os.getpid(),
                     elapsed_seconds=time.perf_counter()-started,test_read=False,parameter_updates=0))
                print(f'{i+1}/500',flush=True)
        terminal=summary(records);verify()
        save(OUTPUT/'results.json',dict(status='terminal_complete_gallery_pending',finished=True,terminal=terminal,
             threads=thread_check(),checkpoint_sha256=PLAN['checkpoint_sha256'],test_read=False))
        print('RESULT',json.dumps(terminal),flush=True)
        from visualization.layouts import plot_case,geometry_details
        gallery=OUTPUT/'gallery';gallery.mkdir(exist_ok=True);manifest=[]
        for i,row in enumerate(rows[:100]):
            if halt():return
            record=records[i];constraints=np.asarray(row['case'].solver_inputs[5]).tolist()
            reference=(row['graph']['reference']*row['graph']['scale']).numpy().tolist()
            variants={name:dict(positions=p,soft_details=geometry_details(p,constraints)) for name,p in
                      [('reference',reference),('raw',record['draft_positions']),('final',record['positions'])]}
            image=gallery/f'{i:03d}.png'
            if not image.exists():plot_case(dict(case_id=f'dev500:{i}',panel='E98 best58 / full repair / CPU1',constraints=constraints,variants=variants),
                 image,stages=(('reference','Reference'),('raw','E98 layout model'),('final','Full repair')),show_metrics=False)
            manifest.append(dict(index=i,member=list(row['member']),image=image.name))
        save(gallery/'manifest.json',dict(cases=manifest,score_controls=False,shared_ids_and_scale=True,test_read=False))
        (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E98 dev500</title><style>img{width:100%}</style>'+''.join(f'<p>{r["index"]}</p><img loading="lazy" src="{r["image"]}">' for r in manifest),encoding='utf-8')
        save(OUTPUT/'results.json',dict(status='complete',finished=True,terminal=terminal,gallery_count=100,
             threads=thread_check(),checkpoint_sha256=PLAN['checkpoint_sha256'],test_read=False,
             elapsed_seconds=time.perf_counter()-started))
        save(OUTPUT/'state.json',dict(status='complete',completed=500,total=500,terminal=terminal,test_read=False))
        (OUTPUT/'report.md').write_text('# E98 best58固定dev500完整后处理评价\n\n'+json.dumps(terminal,ensure_ascii=False,indent=2)+'\n\n测试封存；只评价冻结best，未训练或调权；中性时间Q与完整逐例耗时分列。\n\n[同100例图集](gallery/index.html)\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else evaluate()
    except BaseException as error:
        save(OWNER/'preflight_error.json' if args.preflight else OUTPUT/'error.json',dict(error=repr(error),test_read=False,no_automatic_retry=True))
        raise
