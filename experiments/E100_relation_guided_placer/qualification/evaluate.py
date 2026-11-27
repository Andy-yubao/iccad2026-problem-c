"""One dev500 terminal evaluation and passive relation/geometry diagnostics."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
import json,time,traceback,zipfile
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
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .repair.solver import optimize
from .repair.canonical_native import warmup
from .metrics import module_overlap

OWNER=Path(__file__).resolve().parent
PLAN=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
OUTPUT=ROOT/PLAN['output']

def source_binding():
    files=list(OWNER.rglob('*.py'))+[OWNER/'plan.json']
    files += [HERE/name for name in ('model.py','base_model.py','features.py','batches.py','cache.py','io.py')]
    files += list((ROOT/'scoring').glob('*.py'))+[ROOT/'visualization/layouts.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}

def verify():
    for path,key in (('checkpoint','checkpoint_sha256'),('baseline_checkpoint','baseline_checkpoint_sha256')):
        if sha(ROOT/PLAN[path])!=PLAN[key]:raise ValueError('Frozen checkpoint drift')
    for item in PLAN['copy_manifest']:
        if sha(ROOT/item['copy'])!=item['sha256']:raise ValueError('Copied repair drift')

def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(0);np.random.seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numba
    numba.set_num_threads(1)

def public_graph(inputs,pad=120):
    g=graph_features(inputs);n=int(inputs[0]);scale=g['scale'];spec=g['specs'];locked=g['locked']
    area=torch.where(locked,spec[:,2:].prod(-1),g['areas'])/scale.square()
    known=torch.cat(((spec[:,:2]+spec[:,2:]/2)/scale,
          (spec[:,2].clamp_min(1e-8)/spec[:,3].clamp_min(1e-8)).log()[:,None]),1)*(1-g['mask'])
    data=dict(features=g['features'],adjacency=g['adjacency'],mib=g['relations'][0],group=g['relations'][1],
        mask=g['mask'],area=area,known=known,specs=torch.where(locked[:,None],spec/scale,torch.zeros_like(spec)),
        constraints=torch.as_tensor(inputs[5]).long())
    out={}
    for key,value in data.items():
        shape=(pad,pad) if key in ('adjacency','mib','group') else (pad,*value.shape[1:])
        out[key]=torch.zeros((1,*shape),dtype=value.dtype)
        if key in ('adjacency','mib','group'):out[key][0,:n,:n]=value
        else:out[key][0,:n]=value
    out['valid']=torch.arange(pad)[None,:]<n;out['scale']=scale[None]
    return out

def enforce_public(p,inputs):
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1);pre=c[:,1]!=0
    p[locked,2:]=s[locked,2:];p[pre]=s[pre]
    if not np.isfinite(p).all():raise RuntimeError('Nonfinite draft')
    return p

class Predictor:
    def __init__(self,inputs):
        saved=torch.load(ROOT/PLAN['checkpoint'],map_location='cpu',weights_only=False)
        self.model=GeometryModel(**saved['model_config']).cuda().eval()
        self.model.load_state_dict(saved['model'],strict=True)
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):decode(self.model(self.g)['state'],self.g)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.raw=self.model(self.g);self.output=decode(self.raw['state'],self.g)
        torch.cuda.synchronize()

    def predict(self,inputs):
        host=public_graph(inputs)
        for key,value in host.items():self.g[key].copy_(value)
        self.graph.replay()
        p=(self.output[0,:int(inputs[0])]*self.g['scale'][0]).cpu().numpy().astype(float)
        return enforce_public(p,inputs)

    def diagnostics(self,inputs):
        n=int(inputs[0])
        p=(decode(self.raw['initial_state'],self.g)[0,:n]*self.g['scale'][0]).cpu().numpy().astype(float)
        probabilities=self.raw['relation_logits'][0,:n,:n].softmax(-1).cpu().numpy()
        confidence=self.raw['relation_confidence'][0,:n,:n].cpu().numpy()
        return enforce_public(p,inputs),probabilities,confidence

def overlap(p):
    value=module_overlap(p);area=p[:,2:].prod(-1)
    size=np.maximum(np.minimum((p[:,:2]+p[:,2:])[:,None],(p[:,:2]+p[:,2:])[None])-np.maximum(p[:,:2][:,None],p[:,:2][None]),0)
    inter=size.prod(-1);np.fill_diagonal(inter,0)
    pair=inter.sum(-1)/area
    value.update(pair_intersection_fraction=float(.5*inter.sum()/area.sum()),
        pair_sum_mean=float(pair.mean()),pair_sum_max=float(pair.max()),pair_sum_worst_module=int(pair.argmax()),
        worst_module=int(np.argmax(value['covered_fractions'])),module_count=len(p),
        below10_count=int((np.asarray(value['covered_fractions'])<=.10).sum()))
    return value

def relation_diagnostics(initial,guided,probabilities,confidence,reference,inputs):
    n=len(initial);lo=reference[:,:2];hi=lo+reference[:,2:]
    allowed=np.stack((hi[:,None,0]<=lo[None,:,0]+1e-6,hi[None,:,0]<=lo[:,None,0]+1e-6,
                       hi[:,None,1]<=lo[None,:,1]+1e-6,hi[None,:,1]<=lo[:,None,1]+1e-6),-1)
    movable=~np.asarray(inputs[5])[:,1].astype(bool)
    valid=np.triu(np.ones((n,n),bool),1)&allowed.any(-1)&(movable[:,None]|movable[None,:])
    choice=probabilities.argmax(-1)
    model_correct=np.take_along_axis(allowed,choice[...,None],-1)[...,0]
    wh=initial[:,2:];center=initial[:,:2]+wh/2;delta=center[None]-center[:,None];half=(wh[:,None]+wh[None])/2
    violation=np.maximum(np.stack((half[...,0]-delta[...,0],half[...,0]+delta[...,0],half[...,1]-delta[...,1],half[...,1]+delta[...,1]),-1),0)
    heuristic=violation.argmin(-1);heuristic_correct=np.take_along_axis(allowed,heuristic[...,None],-1)[...,0]
    separation=np.maximum(np.maximum(lo[:,None]-hi[None],lo[None]-hi[:,None]),0)
    local=np.sqrt(reference[:,2:].prod(-1));scale=(local[:,None]+local[None])/2
    initial_overlap=(np.minimum((initial[:,:2]+wh)[:,None],(initial[:,:2]+wh)[None])-np.maximum(initial[:,:2][:,None],initial[:,:2][None])).min(-1)>1e-6
    connected=np.zeros((n,n),bool)
    for i,j,w in np.asarray(inputs[2]):
        if i>=0 and j>=0 and w>0:connected[int(i),int(j)]=connected[int(j),int(i)]=True
    masks=dict(all=valid,near_reference=valid&(np.linalg.norm(separation,axis=-1)<=.25*scale),
               initial_overlap=valid&initial_overlap,connected=valid&connected)
    result={}
    for name,mask in masks.items():
        count=int(mask.sum())
        result[name]=dict(count=count,correct=int((model_correct&mask).sum()),
            coordinate_heuristic_correct=int((heuristic_correct&mask).sum()),confidence_sum=float(confidence[mask].sum()))
    return result

def distribution(values):
    x=np.asarray(values,dtype=float)
    return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),maximum=float(x.max()))

def summarize(records):
    summaries={}
    for name in ('initial','guided','baseline'):
        vals=[r['stages'][name]['overlap'] for r in records]
        worst=max(range(len(records)),key=lambda i:vals[i]['max_covered_fraction'])
        summaries[name]=dict(count=len(records),
            pair_intersection_fraction=distribution([v['pair_intersection_fraction'] for v in vals]),
            module_pass10_mean_layout=float(np.mean([v['module_pass10_fraction'] for v in vals])),
            all_modules_pass10=sum(v['all_modules_pass10'] for v in vals),
            all_modules_below10_fraction=sum(v['below10_count'] for v in vals)/sum(v['module_count'] for v in vals),
            module_coverage_max=distribution([v['max_covered_fraction'] for v in vals]),
            bbox_over_A=distribution([v['bbox_over_A'] for v in vals]),
            worst=dict(index=records[worst]['index'],member=records[worst]['member'],module=vals[worst]['worst_module'],coverage=vals[worst]['max_covered_fraction']),
            hard_legal=sum(r['stages'][name]['metrics']['is_feasible'] for r in records))
    final=[r['metrics'] for r in records];q=[v['cost'] for v in final]
    summaries['final']=dict(count=len(records),hard_legal=sum(v['is_feasible'] for v in final),
        weighted_Q=official_evaluator().compute_total_score(q,[r['n'] for r in records]),
        Q=distribution(q),complete_seconds=distribution([r['complete_seconds'] for r in records]),
        draft_seconds=distribution([r['draft_seconds'] for r in records]),
        at_least250ms=sum(r['complete_seconds']>=.25 for r in records),fallbacks=sum(r['fallback'] for r in records))
    for name in ('initial','guided','baseline'):
        summaries[name]['hpwl_gap']=distribution([r['stages'][name]['metrics']['hpwl_gap'] for r in records])
    summaries['relation']={}
    for name in records[0]['relation']:
        vals=[r['relation'][name] for r in records];count=sum(v['count'] for v in vals)
        summaries['relation'][name]=dict(count=count,
            accuracy=sum(v['correct'] for v in vals)/max(count,1),
            coordinate_heuristic_accuracy=sum(v['coordinate_heuristic_correct'] for v in vals)/max(count,1),
            mean_confidence=sum(v['confidence_sum'] for v in vals)/max(count,1))
    old=[r['baseline_final_Q'] for r in records]
    summaries['comparison']=dict(baseline_weighted_Q=official_evaluator().compute_total_score(old,[r['n'] for r in records]),
        final_Q_wins=sum(a<b-1e-9 for a,b in zip(q,old)),final_Q_losses=sum(a>b+1e-9 for a,b in zip(q,old)),
        initial_to_guided_overlap_improved=sum(r['stages']['guided']['overlap']['pair_intersection_fraction']<r['stages']['initial']['overlap']['pair_intersection_fraction'] for r in records),
        initial_to_guided_overlap_worsened=sum(r['stages']['guided']['overlap']['pair_intersection_fraction']>r['stages']['initial']['overlap']['pair_intersection_fraction'] for r in records))
    return summaries

@torch.inference_mode()
def main():
    setup();verify();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'results.json').exists():raise RuntimeError('Completed evaluation must be reused')
    if (OUTPUT/'error.json').exists():raise RuntimeError('Prior error requires diagnosis')
    with exclusive(OWNER/'evaluation.lock'),threadpool_limits(limits=1):
        started=time.perf_counter()
        rows=load_rows('dev_monitor500');expected=registry()['sets']['dev_monitor500']['records']
        if [list(r['member']) for r in rows]!=PLAN['members'] or [r['member'] for r in expected]!=PLAN['members']:
            raise ValueError('Frozen panel changed')
        binding=dict(plan=PLAN,sources=source_binding(),test_read=False)
        save(OUTPUT/'frozen.json',binding)
        with zipfile.ZipFile(OUTPUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in binding['sources']:archive.write(ROOT/name,name)
        cold=time.perf_counter();warmup();predictor=Predictor(rows[0]['case'].solver_inputs)
        save(OUTPUT/'initialization.json',dict(seconds=time.perf_counter()-cold,threads=threadpool_info(),test_read=False))
        records=[]
        for i,row in enumerate(rows):
            if (OUTPUT/'STOP').exists() or time.perf_counter()-started>PLAN['maximum_seconds']:raise RuntimeError('Evaluation stopped before completion')
            path=OUTPUT/f'terminal/{i:04d}.json'
            if path.exists():
                record=json.loads(path.read_text(encoding='utf-8'))
                if record['member']!=list(row['member']):raise ValueError('Saved member mismatch')
            else:
                case=row['case'];inp=case.solver_inputs
                case.reference_positions=public_hard_reference(inp[1],inp[5],inp[6])
                baseline=json.loads((ROOT/PLAN['baseline_output']/f'terminal/{i:04d}.json').read_text(encoding='utf-8'))
                if baseline['member']!=list(row['member']):raise ValueError('Baseline member mismatch')
                torch.cuda.synchronize();tick=time.perf_counter()
                draft=predictor.predict(inp);draft_seconds=time.perf_counter()-tick
                final,stats=optimize(draft,inp,**PLAN['full_repair'])
                hard=independent_hard_checks(final.tolist(),np.asarray(inp[1]),np.asarray(inp[5]),case.reference_positions)
                if not hard['is_feasible'] or not np.isfinite(final).all() or (final[:,:2]<0).any():raise RuntimeError('Final hard legality failed')
                complete_seconds=time.perf_counter()-tick
                initial,probability,confidence=predictor.diagnostics(inp)
                reference=(row['graph']['reference']*row['graph']['scale']).numpy().astype(float)
                score=evaluate_case(case,final,complete_seconds)
                stages={}
                for name,p in (('initial',initial),('guided',draft),('baseline',np.asarray(baseline['draft_positions']))):
                    metrics=baseline['raw_metrics'] if name=='baseline' else evaluate_case(case,p,draft_seconds if name=='guided' else 0)['metrics']
                    stages[name]=dict(positions=p.tolist(),metrics=metrics,overlap=overlap(p))
                record=dict(index=i,member=list(row['member']),n=int(inp[0]),
                    complete_seconds=complete_seconds,draft_seconds=draft_seconds,repair_seconds=stats['seconds'],
                    metrics=score['metrics'],independent_hard_checks=hard,positions=final.tolist(),stages=stages,
                    relation=relation_diagnostics(initial,draft,probability,confidence,reference,inp),
                    fallback=stats['row_retained'],trace=stats['trace'],baseline_final_Q=baseline['metrics']['cost'])
                save(path,record)
            records.append(record)
            if (i+1)%10==0:
                save(OUTPUT/'state.json',dict(status='evaluating',completed=i+1,total=500,elapsed_seconds=time.perf_counter()-started,test_read=False,parameter_updates=0))
                print(f'{i+1}/500',flush=True)
        verify();summary=summarize(records)
        save(OUTPUT/'results.json',dict(status='metrics_complete_gallery_pending',summary=summary,test_read=False,parameter_updates=0))
        print('SUMMARY',json.dumps(summary),flush=True)
        from visualization.layouts import plot_case,geometry_details
        gallery=OUTPUT/'gallery';gallery.mkdir(exist_ok=True);manifest=[]
        for i,row in enumerate(rows[:100]):
            r=records[i];constraints=np.asarray(row['case'].solver_inputs[5]).tolist()
            reference=(row['graph']['reference']*row['graph']['scale']).numpy().tolist()
            variants={key:dict(positions=p,soft_details=geometry_details(p,constraints)) for key,p in
                [('reference',reference),('baseline',r['stages']['baseline']['positions']),('guided',r['stages']['guided']['positions']),('final',r['positions'])]}
            image=gallery/f'{i:03d}.png'
            if not image.exists():plot_case(dict(case_id=f'dev500:{i}',panel='E100 best61 / E98 best58 / identical full repair',constraints=constraints,variants=variants),image,
                stages=(('reference','Reference'),('baseline','E98 best58 draft'),('guided','E100 best61 draft'),('final','E100 full repair')),show_metrics=False)
            manifest.append(dict(index=i,member=list(row['member']),image=image.name))
            if (i+1)%10==0:
                print(f'gallery {i+1}/100',flush=True)
                save(OUTPUT/'state.json',dict(status='rendering',completed=500,gallery_completed=i+1,test_read=False))
        save(gallery/'manifest.json',dict(cases=manifest,score_controls=False,shared_ids_and_scale=True))
        (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E100 dev500</title><style>body{font-family:system-ui;margin:20px;background:#f4f6f8}img{width:100%;background:white}p{margin:24px 0 6px}</style><h1>E100: reference / E98 / E100 / full repair</h1><p>Fixed first100, common module IDs and scale. No scoring controls.</p>'+''.join(f'<p>dev index {r["index"]} · {r["member"]}</p><img loading="lazy" src="{r["image"]}">' for r in manifest),encoding='utf-8')
        save(OUTPUT/'results.json',dict(status='complete',summary=summary,gallery_count=100,elapsed_seconds=time.perf_counter()-started,test_read=False,parameter_updates=0))
        save(OUTPUT/'state.json',dict(status='complete',completed=500,gallery_completed=100,test_read=False))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUTPUT/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False,no_automatic_retry=True))
        raise
