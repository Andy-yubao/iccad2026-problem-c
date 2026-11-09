"""Frozen own-Loss best, fixed original dev500, unchanged E98 depth repair."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json,time,signal,threading,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..io import ROOT,HERE,sha,save,exclusive
from ..model import GeometryModel
from ..features import graph_features
from ..batches import decode
from ..data import load_rows
from ..dataset_view import ReferenceLayouts
from .solver import optimize
from .repair.canonical_native import warmup
from scoring.evaluation import evaluate_case
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from datasets.reference_layouts import public_hard_reference,mean_area_inputs

OWNER=Path(__file__).resolve().parent

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
    def __init__(self,checkpoint,inputs):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=GeometryModel(**state['model_config']).cuda().eval()
        self.model.load_state_dict(state['model'],strict=True)
        if hasattr(self.model,'active_depth'):
            self.model.active_depth=state.get('active_depth',12)
            if self.model.active_depth!=12:raise ValueError('Only completed depth12 model may enter final dev500')
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        def prediction():
            states=self.model(self.g)
            return decode(states[-1] if states.ndim==4 else states,self.g)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):prediction()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream): self.output=prediction()
        torch.cuda.synchronize()

    def predict(self,inputs):
        for key,value in public_graph(inputs).items():self.g[key].copy_(value)
        self.graph.replay()
        p=(self.output[0,:int(inputs[0])]*self.g['scale'][0]).cpu().numpy().astype(float)
        c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1);pre=c[:,1]!=0
        p[locked,2:]=s[locked,2:];p[pre]=s[pre]
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite draft')
        return p


def main():
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    run=HERE/('runs/20261107_depth12_seed0' if plan['experiment']=='E111_progressive_depth' else 'runs/'+plan['run'])
    checkpoint=run/'checkpoints/best.pt';out=run/'evaluation_depth_dev500'
    if (out/'results.json').exists():
        print('Completed identical evaluation reused:',out,flush=True);return
    if out.exists():raise FileExistsError('Partial/error evaluation requires targeted review, no auto-retry')
    training=json.loads((run/'results.json').read_text())
    if training['status'] not in ('early_stop','budget','max_epochs') or training['error']:
        raise RuntimeError('Training did not complete normally')
    digest=sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    members=[r['member'] for r in ReferenceLayouts('dev_monitor500').records]
    source_identity={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OWNER.rglob('*.py'))}
    source_identity[(OWNER/'budget.json').relative_to(ROOT).as_posix()]=sha(OWNER/'budget.json')
    frozen=dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest,
        best_epoch=saved['progress']['best_epoch'],selection='registered own-dev-Loss best; frozen before Q',
        members=members,evaluator='E98.depth_budget.v1',historical_control='E98 generalization best71 existing fixed-depth dev500; result reused, no control rerun',
        source_identity=source_identity,test_read=False,
        metric_scope='Historical original MIB/mean-area task retained for this controlled study; not new no-MIB scoring',
        time_scope='Includes cached public input copy/idempotent mean-area handling, features, transfer, network, all repair/fallback and independent hard checks; archive IO, init, scoring and plots separate. Historical control lacks identical new input-copy accounting; timing comparison is descriptive.')
    with exclusive(out/'evaluation.lock'):
        save(out/'frozen.json',frozen)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import numba
        numba.set_num_threads(1)
        started=time.perf_counter();closed=threading.Event();requested=[False]
        old_signal=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        state=dict(status='running',completed=0,total=500,pid=os.getpid(),test_read=False)
        def check():
            if requested[0] or (HERE/'STOP').exists() or (out/'STOP').exists():raise InterruptedError('User STOP')
        def beat(**kw):
            state.update(kw);save(out/'state.json',dict(**state,elapsed_seconds=time.perf_counter()-started))
        def heartbeat():
            while not closed.wait(10):beat()
        worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
        try:
            beat(stage='loading_fixed_dev500')
            rows=load_rows('dev_monitor500')
            assert [list(r['member']) for r in rows]==members
            init=time.perf_counter()
            with torch.inference_mode(),threadpool_limits(limits=1):
                warmup();predictor=Predictor(checkpoint,rows[0]['case'].solver_inputs)
            save(out/'initialization.json',dict(seconds=time.perf_counter()-init,torch_threads=1,numba_threads=1))
            records=[]
            with torch.inference_mode(),threadpool_limits(limits=1):
                for i,row in enumerate(rows):
                    check();torch.cuda.synchronize();begin=time.perf_counter()
                    inputs=mean_area_inputs(row['case'].solver_inputs)
                    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin
                    final,info=optimize(draft,inputs)
                    hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),
                        public_hard_reference(inputs[1],inputs[5],inputs[6]))
                    complete_seconds=time.perf_counter()-begin
                    scored=evaluate_case(row['case'],final,complete_seconds)
                    raw=evaluate_case(row['case'],draft,draft_seconds)
                    record=dict(index=i,member=members[i],n=int(inputs[0]),checkpoint_sha256=digest,
                        metrics=scored['metrics'],raw_metrics=raw['metrics'],independent_hard_checks=hard,
                        draft_positions=draft.tolist(),positions=final.tolist(),draft_seconds=draft_seconds,
                        complete_seconds=complete_seconds,repair_seconds=info['seconds'],fallback=info['row_retained'],
                        trace=info['trace'],work=info['work'],test_read=False)
                    save(out/f'cases/{i:04d}.json',record);records.append(record)
                    if not hard['is_feasible'] or not scored['metrics']['is_feasible']:
                        raise RuntimeError(f'Final hard-check failure at case{i}; record retained')
                    if (i+1)%10==0:beat(stage='evaluating',completed=i+1)
                    if (i+1)%50==0:print(f'dev500 {i+1}/500',flush=True)
            costs=[r['metrics']['cost'] for r in records];seconds=np.array([r['complete_seconds'] for r in records])
            weighted=official_evaluator().compute_total_score(costs,[r['n'] for r in records])
            result=dict(status='complete',count=500,checkpoint_sha256=digest,best_epoch=frozen['best_epoch'],
                weighted_Q=weighted,mean_Q=float(np.mean(costs)),mixed_Q=.5*(weighted+float(np.mean(costs))),
                hard_legal=sum(r['metrics']['is_feasible'] for r in records),raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),
                fallbacks=sum(r['fallback'] for r in records),mean_complete_seconds=float(seconds.mean()),
                p95_complete_seconds=float(np.percentile(seconds,95)),max_complete_seconds=float(seconds.max()),
                over250ms=int((seconds>.25).sum()),max_Q=float(max(costs)),p95_Q=float(np.percentile(costs,95)),
                training_selection_loss=saved['progress']['best'],test_read=False,selection_frozen_before_Q=True)
            beat(stage='rendering_same100',completed=500)
            from visualization.layouts import plot_case,geometry_details
            gallery=out/'gallery';gallery.mkdir()
            for i,row in enumerate(rows[:100]):
                check();record=records[i];c=np.asarray(row['case'].solver_inputs[5]).tolist()
                reference=(row['graph']['reference']*row['graph']['scale']).numpy().tolist()
                variants={key:dict(positions=p,soft_details=geometry_details(p,c)) for key,p in
                          [('reference',reference),('raw',record['draft_positions']),('final',record['positions'])]}
                plot_case(dict(case_id=f'dev500:{i}',panel=plan['experiment'],constraints=c,variants=variants),
                          gallery/f'{i:03d}.png',stages=(('reference','Reference'),('raw','L'),('final','Depth repair')),show_metrics=False)
            (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>img{width:100%}</style>'+''.join(
                f'<p>{i}</p><img loading="lazy" src="{i:03d}.png">' for i in range(100)),encoding='utf-8')
            assert sha(checkpoint)==digest
            result['elapsed_seconds']=time.perf_counter()-started
            save(out/'results.json',result)
            (out/'report.md').write_text('# '+plan['experiment']+'：固定dev500终末验证\n\n'
                +'唯一own-Loss best在Q评价前冻结。历史MIB及固定规格路径保留；测试未读，未按Q选权重。\n\n'
                +'```json\n'+json.dumps(result,indent=2)+'\n```\n\n[同100例reference/L/最终图](gallery/index.html)\n',encoding='utf-8')
            beat(status='complete',stage='complete',completed=500)
            print(json.dumps(result),flush=True)
        except BaseException as error:
            save(out/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=state['completed']))
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'error',stage='stopped_no_retry');raise
        finally:
            closed.set();worker.join(timeout=11);signal.signal(signal.SIGINT,old_signal)

if __name__=='__main__':main()
