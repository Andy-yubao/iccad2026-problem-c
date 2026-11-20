"""One frozen FM Loss-best: Euler8 dev500, original depth-budget postprocessor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.E133_flow_matching.control import ROOT,OWNER,read,sha,sources,stopped,budget_exhausted,elapsed_pipeline
from experiments.E133_flow_matching.io import save,exclusive
from experiments.E133_flow_matching.model import build_model
from experiments.E133_flow_matching.features import graph_features
from experiments.E133_flow_matching.batches import decode
from experiments.E133_flow_matching.flow import integrate,member_seed,fixed_bank
from experiments.E133_flow_matching.data import records,load_rows
from experiments.E133_flow_matching.resident import ResidentRows
from experiments.E133_flow_matching.addresses import attach_addresses
from experiments.E133_flow_matching.evaluation.solver import optimize
from experiments.E133_flow_matching.evaluation.repair.canonical_native import warmup
from scoring.research_no_mib import evaluate_case
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from datasets.random_reference import RandomReference,REGISTRY
from datasets.reference_layouts import public_hard_reference
from datasets.cases import _build_case

EVALUATION_ROOT=Path(__file__).resolve().parent
EVALUATION_STARTED=time.perf_counter()
MAX_EVALUATION_SECONDS=1800

def public_graph(inputs,pad=120):
    g=graph_features(inputs);n=int(inputs[0]);scale=g['scale'];spec=g['specs']
    constraints=torch.as_tensor(inputs[5]).long().clone();constraints[:,2]=0
    known=torch.cat(((spec[:,:2]+spec[:,2:]/2)/scale,
        (spec[:,2].clamp_min(1e-8)/spec[:,3].clamp_min(1e-8)).log()[:,None]),1)
    known=torch.cat((known[:,:2]*g['preplaced'][:,None],known[:,2:]*g['locked'][:,None]),1)
    data=dict(features=g['features'],adjacency=g['adjacency'],group=g['relations'][0],
        mask=g['mask'],area=g['areas']/scale.square(),known=known,
        specs=torch.where(g['locked'][:,None],spec/scale,torch.zeros_like(spec)),constraints=constraints)
    out={}
    for key,value in data.items():
        shape=(pad,pad) if key in ('adjacency','group') else (pad,*value.shape[1:])
        out[key]=torch.zeros((1,*shape),dtype=value.dtype)
        if key in ('adjacency','group'):out[key][0,:n,:n]=value
        else:out[key][0,:n]=value
    out['valid']=torch.arange(pad)[None,:]<n
    out['scale']=scale[None];out['address']=torch.arange(pad)[None]
    return out

def repair_inputs(inputs):
    copied=tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(inputs))
    copied[5][:,2]=0
    return copied

class Predictor:
    def __init__(self,checkpoint,inputs):
        state=checkpoint if isinstance(checkpoint,dict) else torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=build_model().cuda().eval();self.model.load_state_dict(state['model'],strict=True)
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        self.noise=torch.zeros(1,120,3,device='cuda')
        def prediction():
            self.state=integrate(self.model,self.g,self.noise,8)
            return decode(self.state,self.g)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):prediction()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=prediction()
        torch.cuda.synchronize()
    def predict(self,inputs,member):
        for key,value in public_graph(inputs).items():self.g[key].copy_(value)
        n=int(inputs[0]);rng=torch.Generator().manual_seed(member_seed(member,'inference'))
        noise=torch.zeros(1,120,3);noise[0,:n]=torch.randn(n,3,generator=rng)
        self.noise.copy_(noise);self.graph.replay()
        output=(self.output[0,:n]*self.g['scale'][0]).cpu().numpy().astype(float)
        if not np.isfinite(output).all():raise RuntimeError('Nonfinite Euler8 draft')
        return output

def case_from_item(item):
    inputs=item['solver_inputs'];reference=item['supervision']['reference_xywh']
    ordinary=[torch.as_tensor(v).clone() for v in inputs[1:6]]
    case=_build_case('random_dev:'+str(item['member']),dict(member=item['member'],role='dev_monitor500'),
        ordinary,reference.tolist(),item['supervision']['stored_metrics'])
    np.testing.assert_array_equal(case.solver_inputs[6],inputs[6])
    return case

def check():
    if stopped() or (EVALUATION_ROOT/'STOP').exists():raise InterruptedError('User STOP')
    if (time.perf_counter()-EVALUATION_STARTED>=MAX_EVALUATION_SECONDS):raise TimeoutError('New user-requested evaluation1800s limit; original pipeline budget unchanged')

@torch.no_grad()
def sensitivity(model,out):
    check();ids=read(OWNER/'plan.json')['preflight_train_indices'];rows=load_rows('train',ids,halted=lambda:stopped() or (time.perf_counter()-EVALUATION_STARTED>=MAX_EVALUATION_SECONDS))
    if rows is None:check();raise RuntimeError('Training sensitivity cache incomplete')
    batch=ResidentRows(rows,'cuda').batch(range(5));attach_addresses(batch,rows,range(5))
    z0,_=fixed_bank(rows,120,domain='inference');results={}
    for steps in (4,8,16):
        check();torch.cuda.synchronize();start=time.perf_counter()
        state,trace=integrate(model,batch,z0,steps,trace=True);torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        error=((model.standardize(state)-model.standardize(batch['target'])).square()*batch['valid'][...,None]).sum((1,2))/(3*batch['valid'].sum(1))
        results[str(steps)]=dict(batch_seconds=seconds,standardized_endpoint_MSE=error.cpu().tolist(),
            states=[s.cpu().tolist() for s in trace],positions=decode(state,batch).cpu().tolist())
    save(out/'five_train_step_sensitivity.json',dict(members=[list(r['member']) for r in rows],steps=results,
        same_initial_noise=True,no_Q=True,not_used_for_selection=True,test_read=False))

def main():
    argparse.ArgumentParser().parse_args()
    plan=read(OWNER/'plan.json');run=OWNER/'runs/fm_seed0';cp=run/'checkpoints/best.pt';out=EVALUATION_ROOT/'dev500'
    frozen=read(OWNER/'evaluation/frozen_model.json')
    if out.exists():raise FileExistsError('Evaluation exists; no automatic retry')
    assert sha(cp)==frozen['checkpoint_sha256'] and sources()==frozen['sources']
    assert sha(REGISTRY)==plan['parent_registry_sha256']
    expected=[r['member'] for r in records('dev_monitor500')]
    with exclusive(out/'evaluation.lock'):
        save(out/'frozen.json',dict(frozen,members=expected,sampling='Euler8 single member-seeded noise',
            postprocessor='E98.depth_budget.v1',scoring='research.original_no_mib.v1',test_read=False))
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import numba
        numba.set_num_threads(1)
        started=time.perf_counter();state=dict(status='running',completed=0,total=500,pid=os.getpid(),test_read=False)
        def beat(**kw):
            state.update(kw);save(out/'state.json',dict(**state,elapsed_seconds=time.perf_counter()-started,pipeline_elapsed_seconds=elapsed_pipeline()))
        try:
            check();beat(stage='reading existing dev500 NPZ scoring objects; not generating data or rebuilding training cache')
            dataset=RandomReference('dev_monitor500')
            assert [dataset.records[i]['member'] for i in dataset.indices]==expected
            assert sha(dataset.folder/'checksums.jsonl')==plan['payload_checksums_sha256']
            checksums={r['id']:r['payload_sha256'] for line in (dataset.folder/'checksums.jsonl').read_text().splitlines() for r in json.loads(line)['records']}
            items=[]
            for i in range(500):
                check();record=dataset.records[dataset.indices[i]]
                assert sha(dataset.folder/record['payload'])==checksums[record['id']]
                items.append(dataset[i])
            cases=[case_from_item(item) for item in items]
            init=time.perf_counter()
            with torch.inference_mode(),threadpool_limits(limits=1):
                check();warmup();predictor=Predictor(cp,items[0]['solver_inputs'])
            save(out/'initialization.json',dict(seconds=time.perf_counter()-init,torch_threads=1,numba_threads=1))
            records_out=[]
            from visualization.layer_comparison import measures
            from scoring.draft_geometry import evaluate_geometry,repair_movement
            with torch.inference_mode(),threadpool_limits(limits=1):
                for i,(item,case) in enumerate(zip(items,cases)):
                    check();torch.cuda.synchronize();begin=time.perf_counter()
                    inputs=tuple(int(v) if k==0 else np.asarray(v).copy() for k,v in enumerate(item['solver_inputs']))
                    draft=predictor.predict(inputs,expected[i]);draft_seconds=time.perf_counter()-begin
                    adjusted=repair_inputs(inputs)
                    assert np.array_equal(adjusted[1],inputs[1]) and np.array_equal(adjusted[6],inputs[6])
                    final,info=optimize(draft,adjusted)
                    hard=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
                    complete_seconds=time.perf_counter()-begin
                    scored=evaluate_case(case,final,complete_seconds);raw=evaluate_case(case,draft,draft_seconds)
                    n=int(inputs[0]);raw_state=predictor.state[0,:n].cpu().numpy()
                    ratio=np.clip(raw_state[:,2],-5,5);area=inputs[1]/inputs[1].sum()
                    wh=np.stack((np.sqrt(area*np.exp(ratio)),np.sqrt(area/np.exp(ratio))),-1)
                    ref=item['supervision']['reference_xywh']
                    ref=np.asarray(ref,float);ref_state=np.column_stack(((ref[:,:2]+ref[:,2:]/2)/np.sqrt(inputs[1].sum()),np.log(ref[:,2]/ref[:,3])))
                    error=(raw_state-ref_state)/predictor.model.state_std.cpu().numpy()
                    geometry,groups,boundaries=evaluate_geometry(draft,inputs,ref,details=False)
                    geometry.update(repair_movement(draft,final,inputs[1]))
                    record=dict(draft_quality=geometry,group_quality=groups,boundary_quality=boundaries,index=i,member=expected[i],n=n,checkpoint_sha256=frozen['checkpoint_sha256'],
                        metrics=scored['metrics'],raw_metrics=raw['metrics'],raw_geometry=measures(draft),independent_hard_checks=hard,
                        draft_state=raw_state.tolist(),draft_positions=draft.tolist(),positions=final.tolist(),
                        standardized_endpoint_MSE=float(np.mean(error**2)),ratio_clipped=int((np.abs(raw_state[:,2])>5).sum()),
                        negative_raw_xy=int(((raw_state[:,:2]-wh/2)<0).any(-1).sum()),
                        draft_seconds=draft_seconds,complete_seconds=complete_seconds,repair_seconds=info['seconds'],
                        fallback=info['row_retained'],trace=info['trace'],work=info['work'],test_read=False)
                    save(out/f'cases/{i:04d}.json',record);records_out.append(record);beat(stage='evaluating',completed=i+1)
                    if not hard['is_feasible'] or not scored['metrics']['is_feasible']:raise RuntimeError(f'Hard legality failed at{i}')
                    if (i+1)%50==0:print(f'FM Euler8 dev500 {i+1}/500',flush=True)
            costs=np.array([r['metrics']['cost'] for r in records_out]);seconds=np.array([r['complete_seconds'] for r in records_out])
            weighted=official_evaluator().compute_total_score(costs.tolist(),[r['n'] for r in records_out])
            result=dict(status='complete',count=500,checkpoint_sha256=frozen['checkpoint_sha256'],best_epoch=frozen['best_epoch'],
                parameters=frozen['parameters'],own_selection_loss=frozen['own_selection_loss'],
                weighted_Q=weighted,mean_Q=float(costs.mean()),mixed_Q=.5*(weighted+float(costs.mean())),
                hard_legal=sum(r['metrics']['is_feasible'] for r in records_out),raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records_out),
                fallbacks=sum(r['fallback'] for r in records_out),mean_complete_seconds=float(seconds.mean()),
                p95_complete_seconds=float(np.quantile(seconds,.95)),max_complete_seconds=float(seconds.max()),over250ms=int((seconds>.25).sum()),
                p95_Q=float(np.quantile(costs,.95)),max_Q=float(costs.max()),
                mean_endpoint_MSE=float(np.mean([r['standardized_endpoint_MSE'] for r in records_out])),
                raw_repeated_coverage=float(np.mean([r['raw_geometry']['repeated_coverage_fraction'] for r in records_out])),
                raw_bbox_over_area=float(np.mean([r['raw_geometry']['bbox_over_area'] for r in records_out])),
                ratio_clipped=sum(r['ratio_clipped'] for r in records_out),negative_raw_xy=sum(r['negative_raw_xy'] for r in records_out),
                elapsed_seconds=time.perf_counter()-started,pipeline_elapsed_seconds=elapsed_pipeline(),test_read=False,
                scoring_policy='original areas/no MIB local research Q; not official contest score',selection_frozen_before_Q=True)
            save(out/'results.json',result);beat(stage='dev500_complete',completed=500)
            (out/'report.md').write_text('# E133 primary dev500 completed; train5 sensitivity follows\n',encoding='utf-8')
            sensitivity(predictor.model,out)
            (out/'report.md').write_text('# E133 FM Euler8固定dev500\n\n唯一FM Loss-best在Q前冻结。单样本8步，不按Q选权重或步数。\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf-8')
            beat(status='complete',stage='complete',completed=500);print(json.dumps(result),flush=True)
        except BaseException as exc:
            save(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=state['completed']))
            beat(status='user_stop' if isinstance(exc,InterruptedError) else 'budget' if isinstance(exc,TimeoutError) else 'error',stage='stopped_no_retry');raise
if __name__=='__main__':main()
