"""Frozen own-Loss best, parent dev500, no-MIB scoring, no training restart."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..io import ROOT, HERE, sha, save, exclusive
from ..model import GeometryModel
from ..features import graph_features
from ..batches import decode
from .solver import optimize
from .repair.canonical_native import warmup
from scoring.research_no_mib import evaluate_case
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from datasets.random_reference import RandomReference, REGISTRY
from datasets.reference_layouts import public_hard_reference
from datasets.cases import _build_case

OWNER=Path(__file__).resolve().parent
from ..loss_config import ARMS

def public_graph(inputs,pad=120):
    """Exactly E114 conditioning/decoding, without constructing any target tensor."""
    g=graph_features(inputs)
    n=int(inputs[0]);scale=g['scale'];spec=g['specs']
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
    out['scale']=scale[None]
    return out

def repair_inputs(inputs):
    copied=tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(inputs))
    copied[5][:,2]=0
    return copied

class Predictor:
    def __init__(self,checkpoint,inputs):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=GeometryModel(**state['model_config']).cuda().eval()
        self.model.load_state_dict(state['model'],strict=True)
        self.g={k:v.cuda() for k,v in public_graph(inputs).items()}
        def prediction():
            self.states=self.model(self.g)
            return decode(self.states[-1],self.g)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):prediction()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=prediction()
        torch.cuda.synchronize()

    def predict(self,inputs):
        for key,value in public_graph(inputs).items():self.g[key].copy_(value)
        self.graph.replay()
        p=(self.output[0,:int(inputs[0])]*self.g['scale'][0]).cpu().numpy().astype(float)
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite raw draft')
        # All-module draft stays untouched: specifications are enforced in optimize only.
        return p

def case_from_item(item):
    inputs=item['solver_inputs'];reference=item['supervision']['reference_xywh']
    ordinary=[torch.as_tensor(v).clone() for v in inputs[1:6]]
    case=_build_case('random_dev:'+str(item['member']),dict(member=item['member'],role='dev_monitor500'),
        ordinary,reference.tolist(),item['supervision']['stored_metrics'])
    np.testing.assert_array_equal(case.solver_inputs[6],inputs[6])
    return case

def main():
    from ..io import sources
    from ..run_queue import verify
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();arm=args.arm
    verify()
    auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
    if not auth.get('training_enabled') or arm not in auth.get('allowed_arms',[]):raise PermissionError('Preparation only')
    plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    run=HERE/'runs'/f'{arm}_seed0';checkpoint=run/'checkpoints/best.pt'
    out=run/'evaluation_no_mib_dev500'
    frozen=json.loads((OWNER/f'frozen_{arm}.json').read_text(encoding='utf-8'))
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    source_identity=sources()
    expected=dict(frozen,source_identity=source_identity,members=plan['members'],
        scoring_policy='research.original_no_mib.v1',postprocessor='E98.depth_budget.v1 original caps; copied MIB column zero; original areas/specifications',
        raw_prediction='All-module predictions; no F/P output overwrite',test_read=False,selection_frozen_before_Q=True)
    if (out/'results.json').exists():
        result=json.loads((out/'results.json').read_text(encoding='utf-8'))
        if result['status']!='complete' or result['count']!=500 or result['checkpoint_sha256']!=frozen['checkpoint_sha256']:
            raise ValueError('Existing evaluation result mismatch')
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    if out.exists() and not args.resume:raise FileExistsError('Partial evaluation requires explicit --resume')
    if args.resume and not out.exists():raise FileNotFoundError('No evaluation to resume')
    prior=0.
    if out.exists():
        if json.loads((out/'frozen.json').read_text(encoding='utf-8'))!=expected:raise ValueError('Evaluation source/member identity changed')
        previous=json.loads((out/'state.json').read_text(encoding='utf-8'))
        if previous['status'] not in ('paused','complete'):raise ValueError('Errors/crashes are not automatically resumed')
        prior=previous.get('elapsed_seconds',0.)
    with exclusive(out/'evaluation.lock'):
        if not (out/'frozen.json').exists():save(out/'frozen.json',expected)
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import numba
        numba.set_num_threads(1)
        started=time.perf_counter();requested=[False]
        signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,lambda *_:requested.__setitem__(0,True))
        records=[]
        state=dict(status='running',arm=arm,completed=0,total=500,pid=os.getpid(),test_read=False)
        def beat(**kw):
            state.update(kw);save(out/'state.json',dict(state,elapsed_seconds=prior+time.perf_counter()-started))
        def paused():
            return requested[0] or any(p.exists() for p in (HERE/'PAUSE_REQUEST',HERE/'STOP',HERE/'QUEUE_STOP',run/'STOP',OWNER/'STOP',out/'STOP'))
        try:
            beat(stage='loading_fixed_dev500_identity')
            dataset=RandomReference('dev_monitor500')
            if [dataset.records[i]['member'] for i in dataset.indices]!=plan['members']:raise ValueError('Dev members changed')
            ledger=dataset.folder/'checksums.jsonl'
            if sha(ledger)!=plan['payload_checksums_sha256']:raise ValueError('Payload ledger changed')
            checksums={record['id']:record['payload_sha256'] for line in ledger.read_text(encoding='utf-8').splitlines()
                       for record in json.loads(line)['records']}
            files=sorted((out/'cases').glob('*.json')) if (out/'cases').exists() else []
            if [p.name for p in files]!=[f'{i:04d}.json' for i in range(len(files))] or len(files)>500:
                raise ValueError('Evaluation commits must form a contiguous case prefix')
            for i,path in enumerate(files):
                row=json.loads(path.read_text(encoding='utf-8'))
                if row['index']!=i or row['member']!=plan['members'][i] or row['checkpoint_sha256']!=frozen['checkpoint_sha256']:
                    raise ValueError('Saved case identity mismatch')
                if not row['independent_hard_checks']['is_feasible'] or not row['metrics']['is_feasible']:
                    raise ValueError('Cannot resume past failed hard check')
                records.append(row)
            beat(completed=len(records),stage='initializing_original_postprocessor')
            predictor=None
            from scoring.layout_geometry import measures
            with torch.inference_mode(),threadpool_limits(limits=1):
                for i in range(len(records),500):
                    if paused():
                        beat(status='paused',stage='accepted_case_boundary',completed=len(records));return
                    record=dataset.records[dataset.indices[i]]
                    if sha(dataset.folder/record['payload'])!=checksums[record['id']]:raise ValueError('Payload changed')
                    item=dataset[i];case=case_from_item(item)
                    if predictor is None:
                        begin_init=time.perf_counter();warmup();predictor=Predictor(checkpoint,item['solver_inputs'])
                        session=out/'initialization_sessions';session.mkdir(exist_ok=True)
                        save(session/f'{os.getpid()}_{time.time_ns()}.json',dict(seconds=time.perf_counter()-begin_init,
                            resumed_from=len(records),excluded_from_per_case_time=True))
                    if paused():
                        beat(status='paused',stage='accepted_case_boundary',completed=len(records));return
                    torch.cuda.synchronize();begin=time.perf_counter()
                    inputs=tuple(int(v) if k==0 else np.asarray(v).copy() for k,v in enumerate(item['solver_inputs']))
                    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin
                    adjusted=repair_inputs(inputs)
                    if not np.array_equal(adjusted[1],inputs[1]) or not np.array_equal(adjusted[6],inputs[6]):raise ValueError('Area/specification changed')
                    final,info=optimize(draft,adjusted)
                    hard=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
                    complete_seconds=time.perf_counter()-begin
                    scored=evaluate_case(case,final,complete_seconds);raw=evaluate_case(case,draft,draft_seconds)
                    row=dict(index=i,member=plan['members'][i],n=int(inputs[0]),checkpoint_sha256=frozen['checkpoint_sha256'],
                        metrics=scored['metrics'],raw_metrics=raw['metrics'],raw_geometry=measures(draft),independent_hard_checks=hard,
                        draft_positions=draft.tolist(),positions=final.tolist(),draft_seconds=draft_seconds,
                        complete_seconds=complete_seconds,repair_seconds=info['seconds'],fallback=info['row_retained'],
                        trace=info['trace'],work=info['work'],test_read=False)
                    save(out/f'cases/{i:04d}.json',row)
                    if not hard['is_feasible'] or not scored['metrics']['is_feasible']:raise RuntimeError(f'Hard legality failure: case{i}')
                    records.append(row);beat(stage='evaluating',completed=len(records))
                    if len(records)%10==0:print(f'{arm} original dev500 {len(records)}/500',flush=True)
            costs=np.asarray([r['metrics']['cost'] for r in records]);seconds=np.asarray([r['complete_seconds'] for r in records])
            weighted=official_evaluator().compute_total_score(costs.tolist(),[r['n'] for r in records])
            result=dict(status='complete',arm=arm,count=500,checkpoint_sha256=frozen['checkpoint_sha256'],
                best_epoch=frozen['best_epoch'],parameters=frozen['parameters'],weighted_Q=weighted,mean_Q=float(costs.mean()),
                mixed_Q=.5*(weighted+float(costs.mean())),hard_legal=sum(r['metrics']['is_feasible'] for r in records),
                raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),fallbacks=sum(r['fallback'] for r in records),
                mean_complete_seconds=float(seconds.mean()),p95_complete_seconds=float(np.percentile(seconds,95)),
                max_complete_seconds=float(seconds.max()),over250ms=int((seconds>.25).sum()),max_Q=float(costs.max()),
                p95_Q=float(np.percentile(costs,95)),raw_repeated_coverage=float(np.mean([r['raw_geometry']['repeated_coverage_fraction'] for r in records])),
                raw_bbox_over_area=float(np.mean([r['raw_geometry']['bbox_over_area'] for r in records])),
                common_final_dev_loss=frozen['common_final_dev_loss'],common_final_new_dev_loss=frozen['own_new_final_dev_loss'],
                own_selection_loss=frozen['own_selection_loss'],training_seconds=frozen['training_seconds'],training_epochs=frozen['training_epochs'],
                scoring_policy='Original areas/no MIB local research Q; neutral runtime1; not official contest score',
                elapsed_seconds=prior+time.perf_counter()-started,test_read=False,selection_frozen_before_Q=True)
            if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Checkpoint changed during evaluation')
            save(out/'results.json',result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            (out/'report.md').write_text('# E135 '+arm+' 原固定dev500\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf-8')
            beat(status='complete',stage='complete',completed=500)
        except BaseException as error:
            save(out/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=len(records)))
            beat(status='error',stage='stopped_no_retry',completed=len(records));raise

if __name__=='__main__':main()

