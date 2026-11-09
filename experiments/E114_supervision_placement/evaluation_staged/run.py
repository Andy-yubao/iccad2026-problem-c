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
ARMS=('staged',)

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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=ARMS,required=True)
    args=parser.parse_args();arm=args.arm
    plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    run=HERE/'runs'/f'{arm}_seed0';checkpoint=run/'checkpoints/best.pt';out=run/'evaluation_no_mib_dev500'
    frozen=plan['models'][arm]
    if (out/'results.json').exists():
        previous=json.loads((out/'results.json').read_text(encoding='utf-8'))
        assert previous['checkpoint_sha256']==frozen['checkpoint_sha256']
        print('Completed identical evaluation reused',flush=True);return
    if out.exists():raise FileExistsError('Partial evaluation: inspect error; no automatic retry')
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    assert sha(REGISTRY)==plan['registry_sha256']
    source_identity={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OWNER.rglob('*.py'))}
    source_identity[(OWNER/'budget.json').relative_to(ROOT).as_posix()]=sha(OWNER/'budget.json')
    with exclusive(out/'evaluation.lock'):
        save(out/'frozen.json',dict(**frozen,source_identity=source_identity,
            members=plan['members'],scoring_policy='research.original_no_mib.v1',
            postprocessor='E98.depth_budget.v1 with copied MIB column zero; original areas/specs retained',
            raw_prediction='No fixed/preplaced output overwrite',test_read=False,selection_frozen_before_Q=True))
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import numba
        numba.set_num_threads(1)
        started=time.perf_counter();requested=[False]
        signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        state=dict(status='running',arm=arm,completed=0,total=500,pid=os.getpid(),test_read=False)
        def beat(**kw):
            state.update(kw);save(out/'state.json',dict(**state,elapsed_seconds=time.perf_counter()-started))
        def check():
            if requested[0] or (OWNER/'STOP').exists() or (out/'STOP').exists():raise InterruptedError('Evaluation STOP')
        try:
            beat(stage='loading_fixed_dev500')
            dataset=RandomReference('dev_monitor500')
            assert [dataset.records[i]['member'] for i in dataset.indices]==plan['members']
            ledger=dataset.folder/'checksums.jsonl'
            assert sha(ledger)==plan['payload_checksums_sha256']
            checksums={record['id']:record['payload_sha256'] for line in ledger.read_text(encoding='utf-8').splitlines()
                       for record in json.loads(line)['records']}
            items=[]
            for i in range(len(dataset)):
                check();record=dataset.records[dataset.indices[i]]
                assert sha(dataset.folder/record['payload'])==checksums[record['id']]
                items.append(dataset[i])
            cases=[case_from_item(item) for item in items]
            init=time.perf_counter()
            with torch.inference_mode(),threadpool_limits(limits=1):
                warmup();predictor=Predictor(checkpoint,items[0]['solver_inputs'])
            save(out/'initialization.json',dict(seconds=time.perf_counter()-init,torch_threads=1,numba_threads=1))
            records=[]
            from visualization.layer_comparison import measures
            with torch.inference_mode(),threadpool_limits(limits=1):
                for i,(item,case) in enumerate(zip(items,cases)):
                    check();torch.cuda.synchronize();begin=time.perf_counter()
                    inputs=tuple(int(v) if k==0 else np.asarray(v).copy() for k,v in enumerate(item['solver_inputs']))
                    draft=predictor.predict(inputs);draft_seconds=time.perf_counter()-begin
                    adjusted=repair_inputs(inputs)
                    assert np.array_equal(adjusted[1],inputs[1]) and np.array_equal(adjusted[6],inputs[6])
                    final,info=optimize(draft,adjusted)
                    hard=independent_hard_checks(final.tolist(),inputs[1],inputs[5],
                        public_hard_reference(inputs[1],inputs[5],inputs[6]))
                    complete_seconds=time.perf_counter()-begin
                    scored=evaluate_case(case,final,complete_seconds)
                    raw=evaluate_case(case,draft,draft_seconds)
                    record=dict(index=i,member=plan['members'][i],n=int(inputs[0]),
                        checkpoint_sha256=frozen['checkpoint_sha256'],metrics=scored['metrics'],raw_metrics=raw['metrics'],
                        raw_geometry=measures(draft),independent_hard_checks=hard,
                        draft_positions=draft.tolist(),positions=final.tolist(),draft_seconds=draft_seconds,
                        complete_seconds=complete_seconds,repair_seconds=info['seconds'],fallback=info['row_retained'],
                        trace=info['trace'],work=info['work'],test_read=False)
                    save(out/f'cases/{i:04d}.json',record);records.append(record)
                    if not hard['is_feasible'] or not scored['metrics']['is_feasible']:
                        raise RuntimeError(f'Hard legality failed at case{i}; evidence preserved')
                    if (i+1)%10==0:beat(stage='evaluating',completed=i+1)
                    if (i+1)%50==0:print(f'{arm} dev500 {i+1}/500',flush=True)
            costs=np.array([r['metrics']['cost'] for r in records]);seconds=np.array([r['complete_seconds'] for r in records])
            weighted=official_evaluator().compute_total_score(costs.tolist(),[r['n'] for r in records])
            result=dict(status='complete',arm=arm,count=500,checkpoint_sha256=frozen['checkpoint_sha256'],
                best_epoch=frozen['best_epoch'],parameters=612307,
                weighted_Q=weighted,mean_Q=float(costs.mean()),mixed_Q=.5*(weighted+float(costs.mean())),
                hard_legal=sum(r['metrics']['is_feasible'] for r in records),raw_hard_legal=sum(r['raw_metrics']['is_feasible'] for r in records),
                fallbacks=sum(r['fallback'] for r in records),mean_complete_seconds=float(seconds.mean()),
                p95_complete_seconds=float(np.percentile(seconds,95)),max_complete_seconds=float(seconds.max()),
                over250ms=int((seconds>.25).sum()),max_Q=float(costs.max()),p95_Q=float(np.percentile(costs,95)),
                raw_repeated_coverage=float(np.mean([r['raw_geometry']['repeated_coverage_fraction'] for r in records])),
                raw_bbox_over_area=float(np.mean([r['raw_geometry']['bbox_over_area'] for r in records])),
                common_final_dev_loss=frozen['common_final_dev_loss'],own_selection_loss=frozen['own_selection_loss'],
                scoring_policy='local research Q without MIB, original areas/specs; not official contest score',
                training_seconds=frozen['training_seconds'],training_epochs=frozen['training_epochs'],
                elapsed_seconds=time.perf_counter()-started,test_read=False,selection_frozen_before_Q=True)
            assert sha(checkpoint)==frozen['checkpoint_sha256']
            save(out/'results.json',result)
            (out/'report.md').write_text('# E114 '+arm+'：固定dev500验证\n\n唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf-8')
            beat(status='complete',stage='complete',completed=500)
            print(json.dumps(result),flush=True)
        except BaseException as error:
            save(out/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=state['completed']))
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'error',stage='stopped_no_retry');raise

if __name__=='__main__':main()
