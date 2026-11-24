"""Frozen own-Loss best, parent dev2000, no-MIB scoring, no training restart."""
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
from ..io import ROOT, HERE, sha, save, exclusive, sources
from ..model import GeometryModel
from research.quality_log import summary as quality_line
from ..features import graph_features
from ..batches import decode
from ..ports import public_ports
from ..role_features import build as build_roles
from .solver import optimize
from .repair.canonical_native import warmup
from scoring.research_no_mib import evaluate_case
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from datasets.random_reference import RandomReference, REGISTRY
from datasets.reference_layouts import public_hard_reference
from datasets.cases import _build_case

OWNER=Path(__file__).resolve().parent

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
    # Match the resident public-role layout, with module slots preceding ports.
    ports = public_ports(inputs, scale)
    roles = build_roles(dict(ports, n=n, constraints=constraints))
    m = len(roles['port_features'])
    out['role_features'] = torch.zeros(1, pad, 16)
    out['role_features'][0, :n] = torch.from_numpy(roles['role_features'])
    out['port_features'] = torch.from_numpy(roles['port_features'])[None]
    out['port_valid'] = torch.ones(1, m, dtype=torch.bool)
    out['role_edges'] = torch.zeros(1, pad, pad + m)
    out['role_edges'][0, :n, :n] = ports['wire_b2b'] * 2
    out['role_edges'][0, :n, pad:] = torch.from_numpy(roles['port_edges'])
    out['role_group'] = torch.zeros_like(out['role_edges'])
    out['role_group'][0, :n, :n] = data['group']
    out['role_hops'] = torch.full((1, pad, pad + m), -1, dtype=torch.int16)
    hops = torch.from_numpy(roles['hops'])
    out['role_hops'][0, :n, :n] = hops[:, :n]
    out['role_hops'][0, :n, pad:] = hops[:, n:]
    return out

def repair_inputs(inputs):
    copied=tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(inputs))
    copied[5][:,2]=0
    return copied

class Predictor:
    def __init__(self,checkpoint):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        configuration=state['model_config']
        self.model=GeometryModel(**configuration).cuda().eval()
        self.model.load_state_dict(state['model'],strict=True)

    def predict(self,inputs):
        graph = {key: value.cuda() for key, value in public_graph(inputs).items()}
        states = self.model(graph)
        p=(decode(states[-1],graph)[0,:int(inputs[0])]*graph['scale'][0]).cpu().numpy().astype(float)
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite raw draft')
        self.stage_states = states[[0, 4, 8, 12]]
        self.graph = graph
        return p

    def stage_layouts(self, n):
        decoded = decode(self.stage_states, self.graph)[:, 0, :n] * self.graph['scale'][0]
        return decoded.cpu().numpy().astype(float)

def case_from_item(item):
    inputs=item['solver_inputs'];reference=item['supervision']['reference_xywh']
    ordinary=[torch.as_tensor(v).clone() for v in inputs[1:6]]
    case=_build_case('random_dev:'+str(item['member']),dict(member=item['member'],role='dev_monitor500'),
        ordinary,reference.tolist(),item['supervision']['stored_metrics'])
    if not np.array_equal(case.solver_inputs[6],inputs[6]):raise ValueError('Public specifications changed')
    return case

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('e165_backend',),required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();arm=args.arm
    auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
    if not auth.get('postprocessing_enabled'):raise PermissionError('Terminal evaluation is not enabled')
    plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    run=OWNER
    out=OWNER/'output'
    frozen=json.loads((OWNER/'frozen.json').read_text(encoding='utf-8'))
    checkpoint=ROOT/frozen['checkpoint']
    members=plan['members']
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    source_identity=frozen['sources']
    if sources()!=source_identity:raise ValueError('Frozen evaluation sources changed')
    expected=dict(frozen,source_identity=source_identity,members=members,
        scoring_policy='research.original_no_mib.v1',postprocessor='E98.depth_budget.v1 original caps; copied MIB column zero; original areas/specifications',
        raw_prediction='All-module predictions; no F/P output overwrite',test_read=False,selection_frozen_before_Q=True)
    if (out/'results.json').exists():
        result=json.loads((out/'results.json').read_text(encoding='utf-8'))
        if result['status']!='complete' or result['count']!=2000 or result['checkpoint_sha256']!=frozen['checkpoint_sha256']:
            raise ValueError('Existing evaluation result mismatch')
        print(quality_line(result,out/'results.json'),flush=True)
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
        state=dict(status='running',arm=arm,completed=0,total=2000,pid=os.getpid(),test_read=False)
        def beat(**kw):
            state.update(kw);save(out/'state.json',dict(state,elapsed_seconds=prior+time.perf_counter()-started))
        def paused():
            return requested[0] or any((folder/name).exists()
                for folder in (HERE,OWNER,out) for name in ('PAUSE_REQUEST','STOP','QUEUE_STOP'))
        try:
            beat(stage='loading_fixed_dev2000_identity')
            dataset=RandomReference('dev')
            if [dataset.records[i]['member'] for i in dataset.indices]!=members:raise ValueError('Dev members changed')
            ledger=dataset.folder/'checksums.jsonl'
            if sha(ledger)!=plan['payload_checksums_sha256']:raise ValueError('Payload ledger changed')
            checksums={record['id']:record['payload_sha256'] for line in ledger.read_text(encoding='utf-8').splitlines()
                       for record in json.loads(line)['records']}
            files=sorted((out/'cases').glob('*.json')) if (out/'cases').exists() else []
            if [p.name for p in files]!=[f'{i:04d}.json' for i in range(len(files))] or len(files)>2000:
                raise ValueError('Evaluation commits must form a contiguous case prefix')
            for i,path in enumerate(files):
                row=json.loads(path.read_text(encoding='utf-8'))
                if row['index']!=i or row['member']!=members[i] or row['checkpoint_sha256']!=frozen['checkpoint_sha256']:
                    raise ValueError('Saved case identity mismatch')
                if not row['independent_hard_checks']['is_feasible'] or not row['metrics']['is_feasible']:
                    raise ValueError('Cannot resume past failed hard check')
                records.append(row)
            beat(completed=len(records),stage='initializing_original_postprocessor')
            predictor=None
            from scoring.layout_geometry import measures
            with torch.inference_mode(),threadpool_limits(limits=1):
                for i in range(len(records),2000):
                    if paused():
                        beat(status='paused',stage='accepted_case_boundary',completed=len(records));return
                    record=dataset.records[dataset.indices[i]]
                    if sha(dataset.folder/record['payload'])!=checksums[record['id']]:raise ValueError('Payload changed')
                    item=dataset[i];case=case_from_item(item)
                    if predictor is None:
                        begin_init=time.perf_counter();warmup();predictor=Predictor(checkpoint)
                        predictor.predict(item['solver_inputs'])
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
                    stage_positions=predictor.stage_layouts(int(inputs[0]))
                    row=dict(index=i,member=members[i],n=int(inputs[0]),checkpoint_sha256=frozen['checkpoint_sha256'],
                        metrics=scored['metrics'],raw_metrics=raw['metrics'],raw_geometry=measures(draft),independent_hard_checks=hard,
                        stage_positions=stage_positions.tolist(),stage_labels=['L0','S0','S1','S2'],
                        draft_positions=draft.tolist(),positions=final.tolist(),draft_seconds=draft_seconds,
                        complete_seconds=complete_seconds,repair_seconds=info['seconds'],fallback=info['row_retained'],
                        trace=info['trace'],work=info['work'],test_read=False)
                    save(out/f'cases/{i:04d}.json',row)
                    if not hard['is_feasible'] or not scored['metrics']['is_feasible']:raise RuntimeError(f'Hard legality failure: case{i}')
                    records.append(row);beat(stage='evaluating',completed=len(records))
                    if len(records)%10==0:print(f'{arm} original dev2000 {len(records)}/2000',flush=True)
            costs=np.asarray([r['metrics']['cost'] for r in records]);seconds=np.asarray([r['complete_seconds'] for r in records])
            weighted=official_evaluator().compute_total_score(costs.tolist(),[r['n'] for r in records])
            result=dict(status='complete',arm=arm,count=2000,checkpoint_sha256=frozen['checkpoint_sha256'],
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
            subsets={}
            monitor=set(plan['monitor500_indices'])
            for label,rows in (('monitor500',[r for r in records if r['index'] in monitor]),('extension1500',[r for r in records if r['index'] not in monitor]),('full2000',records)):
                cq=[r['metrics']['cost'] for r in rows]
                wq=official_evaluator().compute_total_score(cq,[r['n'] for r in rows])
                subsets[label]=dict(count=len(rows),weighted_Q=wq,mean_Q=float(np.mean(cq)),mixed_Q=.5*(wq+float(np.mean(cq))),
                    hard_legal=sum(r['independent_hard_checks']['is_feasible'] for r in rows),
                    grouping_satisfied=sum(r['metrics']['grouping_violations']==0 for r in rows),
                    boundary_satisfied=sum(r['metrics']['boundary_violations']==0 for r in rows),
                    all_soft_satisfied=sum(r['metrics']['total_soft_violations']==0 for r in rows))
            save(out/'subsets.json',subsets)
            result.update(grouping_satisfied=subsets['full2000']['grouping_satisfied'],boundary_satisfied=subsets['full2000']['boundary_satisfied'],
                all_soft_satisfied=subsets['full2000']['all_soft_satisfied'])
            save(out/'results.json',result)
            line=quality_line(result,(out/'results.json').relative_to(ROOT).as_posix())
            logpath=OWNER/'background/quality.log';logpath.parent.mkdir(exist_ok=True)
            with logpath.open('a',encoding='utf-8') as log:log.write(line+'\n')
            print(line,flush=True)
            (out/'report.md').write_text('# E165 frozen joint best 原固定dev2000\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf-8')
            beat(status='complete',stage='complete',completed=2000)
        except BaseException as error:
            save(out/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=len(records)))
            beat(status='error',stage='stopped_no_retry',completed=len(records));raise

if __name__=='__main__':main()

