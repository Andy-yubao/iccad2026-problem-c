"""Resumable paired dev evaluation of a geometry checkpoint with frozen E8."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from datasets.paths import ROOT
from datasets.worksets import LayoutReader, workset_manifest
from .batches import decode
from .inference import prepare_inference
from .model import GeometryModel
from .probe import build_case, worker_call, evaluate_output, summaries, E8
from .records import save, append, read_rows, sha, provenance, snapshot, run_path, OLD_CHECKPOINT, OLD_CHECKPOINT_SHA


def model_digest(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(path,device):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('fixture_only'):raise ValueError('Synthetic preflight checkpoints are not research models')
    model=GeometryModel(**state['model_config']).to(device)
    model.load_state_dict(state['model'])
    return model.eval()


def panel_reader(panel):
    manifest=workset_manifest()
    reader=LayoutReader(manifest,'dev_quality' if panel=='dev100' else 'dev')
    name='20261018_robust_dev100_seed0' if panel=='dev100' else '20261018_robust_confirmation100_seed0'
    baseline=E8/name
    saved=read_rows(baseline/'cases.jsonl')
    lookup={pair:i for i,pair in enumerate(reader.members)}
    indices=[lookup[(row['source']['shard'],row['source']['sample_index'])] for row in saved]
    assert len(indices)==100 and len(set(indices))==100
    return manifest,reader,indices,baseline,{row['case_id']:row for row in saved}


def frozen_backend_hashes():
    inherited=json.loads((Path(__file__).parent/'inherited_sources.json').read_text(encoding='utf-8'))
    hashes={}
    for item in inherited['files']:
        if '/backend/' in item['destination']:
            if sha(ROOT/item['destination'])!=item['sha256']:
                raise ValueError('E9 first round requires the unmodified E8 backend: '+item['destination'])
            hashes[item['destination']]=item['sha256']
    return hashes


def result_summary(rows):
    base=summaries(rows,['model'])['model'] if rows else {}
    if not rows:return base
    q=np.array([r['variants']['model']['final']['metrics']['cost'] for r in rows])
    delta=q-np.array([r['baseline_metrics']['cost'] for r in rows])
    times=np.array([r['inference_seconds']+r['variants']['model']['solve_seconds'] for r in rows])
    base.update(Q_p95=float(np.quantile(q,.95)),Q_median=float(np.median(q)),
                worst10percent_mean=float(np.sort(q)[-max(1,int(np.ceil(.1*len(q)))):].mean()),
                end_to_end_seconds_mean=float(times.mean()),end_to_end_seconds_p95=float(np.quantile(times,.95)),
                end_to_end_seconds_max=float(times.max()),
                mean_difference_vs_E8=float(delta.mean()),wins=int((delta < -1e-8).sum()),
                ties=int((np.abs(delta)<=1e-8).sum()),losses=int((delta > 1e-8).sum()),
                rescue_count=sum(bool(r['variants']['model'].get('stats',{}).get('rescue_used')) for r in rows))
    return base


def evaluate_quality(model,output,panel,device,identity,budget_seconds,stop=lambda:False):
    """No gradient or label-based solver selection; labels only enter scoring."""
    begin=time.monotonic();output=run_path(output)
    manifest,reader,indices,baseline,cached=panel_reader(panel)
    digest=model_digest(model)
    spec={'panel':panel,'members':[reader.members[i] for i in indices],'model_digest':digest,
          'identity':identity,'workset_sha256':manifest['manifest_sha256'],
          'baseline':str(baseline),'baseline_cases_sha256':sha(baseline/'cases.jsonl'),
          'backend_sha256':frozen_backend_hashes(),'solver_seconds':5.,'test_payload_reads':0,
          'source_sha256':provenance()['source_sha256']}
    if output.exists():
        old=json.loads((output/'config.json').read_text(encoding='utf-8'))
        # JSON turns tuples into lists; compare canonical serializations.
        for key in spec:
            if json.dumps(spec[key],sort_keys=True)!=json.dumps(old.get(key),sort_keys=True):
                raise ValueError('Evaluation resume mismatch: '+key)
    else:
        output.mkdir(parents=True);save(output/'config.json',{**provenance(),**spec,
            'timing':'feature preparation + forward + decoding + public spec restoration + postprocessing; file exchange separately',
            'quality_scope':'development only; official neutral runtime, all failures retained at Q=10'})
    rows=read_rows(output/'cases.jsonl') if (output/'cases.jsonl').exists() else []
    done={r['case_id'] for r in rows};prior=0.
    if (output/'results.json').exists():prior=json.loads((output/'results.json').read_text())['cumulative_seconds']
    invocation={'phase':'start','identity':identity,'budget_seconds_this_invocation':budget_seconds}
    append(output/'invocations.jsonl',invocation)
    def persist(status):
        value={'status':status,'completed_cases':len(rows),'expected_cases':len(indices),
               'cumulative_seconds':prior+time.monotonic()-begin,'test_payload_reads':0,'summary':result_summary(rows)}
        save(output/'results.json',value);return value
    if len(rows)==len(indices):return persist('completed')
    model.eval();warm=False;status='completed'
    with torch.no_grad():
        for index in indices:
            pair=reader.members[index];case_id=f'dev:{pair[0]}:{pair[1]}'
            if case_id in done:continue
            if stop() or (output/'STOP').exists():status='user_stop';break
            if time.monotonic()-begin+17>budget_seconds:status='resource_budget';break
            raw=reader.case(index);case=build_case(raw)
            if not warm:
                model(prepare_inference(raw.solver_inputs,device));warm=True
                if device.startswith('cuda'):torch.cuda.synchronize()
            if device.startswith('cuda'):torch.cuda.synchronize()
            tick=time.perf_counter();public=prepare_inference(raw.solver_inputs,device)
            draft=(decode(model(public),public)[0]*public['scale'][0]).cpu().numpy().astype(float)
            c=np.asarray(raw.solver_inputs[5]);specs=np.asarray(raw.solver_inputs[6]);locked=c[:,:2].any(1)
            draft[locked,2:]=specs[locked,2:];draft[c[:,1]!=0]=specs[c[:,1]!=0]
            if device.startswith('cuda'):torch.cuda.synchronize()
            inference=time.perf_counter()-tick
            result=evaluate_output(case,worker_call(output,raw.solver_inputs,draft))
            # Score using complete measured model+postprocessing time. Q itself
            # remains neutral-runtime in the shared official evaluator.
            result['final']['metrics']['runtime_seconds']=inference+result['solve_seconds']
            row={'case_id':case.case_id,'source':case.source,'block_count':case.block_count,
                 'draft':draft.tolist(),'inference_seconds':inference,'variants':{'model':result},
                 'baseline_metrics':cached[case.case_id]['variants']['conic_robust']['final']['metrics'],
                 'baseline_reused_from':{'run':str(baseline),'variant':'conic_robust','case_id':case.case_id}}
            append(output/'cases.jsonl',row);rows.append(row);done.add(case_id)
            persist('running')
            print(json.dumps({'phase':'quality','panel':panel,'completed':len(rows),'Q':result['final']['metrics']['cost']}),flush=True)
    value=persist('completed' if len(rows)==len(indices) else status)
    append(output/'invocations.jsonl',{'phase':'end','status':value['status'],'wall_seconds':time.monotonic()-begin})
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--checkpoint',type=Path)
    choice.add_argument('--selection',type=Path,help='Read the checkpoint path from a completed training selection.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panel',choices=['dev100','confirm100'],default='dev100')
    parser.add_argument('--device',default='cuda');parser.add_argument('--budget-seconds',type=float,default=1800.)
    args=parser.parse_args()
    if args.selection:args.checkpoint=Path(json.loads(args.selection.read_text(encoding='utf-8'))['checkpoint'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.budget_seconds<=0:parser.error('Positive budget required')
    out=run_path(args.output)
    previous=0.
    if (out/'results.json').exists():previous=json.loads((out/'results.json').read_text())['cumulative_seconds']
    stopped=[False]
    def handler(*unused):stopped[0]=True
    signal.signal(signal.SIGINT,handler);signal.signal(signal.SIGTERM,handler)
    if previous>=args.budget_seconds:raise ValueError('Cumulative evaluation budget exhausted')
    model=load_model(args.checkpoint,args.device)
    identity={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint)}
    if (out/'config.json').exists():
        saved_identity=json.loads((out/'config.json').read_text())['identity']
        if saved_identity.get('checkpoint')!=identity['checkpoint'] or saved_identity.get('checkpoint_sha256')!=identity['checkpoint_sha256']:
            raise ValueError('Resume checkpoint changed')
        identity=saved_identity
    result=evaluate_quality(model,out,args.panel,args.device,identity,args.budget_seconds-previous,lambda:stopped[0])
    if not (out/'source_snapshot.zip').exists():snapshot(out,json.loads((out/'config.json').read_text())['source_sha256'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
