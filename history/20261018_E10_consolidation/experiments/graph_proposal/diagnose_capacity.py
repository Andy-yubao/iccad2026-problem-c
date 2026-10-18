"""Bounded train-only capacity and oracle-stage diagnosis, never a deployable oracle."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from datasets.paths import ROOT, resolved_data_paths
from datasets.worksets import LayoutReader, workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case, summarize_variants
from .data import graph_features, encode_target, decode_layout
from .model import GraphProposal
from .pipeline import Optimizer, correct
from .packing import prepare_problem, pack, default_order
from .feasible import is_legal, row_control
from .train_supervised import components
from .train import save_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_draft(inputs, raw, seconds=15):
    """Copy of the frozen one-candidate pipeline after proposal generation.

    Only train labels are permitted for explicitly labelled oracle diagnostics.
    No change is made to the production Optimizer.
    """
    start = time.monotonic(); deadline = start + seconds
    p = prepare_problem(*inputs)
    revised = correct(np.asarray(raw, dtype=float), p)
    centers = revised[:, :2] + revised[:, 2:] / 2
    order = sorted(default_order(p).tolist(), key=lambda i:
                   (round(centers[i, 1]/max(p.scale*.15, 1e-9)), centers[i, 0], i))
    final = pack(p, order=order, ratios=revised[:, 2]/revised[:, 3], target_width=p.scale,
                 shape_trials=1, proposal_centers=centers, deadline=deadline-.1)
    fallback = not is_legal(final, p)
    if fallback: final = row_control(p)
    return revised, final, fallback, time.monotonic()-start


@torch.no_grad()
def prediction_metrics(model, samples):
    model.eval()
    rows = [[float(v) for v in components(model(g), target, g['mask'])] for g,target in samples]
    return dict(zip(('mse','center_mse','log_ratio_mse'), np.mean(rows,axis=0).tolist()))


def twins_bound(graph, target):
    """Exact transposition symmetries of represented graph; rigorous MSE floor."""
    x=graph['features'].cpu(); matrices=[graph['adjacency'].cpu(),*[v.cpu() for v in graph['relations']]]
    target=target.cpu(); mask=graph['mask'].cpu(); n=len(x); parent=list(range(n))
    def find(i):
        while parent[i]!=i: i=parent[i]
        return i
    pairs=[]
    for i in range(n):
        for j in range(i):
            if not torch.equal(x[i],x[j]): continue
            perm=torch.arange(n);perm[i]=j;perm[j]=i
            if all(torch.equal(a,a[perm][:,perm]) for a in matrices):
                pairs.append([j,i]);parent[find(i)]=find(j)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(i)
    groups=[g for g in groups.values() if len(g)>1]
    floor=0.
    for group in groups:
        for col in range(3):
            ids=[i for i in group if mask[i,col]]
            if ids:
                vals=target[ids,col].double()
                floor+=float(((vals-vals.mean())**2).sum())
    return {'exact_twin_groups':groups,'swappable_pairs':pairs,
            'proven_masked_mse_lower_bound':floor/max(float(mask.sum()),1),
            'note':'Only exact represented-graph transposition symmetries; zero bound does not prove sufficient capacity.'}


def fit_probe(width,layers,samples,run,name,args,global_deadline):
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    model=GraphProposal('deterministic',width,layers).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,
                 patience=2,threshold=.002,threshold_mode='rel',min_lr=1e-5)
    started=time.monotonic();deadline=min(started+args.probe_seconds,global_deadline-30)
    curve=[];best=float('inf');best_state=None;best_step=0;bad=0;anchor=float('inf')
    step=0;stop='step_cap'
    while True:
        if step%100==0 or step==args.steps:
            m=prediction_metrics(model,samples)
            scheduler.step(m['mse'])
            row={'step':step,**m,'lr':optimizer.param_groups[0]['lr'],'seconds':time.monotonic()-started}
            curve.append(row)
            with (run/'probe_log.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'probe':name,**row})+'\n')
            print(json.dumps({'phase':'fit_probe','probe':name,**row}),flush=True)
            if m['mse']<best:
                best=m['mse'];best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if m['mse']<anchor*.998: anchor=m['mse'];bad=0
            else: bad+=1
            if m['mse']<1e-5: stop='memory_target';break
            if step>=500 and bad>=8: stop='train_memory_plateau';break
        if step>=args.steps: break
        if time.monotonic()>=deadline: stop='time_cap';break
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=sum(components(model(g),target,g['mask'])[0] for g,target in samples)/len(samples)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();step+=1
    # Select only among pre-recorded train-memory checks, never by Q.
    model.load_state_dict(best_state)
    model=model.cpu().eval()
    torch.save({'model_config':model.config,'model':model.state_dict(),
                'diagnostic_only':True,'selection':'five training cases MSE, not Q'},run/(name+'.pt'))
    return model,{'config':model.config,'parameters':sum(p.numel() for p in model.parameters()),
                  'best_step':best_step,'completed_steps':step,'best_mse':best,'curve':curve,
                  'stop_reason':stop,'seconds':time.monotonic()-started,
                  'note':'Fresh initialization, fit these5train cases only; no generalization or fulltrain capacity claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--probe-seconds',type=float,default=150)
    args=p.parse_args()
    if not 100<=args.steps<=2000 or not 10<=args.probe_seconds<=180: p.error('Bounded short probes only')
    run=args.output.resolve()
    if not run.is_relative_to((ROOT/'experiments/graph_proposal/runs').resolve()): p.error('Output must be experiment runs')
    run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for bounded memory probes')
    started=time.monotonic();deadline=started+600
    manifest=workset_manifest();reader=LayoutReader(manifest,'train')
    # Before reading data or scores: deterministic workers spread over the workset.
    workers=[0,24,49,74,99]
    indices=[next(i for i,(shard,_) in enumerate(reader.members) if shard.startswith(f'worker_{w}/')) for w in workers]
    sources=sorted(set([*Path(__file__).parent.glob('*.py'),*(ROOT/'datasets').glob('*.py'),
        *(ROOT/'scoring').glob('*.py'),ROOT/'protocol/workset_v2.json',ROOT/'protocol/split_v1.json',
        ROOT/'protocol/paths.json',ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']))
    before=digest(args.checkpoint)
    config={'command':[str(Path(os.sys.executable)),*os.sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':before,
            'workset_sha256':manifest['manifest_sha256'],'data_paths':resolved_data_paths(),
            'selection':'firsttrain member in numeric workers0,24,49,74,99, chosen before payloads/scores',
            'members':[reader.members[i] for i in indices],'test_payload_reads':0,
            'budget':{'global_wall_seconds':600,'probe_seconds_each':args.probe_seconds,'steps_each':args.steps,
                      'cases':5,'geometry_seconds_per_variant':15},
            'probes':{'small':{'width':64,'layers':3},'large':{'width':128,'layers':6}},
            'probe_controls':'fresh seed0 each; all5cases eachupdate; AdamW.001/decay.0001/clip1; trainMSE check100; plateauLR.5 patience2 min1e-5; stoptrainMSE<1e-5 or8badchecks after500steps',
            'oracle_note':'Train labels deliberately used for labelled diagnostic variants only; no deployable input-only performance claim.',
            'quality_note':'Official neutralruntime Q, runtime=median_runtime=1; not formal contest score.',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'inference_device':'cpu','threads':1,
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{s.relative_to(ROOT).as_posix():digest(s) for s in sources}}
    save_json(run/'config.json',config)
    with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT))
    cases=[];samples=[];rows=[]
    current=GraphProposal.from_checkpoint(args.checkpoint)
    os.environ['GRAPH_PROPOSAL_CHECKPOINT']=str(args.checkpoint.resolve())
    result={'status':'running','test_payload_reads':0,'cases':rows,'probes':{}}
    with torch.no_grad():
        for index in indices:
            raw=reader.case(index);n,a,b,pb,pins,c,_=raw.solver_inputs
            case=_build_case(raw.case_id,raw.source,(a,b,pb,pins,c),raw.reference_positions,raw.stored_metrics)
            cases.append(case);graph=graph_features(case.solver_inputs)
            target=encode_target(case.reference_positions,graph)
            samples.append((graph,target))
            row={'case_id':case.case_id,'source':case.source,'input_summary':case.input_summary(),'variants':{},
                 'symmetry':twins_bound(graph,target)}
            def record(name,positions,seconds=0,**extra):
                row['variants'][name]={**evaluate_case(case,positions,seconds),'solve_seconds':seconds,**extra}
            record('reference',case.reference_positions,oracle=True)
            record('target_decode',decode_layout(target,graph),oracle=True)
            revised,final,fallback,t=repair_draft(case.solver_inputs,case.reference_positions)
            record('reference_corrected',revised,oracle=True)
            record('reference_repacked',final,t,oracle=True,fallback=fallback)
            for name in ('heuristic','deterministic'):
                opt=Optimizer(name,15,0);t=time.monotonic();pos=opt.solve(*case.solver_inputs);dt=time.monotonic()-t
                d=opt.last_stats['candidate_diagnostics'][0]
                record(name+'_raw',d['raw'])
                record(name+'_corrected',d['corrected'])
                record(name+'_final',pos,dt,fallback=opt.last_stats['fallback'])
                if name=='deterministic':
                    row['prediction_metrics']=dict(zip(('mse','center_mse','log_ratio_mse'),
                        [float(x) for x in components(current(graph),target,graph['mask'])]))
                    again,rep,fb,_=repair_draft(case.solver_inputs,d['raw'])
                    assert np.allclose(rep,pos,rtol=0,atol=1e-10),'Copied postprocessing differs from production'
                    row['copied_pipeline_matches_production']=True
            row['reference_reaches_proven_Q_lower_bound']=abs(row['variants']['reference']['metrics']['cost']-1)<1e-8
            rows.append(row)
            save_json(run/'results.json',result)
            print(json.dumps({'phase':'stages','case_id':case.case_id,'n':n,
                              'Q':{k:v['metrics']['cost'] for k,v in row['variants'].items()}}),flush=True)
    gpu_samples=[({k:([v.cuda() for v in val] if isinstance(val,list) else val.cuda())
                    for k,val in g.items()},t.cuda()) for g,t in samples]
    models={}
    for name,width,layers in [('small',64,3),('large',128,6)]:
        models[name],result['probes'][name]=fit_probe(width,layers,gpu_samples,run,name,args,deadline)
        save_json(run/'results.json',result)
    with torch.no_grad():
        for i,(case,row,(g,target)) in enumerate(zip(cases,rows,samples)):
            for name,model in models.items():
                positions=model.propose(case.solver_inputs)[0]
                revised,final,fallback,t=repair_draft(case.solver_inputs,positions.numpy())
                row[name+'_prediction_metrics']=dict(zip(('mse','center_mse','log_ratio_mse'),
                    [float(x) for x in components(model(g),target,g['mask'])]))
                for stage,pos in [('raw',positions),('corrected',revised),('final',final)]:
                    row['variants'][name+'_'+stage]={**evaluate_case(case,pos,t if stage=='final' else 0),
                        'solve_seconds':t if stage=='final' else 0,'diagnostic_trainfit':True,'fallback':fallback if stage=='final' else False}
    result['summary']=summarize_variants(rows,list(rows[0]['variants']))
    result['current_train5_mse']=prediction_metrics(current,samples)
    result['parameter_count']=sum(p.numel() for p in current.parameters())
    result['status']='completed';result['wall_seconds']=time.monotonic()-started
    result['original_checkpoint_unchanged']=digest(args.checkpoint)==before
    assert result['original_checkpoint_unchanged']
    save_json(run/'results.json',result)
    print(json.dumps({'phase':'completed','wall_seconds':result['wall_seconds'],
       'parameter_count':result['parameter_count'],'current_train5_mse':result['current_train5_mse'],
       'probe_best':{k:{key:v[key] for key in ('best_mse','best_step','completed_steps','stop_reason','parameters')}
                     for k,v in result['probes'].items()},
       'Q':{k:v['neutral_runtime_weighted_cost_all_cases'] for k,v in result['summary'].items()}},indent=2),flush=True)


if __name__=='__main__':main()
