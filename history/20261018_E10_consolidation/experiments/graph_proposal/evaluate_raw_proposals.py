"""Frozen model raw proposals on predeclared train100/dev100; no postprocessing."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from datasets.paths import ROOT,resolved_data_paths
from datasets.worksets import workset_manifest,LayoutReader
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from .model import GraphProposal
from .data import graph_features,encode_target,decode_layout
from .packing import prepare_problem
from .pipeline import heuristic
from .proposal_metrics import evaluate_proposal,definition_checks
from .train_supervised import components
from .train import save_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(rows):
    result={}
    for panel in ('train','dev'):
        selected=[r for r in rows if r['panel']==panel]
        if not selected:continue
        result[panel]={}
        for name in ('reference','heuristic','deterministic'):
            variants=[r['variants'][name] for r in selected]
            keys=sorted(set(k for v in variants for k in v['diagnostics']))
            record={'case_count':len(selected),'feasible_count':sum(v['official_metrics']['is_feasible'] for v in variants),
                    'aggregation':'each layout equal weight; per-layout P95 values are not pooled P95',
                    'diagnostics':{},'official':{}}
            for k in keys:
                vals=[v['diagnostics'][k] for v in variants if v['diagnostics'].get(k) is not None]
                record['diagnostics'][k]={'mean':float(np.mean(vals)) if vals else None,
                    'median':float(np.median(vals)) if vals else None,
                    'p10':float(np.quantile(vals,.1)) if vals else None,
                    'p90':float(np.quantile(vals,.9)) if vals else None,'valid_layouts':len(vals)}
            for k in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations','violations_relative','cost'):
                record['official'][k+'_mean']=float(np.mean([v['official_metrics'][k] for v in variants]))
            for threshold in (.01,.05,.1,.25,.5):
                key=f'translation_lb_exceeds_{threshold:g}'
                record[key]=sum(v['diagnostics']['translation_locked_pair_failure_count']>0 or
                    v['diagnostics']['translation_radius_lb']>threshold for v in variants)
            record['workers']=len(set(r['source']['shard'].split('/')[0] for r in selected))
            record['block_count_mean']=float(np.mean([r['input_summary']['block_count'] for r in selected]))
            if name=='deterministic':
                record['supervised']={k:float(np.mean([v['supervised'][k] for v in variants])) for k in ('mse','center_mse','log_ratio_mse')}
            result[panel][name]=record
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--wall-seconds',type=float,default=600)
    args=p.parse_args()
    if not 60<=args.wall_seconds<=900:p.error('Short evaluation wall budget must be60..900s')
    run=args.output.resolve()
    if not run.is_relative_to((ROOT/'experiments/graph_proposal/runs').resolve()):p.error('Output must be experiment runs')
    run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    started=time.monotonic();deadline=started+args.wall_seconds
    checks=definition_checks()
    manifest=workset_manifest();train=LayoutReader(manifest,'train_monitor');dev=LayoutReader(manifest,'dev_quality')
    train_indices=[]
    for w in range(100):
        candidates=[i for i,(shard,_) in enumerate(train.members) if shard.startswith(f'worker_{w}/')]
        train_indices.append(min(candidates,key=lambda i: hashlib.sha256(
            f'proposal_a_train100_v1:{train.members[i][0]}:{train.members[i][1]}'.encode()).hexdigest()))
    panels={'train':(train,train_indices),'dev':(dev,list(range(len(dev))))}
    sources=sorted(set([*Path(__file__).parent.glob('*.py'),*(ROOT/'datasets').glob('*.py'),
        *(ROOT/'scoring').glob('*.py'),ROOT/'protocol/workset_v2.json',ROOT/'protocol/split_v1.json',
        ROOT/'protocol/paths.json',ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']))
    checkpoint_hash=sha(args.checkpoint)
    config={'command':[os.sys.executable,*os.sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_hash,
        'workset_sha256':manifest['manifest_sha256'],'data_paths':resolved_data_paths(),
        'panels':{name:[reader.members[i] for i in indices] for name,(reader,indices) in panels.items()},
        'train_selection':"one perworker from train_monitor; minimum SHA256('proposal_a_train100_v1:<shard>:<row>')",
        'dev_selection':'unchanged dev_quality100, one perworker',
        'variants':['reference','heuristic','deterministic'],'seed':0,'cpu_threads':1,'device':args.device,
        'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'torch':torch.__version__,
        'wall_seconds':args.wall_seconds,'test_payload_reads':0,'definition_checks':checks,
        'inference':'standard GraphProposal forward and decode_layout, no correct/pack/search/fallback',
        'geometry_storage':'cases.jsonl only; results contains aggregate statistics',
        'interpretation':'RawQ and rawwire/area are diagnostics, not final feasible solver performance. Reference treated nearoptimal.',
        'translation_bound':'Fixed decoded shapes; Euclidean perblock translation <=tau*sqrt(effectivearea), preplaced locked. Max overlapping pair separation/(movable local scales sum). Necessary lowerbound only; excludes shape changes; officialtol1e-6.',
        'normalization':'Layoutscale sqrt(sum effectivearea), local sqrt(effectivearea), pairlocal mean two local scales. Equal layouts, not pooled pairs.',
        'reference_tolerance':1e-6,'figure_selection':'fixeddev workers0,24,49,74,99; plus min/median/max translationLB, ties bycase_id',
        'source_sha256':{s.relative_to(ROOT).as_posix():sha(s) for s in sources},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    save_json(run/'config.json',config)
    with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT))
    model=GraphProposal.from_checkpoint(args.checkpoint,device=args.device)
    if args.device=='cuda':
        with torch.no_grad():model({'features':torch.zeros(3,17,device='cuda'),
            'adjacency':torch.zeros(3,3,device='cuda'),'relations':[torch.zeros(3,3,device='cuda') for _ in range(2)]})
        torch.cuda.synchronize()
    rows=[];case_times=[];stop='completed'
    with (run/'cases.jsonl').open('x',encoding='utf-8',newline='\n') as stream,torch.inference_mode():
        for panel,(reader,indices) in panels.items():
            for index in indices:
                if time.monotonic()>=deadline-15:stop='resource_stopped';break
                tick=time.monotonic();raw=reader.case(index)
                n,a,b,pb,pins,c,_=raw.solver_inputs
                case=_build_case(raw.case_id,raw.source,(a,b,pb,pins,c),raw.reference_positions,raw.stored_metrics)
                graph=graph_features(case.solver_inputs,args.device)
                if args.device=='cuda':torch.cuda.synchronize()
                t=time.monotonic();prediction=model(graph);layout=decode_layout(prediction,graph)
                if args.device=='cuda':torch.cuda.synchronize()
                infer_seconds=time.monotonic()-t
                target=encode_target(case.reference_positions,graph)
                supervised=dict(zip(('mse','center_mse','log_ratio_mse'),[float(v) for v in components(prediction,target,graph['mask'])]))
                pos=layout.cpu().numpy().astype(float);problem=prepare_problem(*case.solver_inputs)
                locked=problem.fixed|problem.preplaced;effective=problem.areas.copy()
                effective[locked]=np.prod(problem.specs[locked,2:],axis=1)
                variants={}
                for name,positions in [('reference',case.reference_positions),('heuristic',heuristic(problem)),('deterministic',pos)]:
                    scored=evaluate_case(case,positions,infer_seconds if name=='deterministic' else 0)
                    diag=evaluate_proposal(scored['positions'],case.reference_positions,effective,c.numpy(),b.numpy(),pb.numpy())
                    if name=='deterministic':
                        prior_xy=prediction[:,:2]*graph['scale']-layout[:,2:]/2
                        diag['pre_decode_negative_mobile_fraction']=float((prior_xy[~graph['preplaced']]<0).any(1).float().mean()) if (~graph['preplaced']).any() else 0.
                        free=~graph['locked'];diag['ratio_clamped_free_fraction']=float((prediction[free,2].abs()>5).float().mean()) if free.any() else 0.
                    variants[name]={'positions':scored['positions'],'official_metrics':scored['metrics'],
                        'independent_hard_checks':scored['independent_hard_checks'],'diagnostics':diag}
                    if name=='deterministic':variants[name]['supervised']=supervised
                row={'panel':panel,'case_id':case.case_id,'source':case.source,'input_summary':case.input_summary(),
                    'constraints':c.int().tolist(),'effective_areas':effective.tolist(),'variants':variants,
                    'timings':{'network_and_decode_seconds':infer_seconds,'case_wall_seconds':time.monotonic()-tick}}
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(row)
                case_times.append(time.monotonic()-tick)
                if len(rows)==5:
                    estimate=float(np.median(case_times[1:]))*195*1.5
                    if time.monotonic()+estimate>=deadline-15:
                        stop='preflight_projected_budget_exceeded';break
                if len(rows)%10==0 or len(rows)==5:
                    progress={'processed':len(rows),'panel':panel,'requested':200,'wall_seconds':time.monotonic()-started,
                        'last_case_seconds':case_times[-1],
                        'last_model_overlap_pairs':variants['deterministic']['diagnostics']['overlap_pair_count']}
                    print(json.dumps(progress),flush=True);save_json(run/'progress.json',progress)
            if stop!='completed':break
    result={'status':stop,'complete':len(rows)==200,'requested_cases':200,'completed_cases':len(rows),
        'summary':summaries(rows),'wall_seconds':time.monotonic()-started,'test_payload_reads':0,
        'checkpoint_unchanged':sha(args.checkpoint)==checkpoint_hash,'definition_checks':checks,
        'postprocessing_calls':0,'training_updates':0,'official_contest_score':None}
    for panel in ('train','dev'):
        selected=[row for row in rows if row['panel']==panel]
        result.setdefault('size_groups',{})[panel]={}
        for label,low,high in [('small',0,49),('medium',50,89),('large',90,100000)]:
            group=[r for r in selected if low<=r['input_summary']['block_count']<=high]
            if group:result['size_groups'][panel][label]=summaries(group).get(panel)
    assert result['checkpoint_unchanged']
    save_json(run/'results.json',result)
    print(json.dumps({'status':stop,'completed_cases':len(rows),'wall_seconds':result['wall_seconds'],
          'summary':{panel:{k:result['summary'][panel]['deterministic']['diagnostics'][k]['mean'] for k in
                    ('center_error_local_mean','pair_relative_local_mean','pair_overlap_area_fraction','translation_radius_lb')}
                     for panel in result['summary']}},indent=2),flush=True)


if __name__=='__main__':main()
