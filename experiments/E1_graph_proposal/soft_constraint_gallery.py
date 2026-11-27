"""Thirty frozen-case illustrations and bounded dev-only soft-weight diagnosis."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union
from datasets.paths import ROOT,resolved_data_paths
from datasets.worksets import workset_manifest,LayoutReader
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case,summarize_variants
from scoring.official import official_evaluator
from .model import GraphProposal
from .data import graph_features,encode_target
from .train_supervised import components
from .pipeline import correct
from .packing import prepare_problem,default_order,pack
from .feasible import is_legal,row_control
from .train import save_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonl(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def soft_details(positions,constraints,metrics):
    p=np.asarray(positions,dtype=float);c=np.asarray(constraints,dtype=int)
    lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0)
    dist=np.column_stack((p[:,0]-lo[0],hi[0]-p[:,0]-p[:,2],hi[1]-p[:,1]-p[:,3],p[:,1]-lo[1]))
    boundary=[]
    for i in np.flatnonzero(c[:,4]):
        bits=[bit for bit in (1,2,4,8) if c[i,4]&bit]
        failed=[bit for bit in bits if abs(dist[i,{1:0,2:1,4:2,8:3}[bit]])>=1e-6]
        boundary.append({'block':int(i),'code':int(c[i,4]),'failed_bits':failed})
    groups={};mibs={}
    for g in np.unique(c[:,3]):
        if g<=0:continue
        polygons=[box(x,y,x+w,y+h) for x,y,w,h in p[c[:,3]==g]]
        union=unary_union(polygons)
        groups[str(g)]=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
    for g in np.unique(c[:,2]):
        if g<=0:continue
        mibs[str(g)]=len({(round(float(w),4),round(float(h),4)) for w,h in p[c[:,2]==g,2:]})
    assert sum(bool(b['failed_bits']) for b in boundary)==metrics['boundary_violations']
    assert sum(n-1 for n in groups.values())==metrics['grouping_violations']
    assert sum(n-1 for n in mibs.values())==metrics['mib_violations']
    return {'boundary':boundary,'group_components':groups,'mib_distinct_shapes':mibs}


def postprocess(problem,raw,boundary_weight=2.,group_weight=2.,seconds=15):
    start=time.monotonic();deadline=start+seconds
    revised=correct(np.asarray(raw,dtype=float),problem)
    centers=revised[:,:2]+revised[:,2:]/2
    order=sorted(default_order(problem).tolist(),key=lambda i:
        (round(centers[i,1]/max(problem.scale*.15,1e-9)),centers[i,0],i))
    final=pack(problem,order=order,ratios=revised[:,2]/revised[:,3],target_width=problem.scale,
        shape_trials=1,proposal_centers=centers,deadline=deadline-.1,
        boundary_weight=boundary_weight,group_weight=group_weight)
    fallback=not is_legal(final,problem)
    if fallback:final=row_control(problem)
    return revised,final,fallback,time.monotonic()-start


def score_decomposition(rows):
    official=official_evaluator();out={};names=('actual','remove_boundary','remove_grouping','remove_mib','remove_all_soft')
    costs={name:[] for name in names};counts=[];logparts=[];capped=0;max_error=0.
    for row in rows:
        m=row['variants']['deterministic']['metrics'];nsoft=max(m['max_possible_violations'],1)
        vb,vg,vm=[m[k] for k in ('boundary_violations','grouping_violations','mib_violations')]
        counts.append(row['input_summary']['block_count']);logparts.append([vb/nsoft,vg/nsoft,vm/nsoft])
        total=vb+vg+vm
        for name,soft in [('actual',total),('remove_boundary',vg+vm),('remove_grouping',vb+vm),('remove_mib',vb+vg),('remove_all_soft',0)]:
            value=official.compute_cost(m['hpwl_gap'],m['area_gap'],soft/nsoft,1.,m['is_feasible'])
            costs[name].append(value)
            if name=='actual':max_error=max(max_error,abs(value-m['cost']))
        capped+=m['cost']>=9.999998 and m['is_feasible']
    weights=np.exp((np.asarray(counts)-max(counts))/12)
    actual=official.compute_total_score(costs['actual'],counts)
    for name in names:
        total=official.compute_total_score(costs[name],counts)
        out[name]={'weighted_Q':total,'relative_reduction':1-total/actual,'mean_Q':float(np.mean(costs[name]))}
    out.update({'case_count':len(rows),'max_recomputed_actual_cost_error':max_error,'capped_cases':int(capped),
        'weighted_normalized_contributions':dict(zip(('boundary','grouping','mib'),(np.asarray(logparts).T@weights/weights.sum()).tolist())),
        'note':'Scoring counterfactual on unchanged saved geometry, feasibility and originalNsoft; no solver improvement. Singlecategory reductions are not additive.'})
    assert max_error<1e-10
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reuse-opened-test-for-gallery',action='store_true')
    args=p.parse_args()
    if not args.reuse_opened_test_for_gallery:p.error('Explicit scoped reuse of already opened test cases required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    run=args.output.resolve()
    if not run.is_relative_to((ROOT/'experiments/graph_proposal/runs').resolve()):p.error('Output must be experiment runs')
    run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+600
    raw_file=ROOT/'experiments/graph_proposal/runs/20261016_raw_proposal_train100_dev100_seed0/cases.jsonl'
    test_file=ROOT/'experiments/graph_proposal/runs/20261016_frozen_best_test2500_seed0/test_cases.jsonl'
    raw_rows=jsonl(raw_file);test_rows=jsonl(test_file)
    raw_config=json.loads((raw_file.parent/'config.json').read_text(encoding='utf-8'))
    test_config=json.loads((test_file.parent/'config.json').read_text(encoding='utf-8'))
    checkpoint_hash=sha(args.checkpoint)
    assert checkpoint_hash==raw_config['checkpoint_sha256']==test_config['identity']['checkpoint_sha256']
    def worker(row):return int(row['source']['shard'].split('/')[0][7:])
    selected=[]
    for panel in ('train','dev'):
        for w in (0,11,22,33,44,55,66,77,88,99):
            selected.append((panel,next(row for row in raw_rows if row['panel']==panel and worker(row)==w)))
    for w in (0,4,8,12,16,21,25,29,33,37):
        pool=[r for r in test_rows if worker(r)==w]
        chosen=min(pool,key=lambda row:hashlib.sha256(('soft_gallery_test_v1:'+row['case_id']).encode()).hexdigest())
        selected.append(('local_test',chosen))
    ablation_workers=(0,22,44,66,99)
    manifest=workset_manifest()
    sources=sorted(set([*Path(__file__).parent.glob('*.py'),*(ROOT/'datasets').glob('*.py'),
        *(ROOT/'scoring').glob('*.py'),ROOT/'protocol/workset_v2.json',ROOT/'protocol/split_v1.json',
        ROOT/'protocol/paths.json',ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']))
    config={'command':[os.sys.executable,*os.sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint_sha256':checkpoint_hash,'checkpoint':str(args.checkpoint.resolve()),'workset_sha256':manifest['manifest_sha256'],
        'data_paths':resolved_data_paths(),'panels':{name:[r['source'] for panel,r in selected if panel==name] for name in ('train','dev','local_test')},
        'selection':'train/dev: fixedworkers0,11,22,33,44,55,66,77,88,99 from previousA; test: alreadyopened workers0,4,8,12,16,21,25,29,33,37, minSHA256(soft_gallery_test_v1:case_id), no scoreselection',
        'test_authorization':'User explicitly requests10test illustrations and soft-penalty analysis; reuse opened cases only; no tuning on test',
        'test_final_geometry':'Reuse original saved frozen evaluation; never rerun test postprocessing',
        'raw_geometry':'Reuse existing GPU train/dev raw proposals; CPU inference only for10alreadyopened test rows to match originalCPU evaluation',
        'ablation':{'panel':'dev only','workers':ablation_workers,'local_soft_weights':[0,2,10],
          'changed':'boundary_weight and group_weight only, jointly; MIB alignment, top-shelf rules and all other settings fixed',
          'purpose':'mechanism comparison, not select new weights or checkpoint','seconds_per_variant':15},
        'wall_seconds':600,'cpu_threads':1,'seed':0,'torch':torch.__version__,
        'canonical_geometry':'cases.jsonl and ablation_cases.jsonl contain only new coordinates; old coordinates referenced via positions_source',
        'source_inputs':{str(f.relative_to(ROOT)):{'sha256':sha(f),'rows':len(raw_rows) if f==raw_file else len(test_rows)} for f in (raw_file,test_file)},
        'source_sha256':{s.relative_to(ROOT).as_posix():sha(s) for s in sources},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    save_json(run/'config.json',config)
    with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT))
    readers={name:LayoutReader(manifest,workset,allow_local_test=name=='local_test') for name,workset in
        [('train','train_monitor'),('dev','dev_quality'),('local_test','local_test')]}
    indices={name:{pair:i for i,pair in enumerate(reader.members)} for name,reader in readers.items()}
    model=GraphProposal.from_checkpoint(args.checkpoint)
    rows=[];ablations=[];raw_test_checks=[];baseline_cache={}
    def ref_to(file,case_id,variant):return {'file':file.relative_to(ROOT).as_posix(),'case_id':case_id,'variant':variant}
    with (run/'cases.jsonl').open('x',encoding='utf-8',newline='\n') as stream,torch.inference_mode():
        for number,(panel,old) in enumerate(selected,1):
            if time.monotonic()>=deadline-20:raise TimeoutError('Bounded gallery stopped before requested case count')
            source=old['source'];index=indices[panel][(source['shard'],source['sample_index'])]
            raw=readers[panel].case(index);n,a,b,pb,pins,c,_=raw.solver_inputs
            case=_build_case(raw.case_id,raw.source,(a,b,pb,pins,c),raw.reference_positions,raw.stored_metrics)
            problem=prepare_problem(*case.solver_inputs)
            if panel=='local_test':
                proposal=model.propose(case.solver_inputs)[0].numpy().astype(float)
                graph=graph_features(case.solver_inputs);target=encode_target(case.reference_positions,graph)
                mse=float(components(model(graph),target,graph['mask'])[0])
                delta=abs(mse-old['supervised']['mse']);assert delta<1e-7
                raw_test_checks.append(delta)
                revised=correct(proposal,problem);final=np.asarray(old['variants']['deterministic']['positions'])
                fallback=old['variants']['deterministic']['fallback'];seconds=old['variants']['deterministic']['solve_seconds']
            else:
                assert old['variants']['reference']['positions']==case.reference_positions
                proposal=np.asarray(old['variants']['deterministic']['positions'])
                revised,final,fallback,seconds=postprocess(problem,proposal)
            out={'panel':panel,'case_id':case.case_id,'source':case.source,'input_summary':case.input_summary(),
                 'constraints':c.int().tolist(),'variants':{},'final_fallback':fallback}
            for name,pos in [('reference',case.reference_positions),('raw',proposal),('corrected',revised),('final',final)]:
                scored=evaluate_case(case,pos,seconds if name=='final' else 0)
                if panel=='local_test' and name=='final':
                    assert abs(scored['metrics']['cost']-old['variants']['deterministic']['metrics']['cost'])<1e-12
                v={'metrics':scored['metrics'],'soft_details':soft_details(pos,c.numpy(),scored['metrics'])}
                if panel!='local_test' and name in ('reference','raw'):
                    v['positions_source']=ref_to(raw_file,old['case_id'],'reference' if name=='reference' else 'deterministic')
                elif panel=='local_test' and name=='final':v['positions_source']=ref_to(test_file,old['case_id'],'deterministic')
                else:v['positions']=scored['positions']
                out['variants'][name]=v
            stream.write(json.dumps(out,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(out)
            if panel=='dev' and worker(old) in ablation_workers:
                baseline_cache[case.case_id]=(case,problem,proposal,out)
            print(json.dumps({'phase':'gallery_data','completed':number,'panel':panel,'final_Q':out['variants']['final']['metrics']['cost']}),flush=True)
    # Fixed5dev weight ablation. No test row is used to choose a candidate.
    with (run/'ablation_cases.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for case_id,(case,problem,proposal,base) in baseline_cache.items():
            if time.monotonic()>=deadline-20:raise TimeoutError('Short dev ablation wall budget exhausted')
            row={'case_id':case_id,'input_summary':case.input_summary(),'variants':{}}
            for weight in (0,2,10):
                name=f'weight_{weight}'
                if weight==2:
                    row['variants'][name]={'metrics':base['variants']['final']['metrics'],
                        'positions_source':{'file':(run/'cases.jsonl').relative_to(ROOT).as_posix(),'case_id':case_id,'variant':'final'},
                        'solve_seconds':base['variants']['final']['metrics']['runtime_seconds'],'fallback':base['final_fallback']}
                else:
                    revised,final,fallback,seconds=postprocess(problem,proposal,weight,weight)
                    row['variants'][name]={**evaluate_case(case,final,seconds),'solve_seconds':seconds,'fallback':fallback}
            ablations.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({'phase':'dev_soft_weight_ablation','case_id':case_id,
                'Q':{k:v['metrics']['cost'] for k,v in row['variants'].items()}}),flush=True)
    assert len(rows)==30 and len(ablations)==5
    decomp=score_decomposition(test_rows)
    staged={}
    for panel in ('train','dev','local_test'):
        panel_rows=[r for r in rows if r['panel']==panel];staged[panel]={}
        for stage in ('reference','raw','corrected','final'):
            vals=[r['variants'][stage]['metrics'] for r in panel_rows]
            staged[panel][stage]={'case_count':len(vals),'feasible_count':sum(m['is_feasible'] for m in vals),
                **{k+'_mean':float(np.mean([m[k] for m in vals])) for k in
                   ('boundary_violations','grouping_violations','mib_violations','violations_relative','hpwl_gap','area_gap','cost')}}
    ablation_summary=summarize_variants(ablations,['weight_0','weight_2','weight_10'])
    for name in ablation_summary:
        ablation_summary[name]['fallback_count']=sum(row['variants'][name]['fallback'] for row in ablations)
    result={'status':'completed','gallery_cases':30,'panels':{p:10 for p in ('train','dev','local_test')},
        'staged_soft_metrics':staged,'saved_test957_scoring_decomposition':decomp,
        'dev5_weight_ablation':ablation_summary,'wall_seconds':time.monotonic()-started,
        'test_existing_final_reused':10,'test_raw_proposal_inferences':10,'new_unopened_test_members':0,
        'max_test_raw_mse_reproduction_error':max(raw_test_checks),'checkpoint_unchanged':sha(args.checkpoint)==checkpoint_hash,
        'training_updates':0,'parameter_or_candidate_selection_from_test':False,
        'test_note':'Only prior957 opened cases are reused; no extension or resume of interrupted2500 test. Test images and decomposition are descriptive, never tuning.'}
    assert result['checkpoint_unchanged']
    save_json(run/'results.json',result)
    print(json.dumps({'status':'completed','wall_seconds':result['wall_seconds'],
        'decomposition':decomp,'dev5_ablation':ablation_summary},indent=2),flush=True)


if __name__=='__main__':main()
