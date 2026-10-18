"""Frozen attention_geometry gallery: five train/dev/previously-opened test cases.

soft_details independently copied from E1 soft_constraint_gallery.py,2026-10-17.
No training, candidate selection, weight changes, or cross-experiment imports.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union
from datasets.paths import ROOT,resolved_data_paths
from datasets.worksets import workset_manifest,LayoutReader
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from .evaluate import load
from .inference import prepare_inference
from .data import decode_layout
from .postprocess import finalize
from .train import save,sha


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
        union=unary_union([box(x,y,x+w,y+h) for x,y,w,h in p[c[:,3]==g]])
        groups[str(g)]=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
    for g in np.unique(c[:,2]):
        if g<=0:continue
        mibs[str(g)]=len({(round(float(w),4),round(float(h),4)) for w,h in p[c[:,2]==g,2:]})
    assert sum(bool(b['failed_bits']) for b in boundary)==metrics['boundary_violations']
    assert sum(n-1 for n in groups.values())==metrics['grouping_violations']
    assert sum(n-1 for n in mibs.values())==metrics['mib_violations']
    return {'boundary':boundary,'group_components':groups,'mib_distinct_shapes':mibs}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to((Path(__file__).parent/'runs').resolve()):raise ValueError('Wrong experiment output')
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    checkpoint=ROOT/'experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0/attention_geometry/checkpoints/best.pt'
    old=ROOT/'experiments/graph_proposal/runs/20261017_soft_constraint_gallery_seed0/cases.jsonl'
    oldrows=[json.loads(line) for line in old.read_text(encoding='utf-8').splitlines()]
    chosen=[]
    for panel in ('train','dev','local_test'):
        pool=[r for r in oldrows if r['panel']==panel]
        assert len(pool)==10
        chosen.extend((panel,pool[i]) for i in (0,2,4,6,9))
    manifest=workset_manifest();sources=[*sorted(Path(__file__).parent.glob('*.py')),
        *sorted((ROOT/'datasets').glob('*.py')),*sorted((ROOT/'scoring').glob('*.py')),
        ROOT/'protocol/workset_v2.json',ROOT/'protocol/split_v1.json',ROOT/'protocol/paths.json',
        ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'command':[__import__('sys').executable,*__import__('sys').argv],
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'model':'attention_geometry best; frozen before test illustrations',
        'selection':'Rows1,3,5,7,10 in each panel of previous30-case gallery; fixed before inference, no scoreselection',
        'selection_source':str(old.relative_to(ROOT)),'selection_source_sha256':sha(old),
        'panels':{p:[r['case_id'] for name,r in chosen if name==p] for p in ('train','dev','local_test')},
        'test_authorization':'User explicitly requests5+5+5 illustrations of frozen new model on2026-10-17; only5alreadyopened test members',
        'test_use':'New frozen-model inference/scoring on5previouslyaccessed members; no tuning, no additional test evaluation',
        'workset_sha256':manifest['manifest_sha256'],'data_paths':resolved_data_paths(),
        'postprocess':'UnchangedE1correct12+pack+fallback;15seconds percase including features/forward; not new repair',
        'wall_budget_seconds':600,'cpu_threads':2,'torch':torch.__version__,'device':torch.cuda.get_device_name(0),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},'status':'running'}
    save(output/'config.json',config)
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT))
    model,_=load(checkpoint);assert model.architecture=='attention'
    readers={p:LayoutReader(manifest,p,allow_local_test=(p=='local_test')) for p in ('train','dev','local_test')}
    indices={p:{member:i for i,member in enumerate(r.members)} for p,r in readers.items()}
    # Match the already-warm terminal evaluation; cold GPU setup is separate.
    first_panel,first_selected=chosen[0];first_source=first_selected['source']
    first_case=readers[first_panel].case(indices[first_panel][(first_source['shard'],first_source['sample_index'])])
    warm_start=time.monotonic()
    with torch.inference_mode():
        warm_g,_=prepare_inference(first_case.solver_inputs,'cuda');model(warm_g);torch.cuda.synchronize()
    config['gpu_warmup_seconds']=time.monotonic()-warm_start
    config['timing']='GPU cold initialization separately recorded; each case includes features/forward/decode/frozen postprocess'
    save(output/'config.json',config)
    rows=[]
    with (output/'cases.jsonl').open('x',encoding='utf-8') as stream,torch.inference_mode():
        for panel,selected in chosen:
            if time.monotonic()-start>600:raise TimeoutError('Gallery600second limit')
            src=selected['source'];key=(src['shard'],src['sample_index'])
            case=readers[panel].case(indices[panel][key]);assert case.case_id==selected['case_id']
            inp=case.solver_inputs
            torch.cuda.synchronize();t=time.monotonic();g,decoder=prepare_inference(inp,'cuda')
            raw=decode_layout(model(g)[0,:case.block_count],decoder).cpu().numpy().astype(float)
            torch.cuda.synchronize();forward=time.monotonic()-t
            final,stats=finalize(raw,inp,max(.05,15-forward));seconds=time.monotonic()-t
            scoring_case=_build_case(case.case_id,case.source,inp[1:6],case.reference_positions,case.stored_metrics)
            row={'panel':panel,'case_id':case.case_id,'source':case.source,'constraints':torch.as_tensor(inp[5]).tolist(),
                'variants':{},'solve_seconds':seconds,'postprocess':stats}
            for stage,positions,runtime in [('reference',case.reference_positions,0),('raw',raw,forward),('final',final,seconds)]:
                scored=evaluate_case(scoring_case,positions,runtime)
                row['variants'][stage]={**scored,'soft_details':soft_details(positions,row['constraints'],scored['metrics'])}
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(row)
            print(json.dumps({'panel':panel,'case_id':case.case_id,'seconds':seconds}),flush=True)
    assert sha(checkpoint)==config['checkpoint_sha256']
    save(output/'results.json',{'status':'completed','case_count':len(rows),'counts':{p:sum(r['panel']==p for r in rows) for p in readers},
        'final_feasible_count':sum(r['variants']['final']['metrics']['is_feasible'] for r in rows),
        'test_sample_reads':5,'test_members_previously_accessed':True,'wall_seconds':time.monotonic()-start,
        'scope':'Illustrations only; no new training or parameter selection; raw/final coordinates saved only in cases.jsonl'})
    config['status']='completed';save(output/'config.json',config)


if __name__=='__main__':main()
