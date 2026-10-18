"""Fixed100dev quality comparison after training; test data stays sealed."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--previous-checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--wall-seconds',type=float,default=600)
    args=p.parse_args()
    if not 0<args.seconds<=15 or not 30<=args.wall_seconds<=900:p.error('Fixed quality budget: per-case<=15s, wall30..900s')
    import torch
    torch.set_num_threads(1)
    from datasets.worksets import workset_manifest,LayoutReader
    from datasets.cases import _build_case
    from scoring.evaluation import evaluate_case,summarize_variants
    from experiments.graph_proposal.pipeline import Optimizer
    from experiments.graph_proposal.train import save_json
    for cp in (args.checkpoint,args.previous_checkpoint):
        if cp and not cp.is_file():p.error('Checkpoint missing: '+str(cp))
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=workset_manifest();reader=LayoutReader(manifest,'dev_quality')
    started=time.monotonic();deadline=started+args.wall_seconds
    choices={'heuristic':None,'deterministic_v2':args.checkpoint}
    if args.previous_checkpoint:choices['deterministic_v1']=args.previous_checkpoint
    source=[*Path(__file__).parent.glob('*.py'),*(ROOT/'datasets').glob('*.py'),*(ROOT/'scoring').glob('*.py'),
            ROOT/'protocol/workset_v2.json',ROOT/'protocol/split_v1.json',ROOT/'protocol/paths.json',
            ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']
    config={'command':[sys.executable,*sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
            'workset_sha256':manifest['manifest_sha256'],'panel':'dev_quality','count':len(reader),
            'variants':list(choices),'cpu_threads':1,'seed':0,'seconds_per_case_variant':args.seconds,
            'wall_seconds':args.wall_seconds,'test_payload_reads':0,
            'checkpoints':{n:{'path':str(cp.resolve()),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()} for n,cp in choices.items() if cp},
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source},
            'runtime_note':'CPU1 inference after GPU training ends; common15s ceiling, actual runtime separate; neutralQ not contest score',
            'comparison_note':'v1 and v2 training data/budget/batch/ML controls differ; cannot attribute delta to a single change',
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    save_json(args.output/'config.json',config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in source:z.write(f,f.relative_to(ROOT))
    result={'status':'running','cases':[],'test_payload_reads':0}
    for index in range(len(reader)):
        raw=reader.case(index)
        n,area,b2b,p2b,pins,c,_=raw.solver_inputs
        case=_build_case(raw.case_id,raw.source,(area,b2b,p2b,pins,c),raw.reference_positions,raw.stored_metrics)
        row={'case_id':case.case_id,'source':case.source,'input_summary':case.input_summary(),'variants':{}}
        for name,cp in choices.items():
            if time.monotonic()>=deadline:
                row['variants'][name]={'error':'aggregate quality wall budget exhausted','solve_seconds':0,
                                       'metrics':{'is_feasible':False,'cost':10.,'hpwl_gap':None,'area_gap':None,
                                                  'boundary_violations':None,'grouping_violations':None,'mib_violations':None}}
                continue
            if cp:os.environ['GRAPH_PROPOSAL_CHECKPOINT']=str(cp.resolve())
            solver=Optimizer('heuristic' if cp is None else 'deterministic',args.seconds,0)
            t=time.monotonic()
            try:
                positions=solver.solve(*case.solver_inputs)
                elapsed=time.monotonic()-t
                assessment=evaluate_case(case,positions,elapsed)
                record={**assessment,'solve_seconds':elapsed,'last_stats':solver.last_stats}
                for diagnostic in record['last_stats'].get('candidate_diagnostics',[]):
                    for stage in ('raw','corrected'):
                        a=evaluate_case(case,diagnostic[stage],0)
                        diagnostic[stage+'_assessment']={k:v for k,v in a.items() if k!='positions'}
                row['variants'][name]=record
            except Exception:
                row['variants'][name]={'error':traceback.format_exc(),'solve_seconds':time.monotonic()-t,
                                       'metrics':{'is_feasible':False,'cost':10.,'hpwl_gap':None,'area_gap':None,
                                                  'boundary_violations':None,'grouping_violations':None,'mib_violations':None}}
        result['cases'].append(row)
        save_json(args.output/'results.json',result)
        if (index+1)%10==0 or index==0:
            status={'completed_cases':index+1,'of':len(reader),'wall_seconds':time.monotonic()-started,
                    'last_case_Q':{k:v['metrics']['cost'] for k,v in row['variants'].items()}}
            print(json.dumps(status),flush=True)
            with (args.output/'run.log').open('a',encoding='utf-8') as f:f.write(json.dumps(status)+'\n')
    result['summary']=summarize_variants(result['cases'],list(choices))
    for name in choices:
        records=[row['variants'][name] for row in result['cases']]
        result['summary'][name]['fallback_count']=sum(bool(v.get('last_stats',{}).get('fallback')) for v in records)
        result['summary'][name]['wins_vs_heuristic']=sum(v['metrics']['cost']<row['variants']['heuristic']['metrics']['cost']-1e-9 for row,v in zip(result['cases'],records))
    result['status']='completed_with_errors' if any('error' in v for row in result['cases'] for v in row['variants'].values()) else 'completed'
    result['wall_seconds']=time.monotonic()-started
    save_json(args.output/'results.json',result)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
