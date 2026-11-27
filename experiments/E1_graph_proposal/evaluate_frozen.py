"""One frozen-model final evaluation. Explicit --final-test authorizes test reads.

No optimizer, model selection, checkpoint changes, or solver tuning exists here.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--wall-seconds',type=float,default=3600)
    args=p.parse_args()
    if not args.final_test:p.error('Test data requires explicit --final-test and a frozen user-approved model')
    if not 0<args.seconds<=15 or not 60<=args.wall_seconds<=3600:p.error('Per-case<=15s; total60..3600s')
    import torch
    torch.set_num_threads(1)
    from datasets.worksets import workset_manifest,LayoutReader
    from datasets.cases import _build_case
    from datasets.paths import resolved_data_paths
    from scoring.evaluation import evaluate_case,summarize_variants
    from experiments.graph_proposal.model import GraphProposal
    from experiments.graph_proposal.data import graph_features,encode_target
    from experiments.graph_proposal.pipeline import Optimizer
    from experiments.graph_proposal.train_supervised import components
    from experiments.graph_proposal.train import save_json
    if not args.checkpoint.is_file():p.error('Missing frozen checkpoint')
    if args.output.exists() and not args.resume:p.error('Output already exists; do not rerun test, use explicit resume of same frozen evaluation')
    manifest=workset_manifest()
    sources=[Path(__file__),*list(Path(__file__).parent.glob('*.py')),*list((ROOT/'datasets').glob('*.py')),
             *list((ROOT/'scoring').glob('*.py')),ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
             ROOT/'protocol/workset_v2.json',ROOT/'protocol/split_v1.json',ROOT/'protocol/paths.json']
    sources=sorted(set(sources))
    identity={'checkpoint_sha256':sha(args.checkpoint),'workset_sha256':manifest['manifest_sha256'],
              'source_sha256':{s.relative_to(ROOT).as_posix():sha(s) for s in sources},
              'cpu_threads':1,'seed':0,'seconds_per_case':args.seconds,
              'protocol':'single validation-best deterministic model; fullworkset finaltest; no test-based decisions',
              'device':'cpu','torch':torch.__version__}
    args.output.mkdir(parents=True,exist_ok=args.resume)
    config_path=args.output/'config.json'
    if args.resume:
        original=json.loads(config_path.read_text(encoding='utf-8'))
        if original['identity']!=identity:raise ValueError('Frozen model/source/workset or evaluation budget changed; cannot resume this test')
    else:
        config={'identity':identity,'command':[sys.executable,*sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
                'checkpoint':str(args.checkpoint.resolve()),'data_paths':resolved_data_paths(),'workset_counts':manifest['counts'],
                'test_authorization':'User: stop training, perform testing and scoring; no further optimization after test is opened',
                'total_wall_seconds':args.wall_seconds,'quality_note':'official runtime=median_runtime=1, independent hard check; actualtime separate; not official contest score',
                'positions_storage':'test_cases.jsonl only; summary has no duplicate geometry','source_snapshot':'source_snapshot.zip'}
        save_json(config_path,config)
        with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    started=time.monotonic();deadline=started+args.wall_seconds
    model=GraphProposal.from_checkpoint(args.checkpoint,device='cpu')
    model.eval()
    saved=json.loads((args.output/'supervised.json').read_text()) if (args.output/'supervised.json').exists() else {}
    # These groups precede first test read. They cannot change the frozen checkpoint.
    with torch.no_grad():
        for name in ('train','dev'):
            if saved.get(name,{}).get('complete'):continue
            reader=LayoutReader(manifest,name);totals=[0.,0.,0.];count=0
            for index in range(len(reader)):
                if time.monotonic()>=deadline-10:break
                inputs,reference=reader[index]
                graph=graph_features(inputs,'cpu');target=encode_target(reference,graph)
                values=components(model(graph),target,graph['mask'])
                for j,v in enumerate(values):totals[j]+=float(v)
                count+=1
                if count%2000==0:print(json.dumps({'phase':'supervised_'+name,'processed':count,'total':len(reader),'wall_seconds':time.monotonic()-started}),flush=True)
            saved[name]={'sample_count':count,'requested_count':len(reader),'complete':count==len(reader),
                         'mse':totals[0]/count if count else None,'center_mse':totals[1]/count if count else None,
                         'log_ratio_mse':totals[2]/count if count else None,'aggregation':'mean per-layout masked MSE'}
            save_json(args.output/'supervised.json',saved)
            if not saved[name]['complete']:
                save_json(args.output/'results.json',{'status':'resource_stopped_before_test','supervised':saved,'test_payload_reads':0})
                return
    reader=LayoutReader(manifest,'local_test',allow_local_test=True)
    os.environ['GRAPH_PROPOSAL_CHECKPOINT']=str(args.checkpoint.resolve())
    positions_path=args.output/'test_cases.jsonl'
    compact=[];totals=[0.,0.,0.]
    def compact_record(row):
        v=row['variants']['deterministic']
        return {'case_id':row['case_id'],'input_summary':row['input_summary'],
                'variants':{'deterministic':{'metrics':v['metrics'],'solve_seconds':v['solve_seconds'],
                                            'fallback':v.get('fallback',False),'error':v['error']} if 'error' in v else
                                           {'metrics':v['metrics'],'solve_seconds':v['solve_seconds'],'fallback':v['fallback']}}}
    if positions_path.exists():
        with positions_path.open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                row=json.loads(line)
                expected=reader.members[i]
                if (row['source']['shard'],row['source']['sample_index'])!=expected:raise ValueError('Test resume member order changed')
                compact.append(compact_record(row))
                for j,k in enumerate(('mse','center_mse','log_ratio_mse')):totals[j]+=row['supervised'][k]
    completed_before=len(compact)
    with positions_path.open('a',encoding='utf-8',newline='\n') as stream,torch.no_grad():
        for index in range(completed_before,len(reader)):
            if time.monotonic()>=deadline-5:break
            raw=reader.case(index)
            n,area,b2b,p2b,pins,c,_=raw.solver_inputs
            case=_build_case(raw.case_id,raw.source,(area,b2b,p2b,pins,c),raw.reference_positions,raw.stored_metrics)
            solver=Optimizer('deterministic',seconds=args.seconds,seed=0)
            t=time.monotonic()
            try:
                positions=solver.solve(*case.solver_inputs)
                elapsed=time.monotonic()-t
                v={**evaluate_case(case,positions,elapsed),'solve_seconds':elapsed,'fallback':solver.last_stats['fallback']}
                v['diagnostics']={k:value for k,value in solver.last_stats.items() if k!='candidate_diagnostics'}
                selected=solver.last_stats.get('candidate_diagnostics',[])
                if selected:
                    selected=selected[solver.last_stats['chosen_candidate']]
                    v['repair_displacement']={k:selected[k] for k in ('continuous_displacement_mean','final_displacement_mean','continuous_seconds','geometry_seconds')}
            except Exception:
                v={'error':traceback.format_exc(),'solve_seconds':time.monotonic()-t,'fallback':False,
                   'metrics':{'is_feasible':False,'cost':10.,'hpwl_gap':None,'area_gap':None,'boundary_violations':None,
                              'grouping_violations':None,'mib_violations':None}}
            graph=graph_features(case.solver_inputs,'cpu');target=encode_target(case.reference_positions,graph)
            values=components(model(graph),target,graph['mask'])
            supervised={k:float(x) for k,x in zip(('mse','center_mse','log_ratio_mse'),values)}
            row={'case_id':case.case_id,'source':case.source,'input_summary':case.input_summary(),
                 'variants':{'deterministic':v},'supervised':supervised}
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            compact.append(compact_record(row))
            for j,k in enumerate(('mse','center_mse','log_ratio_mse')):totals[j]+=supervised[k]
            if (index+1)%50==0 or index<5:
                count=len(compact);elapsed=time.monotonic()-started
                progress={'phase':'local_test','processed':count,'total':len(reader),'last_case_Q':v['metrics']['cost'],
                          'wall_seconds':elapsed,'test_error_count':sum('error' in r['variants']['deterministic'] for r in compact),
                          'last_solve_seconds':v['solve_seconds']}
                print(json.dumps(progress),flush=True)
                with (args.output/'run.log').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
                save_json(args.output/'progress.json',progress)
    count=len(compact)
    summary=summarize_variants(compact,['deterministic']) if count else {}
    if count:
        summary['deterministic']['fallback_count']=sum(r['variants']['deterministic']['fallback'] for r in compact)
        summary['deterministic']['mean_solve_seconds']=summary['deterministic']['total_solve_seconds']/count
    saved['local_test']={'sample_count':count,'requested_count':len(reader),'complete':count==len(reader),
                         **{k:totals[j]/count if count else None for j,k in enumerate(('mse','center_mse','log_ratio_mse'))}}
    save_json(args.output/'supervised.json',saved)
    result={'status':'completed' if count==len(reader) else 'resource_stopped','summary':summary,'supervised':saved,
            'frozen_checkpoint_sha256':identity['checkpoint_sha256'],'test_cases':count,'requested_test_cases':len(reader),
            'wall_seconds_this_invocation':time.monotonic()-started,'test_used_for_tuning':False,'official_contest_score':None,
            'test_independence_note':'This local_test panel has now been evaluated; report reuse explicitly for future models.',
            'formal_complete':count==len(reader)}
    save_json(args.output/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
