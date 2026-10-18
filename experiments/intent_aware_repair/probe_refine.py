"""Five fixed dev-quality cases: frozen proposal, old vs new public-input repair."""
import argparse,json,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import scipy,highspy
from datasets.paths import ROOT
from datasets.worksets import workset_manifest,LayoutReader
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .global_refine import refine
from .train import save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--shape-steps',type=int,default=0)
    parser.add_argument('--mip-seconds',type=float,default=0.)
    parser.add_argument('--panel',choices=['dev5','dev100'],default='dev5')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reweight',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.resume)
    begin=time.monotonic();manifest=workset_manifest();reader=LayoutReader(manifest,'dev_quality')
    source=ROOT/'experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0/evaluation/cases.jsonl'
    existing=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    workers=[0,24,49,74,99]
    selected=existing if args.panel=='dev100' else [r for r in existing if int(r['source']['shard'].split('/')[0][7:]) in workers]
    assert len(selected)==(100 if args.panel=='dev100' else 5)
    files=[*Path(__file__).parent.glob('*.py'),*sorted((ROOT/'scoring').glob('*.py')),*sorted((ROOT/'datasets').glob('*.py'))]
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'case_ids':[r['case_id'] for r in selected],
        'command':[__import__('sys').executable,*__import__('sys').argv],
        'seed':0,'cpu_threads':2,'scipy':scipy.__version__,'highs':highspy.Highs().version(),
        'model_checkpoint_sha256':'16af0c449c766b3828630140c285d4f059a4a5d6fecef725aa07491fc3ad4651',
        'source':str(source),'source_sha256':sha(source),'variant':'attention_geometry','seconds_per_case':args.seconds,
        'shape_steps':args.shape_steps,'mip_seconds':args.mip_seconds,
        'panel':args.panel,
        'reweight':args.reweight,
        'test_reads':0,'selection':'full fixed dev_quality100' if args.panel=='dev100' else 'dev_quality workers0,24,49,74,99 fixed before current repair development',
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    if args.resume:
        old_config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ('case_ids','source_sha256','seconds_per_case','shape_steps','mip_seconds','panel','reweight'):
            if old_config[key]!=config[key]:raise ValueError('Resume configuration mismatch:'+key)
    else:
        save(out/'config.json',config)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT))
    records=[json.loads(s) for s in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()] if args.resume and (out/'cases.jsonl').exists() else []
    done={r['case_id'] for r in records};user_stop=False
    with (out/'cases.jsonl').open('a' if args.resume else 'x',encoding='utf-8') as f:
        for row in selected:
            if row['case_id'] in done:continue
            if (out/'STOP').exists():user_stop=True;break
            src=row['source'];idx=reader.members.index((src['shard'],src['sample_index']));case=reader.case(idx)
            old=row['variants']['attention_geometry'];raw=np.asarray(old['raw']['positions'])
            result,stats=refine(raw,case.solver_inputs,seconds=args.seconds,shape_steps=args.shape_steps,mip_seconds=args.mip_seconds,reweighted=args.reweight)
            stats['cached_model_feature_forward_seconds']=old.get('feature_prep_seconds',0)+old.get('forward_seconds',0)
            stats['end_to_end_seconds_including_recorded_model']=stats['seconds']+stats['cached_model_feature_forward_seconds']
            scoring=_build_case(case.case_id,case.source,case.solver_inputs[1:6],case.reference_positions,case.stored_metrics)
            evaluated=evaluate_case(scoring,result,stats['seconds'])
            item={'case_id':case.case_id,'block_count':case.block_count,'old':old['final']['metrics'],
                'new':evaluated,'trace':stats,'raw_source':str(source)}
            f.write(json.dumps(item,allow_nan=False)+'\n');f.flush();records.append(item)
            print(json.dumps({'case':case.case_id,'old_Q':item['old']['cost'],'new_Q':item['new']['metrics']['cost'],
                'BGM':[item['new']['metrics'][k] for k in ['boundary_violations','grouping_violations','mib_violations']],
                'seconds':stats['seconds'],'legal_candidates':sum(r['legal'] for r in stats['trace']),'completed':len(records)}),flush=True)
    official=official_evaluator();ids=[r['block_count'] for r in records]
    save(out/'results.json',{'status':'user_stop' if user_stop else 'completed','cases':len(records),'test_reads':0,'wall_seconds_this_invocation':time.monotonic()-begin,
        'old_Q':official.compute_total_score([r['old']['cost'] for r in records],ids),
        'new_Q':official.compute_total_score([r['new']['metrics']['cost'] for r in records],ids),
        'new_feasible':sum(r['new']['metrics']['is_feasible'] for r in records),
        'scope':'fixed100dev development evaluation, not independent test' if args.panel=='dev100' else 'five developer cases only, not fixed100 quality or generalization claim',
        'Q_below_1_5_count':sum(r['new']['metrics']['cost']<1.5 for r in records),
        'wins':sum(r['new']['metrics']['cost']<r['old']['cost']-1e-8 for r in records)})


if __name__=='__main__':main()
