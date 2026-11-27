"""User-requested ten-case dev probe; bounded job, no training or test reads.

Kept in this run so adding the probe cannot change the source fingerprint of
the user's potentially active remaining-three training run.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from experiments.repair_friendly_draft.records import save,append,sha,provenance,snapshot
from experiments.repair_friendly_draft.evaluate import load_model,summarize
from experiments.repair_friendly_draft.inference import prepare_inference
from experiments.repair_friendly_draft.batches import decode
from experiments.repair_friendly_draft.occupancy import exact_diagnostics
from experiments.repair_friendly_draft.solver import optimize
from scoring.evaluation import evaluate_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();begin=time.perf_counter();out=Path(__file__).resolve().parent
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest=workset_manifest();reader=LayoutReader(manifest,'dev_quality')
    workers=list(range(0,100,11))
    selected=[i for i,p in enumerate(reader.members) if int(p[0].split('/')[0][7:]) in workers]
    assert len(selected)==10
    checkpoint=ROOT/'experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0/attention_occupancy/checkpoints/best.pt'
    config={'panel':'dev10','workers':workers,'members':[reader.members[i] for i in selected],
        'selection':'fixed workers0,11,22,33,44,55,66,77,88,99 before scores',
        'workset_sha256':manifest['manifest_sha256'],'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'variant':'attention_occupancy','best_step':30625,'seed':0,'threads':2,
        'postprocess':{'seconds':120.,'shape_steps':24,'reweighted':True,'mip_seconds':0.},
        'invocation_budget_seconds':900.,'scope':'10dev diagnostic only; not test50 or formal contest score',
        'test_payload_reads':0,**provenance()}
    config['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]=sha(__file__)
    if (out/'config.json').exists():
        if not args.resume:raise ValueError('Existing probe requires --resume')
        old=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ('members','checkpoint_sha256','postprocess','source_sha256','workset_sha256'):
            if old[key]!=config[key]:raise ValueError('Resume mismatch:'+key)
    else:save(out/'config.json',config);snapshot(out,config['source_sha256'])
    rows=[json.loads(s) for s in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()] if args.resume and (out/'cases.jsonl').exists() else []
    done={r['case_id'] for r in rows};stop=[False];reason='completed'
    signal.signal(signal.SIGINT,lambda *unused:stop.__setitem__(0,True))
    name='attention_occupancy';model=load_model(checkpoint,'cuda');warm=False
    try:
        for idx in selected:
            pair=reader.members[idx];case_id=f'dev:{pair[0]}:{pair[1]}'
            if case_id in done:continue
            if stop[0] or (out/'STOP').exists():reason='user_stop';break
            if time.perf_counter()-begin>900-145:reason='resource_budget';break
            raw=reader.case(idx)
            scoring=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            if not warm:
                with torch.no_grad():model(prepare_inference(raw.solver_inputs,'cuda'))
                torch.cuda.synchronize();warm=True
            row={'case_id':raw.case_id,'source':raw.source,'block_count':raw.block_count,
                'reference_positions':raw.reference_positions,'constraints':np.asarray(raw.solver_inputs[5]).tolist(),'variants':{}}
            try:
                t=time.perf_counter();g=prepare_inference(raw.solver_inputs,'cuda');torch.cuda.synchronize()
                prep=time.perf_counter()-t;t=time.perf_counter()
                with torch.no_grad():pos=(decode(model(g),g)[0]*g['scale'][0]).cpu().numpy().astype(float)
                torch.cuda.synchronize();forward=time.perf_counter()-t
                c=np.asarray(raw.solver_inputs[5]);spec=np.asarray(raw.solver_inputs[6]);locked=c[:,:2].any(1)
                pos[locked,2:]=spec[locked,2:];pos[c[:,1]!=0]=spec[c[:,1]!=0]
                raw_diagnostics=exact_diagnostics(pos)
                t=time.perf_counter();final,stats=optimize(pos,raw.solver_inputs,seconds=120.,shape_steps=24,reweighted=True,mip_seconds=0.)
                post=time.perf_counter()-t
                evaluated=evaluate_case(scoring,final,prep+forward+post)
                areas=np.asarray(raw.solver_inputs[1]).copy();areas[locked]=spec[locked,2:].prod(1)
                distance=np.linalg.norm(final[:,:2]+final[:,2:]/2-pos[:,:2]-pos[:,2:]/2,axis=1)/np.sqrt(areas)
                row['variants'][name]={'raw_positions':pos.tolist(),'raw_diagnostics':raw_diagnostics,'final':evaluated,
                    'final_diagnostics':exact_diagnostics(final),'feature_seconds':prep,'forward_seconds':forward,
                    'postprocess_seconds':post,'end_to_end_seconds':prep+forward+post,'trace':stats,
                    'shape_iterations':sum(s.get('phase')=='shape' for s in stats['trace']),
                    'trace_subproblems':len(stats['trace']),'guard_fallback':int(stats['guard_fallback']),
                    'center_displacement_local_mean':float(distance.mean())}
            except Exception as exc:
                row['variants'][name]={'error':{'type':type(exc).__name__,'message':str(exc)},
                    'final':{'metrics':{'cost':10.,'is_feasible':False}}}
            append(out/'cases.jsonl',row);rows.append(row)
            save(out/'results.json',{'status':'running','completed_cases':len(rows),'expected_cases':10,**summarize(rows,[name])})
            value=row['variants'][name]
            print(json.dumps({'complete':len(rows),'case_id':raw.case_id,'Q':value['final']['metrics']['cost'],
                'feasible':value['final']['metrics']['is_feasible'],'postprocess_seconds':value.get('postprocess_seconds'),
                'elapsed':time.perf_counter()-begin}),flush=True)
    finally:
        save(out/'results.json',{'status':'completed' if len(rows)==10 else reason if reason!='completed' else 'interrupted',
            'completed_cases':len(rows),'expected_cases':10,'wall_seconds_this_invocation':time.perf_counter()-begin,
            'test_payload_reads':0,'scope':config['scope'],**summarize(rows,[name])})


if __name__=='__main__':main()
