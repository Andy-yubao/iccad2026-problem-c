"""Small frozen-dev check for the integrated baseline; no local_test access."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new run directory; existing results are never overwritten')
    from datasets.worksets import LayoutReader, workset_manifest
    from datasets.cases import _build_case
    from scoring.evaluation import evaluate_case
    from experiments.E10_integrated_baseline.pipeline import Baseline, CHECKPOINT_SHA256
    import torch
    import numpy as np
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = {'command':sys.argv, 'seed':0, 'device':args.device, 'threads':2,
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
              'sources':{str(f.relative_to(ROOT)):digest(f) for f in Path(__file__).parent.rglob('*.py')},
              'split_sha256':digest(ROOT/'protocol/split_v1.json'),
              'workset_sha256':digest(ROOT/'protocol/workset_v2.json'),
              'checkpoint_sha256':CHECKPOINT_SHA256, 'seconds_per_case_target':5,
              'panel':'dev_quality workers 0,24,49,74,99', 'local_test_read':False}
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    reader = LayoutReader(workset_manifest(), 'dev_quality')
    indices = [i for i,(shard,_) in enumerate(reader.members) if int(shard.split('/')[0].split('_')[1]) in (0,24,49,74,99)]
    assert len(indices)==5
    model = Baseline(args.device)
    rows=[]
    for i in indices:
        raw=reader.case(i)
        case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
        start=time.perf_counter()
        try:
            positions,stats=model.solve(raw.solver_inputs)
            result=evaluate_case(case,positions,stats['end_to_end_seconds'])
            # Improvement curves duplicate coordinates; retain only final output.
            stats['improvements']=[{k:v for k,v in e.items() if k!='positions'} for e in stats.get('improvements',[])]
            rows.append({'case_id':raw.case_id,'source':raw.source,'stats':stats,'result':result})
        except Exception as exc:
            rows.append({'case_id':raw.case_id,'error':repr(exc),'seconds':time.perf_counter()-start,
                         'result':{'metrics':{'is_feasible':False,'cost':10.}}})
        report={'status':'complete' if len(rows)==5 else 'running','cases':rows,
                'feasible':sum(r['result']['metrics']['is_feasible'] for r in rows),
                'Q_mean':sum(r['result']['metrics']['cost'] for r in rows)/len(rows),
                'scope':'five-case migration check only; neutral-runtime Q; no new quality claim'}
        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(raw.case_id, rows[-1]['result']['metrics']['cost'], flush=True)
    if report['feasible']!=5: raise RuntimeError('Migration check failed')


if __name__=='__main__': main()
