"""Five saved train drafts, two reproductions; no network/GPU, dev, or test."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..io import ROOT,save,sha
from ..generalization_baseline.data import load_rows
from ..dataset_view import ReferenceLayouts
from .solver import optimize,BUDGET
from .repair.canonical_native import warmup
from scoring.evaluation import evaluate_case

OWNER=Path(__file__).resolve().parent

def sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OWNER.rglob('*.py'))}|{
        (OWNER/'budget.json').relative_to(ROOT).as_posix():sha(OWNER/'budget.json')}

def main():
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import numba
    numba.set_num_threads(1)
    records=ReferenceLayouts('train').records
    lookup={tuple(r['member']):i for i,r in enumerate(records)}
    ids=[lookup[tuple(c['member'])] for c in BUDGET['implementation_cases']]
    rows=load_rows('train',ids)
    results=[]
    with threadpool_limits(limits=1):
        warmup()
        for item,row in zip(BUDGET['implementation_cases'],rows):
            saved=json.loads((ROOT/item['record']).read_text())
            draft=np.asarray(saved['draft_positions'],dtype=float)
            outputs=[]
            stats=[]
            for _ in range(2):
                final,info=optimize(draft,row['case'].solver_inputs)
                score=evaluate_case(row['case'],final,info['seconds'])
                assert score['metrics']['is_feasible']
                outputs.append(final)
                stats.append(info)
            np.testing.assert_array_equal(outputs[0],outputs[1])
            assert stats[0]['work']==stats[1]['work']
            results.append(dict(member=item['member'],n=item['n'],identical_geometry=True,
                work=stats[0]['work'],seconds=[x['seconds'] for x in stats],all_hard_legal=True))
    save(OWNER/'preflight.json',dict(status='passed',unique_train_cases=5,cases=results,
        sources=sources(),test_read=False,dev_payloads=0,network_calls=0,
        note='CPU reproduction during separate baseline training; not an uncontended latency benchmark',
        elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(status='passed',cases=results),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
