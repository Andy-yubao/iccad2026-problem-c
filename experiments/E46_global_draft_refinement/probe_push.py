"""Five-case saved-layout probe of a certified whole-piece push polish."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import json
import time
import numpy as np
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from .io import HERE,save,sources
from .backend.frozen.packing import prepare_problem
from .certify_polish import polish
from .canonical_native import warmup


def main():
    data=ReferenceLayouts('dev_monitor500');warmup()
    source=HERE/'runs/20261024_shape_memory_gate5/results_local.json'
    state=json.loads(source.read_text(encoding='utf-8'));rows=[]
    for r in state['records']:
        case=data[r['index']]['case'];p=prepare_problem(*case.solver_inputs);v=r['variants']['control'];before=np.array(v['positions'])
        started=time.perf_counter();after,stats=polish(before,p,seconds=.01,push=True);elapsed=time.perf_counter()-started
        metrics=evaluate_case(case,after,v['complete_seconds']+elapsed)['metrics']
        if metrics['cost']>v['metrics']['cost']+1e-9:raise AssertionError('Certificate violated scored Q')
        rows.append({'index':r['index'],'before_Q':v['metrics']['cost'],'after_Q':metrics['cost'],'extra_seconds':elapsed,'stats':stats,
            'positions':after.tolist(),'metrics':metrics})
        print({k:v for k,v in rows[-1].items() if k not in ('positions','metrics')},flush=True)
    f=HERE/'runs/20261024_certified_push_probe5';f.mkdir(exist_ok=True)
    save(f/'results_local.json',{'source':source.relative_to(HERE).as_posix(),'rows':rows,'sources':sources(),'test_read':False})
    save(f/'summary.json',{'count':len(rows),'accepted_moves':sum(r['stats']['accepted'] for r in rows),'mean_Q_delta':float(np.mean([r['after_Q']-r['before_Q'] for r in rows])),
        'mean_extra_seconds':float(np.mean([r['extra_seconds'] for r in rows])),'test_read':False})


if __name__=='__main__':main()
