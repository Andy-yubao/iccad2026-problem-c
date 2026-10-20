"""Five-case mechanism probes, never a quality acceptance test."""
import json
import time
import numpy as np
from .cases import prepare_cache
from .io import HERE,save
from .solver import optimize
from .fast_geometry import topology
from .backend.frozen.global_refine import topology as old_topology
from .canonical import canonicalize
from .backend.frozen.canonical_geometry import canonicalize as old_canonical
from .backend.frozen.packing import prepare_problem
from scoring.evaluation import evaluate_case


def main():
    rows=prepare_cache()['rows'];order=sorted(range(500),key=lambda i:rows[i]['case'].block_count)
    selected=[order[i] for i in [0,124,249,374,499]]
    configs=[dict(iterations=4),dict(iterations=6,mode=1,reweighted=False,boundary=40.,contact=20.),
        dict(iterations=6,relaxed=True),dict(iterations=0,mode=1,boundary=40.,contact=20.)]
    folder=HERE/'runs/20261020_vector_controls5_seed0';folder.mkdir(exist_ok=False);results=[]
    for index in selected:
        row=rows[index];case=row['case'];p=prepare_problem(*case.solver_inputs)
        for mode in [0,1]:assert topology(row['draft'],p.constraints,mode)==old_topology(row['draft'],p.constraints,mode)
        for ci,config in enumerate(configs):
            start=time.perf_counter();x,stats=optimize(row['draft'],case.solver_inputs,**config)
            elapsed=time.perf_counter()-start+row['inference_seconds'];m=evaluate_case(case,x.tolist(),elapsed)['metrics']
            assert np.array_equal(canonicalize(x,p),old_canonical(x,p))
            r={'index':index,'n':p.n,'config':ci,'Q':m['cost'],'seconds':elapsed,'row':stats['row_retained'],
                'hard_legal':m['is_feasible'],'trace':stats['trace']};results.append(r)
            print(json.dumps({k:v for k,v in r.items() if k!='trace'}),flush=True)
    save(folder/'results.json',{'configs':configs,'records':results,'test_read':False})


if __name__=='__main__': main()
