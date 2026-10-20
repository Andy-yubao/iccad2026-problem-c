"""Compare vectorized Clarabel/HiGHS projections on five saved legal outputs."""
import json
import numpy as np
import torch
from .io import ROOT,HERE,save,sources,snapshot
from .cases import seed_all
from .backend.frozen.packing import prepare_problem
from .backend.frozen.solver import hard_valid
from .fast_geometry import topology,contact_tree,reduce_edges
from .canonical import canonicalize
from .projection import solve
from scoring.evaluation import evaluate_case


def main():
    seed_all()
    rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    previous=json.loads((ROOT/'experiments/E18_fast_postprocess/runs/20261020_full4_robust_fast_dev500/results.json').read_text())['records']
    folder=HERE/'runs/20261020_qp_projection5_e17_diagnostic'
    identity={'sources':sources(),'indices':[21,127,275,447,456],
              'seconds_budget':.13,'test_read':False,'timing':'diagnostic under concurrent training'}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
    records=[]
    for index in identity['indices']:
        case=rows[index]['case'];p=prepare_problem(*case.solver_inputs);seed=np.array(previous[index]['positions'])
        full=topology(seed,p.constraints,array=True);tree=contact_tree(seed,p,full);edges=reduce_edges(full,p.n)
        for engine in ('clarabel','quadratic'):
            x,stats=solve(p,seed,edges,tree,seconds=.13,boundary=40,contact=40,hard_tangent=False,quadratic=engine=='quadratic')
            if x is not None:x=canonicalize(x,p,tolerance=1e-4)
            legal=x is not None and hard_valid(x,p)
            metrics=evaluate_case(case,x.tolist(),stats['seconds'])['metrics'] if legal else None
            records.append({'index':index,'before':previous[index]['metrics'],'engine':engine,'metrics':metrics,'legal':legal,'stats':stats})
            print(index,engine,round(metrics['cost'],6) if legal else 'failed',stats,flush=True)
    save(folder/'results.json',{'records':records,'test_read':False})


if __name__=='__main__':main()
