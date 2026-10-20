"""Five-case test of global anchor-path repair followed by one cheap QP."""
import time
import torch
from .io import ROOT,HERE,sources,save,snapshot
from .cases import seed_all
from .backend.frozen.packing import prepare_problem,_dimensions
from .backend.frozen.solver import hard_valid
from .backend.guard import repair_tiny_negative
from .fast_geometry import topology,reduce_edges,contact_tree,step
from .anchor_topology import repair
from .projection import solve
from .canonical import canonicalize
from scoring.evaluation import evaluate_case
import numpy as np


def main():
    seed_all();rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    folder=HERE/'runs/20261020_anchor_paths_plateau5_e17_diagnostic'
    identity={'sources':sources(),'indices':[21,127,275,447,456],'test_read':False,'timing':'shared-load diagnostic'}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);records=[]
    for index in identity['indices']:
        row=rows[index];case=row['case'];p=prepare_problem(*case.solver_inputs);seed=row['draft'].copy()
        w,h=_dimensions(p,seed[:,2]/seed[:,3],True);seed[:,2]=w;seed[:,3]=h
        start=time.perf_counter()
        for iteration in range(2):
            full=topology(seed,p.constraints,array=True)
            proposed,_=step(p,seed,reduce_edges(full,p.n),full_edges=full,seconds=.1,elastic=100000*(1+iteration/3))
            if proposed is not None:seed=proposed
        full=topology(seed,p.constraints,array=True);full,stats=repair(p,seed,full,seconds=.1)
        tree=contact_tree(seed,p,full)
        x,ps=solve(p,seed,reduce_edges(full,p.n),tree,seconds=.06,quadratic=True,boundary=20.,contact=20.)
        if x is not None:x=canonicalize(x,p,tolerance=1e-4);x,_=repair_tiny_negative(x,p)
        valid=x is not None and hard_valid(x,p);elapsed=time.perf_counter()-start
        metrics=evaluate_case(case,x.tolist(),elapsed)['metrics'] if valid else None
        records.append({'index':index,'stats':stats,'projection':ps,'seconds':elapsed,'metrics':metrics,'legal':valid})
        print(index,stats,ps,metrics['cost'] if valid else 'failed',flush=True)
    save(folder/'results.json',{'records':records,'test_read':False})


if __name__=='__main__':main()
