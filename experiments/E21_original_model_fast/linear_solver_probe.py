"""Five-case native linear algebra comparison on identical public SOCPs."""
import torch
import numpy as np
from .io import ROOT,HERE,save,sources,snapshot
from .cases import seed_all
from .backend.frozen.packing import prepare_problem,_dimensions
from .fast_geometry import topology,reduce_edges,step


def main():
    seed_all();rows=torch.load(ROOT/'experiments/E18_fast_postprocess/runs/public_dev500_v1.pt',weights_only=False)['rows']
    folder=HERE/'runs/20261020_linear_solver5_e17_diagnostic'
    identity={'sources':sources(),'indices':[0,164,366,104,447],'seconds_per_step':1.,'test_read':False,'timing':'shared-load native diagnostic'}
    save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);records=[]
    for index in identity['indices']:
        row=rows[index];p=prepare_problem(*row['case'].solver_inputs);state=row['draft'].copy()
        w,h=_dimensions(p,state[:,2]/state[:,3],True);state[:,2]=w;state[:,3]=h
        full=topology(state,p.constraints,array=True);edges=reduce_edges(full,p.n);previous=None
        for method in ['qdldl','faer']:
            x,stats=step(p,state,edges,full_edges=full,seconds=1.,linear_solver=method)
            error=float(np.abs(x-previous).max()) if previous is not None and x is not None else None
            if x is not None:previous=x
            records.append({'index':index,'n':p.n,'method':method,'stats':stats,'maximum_absolute_difference':error})
            print(index,p.n,method,stats['status'],stats['native_seconds'],'difference',error,flush=True)
    save(folder/'results.json',{'records':records,'test_read':False})


if __name__=='__main__':main()
