"""Large aspect proposals followed by exact fixed-shape soft-constraint restoration."""
import time
import numpy as np
from .fast_geometry import step,topology,reduce_edges,contact_tree
from .projection import solve as project

def refine(archive,deadline,seconds=1.):
    p=archive.p;stop=min(deadline,time.perf_counter()+seconds);trace=[];attempt=0
    while stop-time.perf_counter()>.12 and attempt<5:
        rows=[r for r in archive.rows if sum(r['soft_violations'])==0] or archive.rows
        row=min(rows,key=lambda r:r['public_quality_cost']+.01*r['distance']['D']);seed=np.asarray(row['positions'])
        full=topology(seed,p.constraints,array=True);edges=reduce_edges(full,p.n);tree=contact_tree(seed,p,full)
        y,info=step(p,seed,edges,full_edges=full,seconds=min(.25,stop-time.perf_counter()-.035),
            elastic=0.,conservative=False,reweighted=False,hard_boundary=True,hard_contact=True,
            anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale),
            draft_reference=archive.draft,distance_weight=.003,shape_weight=[0.,.03,.3,.01,.1][attempt])
        accepted=archive.accept(y,f'aggressive_shape/{attempt}')
        trace.append({'phase':'aggressive_shape','attempt':attempt,'accepted':accepted is not None,**info})
        if y is not None and stop-time.perf_counter()>.075:
            z,stats=project(p,y,edges,tree,seconds=min(.24,stop-time.perf_counter()-.025),
                anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale),draft_reference=archive.draft,distance_weight=.003,
                engine='highs',boundary_targets=np.flatnonzero(p.constraints[:,4]),contact_targets=[t[:2] for t in tree])
            accepted=archive.accept(z,f'aggressive_restore/{attempt}')
            trace.append({'phase':'aggressive_restore','attempt':attempt,'accepted':accepted is not None,**stats})
        attempt+=1
    return trace
