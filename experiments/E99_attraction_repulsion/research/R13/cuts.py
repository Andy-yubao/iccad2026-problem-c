"""Cut choices use one current snapshot; final validity checks remain complete."""
import numpy as np
from ...planner.fast_geometry import contact_tree


def update(state,p,full,cuts,removed,initial):
    a,b,axis=full.T
    overlap=np.minimum(state[a,:2]+state[a,2:],state[b,:2]+state[b,2:])-np.maximum(state[a,:2],state[b,:2])
    active=(overlap>1e-6).all(1)
    if initial:active|=p.preplaced[a]|p.preplaced[b]
    for index in np.flatnonzero(active):
        i,j,c=map(int,full[index]);key=(min(i,j),max(i,j))
        if key not in removed and key not in cuts:cuts[key]=(i,j,c)
    if initial:
        for i,j,c in contact_tree(state,p,full):
            key=(min(i,j),max(i,j))
            if key not in removed:cuts.setdefault(key,(i,j,c))
    return np.asarray(list(cuts.values()),dtype=int).reshape(-1,3)
