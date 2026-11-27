"""Model-only global proposals; uncertainty defers to the original iteration."""
from dataclasses import replace
import numpy as np
from .confidence import Guidance
from .fast_geometry import topology
from .global_information import acyclic


def build(draft,p,probabilities,mode,seed):
    probabilities=np.asarray(probabilities,float);i,j=np.tril_indices(p.n,-1)
    if probabilities.shape!=(len(i),4) or not np.all(np.isfinite(probabilities)):raise ValueError('Bad model probabilities')
    base=Guidance.build(draft,p,'uniform');raw=topology(draft,p.constraints,array=True)
    code=probabilities.argmax(1);axis=code//2;first=np.where(code%2==0,i,j);second=np.where(code%2==0,j,i)
    proposed=np.column_stack((first,second,axis));original_code=raw[:,2]*2+(raw[:,0]!=i)
    original_confidence=probabilities[np.arange(len(i)),original_code];proposal_confidence=probabilities.max(1)
    if mode=='learned_shuffle':
        order=np.random.default_rng(seed).permutation(len(i));original_confidence=original_confidence[order];proposal_confidence=proposal_confidence[order]
    elif mode=='learned_geometric':
        def geometric(edges):
            a,b,axis=edges.T;gap=(draft[b,axis]-draft[a,axis]-draft[a,axis+2])/np.maximum((draft[a,axis+2]+draft[b,axis+2])/2,1e-8)
            return 1/(1+np.exp(-np.clip(4*gap,-20,20)))
        original_confidence=geometric(raw);proposal_confidence=geometric(proposed)
    if mode=='learned_uniform':
        selected=proposed.copy();active=np.ones(len(i),bool)
        if not acyclic(selected,p.n):active[:]=False
    else:
        keep=original_confidence>=.9;revise=(original_confidence<.5)&(proposal_confidence>=.9)
        selected=np.where(keep[:,None],raw,proposed);active=keep|revise
    for array in (selected,active,original_confidence):array.flags.writeable=False
    return replace(base,metric_enabled=False,contact_enabled=False,priority=False,commit=False,static_edges=selected,
        active_mask=active,topology_confidence=original_confidence,global_mode=mode)
