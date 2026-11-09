"""A monotone map from convex lower-area geometry to the public area band."""
import numpy as np


def shrink(candidate,p):
    if candidate is None:return None
    out=np.array(candidate,copy=True);locked=p.fixed|p.preplaced;covered=np.zeros(p.n,bool)
    classes=[]
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group);covered[ids]=True;classes.append(ids)
    classes.extend(np.array([i]) for i in np.flatnonzero(~covered))
    for ids in classes:
        if locked[ids].any():continue
        actual=out[ids,2:].prod(1)
        # Public v5 mean-area transformation makes compatible MIB targets equal.
        # A common scale keeps the existing shape ties; individual singletons
        # are the same rule. Clip at1 so no new overlap can be introduced.
        factor=min(1.,float(np.min(p.areas[ids]/actual)))
        out[ids,2:]*=np.sqrt(factor)
    return out
