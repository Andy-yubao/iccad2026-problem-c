"""Public mean-area compatibility and separately derived supervision."""
import numpy as np
import torch


def compatible_areas(areas, constraints, public_wh):
    a=np.asarray(areas,dtype=np.float32).copy()
    c=np.asarray(constraints)
    wh=np.asarray(public_wh,dtype=np.float32)
    locked=(c[:,:2]!=0).any(1)
    if not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('Malformed public areas')
    for label in np.unique(c[:,2]):
        if label<=0:
            continue
        ids=np.flatnonzero(c[:,2]==label)
        mean=float(a[ids].astype(np.float64).mean())
        rigid=ids[locked[ids]]
        if len(rigid):
            if not np.allclose(wh[rigid],wh[rigid[0]],rtol=0,atol=1e-10):
                return None,'locked_dimensions_disagree'
            if abs(float(wh[rigid[0]].astype(np.float64).prod())/mean-1)>=.009999:
                return None,'mean_area_incompatible_with_locked_shape'
        a[ids]=mean
    return a,None


def supervision(original, areas, constraints, specs):
    """Teacher-only shape correction. No result enters public input features."""
    p=np.asarray(original,dtype=np.float32)
    c=np.asarray(constraints)
    spec=np.asarray(specs)
    a=np.asarray(areas,dtype=np.float64)
    center=p[:,:2].astype(np.float64)+p[:,2:].astype(np.float64)/2
    ratio=np.log(p[:,2].astype(np.float64)/p[:,3])
    locked=(c[:,:2]!=0).any(1)
    original_mib_disagreement=0
    for label in np.unique(c[:,2]):
        if label<=0:
            continue
        ids=np.flatnonzero(c[:,2]==label)
        original_mib_disagreement+=int(np.ptp(ratio[ids])>1e-6)
        rigid=ids[locked[ids]]
        value=float(np.log(spec[rigid[0],2]/spec[rigid[0],3])) if len(rigid) else float(ratio[ids].mean())
        ratio[ids]=value
    wh=np.column_stack((np.sqrt(a*np.exp(ratio)),np.sqrt(a/np.exp(ratio))))
    wh[locked]=spec[locked,2:]
    # Compatible MIB classes with immutable members share those exact WH.
    for label in np.unique(c[:,2]):
        if label<=0:
            continue
        ids=np.flatnonzero(c[:,2]==label)
        rigid=ids[locked[ids]]
        if len(rigid):
            wh[ids]=spec[rigid[0],2:]
    q=np.column_stack((center-wh/2,wh)).astype(np.float32)
    pre=c[:,1]!=0
    q[pre]=spec[pre]
    lo=q[:,:2].astype(np.float64);hi=lo+q[:,2:]
    overlap=np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None])
    collides=(overlap[...,0]>1e-7)&(overlap[...,1]>1e-7)
    np.fill_diagonal(collides,False)
    return torch.from_numpy(q),dict(original_mib_shape_disagreement_groups=original_mib_disagreement,
        corrected_overlap_pairs=int(np.triu(collides,1).sum()),
        corrected_negative_modules=int((lo < -1e-7).any(1).sum()),
        changed_shape_modules=int(np.any(np.abs(q[:,2:]-p[:,2:])>1e-5,axis=1).sum()))
