"""Frozen E1 correct+pack control, copied on2026-10-17; no new repair algorithm."""
import time
import numpy as np
from .packing import prepare_problem,pack,default_order
from .feasible import is_legal,row_control

def correct(raw, p, steps=12):
    """Bounded continuous overlap relaxation; immutable dimensions/anchors stay locked."""
    pos = raw.copy()
    for _ in range(steps):
        delta = np.zeros((p.n, 2))
        for i in range(p.n):
            for j in range(i):
                ov = np.minimum(pos[i, :2] + pos[i, 2:], pos[j, :2] + pos[j, 2:]) - np.maximum(pos[i, :2], pos[j, :2])
                if np.all(ov > 0):
                    axis = int(np.argmin(ov))
                    sign = 1 if pos[i, axis] + pos[i, axis+2]/2 >= pos[j, axis] + pos[j, axis+2]/2 else -1
                    mobile = int(not p.preplaced[i]) + int(not p.preplaced[j])
                    if mobile:
                        shift = sign * (ov[axis] + p.scale * 1e-9) / mobile
                        if not p.preplaced[i]: delta[i, axis] += shift
                        if not p.preplaced[j]: delta[j, axis] -= shift
        pos[:, :2] = np.maximum(0, pos[:, :2] + delta)
        pos[p.preplaced] = p.specs[p.preplaced]
    return pos



def finalize(raw,inputs,seconds=15):
    start=time.monotonic();deadline=start+seconds
    p=prepare_problem(*inputs)
    revised=correct(np.asarray(raw,dtype=float),p)
    centers=revised[:,:2]+revised[:,2:]/2
    ratios=revised[:,2]/revised[:,3]
    order=sorted(default_order(p).tolist(),key=lambda i:(round(centers[i,1]/max(p.scale*.15,1e-9)),centers[i,0],i))
    final=pack(p,order=order,ratios=ratios,target_width=p.scale,shape_trials=1,proposal_centers=centers,deadline=deadline-.1)
    fallback=not is_legal(final,p)
    if fallback:final=row_control(p)
    return final,{'fallback':fallback,'postprocess_seconds':time.monotonic()-start,
                  'raw_to_final_center_displacement_mean':float(np.linalg.norm(final[:,:2]+final[:,2:]/2-np.asarray(raw)[:,:2]-np.asarray(raw)[:,2:]/2,axis=1).mean())}
