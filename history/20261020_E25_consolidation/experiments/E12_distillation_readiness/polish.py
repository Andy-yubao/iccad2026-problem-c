"""One tiny shape/position LP: rigorous secant area bounds near SOCP output."""
import time
import numpy as np
from .projector import lp_solve,cleanup


def polish(pos,p,edges,certificate,seconds=.1):
    begin=time.perf_counter();s=p.scale;z=np.asarray(pos)/s;n=p.n
    locked=p.fixed|p.preplaced;costs=[];bounds=[];rows=[];lower=[];upper=[]
    def var(cost=0.,lo=0.,hi=None):
        k=len(costs);costs.append(cost);bounds.append((lo,hi));return k
    def le(row,b):rows.append(row);lower.append(-np.inf);upper.append(b)
    def eq(row,b):rows.append(row);lower.append(b);upper.append(b)
    xy=np.array([[var(),var()] for i in range(n)])
    wh=np.array([[var() for a in (0,1)] for i in range(n)])
    lo=[var(),var()];hi=[var(),var()]
    for i in range(n):
        for a in (0,1):
            bounds[xy[i,a]]=(max(0.,z[i,a]-.001),max(0.,z[i,a])+.001)
            bounds[wh[i,a]]=(z[i,a+2]*(1.-1e-5),z[i,a+2]*(1.+1e-5))
            if locked[i]:bounds[wh[i,a]]=(p.specs[i,a+2]/s,p.specs[i,a+2]/s)
            if p.preplaced[i]:bounds[xy[i,a]]=(p.specs[i,a]/s,p.specs[i,a]/s)
            le({lo[a]:1.,xy[i,a]:-1.},0.)
            le({xy[i,a]:1.,wh[i,a]:1.,hi[a]:-1.},0.)
            for k,target,weight in [(xy[i,a],z[i,a],1.),(wh[i,a],z[i,a+2],10.)]:
                d=var(weight/n);le({k:1.,d:-1.},target);le({k:-1.,d:-1.},-target)
        if not locked[i]:
            area=p.areas[i]/s**2;wl,wu=bounds[wh[i,0]]
            # Reciprocal secant lies ABOVE A/w throughout [wl,wu].
            c=area*.99000002/(wl*wu)
            le({wh[i,1]:-1.,wh[i,0]:-c},-c*(wl+wu))
            # Positive tangent halfspace is sufficient for product <= Amax.
            w,h=z[i,2:]
            le({wh[i,0]:h,wh[i,1]:w},2*np.sqrt(w*h*area*1.00999998))
    labels=np.asarray(certificate['mib_labels'])
    for label in set(labels):
        ids=np.flatnonzero(labels==label)
        for i in ids[1:]:
            for a in (0,1):eq({wh[int(ids[0]),a]:1.,wh[i,a]:-1.},0.)
    # Interior of the EXISTING physical 1e-6 hard-overlap tolerance. This
    # prevents floating point equality noise from breaking group adjacency.
    for i,j,a in edges:le({xy[i,a]:1.,wh[i,a]:1.,xy[j,a]:-1.},4e-7/s)
    for i,j,a in certificate['contacts']:
        eq({xy[i,a]:1.,wh[i,a]:1.,xy[j,a]:-1.},2e-7/s)
        t=1-a
        le({xy[i,t]:1.,xy[j,t]:-1.,wh[j,t]:-1.},-1e-7/s)
        le({xy[j,t]:1.,xy[i,t]:-1.,wh[i,t]:-1.},-1e-7/s)
    for i,mask in enumerate(certificate['boundary']):
        for bit,a,up in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
            if mask&bit:
                if up:le({hi[a]:1.,xy[i,a]:-1.,wh[i,a]:-1.},8e-7/s)
                else:le({xy[i,a]:1.,lo[a]:-1.},8e-7/s)
    # Scale small area/upper-tangent rows with every other row, preserving LP.
    for row,l,u,k in [(r,lower[k],upper[k],k) for k,r in enumerate(rows)]:
        norm=max(abs(v) for v in row.values())
        if norm and norm<.1:
            rows[k]={i:v/norm for i,v in row.items()};lower[k]=l/norm;upper[k]=u/norm
    x,stats=lp_solve(costs,bounds,rows,lower,upper,max(.005,seconds-(time.perf_counter()-begin)))
    stats.update(total_seconds=time.perf_counter()-begin,relative_shape_radius=1e-5,area_bound='secant_inner')
    if x is None:return None,stats
    out=cleanup(np.column_stack((x[xy],x[wh]))*s,p)
    return out,stats
