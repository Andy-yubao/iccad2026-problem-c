"""One local SOCP projection; no wire optimization or iterative topology search."""
import time
import numpy as np
import clarabel
from scipy.sparse import coo_matrix,csc_matrix
from .backend.frozen.feasible import Objective


def joint_step(seed,p,edges,contacts,seconds=.2,certificate=None,allow_polish=False):
    begin=time.perf_counter();n=p.n;s=p.scale;z=seed/s;wh=z[:,2:];areas=p.areas/s**2
    locked=p.fixed|p.preplaced;nsoft=Objective(p).nsoft
    costs=[];bounds=[];rows=[];rhs=[];equal=[];equal_rhs=[]
    def var(cost=0.,lo=0.,hi=None):k=len(costs);costs.append(cost);bounds.append((lo,hi));return k
    def le(row,b):rows.append(row);rhs.append(b)
    def eq(row,b):equal.append(row);equal_rhs.append(b)
    xy=np.array([[var(),var()] for _ in range(n)])
    dims=np.array([[var(0.,wh[i,a]*.8,wh[i,a]*1.25) for a in (0,1)] for i in range(n)])
    lo=[var(-.001),var(-.001)];hi=[var(.001),var(.001)]
    for i in range(n):
        if locked[i]:
            for a in (0,1):bounds[dims[i,a]]=(wh[i,a],wh[i,a])
        else:le({dims[i,0]:wh[i,1],dims[i,1]:wh[i,0]},2*np.sqrt(wh[i].prod()*areas[i]*1.0099999))
        radius=.25*np.sqrt(areas[i])
        for a in (0,1):
            bounds[xy[i,a]]=(max(0.,z[i,a]-radius),z[i,a]+radius)
            if p.preplaced[i]:bounds[xy[i,a]]=(z[i,a],z[i,a])
            le({lo[a]:1.,xy[i,a]:-1.},0.)
            le({xy[i,a]:1.,dims[i,a]:1.,hi[a]:-1.},0.)
            d=var(1./n);le({xy[i,a]:1.,d:-1.},z[i,a]);le({xy[i,a]:-1.,d:-1.},-z[i,a])
            if not locked[i]:
                d=var(10./n);le({dims[i,a]:1.,d:-1.},wh[i,a]);le({dims[i,a]:-1.,d:-1.},-wh[i,a])
    # Preserve classes already coordinated by the public MIB decoder.
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group)
        for k,i in enumerate(ids):
            for j in ids[:k]:
                if np.allclose(wh[i],wh[j],rtol=0.,atol=1e-12):
                    for a in (0,1):eq({dims[i,a]:1.,dims[j,a]:-1.},0.)
                    break
    for i,j,a in edges:le({xy[i,a]:1.,dims[i,a]:1.,xy[j,a]:-1.},1e-8/s)
    bblo=z[:,:2].min(0);bbhi=(z[:,:2]+wh).max(0);active_boundary=0;active_contacts=0
    for i in range(n):
        for bit,a,up in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
            if not p.constraints[i,4]&bit:continue
            if certificate is not None:
                if not certificate['boundary'][i]&bit:continue
                active_boundary+=1
                if up:le({hi[a]:1.,xy[i,a]:-1.,dims[i,a]:-1.},8e-7/s)
                else:le({xy[i,a]:1.,lo[a]:-1.},8e-7/s)
                continue
            gap=bbhi[a]-z[i,a]-wh[i,a] if up else z[i,a]-bblo[a]
            if abs(gap)>.1*np.sqrt(areas[i]):continue
            active_boundary+=1;d=var(12./nsoft)
            if up:le({hi[a]:1.,xy[i,a]:-1.,dims[i,a]:-1.,d:-1.},0.)
            else:le({xy[i,a]:1.,lo[a]:-1.,d:-1.},0.)
    for i,j,a in contacts:
        if certificate is not None:
            t=1-a;active_contacts+=1
            eq({xy[j,a]:1.,xy[i,a]:-1.,dims[i,a]:-1.},0.)
            le({xy[i,t]:1.,xy[j,t]:-1.,dims[j,t]:-1.},-1e-7/s)
            le({xy[j,t]:1.,xy[i,t]:-1.,dims[i,t]:-1.},-1e-7/s)
            continue
        t=1-a;size=min(np.sqrt(areas[i]),np.sqrt(areas[j]))
        gap=abs(z[j,a]-z[i,a]-wh[i,a]);tgap=max(0.,z[i,t]-z[j,t]-wh[j,t],z[j,t]-z[i,t]-wh[i,t])
        if max(gap,tgap)>.1*size:continue
        active_contacts+=1;d=var(6./nsoft)
        le({xy[j,a]:1.,xy[i,a]:-1.,dims[i,a]:-1.,d:-1.},0.)
        shared=.01*min(wh[i,t],wh[j,t])
        le({xy[i,t]:1.,xy[j,t]:-1.,dims[j,t]:-1.,d:-1.},-shared)
        le({xy[j,t]:1.,xy[i,t]:-1.,dims[i,t]:-1.,d:-1.},-shared)
    for k,(low,up) in enumerate(bounds):
        if up is not None and low==up:eq({k:1.},low)
        else:
            if low is not None:le({k:-1.},-low)
            if up is not None:le({k:1.},up)
    allrows=equal+rows;allrhs=equal_rhs+rhs
    cones=[clarabel.ZeroConeT(len(equal)),clarabel.NonnegativeConeT(len(rows))]
    for i in np.flatnonzero(~locked):
        allrows.extend([{dims[i,0]:-1.,dims[i,1]:-1.},{dims[i,0]:-1.,dims[i,1]:1.},{}])
        allrhs.extend([0.,0.,2*np.sqrt(areas[i]*.9900001)]);cones.append(clarabel.SecondOrderConeT(3))
    ii=[];jj=[];vv=[]
    for r,row in enumerate(allrows):
        for k,v in row.items():ii.append(r);jj.append(k);vv.append(v)
    matrix=coo_matrix((vv,(ii,jj)),shape=(len(allrows),len(costs))).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=2
    settings.time_limit=max(.005,seconds-(time.perf_counter()-begin));settings.max_iter=120
    settings.tol_feas=1e-11;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10
    solver=clarabel.DefaultSolver(csc_matrix((len(costs),len(costs))),np.asarray(costs),matrix,np.asarray(allrhs),cones,settings)
    result=solver.solve();stats={'seconds':time.perf_counter()-begin,'native_seconds':result.solve_time,
        'status':str(result.status),'variables':len(costs),'rows':len(allrows),'iterations':result.iterations,
        'active_contacts':active_contacts,'active_boundary':active_boundary,'primal_residual':result.r_prim}
    if result.x is None or str(result.status) not in ('Solved','AlmostSolved','MaxTime') or result.r_prim>(1e-4 if allow_polish else 2e-7):return None,stats
    values=np.asarray(result.x);out=np.column_stack((values[xy],values[dims]))*s
    out[locked,2:]=p.specs[locked,2:];out[p.preplaced]=p.specs[p.preplaced]
    return out,stats
