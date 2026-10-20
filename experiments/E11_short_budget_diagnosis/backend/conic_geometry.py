"""Public-input SOCP geometry; exact convex area lower bounds in one solve.

The conservative mode retains the frozen shape step's upper tangent/trust
region. Relaxed mode drops the upper tangent, then shrinks compatible classes
uniformly. No reference layout or reference score enters either optimization.
"""
import time
import numpy as np
import clarabel
from scipy.sparse import coo_matrix, csc_matrix
from .frozen.feasible import Objective, row_control
from .frozen.packing import prepare_problem, _dimensions
from .frozen.global_refine import topology, contact_tree, solve_lp
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .reduction import reduce_edges
from .observe import invalid_reasons


def step(p, seed, edges, seconds=3., boundary=12., contact=6.,
         elastic=100000., iteration=0, reweighted=True, conservative=True,
         quadratic_wire=False,diagonal_wire=False,guidance=None):
    begin=time.perf_counter();n=p.n;s=p.scale;z=seed/s;wh=z[:,2:]
    areas=p.areas/s**2;locked=p.fixed|p.preplaced
    costs=[];bounds=[];rows=[];rhs=[];equal=[];equal_rhs=[]
    nsoft=Objective(p).nsoft
    def var(cost=0.,lo=0.,hi=None):
        k=len(costs);costs.append(cost);bounds.append((lo,hi));return k
    def le(row,b):rows.append(row);rhs.append(b)
    def eq(row,b):equal.append(row);equal_rhs.append(b)
    xy=np.array([[var(),var()] for _ in range(n)])
    dims=np.array([[var(0.,wh[i,a]*.4,wh[i,a]*2.5) for a in (0,1)] for i in range(n)])
    for i in range(n):
        if locked[i]:
            for a in (0,1):bounds[dims[i,a]]=(wh[i,a],wh[i,a])
        elif conservative:
            le({dims[i,0]:wh[i,1],dims[i,1]:wh[i,0]},2*np.sqrt(wh[i].prod()*areas[i]*1.009999))
    # Same compatible classes as the baseline; union bookkeeping also guarantees
    # uniform shrinking will not destroy equal dimensions in relaxed mode.
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group)
        for index,i in enumerate(ids):
            for j in ids[:index]:
                if np.allclose(wh[i],wh[j],rtol=0,atol=1e-10):
                    for a in (0,1):eq({dims[i,a]:1,dims[j,a]:-1},0.)
                    parent[root(i)]=root(j);break
    lo=np.array([var(-1.),var(-1.)]);hi=np.array([var(1.),var(1.)])
    for i in range(n):
        for a in (0,1):
            le({lo[a]:1,xy[i,a]:-1},0.)
            le({xy[i,a]:1,dims[i,a]:1,hi[a]:-1},0.)
            if p.preplaced[i]:bounds[xy[i,a]]=(z[i,a],z[i,a])
            d=var(.005/n)
            le({xy[i,a]:1,d:-1},z[i,a]);le({xy[i,a]:-1,d:-1},-z[i,a])
        if p.constraints[i,4]:
            cost=boundary/nsoft
            if reweighted:
                bblo=z[:,:2].min(0);bbhi=(z[:,:2]+wh).max(0)
                distance=[z[i,0]-bblo[0],bbhi[0]-z[i,0]-wh[i,0],bbhi[1]-z[i,1]-wh[i,1],z[i,1]-bblo[1]]
                prior=max(distance[k] for k,bit in enumerate([1,2,4,8]) if p.constraints[i,4]&bit)
                cost=2./nsoft/(max(prior,0.)+.02*np.sqrt(wh[i].prod()))
            penalty=var(cost)
            for bit,a,upper in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
                if not p.constraints[i,4]&bit:continue
                if upper:le({hi[a]:1,xy[i,a]:-1,dims[i,a]:-1,penalty:-1},0.)
                else:le({xy[i,a]:1,lo[a]:-1,penalty:-1},0.)
    slack=[]
    for i,j,a in edges:
        terms={xy[i,a]:1,dims[i,a]:1,xy[j,a]:-1}
        if elastic:
            e=var(elastic/n);slack.append(e);terms[e]=-1
        le(terms,0.)
    norm=max(p.total_wire_weight,1e-12)
    wire_rows=[];wire_targets=[];wire_weights=[]
    centers=z[:,:2]+wh/2
    def wire_term(terms,target,weight,prior):
        if quadratic_wire:
            # IRLS is a local quadratic majorizer of smoothed Manhattan length.
            local_weight=.5*weight/np.sqrt(prior**2+.02**2)
            if diagonal_wire and len(terms)==4:
                # (u-v)^2 <= 2(u-m)^2+2(v-m)^2, equality and
                # gradient match at m=(u_old+v_old)/2. This avoids a dense
                # Hessian while retaining an actual majorization step.
                positive={k:v for k,v in terms.items() if v>0}
                negative={k:-v for k,v in terms.items() if v<0}
                values=np.zeros(len(costs));values[xy]=z[:,:2];values[dims]=wh
                mid=.5*(sum(values[k]*v for k,v in positive.items())+sum(values[k]*v for k,v in negative.items()))
                wire_rows.extend([positive,negative]);wire_targets.extend([mid,mid])
                wire_weights.extend([2*local_weight,2*local_weight])
            else:
                wire_rows.append(terms);wire_targets.append(target);wire_weights.append(local_weight)
        else:
            d=var(weight)
            le({**terms,d:-1.},target)
            le({**{k:-v for k,v in terms.items()},d:-1.},-target)
    bi,bj=np.triu_indices(n,1)
    for i,j in zip(bi,bj):
        w=p.adjacency[i,j]
        if not w:continue
        for a in (0,1):
            wire_term({xy[i,a]:1,dims[i,a]:.5,xy[j,a]:-1,dims[j,a]:-.5},
                      0.,w/norm,centers[i,a]-centers[j,a])
    for pin,i,w in p.p2b:
        pin,i=int(pin),int(i)
        for a in (0,1):
            target=p.pins[pin,a]/s
            wire_term({xy[i,a]:1,dims[i,a]:.5},target,w/norm,centers[i,a]-target)
    # The tree needs a direction for every pair even if hard separations reduced.
    tree=contact_tree(seed,p,topology(seed,p.constraints)) if guidance is None else guidance['contacts']
    for i,j,a in tree:
        t=1-a;cost=contact/nsoft
        if reweighted:
            gap=max(0.,z[j,a]-z[i,a]-wh[i,a],z[i,t]-z[j,t]-wh[j,t],z[j,t]-z[i,t]-wh[i,t])
            cost=2./nsoft/(gap+.02*min(np.sqrt(wh[i].prod()),np.sqrt(wh[j].prod())))
        d=var(cost)
        le({xy[j,a]:1,xy[i,a]:-1,dims[i,a]:-1,d:-1},0.)
        shared=.02*min(wh[i,t],wh[j,t])
        le({xy[i,t]:1,xy[j,t]:-1,dims[j,t]:-1,d:-1},-shared)
        le({xy[j,t]:1,xy[i,t]:-1,dims[i,t]:-1,d:-1},-shared)
    for k,(lower,upper) in enumerate(bounds):
        if upper is not None and lower==upper:eq({k:1.},lower)
        else:
            if lower is not None:le({k:-1.},-lower)
            if upper is not None:le({k:1.},upper)
    allrows=equal+rows;allrhs=equal_rhs+rhs
    cones=[clarabel.ZeroConeT(len(equal)),clarabel.NonnegativeConeT(len(rows))]
    for i in np.flatnonzero(~locked):
        # A*x+s=b: (s0,s1,s2)=(w+h,w-h,2 sqrt(alo)).
        allrows.extend([{dims[i,0]:-1.,dims[i,1]:-1.},
                        {dims[i,0]:-1.,dims[i,1]:1.},{}])
        allrhs.extend([0.,0.,2*np.sqrt(areas[i]*.990001)])
        cones.append(clarabel.SecondOrderConeT(3))
    ii=[];jj=[];data=[]
    for r,row in enumerate(allrows):
        for k,v in row.items():ii.append(r);jj.append(k);data.append(v)
    matrix=coo_matrix((data,(ii,jj)),shape=(len(allrows),len(costs))).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=2
    settings.time_limit=max(.005,seconds-(time.perf_counter()-begin));settings.max_iter=150
    settings.tol_feas=1e-10;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
    hessian=csc_matrix((len(costs),len(costs)));linear=np.asarray(costs)
    if wire_rows:
        wi=[];wj=[];wv=[]
        for r,row in enumerate(wire_rows):
            for k,v in row.items():wi.append(r);wj.append(k);wv.append(v)
        incidence=coo_matrix((wv,(wi,wj)),shape=(len(wire_rows),len(costs))).tocsc()
        weighted=incidence.multiply(np.asarray(wire_weights)[:,None])
        hessian=(2*incidence.T@weighted).tocsc()
        linear-=2*np.asarray(incidence.T@(np.asarray(wire_weights)*np.asarray(wire_targets))).reshape(-1)
    solver=clarabel.DefaultSolver(hessian,linear,
                                 matrix,np.asarray(allrhs),cones,settings)
    result=solver.solve()
    stats={'status':str(result.status),'iterations':result.iterations,
           'native_seconds':result.solve_time,'seconds':time.perf_counter()-begin,
           'variables':len(costs),'constraints':len(allrows),'conservative':conservative}
    if str(result.status) not in ('Solved','AlmostSolved','MaxTime') or result.r_prim>2e-6:
        return None,{**stats,'primal_residual':result.r_prim}
    values=np.asarray(result.x);out=np.column_stack((values[xy],values[dims]))*s
    out[locked,2:]=p.specs[locked,2:];out[p.preplaced]=p.specs[p.preplaced]
    # A uniform class scale preserves MIB and lower-left shrinking preserves hard
    # separations. The final guard still decides legality after elastic solves.
    for r in set(root(i) for i in range(n)):
        ids=np.array([i for i in range(n) if root(i)==r]);free=ids[~locked[ids]]
        if not len(free):continue
        actual=out[free,2:].prod(1)
        lower=float(np.max(p.areas[free]*.9900001/actual))
        upper=float(np.min(p.areas[free]*1.009999/actual))
        if locked[ids].any():factor=1.
        elif lower<=upper:factor=float(np.clip(1.,lower,upper))
        else:factor=1.  # Strict guard rejects inconsistent classes.
        out[free,2:]*=np.sqrt(factor)
    stats.update(elastic_sum=float(values[slack].sum()) if slack else 0.,lp_objective=float(result.obj_val))
    return out,stats


def optimize(raw,inputs,seconds=1.3,conservative=True,reduced=True,quadratic_wire=False,diagonal_wire=False,guidance=None):
    begin=time.perf_counter();deadline=begin+seconds
    p=prepare_problem(*inputs);objective=Objective(p);best=row_control(p)
    value=objective(best);trace=[];events=[]
    def accept(candidate,label):
        nonlocal best,value
        if candidate is None:return False
        candidate=canonicalize(candidate,p)
        movable=~p.preplaced
        xy=candidate[:,:2];mask=(xy<0)&(xy>=-1e-9)&movable[:,None];xy[mask]=0.
        if not hard_valid(candidate,p):return False
        score=objective(candidate)
        if score<value-1e-10:
            best=candidate.copy();value=score
            events.append({'elapsed':time.perf_counter()-begin,'proxy':float(value),'phase':label,'positions':best.tolist()})
        return True
    accept(raw,'raw')
    dims=_dimensions(p,np.asarray(raw)[:,2]/np.asarray(raw)[:,3],True)
    state=np.column_stack((np.asarray(raw)[:,:2],*dims))
    for iteration in range(24):
        remaining=deadline-time.perf_counter()-.025
        if remaining<.08:break
        full=topology(state,p.constraints) if guidance is None else guidance['edges']
        edges=reduce_edges(full,p.n) if reduced else full
        candidate,stats=step(p,state,edges,seconds=max(.01,remaining-.08),
                             conservative=conservative,elastic=100000*(1+iteration/3),
                             quadratic_wire=quadratic_wire,diagonal_wire=diagonal_wire,guidance=guidance)
        legal=accept(candidate,'conic_shape')
        trace.append({'phase':'conic_shape','iteration':iteration,'edges':len(edges),'legal':legal,**stats,
                      'rejection':invalid_reasons(candidate,p) if candidate is not None and not legal else None})
        if candidate is None:break
        state=candidate
        remaining=deadline-time.perf_counter()-.025
        if remaining<.025:break
        if quadratic_wire and iteration%4!=3:continue
        # Projection also restores exact contacts lost in the interior solver.
        projection_edges=topology(candidate,p.constraints) if guidance is None else guidance['edges']
        projected,stats=solve_lp(p,candidate,reduce_edges(projection_edges,p.n),
                                contact=0.,limit=min(.3,remaining))
        legal=accept(projected,'conic_projection')
        trace.append({'phase':'projection','iteration':iteration,'legal':legal,**stats})
        if legal:state=projected
    if not hard_valid(best,p):raise AssertionError('Invalid conic incumbent')
    return best,{'seconds':time.perf_counter()-begin,'proxy':float(value),'trace':trace,
                 'guard_fallback':False,'improvements':events}
