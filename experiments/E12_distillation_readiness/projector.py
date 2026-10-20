"""Small fixed-shape projection and separation-aware exact contact arithmetic."""
import time
import math
import numpy as np
import highspy
from scipy.sparse import coo_matrix
from .backend.frozen.packing import prepare_problem
from .backend.frozen.global_refine import topology,contact_tree
from .backend.frozen.feasible import Objective,row_control
from .backend.frozen.solver import hard_valid


def lp_solve(costs,bounds,rows,lower,upper,seconds):
    start=time.perf_counter();ii=[];jj=[];vv=[]
    for r,row in enumerate(rows):
        for col,val in row.items():ii.append(r);jj.append(col);vv.append(val)
    matrix=coo_matrix((vv,(ii,jj)),shape=(len(rows),len(costs))).tocsc()
    model=highspy.HighsLp();model.num_col_=len(costs);model.num_row_=len(rows)
    model.col_cost_=costs;model.col_lower_=[a for a,b in bounds]
    model.col_upper_=[highspy.kHighsInf if b is None else b for a,b in bounds]
    model.row_lower_=lower;model.row_upper_=upper
    model.a_matrix_.format_=highspy.MatrixFormat.kColwise
    model.a_matrix_.start_=matrix.indptr;model.a_matrix_.index_=matrix.indices;model.a_matrix_.value_=matrix.data
    solver=highspy.Highs();solver.setOptionValue('output_flag',False);solver.setOptionValue('threads',2)
    solver.setOptionValue('primal_feasibility_tolerance',1e-10);solver.setOptionValue('dual_feasibility_tolerance',1e-10)
    solver.setOptionValue('time_limit',max(.005,seconds-(time.perf_counter()-start)))
    solver.passModel(model);solver.run();solution=solver.getSolution();info=solver.getInfo()
    stats={'seconds':time.perf_counter()-start,'native_seconds':solver.getRunTime(),
           'status':solver.modelStatusToString(solver.getModelStatus()),'variables':len(costs),'rows':len(rows),
           'simplex_iterations':info.simplex_iteration_count}
    return (np.asarray(solution.col_value) if solution.value_valid else None),stats


def cleanup(pos,p):
    out=np.asarray(pos,dtype=float).copy();out[p.fixed|p.preplaced,2:]=p.specs[p.fixed|p.preplaced,2:]
    out[p.preplaced]=p.specs[p.preplaced]
    xy=out[:,:2];mask=(xy<0)&(xy>=-1e-9)&(~p.preplaced)[:,None];xy[mask]=0.
    return out


def separated_snap(pos,p,edges,seconds=.15,certificate=None):
    """Collapse dyadic near-contact equalities, then enforce all separations.

    A 1e-7 physical separation allowance is inside the unchanged 1e-6 hard
    overlap tolerance; returned geometry still must pass the strict guard.
    """
    begin=time.perf_counter();locked=p.fixed|p.preplaced
    public=[*p.specs[locked,2:].reshape(-1),*p.specs[p.preplaced,:2].reshape(-1)]
    exponent=max([36,*[float(v).as_integer_ratio()[1].bit_length()-1 for v in public]])
    quantum=2.**(-exponent);units=np.rint(pos/quantum).astype(np.int64)
    units[locked,2:]=np.rint(p.specs[locked,2:]/quantum).astype(np.int64)
    if np.max(np.abs(units))>=2**51:return None,{'reason':'grid_range_unsupported'}
    result=units.copy();traces=[];n=p.n
    for axis in (0,1):
        values=[int(v) for v in units[:,axis]]
        values.extend([min(values),max(int(units[i,axis]+units[i,axis+2]) for i in range(n))])
        parent=list(range(n+2));offset=[0]*(n+2)
        anchor={int(i):int(round(float(p.specs[i,axis])/quantum)) for i in np.flatnonzero(p.preplaced)}
        if abs(values[n]*quantum)<1e-6:anchor[n]=0
        def find(i):
            if parent[i]!=i:
                root,delta=find(parent[i]);offset[i]+=delta;parent[i]=root
            return parent[i],offset[i]
        def join(i,j,d):
            ri,di=find(i);rj,dj=find(j)
            if ri==rj:return dj-di==d
            shift=d+di-dj
            if ri in anchor and rj in anchor and anchor[rj]-anchor[ri]!=shift:return False
            parent[rj]=ri;offset[rj]=shift
            if rj in anchor:anchor[ri]=anchor[rj]-shift;del anchor[rj]
            return True
        events=[]
        if certificate is not None:
            for a,b,ax in certificate['contacts']:
                if ax==axis:events.append((0.,a,b,int(units[a,axis+2])))
        for i in range(n):
            group=p.constraints[i,3]
            for j in range(i):
                if certificate is not None or group<=0 or p.constraints[j,3]!=group:continue
                tangent=1-axis
                overlap=min(pos[i,tangent]+pos[i,tangent+2],pos[j,tangent]+pos[j,tangent+2])-max(pos[i,tangent],pos[j,tangent])
                if overlap<=1e-6:continue
                for a,b in ((i,j),(j,i)):
                    gap=pos[b,axis]-pos[a,axis]-pos[a,axis+2]
                    if abs(gap)<1e-6:events.append((abs(gap),a,b,int(units[a,axis+2])))
            lowerbit=1 if axis==0 else 8;upperbit=2 if axis==0 else 4
            for endpoint,delta,bit in ((n,0,lowerbit),(n+1,int(units[i,axis+2]),upperbit)):
                if not (p.constraints[i,4] if certificate is None else certificate['boundary'][i])&bit:continue
                gap=(values[endpoint]-values[i]-delta)*quantum
                if abs(gap)<1e-6:events.append((abs(gap),i,endpoint,delta))
        for _,i,j,d in sorted(events):join(i,j,d)
        components={}
        for i in range(n+2):
            root,delta=find(i);components.setdefault(root,[]).append((i,delta))
        roots=list(components);index={r:k for k,r in enumerate(roots)};count=len(roots)
        costs=[0.]*count;bounds=[];rows=[];lower=[];upper=[]
        for root in roots:
            members=components[root]
            lo=max([-d*quantum/p.scale for i,d in members if i<n]+[0. if any(i==n for i,d in members) else -highspy.kHighsInf])
            if root in anchor:v=anchor[root]*quantum/p.scale;bounds.append((v,v))
            else:bounds.append((lo,None))
        def add(row,rhs):rows.append(row);lower.append(-highspy.kHighsInf);upper.append(rhs)
        for root in roots:
            col=index[root];members=components[root]
            target=float(np.median([(values[i]-d)*quantum/p.scale for i,d in members]))
            dev=len(costs);costs.append(max(1,len(members))/n);bounds.append((0.,None))
            add({col:1.,dev:-1.},target);add({col:-1.,dev:-1.},-target)
        def separation(i,j,width):
            ri,di=find(i);rj,dj=find(j)
            rhs=((dj-di)*quantum-width+1e-7)/p.scale
            if ri==rj:
                if rhs < -1e-10:return False
            else:add({index[ri]:1.,index[rj]:-1.},rhs)
            return True
        feasible=True
        for i,j,a in edges:
            if a==axis and not separation(i,j,int(units[i,axis+2])*quantum):feasible=False;break
        for i in range(n):
            feasible=feasible and separation(n,i,0.) and separation(i,n+1,int(units[i,axis+2])*quantum)
        if not feasible:return None,{'reason':'contact_component_conflicts_with_separation','axis':axis}
        remaining=seconds-(time.perf_counter()-begin)
        if remaining<.005:return None,{'reason':'snap_budget_exhausted','trace':traces}
        solved,stats=lp_solve(costs,bounds,rows,lower,upper,remaining);traces.append(stats)
        if solved is None:return None,{'reason':'snap_lp_failed','trace':traces}
        for root,members in components.items():
            base=int(round(float(solved[index[root]])*p.scale/quantum))
            for i,d in members:
                if i<n:result[i,axis]=base+d
    out=cleanup(result.astype(float)*quantum,p)
    return out,{'seconds':time.perf_counter()-begin,'trace':traces,'max_abs_change':float(np.max(np.abs(out-pos))),
                'strict_valid':hard_valid(out,p)}


def position_lp(seed,p,edges,contacts,seconds=.25):
    begin=time.perf_counter();n=p.n;s=p.scale;z=seed/s;wh=z[:,2:];nsoft=Objective(p).nsoft
    costs=[];bounds=[];rows=[];lower=[];upper=[]
    def var(cost=0.,lo=0.,hi=None):
        k=len(costs);costs.append(cost);bounds.append((lo,hi));return k
    def add(row,rhs):rows.append(row);lower.append(-highspy.kHighsInf);upper.append(rhs)
    xy=np.array([[var(),var()] for _ in range(n)]);lo=[var(-.001),var(-.001)];hi=[var(.001),var(.001)]
    bblo=z[:,:2].min(0);bbhi=(z[:,:2]+wh).max(0);active_boundary=0
    for i in range(n):
        radius=.25*np.sqrt(p.areas[i])/s
        for a in (0,1):
            bounds[xy[i,a]]=(max(0.,z[i,a]-radius),z[i,a]+radius)
            if p.preplaced[i]:bounds[xy[i,a]]=(z[i,a],z[i,a])
            add({lo[a]:1.,xy[i,a]:-1.},0.);add({xy[i,a]:1.,hi[a]:-1.},-wh[i,a])
            d=var(1./n);add({xy[i,a]:1.,d:-1.},z[i,a]);add({xy[i,a]:-1.,d:-1.},-z[i,a])
        for bit,a,up in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
            if not p.constraints[i,4]&bit:continue
            gap=bbhi[a]-z[i,a]-wh[i,a] if up else z[i,a]-bblo[a]
            if abs(gap)>.1*np.sqrt(p.areas[i])/s:continue
            active_boundary+=1;d=var(12./nsoft)
            if up:add({hi[a]:1.,xy[i,a]:-1.,d:-1.},wh[i,a])
            else:add({xy[i,a]:1.,lo[a]:-1.,d:-1.},0.)
    for i,j,a in edges:add({xy[i,a]:1.,xy[j,a]:-1.},-wh[i,a]+1e-7/s)
    active_contacts=0
    for i,j,a in contacts:
        t=1-a;size=min(np.sqrt(p.areas[i]),np.sqrt(p.areas[j]))/s
        gap=abs(z[j,a]-z[i,a]-wh[i,a]);tgap=max(0.,z[i,t]-z[j,t]-wh[j,t],z[j,t]-z[i,t]-wh[i,t])
        if max(gap,tgap)>.1*size:continue
        active_contacts+=1;d=var(6./nsoft)
        add({xy[j,a]:1.,xy[i,a]:-1.,d:-1.},wh[i,a])
        shared=.01*min(wh[i,t],wh[j,t])
        add({xy[i,t]:1.,xy[j,t]:-1.,d:-1.},wh[j,t]-shared)
        add({xy[j,t]:1.,xy[i,t]:-1.,d:-1.},wh[i,t]-shared)
    solution,stats=lp_solve(costs,bounds,rows,lower,upper,max(.005,seconds-(time.perf_counter()-begin)))
    stats.update(active_contacts=active_contacts,active_boundary=active_boundary,total_seconds=time.perf_counter()-begin)
    if solution is None:return None,stats
    out=seed.copy();out[:,:2]=solution[xy]*s
    return cleanup(out,p),stats


def project(seed,inputs,seconds=.35,guidance=None,snap_mode='separated',joint=False,polished=False):
    begin=time.perf_counter();p=prepare_problem(*inputs);objective=Objective(p)
    seed=cleanup(np.asarray(seed,dtype=float),p)
    edges=topology(seed,p.constraints) if guidance is None else [tuple(v) for v in guidance['edges']]
    contacts=contact_tree(seed,p,edges) if guidance is None else [tuple(v) for v in guidance['contacts']]
    certificate=guidance if guidance is not None and guidance.get('certified') else None
    best=row_control(p);best_score=objective(best);traces=[];accepted=[]
    def offer(value,label):
        nonlocal best,best_score
        if value is None:return
        value=cleanup(value,p)
        if not hard_valid(value,p):return
        score=objective(value)
        if score<best_score-1e-10:
            best=value.copy();best_score=score;accepted.append({'phase':label,'elapsed':time.perf_counter()-begin})
    offer(seed,'seed')
    remaining=seconds-(time.perf_counter()-begin)
    if joint:
        from .joint import joint_step
        candidate,stats=joint_step(seed,p,edges,contacts,min(.20,max(.005,remaining*.6)),certificate=certificate,allow_polish=polished)
    else:candidate,stats=position_lp(seed,p,edges,contacts,min(.20,max(.005,remaining*.6)))
    traces.append({'phase':'joint_socp' if joint else 'position_lp',**stats})
    offer(candidate,'native_lp')
    if candidate is not None and polished and certificate is not None:
        from .polish import polish
        refined,stats=polish(candidate,p,edges,certificate,min(.1,max(.005,seconds-(time.perf_counter()-begin))))
        traces.append({'phase':'secant_polish',**stats});offer(refined,'secant_polish')
        if refined is not None:candidate=refined
    if candidate is not None:
        remaining=seconds-(time.perf_counter()-begin)
        if snap_mode=='legacy':
            from .backend.frozen.canonical_geometry import canonicalize
            start=time.perf_counter();snapped=canonicalize(candidate,p);stats={'seconds':time.perf_counter()-start}
        else:
            snapped,stats=separated_snap(candidate,p,edges,max(.005,remaining),certificate=certificate)
            # Independent control: legacy snapping is also eligible only when
            # it passes the unchanged complete hard guard.
            from .backend.frozen.canonical_geometry import canonicalize
            offer(canonicalize(candidate,p),'legacy_snap_control')
        traces.append({'phase':'contact_snap',**stats});offer(snapped,'contact_snap')
    assert hard_valid(best,p)
    return best,{'seconds':time.perf_counter()-begin,'proxy':float(best_score),'trace':traces,
                 'accepted':accepted,'row_retained':not accepted,'fixed_shape':not joint,'snap_mode':snap_mode,
                 'hard_valid':True,'joint':joint}
