"""Mixed-integer separation/contact and dynamic-boundary optimization.

All constants are public input or draft-derived. No scoring references allowed.
"""
import math,time
import numpy as np
import highspy
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from .feasible import Objective
from .global_refine import topology,contact_tree


def optimize(p,seed,seconds=20.,freedom=.4,boundary_weight=8.,contact_weight=4.,shape=False):
    n=p.n;scale=p.scale;z=seed/scale;wh=z[:,2:];codes=p.constraints[:,4]
    maxdim=max(3.,float((z[:,:2]+wh).max())*1.5);big=2*maxdim
    costs=[];low=[];up=[];integer=[];rows=[];rhs=[]
    nsoft=Objective(p).nsoft
    def var(cost=0.,lo=0.,hi=maxdim,binary=False):
        k=len(costs);costs.append(cost);low.append(lo);up.append(hi);integer.append(int(binary));return k
    def le(row,b):rows.append(row);rhs.append(b)
    def eq(row,b):le(row,b);le({k:-v for k,v in row.items()},-b)
    xy=np.array([[var(),var()] for _ in range(n)])
    # Fixed-size first; shape variation is a bounded first-order area step.
    dw=[]
    for i in range(n):dw.append(var(0.,-.15,.15) if shape and not(p.fixed[i] or p.preplaced[i]) else None)
    size=lambda i,a:wh[i,a]*(1 if a==0 else -1)
    def size_term(row,i,a,factor=1.):
        if dw[i] is not None:row[dw[i]]=row.get(dw[i],0)+factor*size(i,a)
    lo=np.array([var(-1.),var(-1.)]);hi=np.array([var(1.),var(1.)])
    for i in range(n):
        for a in (0,1):
            le({lo[a]:1,xy[i,a]:-1},0)
            r={xy[i,a]:1,hi[a]:-1};size_term(r,i,a);le(r,-wh[i,a])
            if p.preplaced[i]:low[xy[i,a]]=up[xy[i,a]]=z[i,a]
            d=var(.02/n,hi=2*maxdim);le({xy[i,a]:1,d:-1},z[i,a]);le({xy[i,a]:-1,d:-1},-z[i,a])
        if codes[i]:
            satisfied=var(-boundary_weight/nsoft,hi=1,binary=True)
            for bit,a,upper in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
                if not codes[i]&bit:continue
                if upper:
                    r={hi[a]:1,xy[i,a]:-1,satisfied:big};size_term(r,i,a,-1);le(r,big+wh[i,a])
                else:le({xy[i,a]:1,lo[a]:-1,satisfied:big},big)
    selected=topology(seed,p.constraints,0);direction_vars={};fixed=0;flexible=0;initial_binary={}
    for i,j,a in selected:
        center=z[:,:2]+wh/2;delta=center[j]-center[i];margin=np.abs(delta)-(wh[i]+wh[j])/2
        local=(np.sqrt(np.prod(wh[i]))+np.sqrt(np.prod(wh[j])))/2
        # Keep only confidently separated pairs fixed. Boundary pairs can swap.
        ambiguous=margin.max()<freedom*local
        if not ambiguous:
            r={xy[i,a]:1,xy[j,a]:-1};size_term(r,i,a);le(r,-wh[i,a]);fixed+=1;continue
        flexible+=1;opts=[]
        for u,v,axis in [(i,j,0),(j,i,0),(i,j,1),(j,i,1)]:
            b=var(0.,hi=1,binary=True);opts.append(b);direction_vars[u,v,axis]=b
            initial_binary[b]=int((u,v,axis)==(i,j,a))
            r={xy[u,axis]:1,xy[v,axis]:-1,b:big};size_term(r,u,axis);le(r,big-wh[u,axis])
        eq({b:1 for b in opts},1)
    norm=max(p.total_wire_weight,1e-12)
    # Consolidated adjacency avoids duplicate directed input edges.
    bi,bj=np.triu_indices(n,1)
    for i,j in zip(bi,bj):
        w=p.adjacency[i,j]
        if not w:continue
        for a in (0,1):
            d=var(w/norm,hi=2*maxdim);offset=(wh[i,a]-wh[j,a])/2
            r={xy[i,a]:1,xy[j,a]:-1,d:-1};size_term(r,i,a,.5);size_term(r,j,a,-.5);le(r,-offset)
            r={xy[i,a]:-1,xy[j,a]:1,d:-1};size_term(r,i,a,-.5);size_term(r,j,a,.5);le(r,offset)
    for pin,i,w in p.p2b:
        pin,i=int(pin),int(i)
        for a in (0,1):
            d=var(w/norm,hi=2*maxdim);target=p.pins[pin,a]/scale-wh[i,a]/2
            r={xy[i,a]:1,d:-1};size_term(r,i,a,.5);le(r,target)
            r={xy[i,a]:-1,d:-1};size_term(r,i,a,-.5);le(r,-target)
    # A fixed candidate spanning tree limits redundant contact rewards.
    for i,j,preferred in contact_tree(seed,p,selected):
        options=[]
        for u,v,a in [(i,j,0),(j,i,0),(i,j,1),(j,i,1)]:
            sep=direction_vars.get((u,v,a))
            if sep is None and (u,v,a)!=(i,j,preferred):continue
            b=var(-contact_weight/nsoft,hi=1,binary=True);options.append(b)
            if sep is not None:le({b:1,sep:-1},0)
            r={xy[v,a]:1,xy[u,a]:-1,b:big};size_term(r,u,a,-1);le(r,big+wh[u,a])
            t=1-a;eps=.005*min(wh[u,t],wh[v,t])
            r={xy[u,t]:1,xy[v,t]:-1,b:big};size_term(r,v,t,-1);le(r,big+wh[v,t]-eps)
            r={xy[v,t]:1,xy[u,t]:-1,b:big};size_term(r,u,t,-1);le(r,big+wh[u,t]-eps)
        if options:le({b:1 for b in options},1)
    ii=[];jj=[];data=[]
    for r,row in enumerate(rows):
        for k,v in row.items():ii.append(r);jj.append(k);data.append(v)
    mat=coo_matrix((data,(ii,jj)),shape=(len(rows),len(costs))).tocsc()
    start=time.monotonic()
    solver=highspy.Highs();solver.setOptionValue('output_flag',False);solver.setOptionValue('threads',2)
    solver.setOptionValue('time_limit',seconds);solver.setOptionValue('mip_rel_gap',.02);solver.setOptionValue('random_seed',0)
    lp=highspy.HighsLp();lp.num_col_=len(costs);lp.num_row_=len(rows)
    lp.col_cost_=costs;lp.col_lower_=low;lp.col_upper_=up;lp.row_lower_=[-highspy.kHighsInf]*len(rows);lp.row_upper_=rhs
    lp.integrality_=[highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integer]
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=mat.indptr
    lp.a_matrix_.index_=mat.indices;lp.a_matrix_.value_=mat.data;solver.passModel(lp)
    hint={int(xy[i,a]):float(z[i,a]) for i in range(n) for a in (0,1)}
    hint.update({k:float(initial_binary.get(k,0)) for k in range(len(costs)) if integer[k]})
    hint.update({k:0. for k in dw if k is not None})
    solver.setSolution(len(hint),np.asarray(list(hint),dtype=np.int32),np.asarray(list(hint.values())))
    solver.run();solution=solver.getSolution();info=solver.getInfo()
    stats={'status':str(solver.modelStatusToString(solver.getModelStatus())),'seconds':time.monotonic()-start,'variables':len(costs),
        'binary':sum(integer),'fixed_pairs':fixed,'flexible_pairs':flexible}
    if not solution.value_valid:return None,stats
    values=np.asarray(solution.col_value)
    out=seed.copy();out[:,:2]=values[xy]*scale
    if shape:
        for i,v in enumerate(dw):
            if v is not None:out[i,2:]=seed[i,2:]*np.exp([values[v],-values[v]])
    out[p.preplaced]=p.specs[p.preplaced]
    stats['objective']=float(info.objective_function_value);stats['gap']=float(info.mip_gap)
    # Fixed-size MIP coordinates are repaired/snap-checked by caller.
    return out,stats
