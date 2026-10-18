"""Sequential convex position/shape steps with area tolerance and immutable specs.

Area lower bound uses supporting hyperbola cuts; a tangent ellipse upper bound
guarantees product<=upper. Actual area and legality are independently checked.
"""
import numpy as np
import time
import highspy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from .frozen.feasible import Objective
from .frozen.global_refine import contact_tree, topology


def step(p,seed,edges,seconds=3.,boundary=12.,contact=6.,elastic=100.,iteration=0,reweighted=False,tangents=5):
    call_begin=time.monotonic()
    n=p.n;s=p.scale;z=seed/s;wh=z[:,2:];areas=p.areas/s**2
    costs=[];bounds=[];rows=[];rhs=[];nsoft=Objective(p).nsoft
    def var(cost=0.,lo=0.,hi=None):
        k=len(costs);costs.append(cost);bounds.append((lo,hi));return k
    def le(row,b):rows.append(row);rhs.append(b)
    def eq(row,b):le(row,b);le({k:-v for k,v in row.items()},-b)
    xy=np.array([[var(),var()] for _ in range(n)])
    dims=np.array([[var(0.,wh[i,a]*.4,wh[i,a]*2.5) for a in (0,1)] for i in range(n)])
    locked=p.fixed|p.preplaced
    for i in range(n):
        if locked[i]:
            for a in (0,1):bounds[dims[i,a]]=(wh[i,a],wh[i,a])
        else:
            a0=wh[i].prod();alo=areas[i]*.990001;ahi=areas[i]*1.009999
            le({dims[i,0]:wh[i,1],dims[i,1]:wh[i,0]},2*np.sqrt(a0*ahi))
            for w in wh[i,0]*np.exp(np.linspace(-1,1,tangents)):
                le({dims[i,0]:-alo/w**2,dims[i,1]:-1},-2*alo/w)
    # Preserve already-compatible MIB shape classes, including rigid members.
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group)
        for index,i in enumerate(ids):
            for j in ids[:index]:
                if np.allclose(wh[i],wh[j],rtol=0,atol=1e-10):
                    for a in (0,1):eq({dims[i,a]:1,dims[j,a]:-1},0)
                    break
    lo=np.array([var(-1.),var(-1.)]);hi=np.array([var(1.),var(1.)])
    for i in range(n):
        for a in (0,1):
            le({lo[a]:1,xy[i,a]:-1},0);le({xy[i,a]:1,dims[i,a]:1,hi[a]:-1},0)
            if p.preplaced[i]:bounds[xy[i,a]]=(z[i,a],z[i,a])
            d=var(.005/n);le({xy[i,a]:1,d:-1},z[i,a]);le({xy[i,a]:-1,d:-1},-z[i,a])
        if p.constraints[i,4]:
            cost=boundary/nsoft
            if reweighted:
                bblo=z[:,:2].min(0);bbhi=(z[:,:2]+wh).max(0)
                distances=[z[i,0]-bblo[0],bbhi[0]-z[i,0]-wh[i,0],bbhi[1]-z[i,1]-wh[i,1],z[i,1]-bblo[1]]
                prior=max(distances[k] for k,bit in enumerate([1,2,4,8]) if p.constraints[i,4]&bit)
                cost=2./nsoft/(max(prior,0)+.02*np.sqrt(wh[i].prod()))
            penalty=var(cost)
            for bit,a,upper in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
                if not p.constraints[i,4]&bit:continue
                if upper:le({hi[a]:1,xy[i,a]:-1,dims[i,a]:-1,penalty:-1},0)
                else:le({xy[i,a]:1,lo[a]:-1,penalty:-1},0)
    slack=[]
    for i,j,a in edges:
        e=var(elastic/n);slack.append(e);le({xy[i,a]:1,dims[i,a]:1,xy[j,a]:-1,e:-1},0)
    norm=max(p.total_wire_weight,1e-12)
    bi,bj=np.triu_indices(n,1)
    for i,j in zip(bi,bj):
        w=p.adjacency[i,j]
        if not w:continue
        for a in (0,1):
            d=var(w/norm)
            le({xy[i,a]:1,dims[i,a]:.5,xy[j,a]:-1,dims[j,a]:-.5,d:-1},0)
            le({xy[i,a]:-1,dims[i,a]:-.5,xy[j,a]:1,dims[j,a]:.5,d:-1},0)
    for pin,i,w in p.p2b:
        pin,i=int(pin),int(i)
        for a in (0,1):
            d=var(w/norm);target=p.pins[pin,a]/s
            le({xy[i,a]:1,dims[i,a]:.5,d:-1},target);le({xy[i,a]:-1,dims[i,a]:-.5,d:-1},-target)
    tree=contact_tree(seed,p,topology(seed,p.constraints))
    for i,j,a in tree:
        t=1-a;cost=contact/nsoft
        if reweighted:
            gap=max(0.,z[j,a]-z[i,a]-wh[i,a],z[i,t]-z[j,t]-wh[j,t],z[j,t]-z[i,t]-wh[i,t])
            cost=2./nsoft/(gap+.02*min(np.sqrt(wh[i].prod()),np.sqrt(wh[j].prod())))
        d=var(cost)
        le({xy[j,a]:1,xy[i,a]:-1,dims[i,a]:-1,d:-1},0)
        shared=.02*min(wh[i,t],wh[j,t])
        le({xy[i,t]:1,xy[j,t]:-1,dims[j,t]:-1,d:-1},-shared)
        le({xy[j,t]:1,xy[i,t]:-1,dims[i,t]:-1,d:-1},-shared)
    ii=[];jj=[];data=[]
    for r,row in enumerate(rows):
        for k,v in row.items():ii.append(r);jj.append(k);data.append(v)
    begin=call_begin;cut_rounds=0
    if time.monotonic()-begin>=seconds:return None,{"status":"budget_in_build"}
    mat=coo_matrix((data,(ii,jj)),shape=(len(rhs),len(costs))).tocsc()
    solver=highspy.Highs();solver.setOptionValue('output_flag',False);solver.setOptionValue('threads',2)
    solver.setOptionValue('primal_feasibility_tolerance',1e-9);solver.setOptionValue('dual_feasibility_tolerance',1e-9)
    lp=highspy.HighsLp();lp.num_col_=len(costs);lp.num_row_=len(rhs);lp.col_cost_=costs
    lp.col_lower_=[a for a,b in bounds];lp.col_upper_=[highspy.kHighsInf if b is None else b for a,b in bounds]
    lp.row_lower_=[-highspy.kHighsInf]*len(rhs);lp.row_upper_=rhs
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=mat.indptr
    lp.a_matrix_.index_=mat.indices;lp.a_matrix_.value_=mat.data;solver.passModel(lp)
    signature=(len(costs),len(rhs))
    cached=getattr(p,'_shape_basis_cache',None)
    if cached is not None and cached[0]==signature:solver.setBasis(cached[1])
    while True:
        solver.setOptionValue('time_limit',max(.01,seconds-(time.monotonic()-begin)));solver.run()
        solution=solver.getSolution();status=solver.modelStatusToString(solver.getModelStatus())
        if not solution.value_valid:return None,{'status':status}
        if cut_rounds==0 and solver.getModelStatus()==highspy.HighsModelStatus.kOptimal:
            p._shape_basis_cache=(signature,solver.getBasis())
        values=np.asarray(solution.col_value);product=values[dims].prod(1)
        bad=np.flatnonzero((~locked)&(product<areas*.9900002))
        if not len(bad) or cut_rounds>=6 or time.monotonic()-begin>=seconds:break
        for i in bad:
            w=values[dims[i,0]];alo=areas[i]*.990001
            solver.addRow(-highspy.kHighsInf,-2*alo/w,2,np.asarray([dims[i,0],dims[i,1]],dtype=np.int32),np.asarray([-alo/w**2,-1.]))
        cut_rounds+=1
    out=np.column_stack((values[xy],values[dims]))*s;out[locked,2:]=p.specs[locked,2:];out[p.preplaced]=p.specs[p.preplaced]
    # Clip ordinary area safely inside actual allowed interval. The LP relaxation
    # is never itself accepted as proof of area feasibility.
    for i in np.flatnonzero(~locked):
        actual=out[i,2:].prod();target=np.clip(actual,p.areas[i]*.99000005,p.areas[i]*1.0099995)
        out[i,2:]*=np.sqrt(target/actual)
    return out,{'status':status,'elastic_sum':float(values[slack].sum()),'lp_objective':float(solver.getObjectiveValue()),'area_cut_rounds':cut_rounds}

