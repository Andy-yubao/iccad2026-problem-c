"""One vectorized fixed-shape LP with exact Manhattan wire terms."""
import time
import numpy as np
import clarabel
from scipy.sparse import coo_matrix,csc_matrix
from .fast_geometry import Rows
from .backend.frozen.feasible import Objective


def solve(p,seed,edges,tree,*,seconds=.1,boundary=12.,contact=6.,anchors=None,hard_tangent=False):
    start=time.perf_counter();n=p.n;s=p.scale;z=seed/s;wh=z[:,2:];ns=Objective(p).nsoft
    edges=np.asarray(edges,dtype=int).reshape(-1,3);tree=np.asarray(tree,dtype=int).reshape(-1,3)
    bi,bj=np.nonzero(np.triu(p.adjacency,1));bw=p.adjacency[bi,bj]
    if len(p.p2b): pi,pj,pw=p.p2b.T;pi=pi.astype(int);pj=pj.astype(int)
    else:pi=pj=np.zeros(0,dtype=int);pw=np.zeros(0)
    xy=np.arange(2*n).reshape(n,2);lo=2*n+np.arange(2);hi=2*n+2+np.arange(2)
    move=2*n+4+np.arange(2*n).reshape(n,2);requested=np.flatnonzero(p.constraints[:,4])
    bvar=4*n+4+np.arange(len(requested));cvar=4*n+4+len(requested)+np.arange(len(tree))
    wb=4*n+4+len(requested)+len(tree)+np.arange(2*len(bi)).reshape(-1,2)
    wp=4*n+4+len(requested)+len(tree)+2*len(bi)+np.arange(2*len(pi)).reshape(-1,2)
    nv=4*n+4+len(requested)+len(tree)+2*len(bi)+2*len(pi)
    costs=np.zeros(nv)
    box_weight=np.ones(2) if anchors is None else .5*((z[:,:2]+wh).max(0)-z[:,:2].min(0))[::-1]/anchors[0]
    costs[lo]=-box_weight;costs[hi]=box_weight;costs[move]=.005/n
    costs[bvar]=boundary/ns;costs[cvar]=contact/ns
    norm=max(p.total_wire_weight if anchors is None else 2*anchors[1],1e-12)
    costs[wb]=bw[:,None]/norm;costs[wp]=pw[:,None]/norm
    eq=Rows();ineq=Rows();eq.add(xy[p.preplaced].ravel(),1,z[p.preplaced,:2].ravel())
    ineq.add(np.arange(nv),-1,0);axis=np.tile(np.arange(2),n)
    ineq.add(np.column_stack((lo[axis],xy.ravel())),[1,-1],0)
    ineq.add(np.column_stack((xy.ravel(),hi[axis])),[1,-1],-wh.ravel())
    ineq.add(np.column_stack((xy.ravel(),move.ravel())),[1,-1],z[:,:2].ravel())
    ineq.add(np.column_stack((xy.ravel(),move.ravel())),[-1,-1],-z[:,:2].ravel())
    if len(edges):
        i,j,a=edges.T;ineq.add(np.column_stack((xy[i,a],xy[j,a])),[1,-1],-wh[i,a])
    for bit,a,upper in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
        use=(p.constraints[requested,4].astype(int)&bit)!=0;ids=requested[use];v=bvar[use]
        if upper:ineq.add(np.column_stack((np.full(len(ids),hi[a]),xy[ids,a],v)),[1,-1,-1],wh[ids,a])
        else:ineq.add(np.column_stack((xy[ids,a],np.full(len(ids),lo[a]),v)),[1,-1,-1],0)
    if len(tree):
        i,j,a=tree.T;b=1-a;shared=.02*np.minimum(wh[i,b],wh[j,b])
        ineq.add(np.column_stack((xy[j,a],xy[i,a],cvar)),[1,-1,-1],wh[i,a])
        # Soft contact preference cannot make hard-feasible placement impossible.
        if hard_tangent:
            ineq.add(np.column_stack((xy[i,b],xy[j,b])),[1,-1],wh[j,b]-shared)
            ineq.add(np.column_stack((xy[j,b],xy[i,b])),[1,-1],wh[i,b]-shared)
        else:
            ineq.add(np.column_stack((xy[i,b],xy[j,b],cvar)),[1,-1,-1],wh[j,b]-shared)
            ineq.add(np.column_stack((xy[j,b],xy[i,b],cvar)),[1,-1,-1],wh[i,b]-shared)
    for a in [0,1]:
        offset=(wh[bi,a]-wh[bj,a])/2
        ineq.add(np.column_stack((xy[bi,a],xy[bj,a],wb[:,a])),[1,-1,-1],-offset)
        ineq.add(np.column_stack((xy[bi,a],xy[bj,a],wb[:,a])),[-1,1,-1],offset)
        target=p.pins[pi,a]/s-wh[pj,a]/2
        ineq.add(np.column_stack((xy[pj,a],wp[:,a])),[1,-1],target)
        ineq.add(np.column_stack((xy[pj,a],wp[:,a])),[-1,-1],-target)
    ii=[];jj=[];vv=[];rhs=[];offset=0
    for block in [eq,ineq]:
        if block.count:ii.append(np.concatenate(block.i)+offset);jj.extend(block.j);vv.extend(block.v);rhs.extend(block.b)
        offset+=block.count
    matrix=coo_matrix((np.concatenate(vv),(np.concatenate(ii),np.concatenate(jj))),shape=(offset,nv)).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=2;settings.max_iter=80
    settings.time_limit=max(.005,seconds-(time.perf_counter()-start));settings.tol_feas=1e-10;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
    solver=clarabel.DefaultSolver(csc_matrix((nv,nv)),costs,matrix,np.concatenate(rhs),
        [clarabel.ZeroConeT(eq.count),clarabel.NonnegativeConeT(ineq.count)],settings)
    result=solver.solve();stats={'status':str(result.status),'native_seconds':result.solve_time,'seconds':time.perf_counter()-start}
    if str(result.status) not in ('Solved','AlmostSolved','MaxTime') or result.r_prim>1e-7:return None,stats
    out=seed.copy();out[:,:2]=np.asarray(result.x)[xy]*s;out[p.preplaced]=p.specs[p.preplaced]
    return out,stats
