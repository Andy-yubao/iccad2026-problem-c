"""Snapshot-only central pair forces plus public contact organization."""
import time
import numpy as np
from ...relaxation import _pairs
from ...planner.backend.frozen.packing import _dimensions


class PairParameters:
    attraction_range=.5
    attraction_strength=.25
    repulsion_strength=1.


def organize(draft,p,settings):
    start=time.perf_counter();x=np.array(draft,copy=True)
    w,h=_dimensions(p,x[:,2]/x[:,3],True)
    initial_wh=np.column_stack((w,h));locked=p.fixed|p.preplaced
    c0=x[:,:2]+x[:,2:]/2
    x[:,2:]=initial_wh;x[:,:2]=np.maximum(0,c0-initial_wh/2);x[p.preplaced]=p.specs[p.preplaced]
    origin=x.copy();c0=x[:,:2]+x[:,2:]/2
    local=x[:,2:].min(1);area=x[:,2:].prod(1);mass=area/np.median(area)
    path=np.zeros(p.n);q=np.zeros(p.n);qpath=np.zeros(p.n)
    classes=[np.flatnonzero(p.constraints[:,2]==g) for g in np.unique(p.constraints[:,2]) if g>0]
    groups=[np.flatnonzero(p.constraints[:,3]==g) for g in np.unique(p.constraints[:,3]) if g>0]
    codes=p.constraints[:,4].astype(int);total_shape=np.log(settings['shape_total_factor'])
    for iteration in range(settings['iterations']):
        center=x[:,:2]+x[:,2:]/2
        i,j,unit,gap,scalar=_pairs(x,PairParameters)
        force=scalar[:,None]*unit;net=np.zeros((p.n,2))
        np.add.at(net,i,force);np.add.at(net,j,-force)
        degree=np.bincount(np.r_[i[scalar!=0],j[scalar!=0]],minlength=p.n)
        damping=max(1,int(degree.max()))
        net/=damping
        # One nearest face-contact proposal per module in each public group.
        # Choices and all forces depend only on this snapshot, never earlier updates.
        for ids in groups:
            delta=center[ids][None]-center[ids][:,None]
            extent=(x[ids,2:][None]+x[ids,2:][:,None])/2
            gaps=np.abs(delta)-extent
            distance=np.linalg.norm(np.maximum(gaps,0),axis=-1)
            np.fill_diagonal(distance,np.inf)
            partners=np.argmin(distance,axis=1)
            targets=ids[partners];d=center[targets]-center[ids]
            size=(x[ids,2:]+x[targets,2:])/2
            axis=np.argmax(np.abs(d)-size,axis=1)
            separation=np.maximum(0,(np.abs(d)-size)[np.arange(len(ids)),axis])
            u=np.divide(d,np.linalg.norm(d,axis=1)[:,None],out=np.zeros_like(d),where=np.linalg.norm(d,axis=1)[:,None]>0)
            pull=settings['group_strength']*np.minimum(separation,local[ids])[:,None]*u
            np.add.at(net,ids,pull);np.add.at(net,targets,-pull)
        lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0)
        for bit,axis,upper in ((1,0,False),(2,0,True),(4,1,True),(8,1,False)):
            ids=np.flatnonzero(codes&bit)
            shift=hi[axis]-x[ids,axis]-x[ids,axis+2] if upper else lo[axis]-x[ids,axis]
            net[ids,axis]+=settings['boundary_strength']*np.clip(shift,-local[ids],local[ids])
        move=.5*net/mass[:,None];move[p.preplaced]=0
        length=np.linalg.norm(move,axis=1)
        budget=np.minimum(settings['step_fraction']*local,np.maximum(0,settings['total_fraction']*local-path))
        factor=min(1.,float(np.divide(budget,length,out=np.ones(p.n),where=length>0).min()))
        move*=factor;next_center=np.maximum(center+move,x[:,2:]/2);next_center[p.preplaced]=c0[p.preplaced]
        anisotropy=unit[:,0]**2-unit[:,1]**2;stress=np.zeros(p.n)
        for ids in (i,j):
            radial=np.divide(x[ids,2:]/2,np.abs(unit),out=np.full_like(unit,np.inf),where=np.abs(unit)>0).min(1)
            np.add.at(stress,ids,scalar*radial*anisotropy)
        dq=.5*stress/area/damping;dq[locked]=0
        for ids in classes:
            dq[ids]=0 if locked[ids].any() else np.average(dq[ids],weights=area[ids])
        remaining=np.minimum(settings['shape_step_log'],np.maximum(0,total_shape-qpath))
        dq=np.clip(dq,-remaining,remaining)
        lower=np.minimum(q,np.log(initial_wh[:,1]/(2*next_center[:,1])))
        upper=np.maximum(q,np.log(2*next_center[:,0]/initial_wh[:,0]))
        for ids in classes:
            lower[ids]=lower[ids].max();upper[ids]=upper[ids].min()
        wanted=np.clip(q+dq,lower,upper);wanted[locked]=0;dq=wanted-q
        q=wanted;wh=initial_wh*np.column_stack((np.exp(q),np.exp(-q)))
        x=np.column_stack((np.maximum(0,next_center-wh/2),wh));x[p.preplaced]=p.specs[p.preplaced]
        path+=np.linalg.norm(x[:,:2]+x[:,2:]/2-center,axis=1);qpath+=np.abs(dq)
    assert np.max(path/local)<=settings['total_fraction']+1e-10
    assert np.max(qpath)<=total_shape+1e-10
    np.testing.assert_allclose(x[:,2:].prod(1),origin[:,2:].prod(1),rtol=2e-12,atol=1e-12)
    np.testing.assert_array_equal(x[locked,2:],origin[locked,2:])
    np.testing.assert_array_equal(x[p.preplaced],origin[p.preplaced])
    for ids in classes:np.testing.assert_allclose(x[ids,2:],np.broadcast_to(x[ids[0],2:],(len(ids),2)),rtol=1e-12,atol=1e-12)
    return x,dict(seconds=time.perf_counter()-start,iterations=settings['iterations'],max_center_path_fraction=float((path/local).max()),max_shape_factor=float(np.exp(np.abs(q)).max()),mass='initial_area_over_median_area')
