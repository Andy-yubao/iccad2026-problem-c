"""V3: synchronous central-force translation and area-preserving shape stress.

Old v1/v2 code and evidence remain separate. Shape is driven by anisotropic
pair stress, not net force: balanced opposing forces can still deform a block.
"""
from dataclasses import asdict,dataclass
from numbers import Integral
import time
import numpy as np
from .relaxation import _positions,_pairs


@dataclass(frozen=True)
class Config:
    iterations: int = 10
    step_size: float = .5
    step_fraction: float = .03
    total_fraction: float = .1999
    attraction_range: float = .5
    attraction_strength: float = .25
    repulsion_strength: float = 1.
    mass_weighted: bool = True
    shape_enabled: bool = True
    shape_step_size: float = .5
    shape_step_fraction: float = .015
    shape_total_fraction: float = .15

    def __post_init__(self):
        if isinstance(self.iterations,bool) or not isinstance(self.iterations,Integral) or not 0<=self.iterations<=10:
            raise ValueError('v3 iterations must be an integer in [0,10]')
        if not isinstance(self.mass_weighted,bool) or not self.mass_weighted or not isinstance(self.shape_enabled,bool):
            raise ValueError('v3 requires area mass and boolean shape_enabled')
        limits=dict(step_size=1.,step_fraction=.03,total_fraction=.1999,attraction_range=1.,
                    attraction_strength=1.,repulsion_strength=1.,shape_step_size=1.,
                    shape_step_fraction=.015,shape_total_fraction=.15)
        for key,maximum in limits.items():
            value=getattr(self,key)
            if isinstance(value,bool) or not np.isfinite(value) or not 0<value<=maximum:
                raise ValueError(f'{key} must be finite in (0,{maximum}]')


def _mask(value,n,name):
    a=np.asarray(value)
    if a.shape!=(n,) or not np.isin(a,[0,1]).all():raise ValueError(f'{name} must be N boolean/0/1 values')
    return a.astype(bool)


def relax(positions,preplaced=None,*,fixed_shape=None,config=None,return_history=False):
    """Return new XYWH and diagnostics; supplied input is never modified.

    Movement budgets refer to physical centers, not lower-left coordinates.
    Short-side scales and area mass are frozen at the ORIGINAL input. For a
    soft block w=w0*exp(q), h=h0*exp(-q). Per-step |dq|<=log(1.015), cumulative
    |dq| path<=log(1.15), also bounding each dimension to [1/1.15,1.15] times
    its input value. Public fixed dimensions and preplaced blocks are locked.
    """
    start=time.perf_counter();config=Config() if config is None else config
    if not isinstance(config,Config):raise TypeError('v3 Config required')
    p=_positions(positions);original=p.copy();n=len(p)
    pre=np.zeros(n,bool) if preplaced is None else _mask(preplaced,n,'preplaced')
    if config.shape_enabled and fixed_shape is None:
        raise ValueError('explicit fixed_shape mask is required when deformation is enabled')
    fixed=np.zeros(n,bool) if fixed_shape is None else _mask(fixed_shape,n,'fixed_shape')
    shape_locked=fixed|pre
    local=p[:,2:].min(1);area=p[:,2]*p[:,3]
    if not np.isfinite(area).all() or (area<=0).any():raise ValueError('finite positive module area required')
    mass=area/np.median(area)
    c0=p[:,:2]+p[:,2:]/2;path=np.zeros(n);q=np.zeros(n);qpath=np.zeros(n)
    shape_step=np.log1p(config.shape_step_fraction);shape_total=np.log1p(config.shape_total_fraction)
    history=[p.copy()] if return_history else None;trace=[];stop='iteration_limit'
    for iteration in range(config.iterations):
        center=p[:,:2]+p[:,2:]/2
        i,j,unit,gap,scalar=_pairs(p,config)
        force=scalar[:,None]*unit;net=np.zeros((n,2))
        np.add.at(net,i,force);np.add.at(net,j,-force)
        active=scalar!=0;degree=np.bincount(np.r_[i[active],j[active]],minlength=n)
        damping=max(1,int(degree.max()))
        move=config.step_size*net/damping/mass[:,None];move[pre]=0
        length=np.hypot(move[:,0],move[:,1])
        budget=np.minimum(config.step_fraction*local,np.maximum(0,config.total_fraction*local-path))
        scale=min(1.,float(np.divide(budget,length,out=np.ones(n),where=length>0).min()))
        move*=scale
        proposed_center=np.maximum(center+move,p[:,2:]/2)
        proposed_center[pre]=c0[pre]
        dq=np.zeros(n)
        if config.shape_enabled:
            # Pair force projected onto a rectangle's radial surface point.
            # Generalized log-width force: f*r_surface*(ux^2-uy^2).
            # Both endpoints receive the same tensile/compressive sign.
            anisotropy=unit[:,0]**2-unit[:,1]**2
            stress=np.zeros(n)
            for ids in (i,j):
                radial=np.divide(p[ids,2:]/2,np.abs(unit),out=np.full((len(ids),2),np.inf),where=np.abs(unit)>0).min(1)
                np.add.at(stress,ids,scalar*radial*anisotropy)
            dq=config.shape_step_size*stress/area/damping
            dq[shape_locked]=0
            remaining=np.maximum(0,shape_total-qpath)
            dq=np.clip(dq,-np.minimum(shape_step,remaining),np.minimum(shape_step,remaining))
            wanted=np.clip(q+dq,-shape_total,shape_total)
            # Contain a newly shaped rectangle in XY>=0 without a hidden center
            # translation. Current q is always retained as a feasible identity.
            lower=np.log(original[:,3]/(2*proposed_center[:,1]))
            upper=np.log(2*proposed_center[:,0]/original[:,2])
            lower=np.minimum(lower,q);upper=np.maximum(upper,q)
            wanted=np.minimum(np.maximum(wanted,lower),upper)
            wanted[shape_locked]=0
            dq=wanted-q
        next_q=q+dq
        wh=original[:,2:]*np.stack((np.exp(next_q),np.exp(-next_q)),-1)
        wh[shape_locked]=original[shape_locked,2:]
        proposed=np.c_[np.maximum(0,proposed_center-wh/2),wh]
        proposed[pre]=original[pre]
        actual_center=proposed[:,:2]+proposed[:,2:]/2
        actual=np.linalg.norm(actual_center-center,axis=1)
        if not np.isfinite(proposed).all():raise FloatingPointError('nonfinite geometry')
        path+=actual;qpath+=np.abs(dq);q=next_q;p=proposed
        trace.append(dict(iteration=iteration+1,attractive_pairs=int((scalar>0).sum()),repulsive_pairs=int((scalar<0).sum()),
            damping=damping,translation_safety_scale=scale,max_center_step_fraction=float((actual/local).max()),
            max_center_path_fraction=float((path/local).max()),max_shape_step_log=float(np.abs(dq).max()),
            max_shape_path_log=float(qpath.max()),deformed_modules=int((np.abs(q)>1e-12).sum())))
        if return_history:history.append(p.copy())
        if not actual.any() and not dq.any():stop='no_movement';break
    displacement=np.linalg.norm(p[:,:2]+p[:,2:]/2-c0,axis=1)
    eps=256*np.finfo(float).eps*np.maximum(1,np.abs(original).max(1))
    if (path>config.total_fraction*local+eps).any() or (displacement>config.total_fraction*local+eps).any():
        raise AssertionError('center movement budget exceeded')
    if (qpath>shape_total+1e-12).any():raise AssertionError('shape budget exceeded')
    area_error=np.abs(p[:,2]*p[:,3]/area-1)
    if area_error.max()>2e-12:raise AssertionError('area preservation failed')
    if not np.array_equal(p[pre],original[pre]) or not np.array_equal(p[shape_locked,2:],original[shape_locked,2:]):
        raise AssertionError('public immutable geometry changed')
    d=dict(config=asdict(config),iterations=len(trace),stop_reason=stop,trace=trace,
        mass_model='initial_area_over_median_area',max_center_displacement_fraction=float((displacement/local).max()),
        mean_center_displacement_fraction=float((displacement/local).mean()),max_center_path_fraction=float((path/local).max()),
        max_shape_path_log=float(qpath.max()),max_dimension_factor=float(np.maximum(p[:,2:]/original[:,2:],original[:,2:]/p[:,2:]).max()),
        max_relative_area_error=float(area_error.max()),deformed_modules=int((np.abs(q)>1e-12).sum()),
        free_shape_modules=int((~shape_locked).sum()),seconds=time.perf_counter()-start)
    if return_history:d['history']=np.stack(history)
    return p,d


def preprocess(draft,inputs,*,config=None,return_history=False):
    n,_,_,_,_,constraints,specs=inputs
    p=np.asarray(draft,dtype=float);c=np.asarray(constraints);s=np.asarray(specs)
    if p.shape!=(int(n),4) or c.shape!=(int(n),5) or s.shape!=p.shape:raise ValueError('public input shape mismatch')
    pre=c[:,1]!=0;locked=(c[:,:2]!=0).any(1)
    if not np.array_equal(p[locked,2:],s[locked,2:]) or not np.array_equal(p[pre],s[pre]):
        raise ValueError('restore public immutable geometry first')
    return relax(p,pre,fixed_shape=c[:,0]!=0,config=config,return_history=return_history)
