"""Pure center forces, fixed initial-area mobility, synchronous area-safe shape."""
from dataclasses import dataclass,asdict,replace
import time
import numpy as np
from numba import njit


@dataclass(frozen=True)
class Config:
    iterations:int=20
    step_size:float=.5
    step_fraction:float=.05
    total_fraction:float=.4
    attraction_range:float=.5
    attraction_strength:float=.25
    repulsion_strength:float=1.
    shape_step_size:float=.5
    shape_step_fraction:float=.03
    shape_total_fraction:float=.5
    exclude_exhausted:bool=False
    area_attraction:bool=False
    rectangle_faces:bool=False


def variants():
    return dict(rectangle_faces=Config(exclude_exhausted=True,attraction_strength=1.,rectangle_faces=True))


MODULE_COLUMNS=['active_degree','attraction_fx','attraction_fy','repulsion_fx','repulsion_fy','sum_pair_force_norms',
    'raw_center_dx','raw_center_dy','available_step_length','applied_common_scale','actual_center_dx','actual_center_dy',
    'center_path_length','attraction_shape_stress','repulsion_shape_stress','raw_shape_dq','actual_shape_dq',
    'shape_path_log','shape_q','excluded_exhausted','sum_attraction_magnitudes','sum_repulsion_magnitudes']
ROUND_COLUMNS=['attractive_pairs','repulsive_pairs','damping','old_common_scale','actual_common_scale',
    'old_limiter_id','actual_limiter_id','excluded_exhausted_modules','shape_step_clipped_modules','coordinate_clipped_modules',
    'near_diagonal_active_pairs','near_diagonal_pair_force_sum','all_pair_force_sum']


@njit(cache=True)
def kernel(original,pre,locked,iterations,eta,step_frac,total_frac,range_frac,ka,kr,eta_q,shape_step,shape_total,gate,area_source,rectangle_faces,capture):
    x=original.copy();n=len(x);wh0=x[:,2:].copy();area=wh0[:,0]*wh0[:,1]
    mass=area/np.median(area);local=np.minimum(wh0[:,0],wh0[:,1]);path=np.zeros(n);q=np.zeros(n);qpath=np.zeros(n)
    history=np.zeros((iterations+1,n,4)) if capture else np.empty((0,0,0))
    log=np.zeros((iterations,n,22)) if capture else np.empty((0,0,0))
    rounds=np.zeros((iterations,13)) if capture else np.empty((0,0))
    if capture:history[0]=x
    for iteration in range(iterations):
        center=x[:,:2]+x[:,2:]/2
        attraction=np.zeros((n,2));repulsion=np.zeros((n,2));sa=np.zeros(n);sr=np.zeros(n)
        degree=np.zeros(n,np.int64);sum_a=np.zeros(n);sum_r=np.zeros(n);na=0;nr=0;ndiag=0;fdiag=0.;fall=0.
        for i in range(n):
            for j in range(i+1,n):
                dx=center[j,0]-center[i,0];dy=center[j,1]-center[i,1];distance=np.hypot(dx,dy)
                if distance>0:ux=dx/distance;uy=dy/distance
                else:
                    angle=(((i+1)*73856093 ^ (j+1)*19349663)%104729)*(2*np.pi/104729)
                    ux=np.cos(angle);uy=np.sin(angle)
                contact_x=(x[i,2]+x[j,2])/(2*abs(ux)) if ux else np.inf
                contact_y=(x[i,3]+x[j,3])/(2*abs(uy)) if uy else np.inf
                contact=min(contact_x,contact_y)
                gap=distance-contact
                if abs(gap)<=32*np.finfo(np.float64).eps*max(distance,contact):gap=0.
                short=min(x[i,2],x[i,3],x[j,2],x[j,3]);cutoff=range_frac*short
                t=min(1.,max(0.,gap/cutoff))
                coupling=2*mass[i]*mass[j]/(mass[i]+mass[j]) if area_source else 1.
                fa=ka*max(gap,0.)*(1-t)**2*coupling
                fr=-kr*short*np.tanh(max(-gap,0.)/short)
                if fa>0:na+=1
                if fr<0:nr+=1
                if fa+fr!=0:degree[i]+=1;degree[j]+=1
                attraction[i,0]+=fa*ux;attraction[i,1]+=fa*uy;attraction[j,0]-=fa*ux;attraction[j,1]-=fa*uy
                repulsion[i,0]+=fr*ux;repulsion[i,1]+=fr*uy;repulsion[j,0]-=fr*ux;repulsion[j,1]-=fr*uy
                anisotropy=ux*ux-uy*uy
                fall+=abs(fa)+abs(fr)
                if fa+fr!=0 and abs(anisotropy)<.1:ndiag+=1;fdiag+=abs(fa)+abs(fr)
                for k in (i,j):
                    radial=min(x[k,2]/(2*abs(ux)) if ux else np.inf,x[k,3]/(2*abs(uy)) if uy else np.inf)
                    if rectangle_faces:
                        tie=np.isfinite(contact_x) and np.isfinite(contact_y) and abs(contact_x-contact_y)<=1e-11*max(contact_x,contact_y)
                        if tie:coefficient=.5*(x[k,2]/(2*abs(ux))-x[k,3]/(2*abs(uy)))
                        elif contact_x<contact_y:coefficient=x[k,2]/(2*abs(ux))
                        else:coefficient=-x[k,3]/(2*abs(uy))
                        sa[k]+=fa*coefficient;sr[k]+=fr*coefficient
                    else:
                        sa[k]+=fa*radial*anisotropy;sr[k]+=fr*radial*anisotropy
                    sum_a[k]+=abs(fa);sum_r[k]+=abs(fr)
        damping=max(1,degree.max());move=eta*(attraction+repulsion)/damping/mass.reshape(n,1)
        budgets=np.zeros(n);old_alpha=1.;alpha=1.;old_limiter=-1;limiter=-1;excluded=0
        for i in range(n):
            if pre[i]:move[i,:]=0.
            remaining=max(0.,total_frac*local[i]-path[i]);budgets[i]=min(step_frac*local[i],remaining)
            length=np.hypot(move[i,0],move[i,1])
            if length>0 and budgets[i]/length<old_alpha:old_alpha=budgets[i]/length;old_limiter=i
            eps=256*np.finfo(np.float64).eps*max(1.,np.max(np.abs(original[i])))
            exhausted=gate and not pre[i] and remaining<=eps
            if capture:
                log[iteration,i,0]=degree[i]
                log[iteration,i,1:3]=attraction[i];log[iteration,i,3:5]=repulsion[i]
                log[iteration,i,5]=sum_a[i]+sum_r[i];log[iteration,i,6:8]=move[i]
                log[iteration,i,8]=budgets[i];log[iteration,i,19]=1. if exhausted else 0.
                log[iteration,i,20]=sum_a[i];log[iteration,i,21]=sum_r[i]
            if exhausted:move[i,:]=0.;excluded+=1;length=0.
            if length>0 and budgets[i]/length<alpha:alpha=budgets[i]/length;limiter=i
        next_center=np.maximum(center+min(1.,alpha)*move,x[:,2:]/2)
        for i in range(n):
            if pre[i]:next_center[i,:]=original[i,:2]+original[i,2:]/2
        shape_clipped=0;coordinate_clipped=0
        for i in range(n):
            raw=eta_q*(sa[i]+sr[i])/area[i]/damping
            if locked[i]:raw=0.
            remaining=min(shape_step,max(0.,shape_total-qpath[i]));dq=min(remaining,max(-remaining,raw))
            if abs(dq-raw)>1e-14:shape_clipped+=1
            wanted=min(shape_total,max(-shape_total,q[i]+dq))
            lower=min(q[i],np.log(wh0[i,1]/(2*next_center[i,1])))
            upper=max(q[i],np.log(2*next_center[i,0]/wh0[i,0]))
            contained=min(upper,max(lower,wanted))
            if abs(contained-wanted)>1e-14:coordinate_clipped+=1
            if locked[i]:contained=0.
            dq=contained-q[i];q[i]=contained;qpath[i]+=abs(dq)
            x[i,2]=wh0[i,0]*np.exp(q[i]);x[i,3]=wh0[i,1]*np.exp(-q[i])
            if locked[i]:x[i,2:]=wh0[i]
            for a in range(2):x[i,a]=max(0.,next_center[i,a]-x[i,a+2]/2)
            if pre[i]:x[i,:]=original[i]
            actual=x[i,:2]+x[i,2:]/2-center[i];path[i]+=np.hypot(actual[0],actual[1])
            if capture:
                log[iteration,i,9]=min(1.,alpha);log[iteration,i,10:12]=actual;log[iteration,i,12]=path[i]
                log[iteration,i,13]=sa[i];log[iteration,i,14]=sr[i];log[iteration,i,15]=raw
                log[iteration,i,16]=dq;log[iteration,i,17]=qpath[i];log[iteration,i,18]=q[i]
        if capture:
            history[iteration+1]=x
            rounds[iteration,:]=np.array([na,nr,damping,old_alpha,alpha,old_limiter,limiter,excluded,shape_clipped,coordinate_clipped,ndiag,fdiag,fall])
    return x,history,log,rounds,path,qpath


def preprocess(draft,inputs,config=Config(),capture=False):
    start=time.perf_counter();x=np.asarray(draft,dtype=np.float64);c=np.asarray(inputs[5]);specs=np.asarray(inputs[6])
    pre=c[:,1]!=0;locked=(c[:,:2]!=0).any(1)
    if x.shape!=(int(inputs[0]),4) or not np.isfinite(x).all() or (x[:,2:]<=0).any():raise ValueError('Invalid geometry')
    np.testing.assert_array_equal(x[locked,2:],specs[locked,2:]);np.testing.assert_array_equal(x[pre],specs[pre])
    result,history,modules,rounds,path,qpath=kernel(x,pre,locked,config.iterations,config.step_size,config.step_fraction,config.total_fraction,
        config.attraction_range,config.attraction_strength,config.repulsion_strength,config.shape_step_size,
        np.log1p(config.shape_step_fraction),np.log1p(config.shape_total_fraction),config.exclude_exhausted,config.area_attraction,config.rectangle_faces,capture)
    eps=256*np.finfo(float).eps*np.maximum(1,np.abs(x).max(1));local=x[:,2:].min(1)
    assert np.isfinite(result).all() and (result[:,:2]>=0).all() and (result[:,2:]>0).all()
    assert (path<=config.total_fraction*local+eps).all() and (qpath<=np.log1p(config.shape_total_fraction)+1e-10).all()
    np.testing.assert_allclose(result[:,2:].prod(1),x[:,2:].prod(1),rtol=2e-12,atol=1e-12)
    np.testing.assert_array_equal(result[locked,2:],x[locked,2:]);np.testing.assert_array_equal(result[pre],x[pre])
    stats=dict(seconds=time.perf_counter()-start,config=asdict(config),max_center_path_fraction=float((path/local).max()),
        mean_center_path_fraction=float((path/local).mean()),max_dimension_factor=float(np.maximum(result[:,2:]/x[:,2:],x[:,2:]/result[:,2:]).max()),
        max_relative_area_error=float(np.abs(result[:,2:].prod(1)/x[:,2:].prod(1)-1).max()))
    if capture:stats.update(history=history,module_log=modules,round_log=rounds)
    return result,stats
