"""Same R1 equations in an experiment-local FP64, no-fastmath native loop."""
import time
import numpy as np
from numba import njit
from ...planner.backend.frozen.packing import _dimensions


@njit(cache=True)
def _field(x,pre,locked,mib,groups,codes,iterations,step_fraction,total_fraction,shape_step,total_shape,group_strength,boundary_strength):
    n=len(x);origin=x.copy();wh=x[:,2:].copy();area=wh[:,0]*wh[:,1];mass=area/np.median(area)
    local=np.minimum(wh[:,0],wh[:,1]);path=np.zeros(n);q=np.zeros(n);qpath=np.zeros(n)
    for iteration in range(iterations):
        center=x[:,:2]+x[:,2:]/2;net=np.zeros((n,2));stress=np.zeros(n);degree=np.zeros(n,np.int64)
        for i in range(n):
            for j in range(i+1,n):
                dx=center[j,0]-center[i,0];dy=center[j,1]-center[i,1];dist=np.hypot(dx,dy)
                if dist>0:ux=dx/dist;uy=dy/dist
                else:
                    angle=(((i+1)*73856093 ^ (j+1)*19349663)%104729)*(2*np.pi/104729)
                    ux=np.cos(angle);uy=np.sin(angle)
                radius=min((x[i,2]+x[j,2])/(2*abs(ux)) if ux else np.inf,(x[i,3]+x[j,3])/(2*abs(uy)) if uy else np.inf)
                gap=dist-radius
                if abs(gap)<=32*np.finfo(np.float64).eps*max(dist,radius):gap=0.
                short=min(x[i,2],x[i,3],x[j,2],x[j,3]);t=min(1.,max(0.,gap/(.5*short)))
                scalar=.25*max(gap,0.)*(1-t)**2-short*np.tanh(max(-gap,0.)/short)
                net[i,0]+=scalar*ux;net[i,1]+=scalar*uy;net[j,0]-=scalar*ux;net[j,1]-=scalar*uy
                if scalar!=0:degree[i]+=1;degree[j]+=1
                anisotropy=ux*ux-uy*uy
                for k in (i,j):
                    r=min(x[k,2]/(2*abs(ux)) if ux else np.inf,x[k,3]/(2*abs(uy)) if uy else np.inf)
                    stress[k]+=scalar*r*anisotropy
        damping=max(1,degree.max());net/=damping
        for i in range(n):
            if groups[i]<=0:continue
            best=np.inf;target=-1
            for j in range(n):
                if j==i or groups[i]!=groups[j]:continue
                gx=abs(center[j,0]-center[i,0])-(x[i,2]+x[j,2])/2
                gy=abs(center[j,1]-center[i,1])-(x[i,3]+x[j,3])/2
                distance=np.hypot(max(0.,gx),max(0.,gy))
                if distance<best:best=distance;target=j
            if target<0:continue
            j=target;dx=center[j,0]-center[i,0];dy=center[j,1]-center[i,1];dist=np.hypot(dx,dy)
            gap=max(abs(dx)-(x[i,2]+x[j,2])/2,abs(dy)-(x[i,3]+x[j,3])/2,0.)
            force=group_strength*min(gap,local[i])
            if dist>0:
                fx=force*dx/dist;fy=force*dy/dist
                net[i,0]+=fx;net[i,1]+=fy;net[j,0]-=fx;net[j,1]-=fy
        lo=np.zeros(2)
        hi=np.zeros(2)
        for a in range(2):
            lo[a]=x[:,a].min();hi[a]=(x[:,a]+x[:,a+2]).max()
        for i in range(n):
            for bit,a,upper in ((1,0,False),(2,0,True),(4,1,True),(8,1,False)):
                if codes[i]&bit:
                    shift=hi[a]-x[i,a]-x[i,a+2] if upper else lo[a]-x[i,a]
                    net[i,a]+=boundary_strength*min(local[i],max(-local[i],shift))
        move=.5*net/mass.reshape(n,1);factor=1.
        for i in range(n):
            if pre[i]:move[i,:]=0.
            length=np.hypot(move[i,0],move[i,1]);budget=min(step_fraction*local[i],max(0.,total_fraction*local[i]-path[i]))
            if length>0:factor=min(factor,budget/length)
        next_center=np.maximum(center+min(1.,factor)*move,x[:,2:]/2)
        for i in range(n):
            if pre[i]:next_center[i,:]=origin[i,:2]+origin[i,2:]/2
        dq=.5*stress/area/damping
        for i in range(n):
            if locked[i]:dq[i]=0.
        for g in np.unique(mib):
            if g<=0:continue
            num=0.;den=0.;any_locked=False
            for i in range(n):
                if mib[i]==g:num+=dq[i]*area[i];den+=area[i];any_locked=any_locked or locked[i]
            for i in range(n):
                if mib[i]==g:dq[i]=0. if any_locked else num/den
        lower=np.minimum(q,np.log(wh[:,1]/(2*next_center[:,1])))
        upper=np.maximum(q,np.log(2*next_center[:,0]/wh[:,0]))
        for g in np.unique(mib):
            if g<=0:continue
            low=-np.inf;high=np.inf
            for i in range(n):
                if mib[i]==g:low=max(low,lower[i]);high=min(high,upper[i])
            for i in range(n):
                if mib[i]==g:lower[i]=low;upper[i]=high
        for i in range(n):
            remaining=min(shape_step,max(0.,total_shape-qpath[i]));delta=min(remaining,max(-remaining,dq[i]))
            wanted=min(upper[i],max(lower[i],q[i]+delta))
            if locked[i]:wanted=0.
            qpath[i]+=abs(wanted-q[i]);q[i]=wanted
            x[i,2]=wh[i,0]*np.exp(wanted);x[i,3]=wh[i,1]*np.exp(-wanted)
            for a in range(2):x[i,a]=max(0.,next_center[i,a]-x[i,a+2]/2)
            if pre[i]:x[i,:]=origin[i,:]
            path[i]+=np.hypot(x[i,0]+x[i,2]/2-center[i,0],x[i,1]+x[i,3]/2-center[i,1])
    return x,path,q,qpath


def organize(draft,p,settings):
    start=time.perf_counter();x=np.array(draft,copy=True);w,h=_dimensions(p,x[:,2]/x[:,3],True)
    center=x[:,:2]+x[:,2:]/2;x[:,2:]=np.column_stack((w,h));x[:,:2]=np.maximum(0,center-x[:,2:]/2);x[p.preplaced]=p.specs[p.preplaced]
    local=x[:,2:].min(1)
    x,path,q,qpath=_field(x,p.preplaced,p.fixed|p.preplaced,p.constraints[:,2],p.constraints[:,3],p.constraints[:,4],
        settings['iterations'],settings['step_fraction'],settings['total_fraction'],settings['shape_step_log'],np.log(settings['shape_total_factor']),settings['group_strength'],settings['boundary_strength'])
    return x,dict(seconds=time.perf_counter()-start,iterations=settings['iterations'],max_center_path_fraction=float((path/local).max()),max_shape_factor=float(np.exp(np.abs(q)).max()),max_shape_path_log=float(qpath.max()),mass='initial_area_over_median_area',engine='numba_FP64_no_fastmath')
