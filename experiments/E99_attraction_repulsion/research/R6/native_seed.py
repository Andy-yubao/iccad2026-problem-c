"""Order0 legal_seed equations in FP64 native loops, without fastmath."""
import numpy as np
from numba import njit
from ...planner.backend.frozen.packing import _dimensions
from ...planner.canonical import canonicalize


@njit(cache=True)
def construct(result,initial_placed,order,codes,groups,target_hi):
    n=len(result);placed=np.empty(n,np.int64);m=len(initial_placed)
    placed[:m]=initial_placed
    for i in order:
        w=result[i,2];h=result[i,3];dx=max(0.,result[i,0]);dy=max(0.,result[i,1])
        collision=False
        for k in range(m):
            j=placed[k]
            if min(dx+w,result[j,0]+result[j,2])-max(dx,result[j,0])>1e-8 and min(dy+h,result[j,1]+result[j,3])-max(dy,result[j,1])>1e-8:collision=True
        if m==0 or not collision:
            result[i,0]=dx;result[i,1]=dy;placed[m]=i;m+=1;continue
        lo=np.empty((m,2));hi=np.empty((m,2));xs=np.empty(2+2*m)
        xs[0]=dx;xs[1]=0.
        for k in range(m):
            j=placed[k]
            for a in range(2):lo[k,a]=result[j,a];hi[k,a]=result[j,a]+result[j,a+2]
            xs[2+k]=hi[k,0];xs[2+m+k]=lo[k,0]-w
        xs=np.unique(xs);xs=xs[xs>=0];count=len(xs)
        cross=np.empty((count,m),np.bool_);up=np.full(count,dy);down=np.full(count,dy)
        for k in range(count):
            for j in range(m):cross[k,j]=xs[k]<hi[j,0]-1e-10 and xs[k]+w>lo[j,0]+1e-10
        for _ in range(m+1):
            hit_any=False
            for k in range(count):
                new=up[k]
                for j in range(m):
                    if cross[k,j] and up[k]<hi[j,1]-1e-10 and up[k]+h>lo[j,1]+1e-10:
                        hit_any=True;new=max(new,hi[j,1])
                up[k]=new
            if not hit_any:break
        for _ in range(m+1):
            hit_any=False
            for k in range(count):
                new=down[k]
                for j in range(m):
                    if cross[k,j] and down[k]<hi[j,1]-1e-10 and down[k]+h>lo[j,1]+1e-10:
                        hit_any=True;new=min(new,lo[j,1]-h)
                down[k]=new
            if not hit_any:break
        has_group=False
        for j in range(m):
            if groups[i]>0 and groups[placed[j]]==groups[i]:has_group=True
        best=np.inf;bx=dx;by=dy
        for k in range(2*count):
            x=xs[k%count];y=up[k] if k<count else down[k-count]
            if y<0:continue
            cost=((x-dx)/max(w,1e-8))**2+((y-dy)/max(h,1e-8))**2
            if codes[i]&1:cost+=.3*(x/max(w,1e-8))**2
            if codes[i]&8:cost+=.3*(y/max(h,1e-8))**2
            if codes[i]&2:cost+=.3*((x+w-target_hi[0])/max(w,1e-8))**2
            if codes[i]&4:cost+=.3*((y+h-target_hi[1])/max(h,1e-8))**2
            if has_group:
                touches=False
                for j in range(m):
                    if groups[placed[j]]!=groups[i]:continue
                    ix=min(x+w,hi[j,0])-max(x,lo[j,0]);iy=min(y+h,hi[j,1])-max(y,lo[j,1])
                    if (abs(ix)<1e-7 and iy>1e-7) or (abs(iy)<1e-7 and ix>1e-7):touches=True
                if not touches:cost+=.15
            if cost<best:best=cost;bx=x;by=y
        result[i,0]=bx;result[i,1]=by;placed[m]=i;m+=1
    return result


def seed(p,draft,order_mode=0,preplaced_only=False,preserve_centers=False):
    if order_mode!=0 or preplaced_only:raise ValueError('Registered native constructor supports order0 full placement only')
    widths,heights=_dimensions(p,draft[:,2]/draft[:,3],True)
    result=np.column_stack((draft[:,:2],widths,heights));result[p.preplaced]=p.specs[p.preplaced]
    if preserve_centers:
        result[~p.preplaced,:2]=np.maximum(0,draft[~p.preplaced,:2]+draft[~p.preplaced,2:]/2-result[~p.preplaced,2:]/2)
    ids=np.flatnonzero(~p.preplaced);codes=p.constraints[:,4].astype(int)
    order=ids[np.lexsort((ids,-p.areas[ids],codes[ids]==0))]
    result=construct(result,np.flatnonzero(p.preplaced),order,codes,p.constraints[:,3],(draft[:,:2]+draft[:,2:]).max(0))
    return canonicalize(result,p)
