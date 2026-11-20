"""Independent copies of previously defined diagnostic primitives; no training."""

import numpy as np

def ratio(numerator,denominator):return float(numerator/denominator) if denominator>0 else np.nan

def center_setup(reference_centers,areas,pre=None):
    ref=np.asarray(reference_centers,float);areas=np.asarray(areas,float);n=len(ref)
    scale=(np.sqrt(areas)[:,None]+np.sqrt(areas)[None,:])/2
    delta=(ref[None,:,:]-ref[:,None,:])/scale[...,None]
    dist=np.linalg.norm(delta,axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')
    neighbors={k:np.zeros((n,n),bool) for k in (4,6,8)}
    for k,mask in neighbors.items():mask[np.arange(n)[:,None],order[:,:min(k,n-1)]]=True
    pairs=np.triu(np.ones((n,n),bool),1)
    if pre is not None:pairs&=~(pre[:,None]&pre[None,:])
    local=pairs&(neighbors[6]|neighbors[6].T)
    return dict(ref=ref,areas=areas,scale=scale,delta=delta,order=order,neighbors=neighbors,pairs=pairs,local=local,
        axes=np.abs(delta)>=.10)

def center_metrics(centers,setup):
    p=np.asarray(centers,float);n=len(p);d=(p[None,:,:]-p[:,None,:])/setup['scale'][...,None]
    known=np.abs(d)>=.02;correct=(np.sign(d)==np.sign(setup['delta']))&known
    result={}
    for name,mask in (('global',setup['pairs']),('local',setup['local'])):
        eligible=mask[...,None]&setup['axes']
        result[name+'_axis_accuracy']=ratio((correct&eligible).sum(),eligible.sum())
        result[name+'_axis_resolved']=ratio((known&eligible).sum(),eligible.sum())
    dist=np.linalg.norm(d,axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')
    for k,reference in setup['neighbors'].items():
        actual=np.zeros((n,n),bool);actual[np.arange(n)[:,None],order[:,:min(k,n-1)]]=True
        result[f'knn{k}_recall']=float(((actual&reference).sum(1)/min(k,n-1)).mean())
    return result,d,correct

def intent(positions,setup):
    p=np.asarray(positions,float);center=p[:,:2]+p[:,2:]/2
    delta=center[None,:,:]-center[:,None,:]
    half=(p[:,None,2:]+p[None,:,2:])/2
    gaps=np.stack((delta[...,0]-half[...,0],-delta[...,0]-half[...,0],delta[...,1]-half[...,1],-delta[...,1]-half[...,1]),-1)/setup['scale'][...,None]
    choice=gaps.argmax(-1);top=np.sort(gaps,axis=-1)
    return gaps,choice,top[...,-1]-top[...,-2]>.02

def largest_empty(p):
    from numba import njit
    global _empty_kernel
    if '_empty_kernel' not in globals():
        @njit
        def kernel(occupied,x,y):
            ny,nx=occupied.shape
            heights=np.zeros(nx);best=0.;rect=np.zeros(4)
            stack_i=np.empty(nx+1,np.int64);stack_h=np.empty(nx+1)
            for row in range(ny):
                for col in range(nx):
                    heights[col]=0. if occupied[row,col] else heights[col]+y[row+1]-y[row]
                size=0
                for col in range(nx+1):
                    h=heights[col] if col<nx else 0.
                    left=col
                    while size>0 and stack_h[size-1]>h:
                        size-=1;left=stack_i[size];old=stack_h[size]
                        area=(x[col]-x[left])*old
                        if area>best:
                            best=area;rect[0]=x[left];rect[1]=y[row+1]-old;rect[2]=x[col]-x[left];rect[3]=old
                    if h>0 and (size==0 or stack_h[size-1]<h):
                        stack_i[size]=left;stack_h[size]=h;size+=1
            return best,rect
        _empty_kernel=kernel
    lo=p[:,:2];hi=lo+p[:,2:]
    x=np.unique(np.r_[lo[:,0],hi[:,0]]);y=np.unique(np.r_[lo[:,1],hi[:,1]])
    a=np.searchsorted(x,lo[:,0]);b=np.searchsorted(x,hi[:,0])
    c=np.searchsorted(y,lo[:,1]);d=np.searchsorted(y,hi[:,1])
    diff=np.zeros((len(x),len(y)),np.int32)
    for xx,yy,v in ((a,c,1),(b,c,-1),(a,d,-1),(b,d,1)):np.add.at(diff,(xx,yy),v)
    occupied=(diff.cumsum(0).cumsum(1)[:-1,:-1]>0).T.copy()
    area,rect=_empty_kernel(occupied,x,y)
    return dict(area=float(area),bbox_fraction=float(area/((x[-1]-x[0])*(y[-1]-y[0]))),rectangle=rect.tolist())

def blockers(p,i,j,foreign,areas):
    center=p[:,:2]+p[:,2:]/2
    a=center[i];v=center[j]-a
    inset=.02*np.sqrt(areas[foreign])[:,None]
    lo=p[foreign,:2]+inset;hi=p[foreign,:2]+p[foreign,2:]-inset
    lower=np.full(len(foreign),.15);upper=np.full(len(foreign),.85)
    for axis in range(2):
        if abs(v[axis])<1e-12:
            valid=(a[axis]>lo[:,axis])&(a[axis]<hi[:,axis])
            upper=np.where(valid,upper,-1.)
        else:
            t1=(lo[:,axis]-a[axis])/v[axis];t2=(hi[:,axis]-a[axis])/v[axis]
            lower=np.maximum(lower,np.minimum(t1,t2));upper=np.minimum(upper,np.maximum(t1,t2))
    return upper>lower

def boundary(p,flags,areas):
    lo=p[:,:2];hi=lo+p[:,2:];center=lo+p[:,2:]/2
    bblo=lo.min(0);bbhi=hi.max(0);local=np.sqrt(areas)
    result=[]
    for i in np.flatnonzero(flags[:,4]):
        for bit,axis,outward in ((1,0,-1),(2,0,1),(4,1,1),(8,1,-1)):
            if not flags[i,4]&bit:continue
            tangent=1-axis
            gap=(lo[i,axis]-bblo[axis] if outward<0 else bbhi[axis]-hi[i,axis])/local[i]
            overlap=np.minimum(hi[i,tangent],hi[:,tangent])-np.maximum(lo[i,tangent],lo[:,tangent])
            farther=(center[:,axis]-center[i,axis])*outward>.05*local[i]
            shadow=farther&(overlap>=.1*np.minimum(p[i,2+tangent],p[:,2+tangent]))
            shadow[i]=False
            result.append(dict(module=int(i),bit=bit,gap=float(gap),shadow=int(shadow.sum())))
    return result

def pair_motion(p,areas,flags):
    n=len(areas);center=p[:,:,:2]+p[:,:,2:]/2
    scale=(np.sqrt(areas)[:,None]+np.sqrt(areas)[None,:])/2
    d=center[:,:,None,:]-center[:,None,:,:]
    half=(p[:,:,None,2:]+p[:,None,:,2:])/2
    pairs=np.triu(np.ones((n,n),bool),1)&~((flags[:,1]!=0)[:,None]&(flags[:,1]!=0)[None,:])
    dist=np.linalg.norm(d[0]/scale[...,None],axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')[:,:min(6,n-1)]
    near=np.zeros((n,n),bool);near[np.arange(n)[:,None],order]=True
    near=(near|near.T)&pairs
    clear=np.where(np.abs(d)-half>.02*scale[None,:,:,None],np.sign(d),0)
    last=clear[0].copy();pending=np.zeros((n,n,2),bool)
    output=[];events=[]
    for k in range(1,10):
        old=d[k-1];new=d[k]
        orderflip=(old*new<0)&(np.abs(old)>.05*scale[...,None])&(np.abs(new)>.05*scale[...,None])
        eligible=(np.abs(old)>.05*scale[...,None])&near[...,None]
        for axis in range(2):
            cross=old[...,axis]*new[...,axis]<0
            t=np.divide(old[...,axis],old[...,axis]-new[...,axis],out=np.zeros_like(scale),where=cross)
            orth=1-axis
            rel=old[...,orth]+t*(new[...,orth]-old[...,orth])
            width=half[k-1,...,orth]+t*(half[k,...,orth]-half[k-1,...,orth])
            through=cross&(width-np.abs(rel)>.02*scale)&pairs
            pending[...,axis]|=through
        exchange=(last*clear[k]<0)&pending&pairs[...,None]
        for i,j,axis in np.argwhere(exchange):events.append(dict(layer=k+3,i=int(i),j=int(j),axis=int(axis),local=bool(near[i,j])))
        resolved=clear[k]!=0
        last=np.where(resolved,clear[k],last);pending=np.where(resolved,False,pending)
        movement=(center[k]-center[k-1]);raw=np.linalg.norm(movement,axis=-1)/np.sqrt(areas)
        relative=np.linalg.norm(movement-np.median(movement,axis=0),axis=-1)/np.sqrt(areas)
        changed_nodes=np.any(orderflip&near[...,None],axis=(1,2))|np.any(orderflip&near[...,None],axis=(0,2))
        full_nodes=np.any(exchange,axis=(1,2))|np.any(exchange,axis=(0,2))
        output.append(dict(layer=k+3,local_pair_axes=int(eligible.sum()),
            local_clear_order_reversals=int((orderflip&near[...,None]).sum()),
            local_clear_order_reversal_rate=float((orderflip&near[...,None]).sum()/max(eligible.sum(),1)),
            full_side_exchanges=int(exchange.sum()),local_full_side_exchanges=int((exchange&near[...,None]).sum()),
            modules_in_full_exchange=int(full_nodes.sum()),modules_with_order_change=float(changed_nodes.mean()),
            raw_move_mean=float(raw.mean()),translation_removed_move_mean=float(relative.mean()),
            large_move025=float((relative>=.25).mean()),large_move05=float((relative>=.5).mean()),large_move1=float((relative>=1.).mean())))
    cumulative=(d[0]*d[-1]<0)&(np.abs(d[0])>.05*scale[...,None])&(np.abs(d[-1])>.05*scale[...,None])&near[...,None]
    return output,events,dict(local_final_order_reversal_rate=float(cumulative.sum()/max((near[...,None]&(np.abs(d[0])>.05*scale[...,None])).sum(),1)))

def relation_metrics(trajectory,reference,baseline3,areas,flags):
    n=len(areas);scale=(np.sqrt(areas)[:,None]+np.sqrt(areas)[None,:])/2
    pairs=np.triu(np.ones((n,n),bool),1)&~((flags[:,1]!=0)[:,None]&(flags[:,1]!=0)[None,:])
    def coded(p):
        centers=p[:,:2]+p[:,2:]/2;d=centers[:,None]-centers[None,:];half=(p[:,None,2:]+p[None,:,2:])/2
        return np.where(np.abs(d)-half>.02*scale[...,None],np.sign(d),0),d,half
    refcode,refd,_=coded(reference);basecode,_,_=coded(baseline3);owncode,_,_=coded(trajectory[3])
    basebad=pairs[...,None]&(refcode!=0)&(basecode==-refcode)
    ownbad=pairs[...,None]&(refcode!=0)&(owncode==-refcode)
    metrics={}
    for name,mask in [('fixed_baseline_S3_wrong',basebad),('own_S3_wrong',ownbad)]:
        strong=[];direction=[]
        for p in trajectory:
            code,d,_=coded(p)
            strong.append(int(((code==refcode)&mask).sum()))
            direction.append(int(((np.sign(d)==np.sign(refd))&mask).sum()))
        metrics[name]=dict(count=int(mask.sum()),strong_repaired=strong,direction_repaired=direction)
    last,d0,h0=coded(trajectory[3]);pending=np.zeros_like(last,dtype=bool);exchanges=[]
    for layer in range(4,13):
        code,d,h=coded(trajectory[layer])
        for axis in range(2):
            cross=d0[...,axis]*d[...,axis]<0
            t=np.divide(d0[...,axis],d0[...,axis]-d[...,axis],out=np.zeros_like(scale),where=cross)
            orth=1-axis;dd=d0[...,orth]+t*(d[...,orth]-d0[...,orth]);hh=h0[...,orth]+t*(h[...,orth]-h0[...,orth])
            pending[...,axis]|=cross&(hh-np.abs(dd)>.02*scale)&pairs
        event=(last*code<0)&pending&pairs[...,None]
        exchanges.append(int(event.sum()))
        resolved=code!=0;last=np.where(resolved,code,last);pending=np.where(resolved,False,pending)
        d0,h0=d,h
    metrics['complete_side_exchanges_by_arrival_S4_to_S12']=exchanges
    return metrics
