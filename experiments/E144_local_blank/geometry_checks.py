"""Independent small-scene union-area oracle for E144 feedback semantics."""
import numpy as np
import torch
from .geometry_feedback import module_features,FEATURE_NAMES
from .blank_geometry import blank_strips


def union_area(rects):
    if not rects:return 0.
    xs=sorted({x for a in rects for x in (a[0],a[2])})
    total=0.
    for l,r in zip(xs,xs[1:]):
        intervals=sorted((a[1],a[3]) for a in rects if a[0]<(l+r)/2<a[2])
        if not intervals:continue
        start,end=intervals[0];height=0.
        for a,b in intervals[1:]:
            if a>end:height+=end-start;start,end=a,b
            else:end=max(end,b)
        total+=(r-l)*(height+end-start)
    return total


def oracle(q,valid):
    out=np.zeros((len(q),4));real=q[valid]
    if not len(real):return out
    bounds_lo=real[:,:2].min(0);bounds_hi=(real[:,:2]+real[:,2:]).max(0)
    for i,(x,y,w,h) in enumerate(q):
        if not valid[i]:continue
        strips=((x-.25*w,y,x,y+h),(x+w,y,x+1.25*w,y+h),
            (x,y+h,x+w,y+1.25*h),(x,y-.25*h,x+w,y))
        for d,strip in enumerate(strips):
            low=np.maximum(strip[:2],bounds_lo);high=np.minimum(strip[2:],bounds_hi)
            size=np.maximum(high-low,0).prod();rects=[]
            if size<=0:continue
            for j in np.flatnonzero(valid):
                if j==i:continue
                lo=np.maximum(low,q[j,:2]);hi=np.minimum(high,q[j,:2]+q[j,2:])
                if np.all(hi>lo):rects.append([*lo,*hi])
            out[i,d]=(size-union_area(rects))/(w*h)
    return out


def synthetic():
    from experiments.E142_dual_geometry.geometry_feedback import module_features as parent_features
    rng=np.random.default_rng(20261116);count=0
    scenes=[np.asarray([[2,2,2,2],[4,2,2,1],[4,2,2,1],[0,0,1,1],[9,9,1,1]],float)]
    scenes += [np.column_stack((rng.uniform(-2,8,(8,2)),rng.uniform(.1,5,(8,2)))) for _ in range(16)]
    for q in scenes:
        p=torch.from_numpy(q[None]);valid=torch.ones((1,len(q)),dtype=torch.bool)
        valid[0,-1]=False
        lo=p[...,:2].masked_fill(~valid[...,None],float('inf')).amin(1)
        hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-float('inf')).amax(1)
        actual=blank_strips(p,valid,lo,hi)[0].numpy()
        np.testing.assert_allclose(actual,oracle(q,valid[0].numpy()),rtol=1e-10,atol=2e-12)
        np.testing.assert_allclose(blank_strips(p.float().cuda(),valid.cuda(),lo.float().cuda(),hi.float().cuda()).cpu()[0],actual,rtol=2e-4,atol=2e-6)
        count+=len(q)*4
    p=torch.tensor([[[2.,2.,2.,2.],[3.,2.5,2.,2.],[7.,6.,1.,1.],[80.,80.,1.,1.]]],dtype=torch.double)
    valid=torch.tensor([[True,True,True,False]])
    group=torch.zeros((1,4,4),dtype=torch.double);group[:,0,2]=group[:,2,0]=1
    c=torch.zeros((1,4,5),dtype=torch.long);c[:,0,4]=9
    g=dict(area=p[...,2:].prod(-1),valid=valid,group=group,constraints=c)
    z=module_features(p,g);old=parent_features(p,g)
    assert z.shape==(1,4,45) and len(FEATURE_NAMES)==45 and not z.requires_grad
    torch.testing.assert_close(z[0,0,8:12],torch.tensor([.5,1.,1.,.75],dtype=torch.double),rtol=0,atol=0)
    torch.testing.assert_close(z[...,:8],old[...,:8],rtol=0,atol=0)
    torch.testing.assert_close(z[...,16:],old[...,20:],rtol=0,atol=0)
    for factor in (.1,10.):torch.testing.assert_close(module_features(p*factor,dict(g,area=g['area']*factor**2)),z,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(module_features(p+torch.tensor([4.2,-.7,0,0]),g),z,rtol=1e-6,atol=1e-7)
    perm=torch.tensor([2,3,0,1]);gg=dict(area=g['area'][:,perm],valid=valid[:,perm],group=group[:,perm][:,:,perm],constraints=c[:,perm])
    torch.testing.assert_close(module_features(p[:,perm],gg),z[:,perm],rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(module_features(p[...,[1,0,3,2]],dict(g,constraints=torch.zeros_like(c)))[...,8:16],
        module_features(p,dict(g,constraints=torch.zeros_like(c)))[...,[11,10,9,8,15,14,13,12]],rtol=1e-10,atol=1e-12)
    one=dict(area=g['area'][:,:1],valid=valid[:,:1],group=group[:,:1,:1],constraints=c[:,:1])
    assert torch.count_nonzero(module_features(p[:,:1],one)[...,8:16])==0
    assert torch.count_nonzero(module_features(p,dict(g,valid=torch.zeros_like(valid))))==0
    assert torch.count_nonzero(z[:,-1])==0 and bool(torch.isfinite(z).all())
    # Predicted own area, not public target area, is the blank denominator.
    lo=p[...,:2].masked_fill(~valid[...,None],float('inf')).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-float('inf')).amax(1)
    torch.testing.assert_close(module_features(p,dict(g,area=g['area']*4))[...,12:16],blank_strips(p,valid,lo,hi),rtol=0,atol=0)
    return dict(independent_union_strip_queries=count,CPU_FP64_and_CUDA_FP32=True,overlapping_blockers_union=True,
        bounded_escape_side_ratio=True,new_strips_only=True,bbox_outside_excluded=True,predicted_area_denominator=True,
        retained_E142_features_exact=True,scale_translation_axis_permutation=True,singleton_empty_padding=True)
