"""Exact union-area queries on current layout; public, detached, no rasterization."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def blank_strips(p,valid,bblo,bbhi):
    """Newly entered .25-side strips, within current bbox, divided by own XYWH area.

    Coordinate compression gives a union occupancy grid in O(BN^2) storage.
    Integrating its piecewise constant cells gives exact rectangular queries
    (up to floating point). Own rectangle never enters the new strips' interiors.
    """
    b,n,_=p.shape
    lo=p[...,:2];wh=p[...,2:];hi=lo+wh
    # Shift coordinates to bbox origin; FP64 prefix areas avoid catastrophic
    # subtraction of large prefix areas for small strip queries.
    lower=(lo-bblo[:,None]).double();upper=(hi-bblo[:,None]).double()
    extent=(bbhi-bblo).double()
    endpoints=torch.cat((lower,upper),1)
    endpoints=torch.where(torch.cat((valid,valid),1)[...,None],endpoints,extent[:,None])
    x,x_order=endpoints[...,0].sort(dim=1,stable=True)
    y,y_order=endpoints[...,1].sort(dim=1,stable=True)
    m=2*n
    # Invert the permutations without scatter assignment: the Windows
    # deterministic scatter-assignment dispatcher synchronizes the host.
    xr=x_order.argsort(dim=1,stable=True)
    yr=y_order.argsort(dim=1,stable=True)
    xl,xh=xr[:,:n],xr[:,n:];yl,yh=yr[:,:n],yr[:,n:]
    slabs=torch.arange(m-1,device=p.device)[None,:,None]
    cx=((slabs>=xl[:,None])&(slabs<xh[:,None])&valid[:,None]).float()
    cy=((slabs>=yl[:,None])&(slabs<yh[:,None])&valid[:,None]).float()
    # Products/sums of 0/1 are exact for N<=120. This deterministic GEMM
    # avoids the host synchronization in Windows' deterministic scatter_add.
    occupied=(cx@cy.transpose(1,2))>0
    cells=occupied*(x[:,1:]-x[:,:-1])[:,:,None]*(y[:,1:]-y[:,:-1])[:,None,:]
    integral=F.pad(cells.cumsum(1).cumsum(2),(1,0,1,0))

    widths=wh.double()
    # L, R, T, B. Only the new strip, never the retained .75 of own rectangle.
    qlo=torch.stack((lower-torch.stack((.25*widths[...,0],torch.zeros_like(widths[...,1])),-1),
        torch.stack((upper[...,0],lower[...,1]),-1),
        torch.stack((lower[...,0],upper[...,1]),-1),
        lower-torch.stack((torch.zeros_like(widths[...,0]),.25*widths[...,1]),-1)),2)
    qhi=torch.stack((torch.stack((lower[...,0],upper[...,1]),-1),
        upper+torch.stack((.25*widths[...,0],torch.zeros_like(widths[...,1])),-1),
        upper+torch.stack((torch.zeros_like(widths[...,0]),.25*widths[...,1]),-1),
        torch.stack((upper[...,0],lower[...,1]),-1)),2)
    qlo=torch.maximum(qlo,torch.zeros_like(qlo));qlo=torch.minimum(qlo,extent[:,None,None])
    qhi=torch.maximum(qhi,torch.zeros_like(qhi));qhi=torch.minimum(qhi,extent[:,None,None])

    def prefix(q):
        q=q.reshape(b,-1,2)
        ix=torch.searchsorted(x.contiguous(),q[...,0].contiguous(),right=True).sub(1).clamp(0,m-2)
        iy=torch.searchsorted(y.contiguous(),q[...,1].contiguous(),right=True).sub(1).clamp(0,m-2)
        x0=x.gather(1,ix);x1=x.gather(1,ix+1)
        y0=y.gather(1,iy);y1=y.gather(1,iy+1)
        tx=((q[...,0]-x0)/(x1-x0).clamp_min(1e-24)).clamp(0,1)
        ty=((q[...,1]-y0)/(y1-y0).clamp_min(1e-24)).clamp(0,1)
        flat=integral.reshape(b,-1)
        z00=flat.gather(1,ix*m+iy);z10=flat.gather(1,(ix+1)*m+iy)
        z01=flat.gather(1,ix*m+iy+1);z11=flat.gather(1,(ix+1)*m+iy+1)
        return ((1-tx)*(1-ty)*z00+tx*(1-ty)*z10+(1-tx)*ty*z01+tx*ty*z11).reshape(b,n,4)
    covered=prefix(qhi)-prefix(torch.stack((qlo[...,0],qhi[...,1]),-1))-prefix(torch.stack((qhi[...,0],qlo[...,1]),-1))+prefix(qlo)
    query_area=(qhi-qlo).clamp_min(0).prod(-1)
    blank=(query_area-covered).clamp_min(0)
    return (blank/widths.prod(-1).clamp_min(1e-24)[...,None]).clamp(0,.25).to(p.dtype)*valid[...,None]
