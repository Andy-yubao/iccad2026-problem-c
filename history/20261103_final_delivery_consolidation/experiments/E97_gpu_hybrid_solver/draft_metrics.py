"""Exact covered-area union per module; reporting only, not a hidden Loss."""
import numpy as np

def union_area(rectangles):
    if not len(rectangles):return 0.
    r=np.asarray(rectangles);xs=np.unique(np.r_[r[:,0],r[:,0]+r[:,2]]);area=0.
    for left,right in zip(xs[:-1],xs[1:]):
        active=r[(r[:,0]<right)&(r[:,0]+r[:,2]>left)]
        if not len(active):continue
        intervals=sorted(zip(active[:,1],active[:,1]+active[:,3]));a,b=intervals[0];height=0.
        for c,d in intervals[1:]:
            if c>b:height+=b-a;a,b=c,d
            else:b=max(b,d)
        area+=(right-left)*(height+b-a)
    return float(area)

def module_overlap(positions):
    p=np.asarray(positions,dtype=float);lo=p[:,:2];hi=lo+p[:,2:];area=p[:,2:].prod(-1)
    low=np.maximum(lo[:,None],lo[None,:]);size=np.maximum(np.minimum(hi[:,None],hi[None,:])-low,0)
    inter=size.prod(-1);np.fill_diagonal(inter,0);pair=inter.sum(1)/area
    covered=[]
    for i in range(len(p)):
        ids=np.flatnonzero(inter[i]>0);covered.append(union_area(np.c_[low[i,ids],size[i,ids]])/area[i])
    covered=np.asarray(covered)
    bbox=float(np.prod(hi.max(0)-lo.min(0)))
    return dict(N5=int((covered>.05).sum()),N10=int((covered>.10).sum()),
                module_covered_mean=float(covered.mean()),module_covered_p95=float(np.percentile(covered,95)),
                N5_pair_sum=int((pair>.05).sum()),
                max_covered_fraction=float(covered.max()),tail_excess=float(np.maximum(covered-.05,0).sum()),
                bbox_over_A=float(bbox/area.sum()),exact_bbox_void_fraction=1-union_area(p)/bbox)
