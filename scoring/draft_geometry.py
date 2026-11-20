"""Continuous layout diagnostics. No official scoring, inference or repair.

All areas are exact rectangle unions. Contact distance is the smallest relative
translation (fixed current shapes) that produces a positive-length shared edge;
it is a pairwise geometric proxy, not a joint feasibility certificate.
"""
import numpy as np
from shapely.geometry import box, MultiPoint
from shapely.ops import unary_union

EPSILONS = np.array([0., .005, .01, .02, .05, .1, .2, .5, 1., 2.])
SIDES = ((1, 'L'), (2, 'R'), (4, 'T'), (8, 'B'))


def rectangle_polys(p):
    return [box(x, y, x+w, y+h) for x, y, w, h in p]


def pair_geometry(p):
    lo=p[:, :2]; hi=lo+p[:, 2:]
    overlap=np.minimum(hi[:, None], hi[None])-np.maximum(lo[:, None], lo[None])
    inter=np.maximum(overlap, 0).prod(-1)
    np.fill_diagonal(inter, 0)
    return lo, hi, overlap, inter


def contact_distances(p, fraction=.01):
    """Return normalized distances and moving-first-rectangle displacement.

    Tangential contact length is fraction * min(two corresponding side lengths).
    This length is a robustness probe only; it does not alter contest rules.
    """
    lo=p[:, :2]; hi=lo+p[:, 2:]
    local=np.sqrt(p[:, 2]*p[:, 3]); scale=(local[:, None]+local[None])/2
    lx=fraction*np.minimum(p[:, None, 2], p[None, :, 2])
    ly=fraction*np.minimum(p[:, None, 3], p[None, :, 3])
    # Admissible translations on the tangent axis form a closed interval.
    ylow=lo[None, :, 1]+ly-hi[:, None, 1]
    yhigh=hi[None, :, 1]-ly-lo[:, None, 1]
    dy=np.maximum(ylow, np.minimum(0., yhigh))
    xlow=lo[None, :, 0]+lx-hi[:, None, 0]
    xhigh=hi[None, :, 0]-lx-lo[:, None, 0]
    dx=np.maximum(xlow, np.minimum(0., xhigh))
    candidates=np.stack((
        np.stack((lo[None, :, 0]-hi[:, None, 0], dy), -1),
        np.stack((hi[None, :, 0]-lo[:, None, 0], dy), -1),
        np.stack((dx, lo[None, :, 1]-hi[:, None, 1]), -1),
        np.stack((dx, hi[None, :, 1]-lo[:, None, 1]), -1)), 0)
    lengths=np.linalg.norm(candidates, axis=-1)
    choice=lengths.argmin(0)
    delta=np.take_along_axis(candidates, choice[None, ..., None], axis=0)[0]
    distance=lengths.min(0)/scale
    np.fill_diagonal(distance, np.inf)
    return distance, delta


def minimum_tree(d):
    """Deterministic dense Prim; unlike scipy sparse MST, preserves zero edges."""
    n=len(d)
    if n<2:return []
    visited=np.zeros(n, bool); visited[0]=True
    best=d[0].copy(); parent=np.zeros(n, int); edges=[]
    for _ in range(n-1):
        j=int(np.argmin(np.where(visited, np.inf, best)))
        edges.append((int(parent[j]), j, float(best[j])))
        visited[j]=True
        improve=(~visited)&(d[j]<best)
        best[improve]=d[j, improve]; parent[improve]=j
    return edges


def swept_increment(rect, delta):
    x,y,w,h=rect; dx,dy=delta
    corners=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    swept=MultiPoint(corners+[(a+dx,b+dy) for a,b in corners]).convex_hull
    return swept.difference(box(x,y,x+w,y+h))


def intrusion(rect, delta, obstacles):
    sweep=swept_increment(rect, delta)
    if sweep.area<1e-16:return 0.
    return float(sweep.intersection(obstacles).area/max(rect[2]*rect[3], 1e-12))


def wirelength(p, b2b, p2b, pins):
    center=p[:, :2]+p[:, 2:]/2
    b2b=np.asarray(b2b); p2b=np.asarray(p2b)
    b2b=b2b[~(b2b==-1).all(1)] if len(b2b) else b2b
    p2b=p2b[~(p2b==-1).all(1)] if len(p2b) else p2b
    internal=float((b2b[:, 2]*np.abs(center[b2b[:,0].astype(int)]-center[b2b[:,1].astype(int)]).sum(-1)).sum()) if len(b2b) else 0.
    external=float((p2b[:, 2]*np.abs(np.asarray(pins)[p2b[:,0].astype(int), :2]-center[p2b[:,1].astype(int)]).sum(-1)).sum()) if len(p2b) else 0.
    return internal, external


def evaluate_geometry(p, inputs, reference=None, details=True, contact_fraction=.01):
    p=np.asarray(p, float); n,areas,b2b,p2b,pins,constraints,specs=inputs
    areas=np.asarray(areas, float); c=np.asarray(constraints, int); specs=np.asarray(specs, float)
    assert p.shape==(n,4) and np.isfinite(p).all() and (p[:,2:]>0).all()
    L=np.sqrt(areas.sum()); actual=p[:,2:].prod(-1); local=np.sqrt(actual)
    lo,hi,signed,inter=pair_geometry(p)
    bblo=lo.min(0); bbhi=hi.max(0); extent=bbhi-bblo; bbox=float(extent.prod())
    polys=rectangle_polys(p); union=unary_union(polys); total=actual.sum()
    covered=np.zeros(n)
    for i in range(n):
        neighbors=np.flatnonzero(inter[i]>0)
        if len(neighbors):covered[i]=polys[i].intersection(unary_union([polys[j] for j in neighbors])).area/actual[i]
    wi,we=wirelength(p,b2b,p2b,pins)
    out=dict(bbox_area=bbox,bbox_over_public_area=bbox/areas.sum(),
        actual_over_public_area=total/areas.sum(),bbox_aspect=float(extent[0]/extent[1]),
        duplicate_coverage=float(max(0,total-union.area)/total),
        pair_intersection=float(inter.sum()/2/total),
        covered_module_mean=float(covered.mean()),covered_module_p95=float(np.quantile(covered,.95)),
        covered_module_max=float(covered.max()),covered_module_over10=float((covered>.1).mean()),
        whitespace_bbox=float(max(0,bbox-union.area)/bbox),
        whitespace_public=float(max(0,bbox-union.area)/areas.sum()),
        hpwl_internal=wi,hpwl_external=we,hpwl_total=wi+we,
        abs_log_aspect_p95=float(np.quantile(np.abs(np.log(p[:,2]/p[:,3])),.95)),
        aspect_over10_fraction=float((np.maximum(p[:,2]/p[:,3],p[:,3]/p[:,2])>10).mean()),
        worst_covered_module=int(covered.argmax()))
    if reference is not None:
        ref=np.asarray(reference,float); reflo=ref[:,:2].min(0); refhi=(ref[:,:2]+ref[:,2:]).max(0)
        ri,re=wirelength(ref,b2b,p2b,pins)
        out.update(bbox_reference=bbox/float((refhi-reflo).prod()),hpwl_reference=(wi+we)/(ri+re),
            hpwl_internal_reference=wi/ri if ri else None,hpwl_external_reference=we/re if re else None)
    locked=(c[:,0]!=0)|(c[:,1]!=0); pre=c[:,1]!=0
    if locked.any():
        dimension=np.max(np.abs(p[locked,2:]/specs[locked,2:]-1),-1)
        out.update(fixed_dimension_relative_mean=float(dimension.mean()),fixed_dimension_relative_max=float(dimension.max()))
    if pre.any():
        drift=np.linalg.norm(p[pre,:2]-specs[pre,:2],axis=-1)/np.sqrt(specs[pre,2:].prod(-1))
        out.update(preplaced_position_local_mean=float(drift.mean()),preplaced_position_local_max=float(drift.max()))
    distance,delta=contact_distances(p,contact_fraction)
    group_rows=[]; boundary_rows=[]
    for group in sorted(set(c[:,3])-{0}):
        members=np.flatnonzero(c[:,3]==group); m=len(members)
        if m<2:continue
        edges=minimum_tree(distance[np.ix_(members,members)])
        costs=np.array([e[2] for e in edges]); gpoly=unary_union([polys[i] for i in members])
        gbbox=box(*gpoly.bounds); foreign=np.flatnonzero(c[:,3]!=group)
        outside=unary_union([polys[j] for j in foreign])
        row=dict(group=int(group),size=m,contact_bottleneck=float(costs.max()),
            contact_mean=float(costs.mean()),group_bbox_over_area=float(gbbox.area/actual[members].sum()),
            foreign_bbox_fraction=float(outside.intersection(gbbox).area/gbbox.area),
            foreign_empty_bbox_fraction=float(outside.intersection(gbbox.difference(gpoly)).area/gbbox.area),
            same_group_overlap=float((actual[members].sum()-gpoly.area)/actual[members].sum()),
            foreign_center_count=int(sum(gbbox.contains(MultiPoint([tuple(lo[j]+p[j,2:]/2)])) for j in foreign)))
        for eps in EPSILONS:
            row['connectivity_'+str(eps)]=float((costs<=eps+1e-12).mean())
            row['connected_'+str(eps)]=int(costs.max()<=eps+1e-12)
        if details:
            loads=[]; blocked=[]
            for a,b,_ in edges:
                i,j=int(members[a]),int(members[b]); d=delta[i,j]
                # Three specified straight-line allocations; not an all-path search.
                choices=[]
                for amount in (0.,.5,1.):
                    if (pre[i] and amount>0) or (pre[j] and amount<1):continue
                    choices.append(intrusion(p[i],amount*d,outside)+intrusion(p[j],-(1-amount)*d,outside))
                loads.append(min(choices) if choices else np.nan)
                blocked.append(float(min(choices)>1e-12) if choices else np.nan)
            finite=np.isfinite(loads)
            row['tree_swept_foreign_load']=float(np.asarray(loads)[finite].mean()) if finite.any() else None
            row['tree_blocked_fraction']=float(np.asarray(blocked)[finite].mean()) if finite.any() else None
            row['immovable_tree_edges']=int((~finite).sum())
        group_rows.append(row)
    margins=np.stack((lo[:,0]-bblo[0],bbhi[0]-hi[:,0],bbhi[1]-hi[:,1],lo[:,1]-bblo[1]),-1)
    for i in np.flatnonzero(c[:,4]):
        required=[j for j,(bit,_) in enumerate(SIDES) if c[i,4]&bit]
        row=dict(module=int(i),preplaced=int(pre[i]),fixed_shape=int(locked[i]),corner=int(len(required)>1),
            gap_local=float(margins[i,required].max()/local[i]),gap_global=float(margins[i,required].max()/L))
        others=[j for j in range(n) if j!=i]; obstacle=unary_union([polys[j] for j in others]) if details else None
        rank=[]; strip=[]; anchor_depth=[]; anchor_rank=[]
        for j in required:
            name=SIDES[j][1]; gap=margins[i,j]
            excess=(lo[i,0]-lo[:,0] if j==0 else hi[:,0]-hi[i,0] if j==1 else hi[:,1]-hi[i,1] if j==2 else lo[i,1]-lo[:,1])
            count=int((excess>1e-9*L).sum());rank.append(count/max(n-1,1))
            row['gap_'+name]=float(gap/local[i]);row['outside_'+name]=count
            if details:
                d=np.array([[-gap,0],[gap,0],[0,gap],[0,-gap]][j])
                strip.append(intrusion(p[i],d,obstacle))
            if pre[i]:
                sl=specs[i,:2]; sh=sl+specs[i,2:]
                ex=(sl[0]-lo[:,0] if j==0 else hi[:,0]-sh[0] if j==1 else hi[:,1]-sh[1] if j==2 else sl[1]-lo[:,1])
                ex[i]=0
                anchor_depth.append(max(0,float(ex.max()))/np.sqrt(specs[i,2:].prod()))
                anchor_rank.append(float((ex>1e-9*L).sum()/max(n-1,1)))
        row['outside_fraction']=float(max(rank))
        if details:row['swept_obstacle_load']=float(max(strip))
        if anchor_depth:row.update(anchor_outside_depth=max(anchor_depth),anchor_outside_fraction=max(anchor_rank))
        boundary_rows.append(row)
    if group_rows:
        for name in ('contact_bottleneck','contact_mean','group_bbox_over_area','foreign_bbox_fraction','foreign_empty_bbox_fraction','same_group_overlap','tree_swept_foreign_load','tree_blocked_fraction'):
            vals=[r[name] for r in group_rows if r.get(name) is not None]
            if vals:out['group_'+name]=float(np.mean(vals))
        for eps in EPSILONS:
            out['group_connectivity_'+str(eps)]=float(np.average([r['connectivity_'+str(eps)] for r in group_rows],weights=[r['size']-1 for r in group_rows]))
            out['group_connected_'+str(eps)]=float(np.mean([r['connected_'+str(eps)] for r in group_rows]))
    if boundary_rows:
        for name in ('gap_local','outside_fraction','swept_obstacle_load','anchor_outside_depth'):
            vals=[r[name] for r in boundary_rows if name in r]
            if vals:
                out['boundary_'+name]=float(np.mean(vals));out['boundary_'+name+'_p95']=float(np.quantile(vals,.95))
        for eps in EPSILONS:
            out['boundary_near_'+str(eps)]=float(np.mean([r['gap_local']<=eps+1e-12 for r in boundary_rows]))
    return out,group_rows,boundary_rows


def repair_movement(draft, final, areas):
    a=np.asarray(draft,float); b=np.asarray(final,float); L=np.sqrt(np.sum(areas))
    centers_a=a[:,:2]+a[:,2:]/2;centers_b=b[:,:2]+b[:,2:]/2
    move=np.linalg.norm(centers_b-centers_a,axis=-1)/L
    shape=np.abs(np.log(b[:,2]/b[:,3])-np.log(a[:,2]/a[:,3]))
    return dict(move_global_mean=float(move.mean()),move_global_p95=float(np.quantile(move,.95)),
        move_global_max=float(move.max()),shape_log_change_mean=float(shape.mean()),shape_log_change_p95=float(np.quantile(shape,.95)))
