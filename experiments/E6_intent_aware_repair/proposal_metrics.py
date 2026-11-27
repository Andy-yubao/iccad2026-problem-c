"""E1 raw-proposal geometry diagnostics; no legalization or candidate selection."""
from __future__ import annotations
import numpy as np

TOL = 1e-6


def mean(values):
    a=np.asarray(values,dtype=float)
    return float(a.mean()) if a.size else None


def percentile(values,q):
    a=np.asarray(values,dtype=float)
    return float(np.quantile(a,q)) if a.size else None


def weighted(values,weights):
    v=np.asarray(values,dtype=float);w=np.asarray(weights,dtype=float)
    return float(v@w/w.sum()) if v.size and w.sum()>0 else None


def movement_bound_with_shape_relaxation(positions,effective_areas,constraints,eta):
    """Necessary center-movement bound allowing bounded log-width/height changes.

    Relaxation: both free dimensions may shrink by exp(-eta) independently,
    dropping area preservation. Thus the bound is optimistic even for joint
    position/shape repair; a small value is NOT a repair witness. Public fixed
    dimensions are assumed already locked in positions. Centers move by at most
    tau*sqrt(effective_area), preplaced centers by0. Each pair must separate on
    at least one axis; its optimistic least remaining penetration divided by
    the sum of movable local scales lower-bounds tau. Constraints coupling
    different pairs can only increase the true required budget.
    """
    p=np.asarray(positions,dtype=float);c=np.asarray(constraints);a=np.asarray(effective_areas,dtype=float)
    if eta<0 or (a<=0).any():raise ValueError('Expected nonnegative eta and positive areas')
    i,j=np.triu_indices(len(p),1);center=p[:,:2]+p[:,2:]/2
    free=~c[:,:2].any(1);factor=np.where(free,np.exp(-eta),1.)
    minimum_wh=p[:,2:]*factor[:,None]
    penetration=(minimum_wh[i]+minimum_wh[j])/2-np.abs(center[i]-center[j])
    needed=(penetration.min(1)-TOL).clip(min=0)
    movement=np.sqrt(a)*(c[:,1]==0);denom=movement[i]+movement[j]
    locked=int(((denom==0)&(needed>0)).sum())
    values=needed[denom>0]/denom[denom>0]
    return {'lower_bound':None if locked else float(values.max()) if len(values) else 0.,'locked_pair_failures':locked}


def joint_bound_checks():
    a=np.ones(2);c=np.zeros((2,5),dtype=int)
    p=np.array([[0,0,1,1],[0,0,1,1]],dtype=float)
    b0=movement_bound_with_shape_relaxation(p,a,c,0)['lower_bound']
    b5=movement_bound_with_shape_relaxation(p,a,c,.05)['lower_bound']
    assert abs(b0-(1-TOL)/2)<1e-12 and abs(b5-(np.exp(-.05)-TOL)/2)<1e-12
    p[1,0]=1
    assert movement_bound_with_shape_relaxation(p,a,c,.05)['lower_bound']==0
    p[1,0]=0;c[0,0]=1
    assert abs(movement_bound_with_shape_relaxation(p,a,c,.05)['lower_bound']-((1+np.exp(-.05))/2-TOL)/2)<1e-12
    c[:,1]=1
    assert movement_bound_with_shape_relaxation(p,a,c,.05)['locked_pair_failures']==1
    return ['coincident_free_boxes_optimistic_shrink_bound','touching_boxes_need_zero_movement',
            'fixed_dimensions_do_not_shrink','locked_conflict_is_explicit']


def mst_length(dist):
    if len(dist)<2:return 0.
    used=np.zeros(len(dist),dtype=bool);best=np.full(len(dist),np.inf);best[0]=0.;total=0.
    for _ in range(len(dist)):
        i=int(np.argmin(np.where(used,np.inf,best)))
        total+=float(best[i]);used[i]=True;best=np.minimum(best,dist[i])
    return total


def evaluate_proposal(positions,reference,areas,constraints,b2b=None,p2b=None):
    p=np.asarray(positions,dtype=float);r=np.asarray(reference,dtype=float)
    c=np.asarray(constraints,dtype=int);a=np.asarray(areas,dtype=float)
    n=len(p)
    if p.shape!=(n,4) or r.shape!=p.shape or c.shape[0]!=n or not np.isfinite(p).all() or (p[:,2:]<=0).any():
        raise ValueError('Expected finite positive XYWH rectangles')
    local=np.sqrt(a);scale=np.sqrt(a.sum());pre=c[:,1]!=0;free=~((c[:,0]!=0)|pre);movable=~pre
    pc=p[:,:2]+p[:,2:]/2;rc=r[:,:2]+r[:,2:]/2
    d=pc-rc;norm=np.linalg.norm(d,axis=1)/local
    shift=d[movable].mean(0) if movable.any() else np.zeros(2)
    i,j=np.triu_indices(n,1);pair_scale=(local[i]+local[j])/2
    pd=pc[j]-pc[i];rd=rc[j]-rc[i]
    pair_error=np.linalg.norm(pd-rd,axis=1)
    group_pair=(c[i,3]>0)&(c[i,3]==c[j,3])
    ref_margin=np.column_stack((r[j,0]-r[i,0]-r[i,2],r[i,0]-r[j,0]-r[j,2],
                                r[j,1]-r[i,1]-r[i,3],r[i,1]-r[j,1]-r[j,3]))
    pred_margin=np.column_stack((p[j,0]-p[i,0]-p[i,2],p[i,0]-p[j,0]-p[j,2],
                                 p[j,1]-p[i,1]-p[i,3],p[i,1]-p[j,1]-p[j,3]))
    allowed=ref_margin>=-TOL;valid=allowed.any(1)
    center_order=np.column_stack((pd[:,0]>TOL,pd[:,0]<-TOL,pd[:,1]>TOL,pd[:,1]<-TOL))
    best_violation=np.where(allowed,np.maximum(-pred_margin-TOL,0),np.inf).min(1)
    lo=np.maximum(p[i,:2],p[j,:2]);hi=np.minimum(p[i,:2]+p[i,2:],p[j,:2]+p[j,2:]);intersection=hi-lo
    overlaps=(intersection>TOL).all(1)
    oa=np.where(overlaps,np.maximum(intersection[:,0],0)*np.maximum(intersection[:,1],0),0)
    affected=np.zeros(n,dtype=bool);affected[i[overlaps]]=True;affected[j[overlaps]]=True
    per_block=np.zeros(n);np.add.at(per_block,i,oa);np.add.at(per_block,j,oa)
    # Half-extent sum, not intersection width: handles rectangle containment.
    penetration=(p[i,2:]+p[j,2:])/2-np.abs(pd)
    separation=np.maximum(penetration.min(1)-TOL,0)
    movement_scale=local*movable
    denom=movement_scale[i]+movement_scale[j]
    locked_conflicts=overlaps&(denom==0)
    lb=separation[overlaps&~locked_conflicts]/denom[overlaps&~locked_conflicts]
    gap=np.maximum(np.abs(pd)-(p[i,2:]+p[j,2:])/2,0)
    dist=np.linalg.norm(gap,axis=1);dist_matrix=np.zeros((n,n));dist_matrix[i,j]=dist;dist_matrix[j,i]=dist
    ref_intersection=np.minimum(r[i,:2]+r[i,2:],r[j,:2]+r[j,2:])-np.maximum(r[i,:2],r[j,:2])
    def contacts(x):
        return ((np.abs(x[:,0])<=TOL)&(x[:,1]>TOL))|((np.abs(x[:,1])<=TOL)&(x[:,0]>TOL))
    ref_contact=contacts(ref_intersection)&group_pair;pred_contact=contacts(intersection)
    # Near edge contact is a continuous diagnostic; corner contact can have zero
    # distance, so the strict positive-shared-edge recall is always also given.
    plane=np.abs(np.abs(pd)-(p[i,2:]+p[j,2:])/2)
    contact_distance=np.minimum(np.hypot(plane[:,0],gap[:,1]),np.hypot(plane[:,1],gap[:,0]))/pair_scale
    boundaries=[];bbox_lo=p[:,:2].min(0);bbox_hi=(p[:,:2]+p[:,2:]).max(0)
    side_dist=np.column_stack((p[:,0]-bbox_lo[0],bbox_hi[0]-p[:,0]-p[:,2],
                               bbox_hi[1]-p[:,1]-p[:,3],p[:,1]-bbox_lo[1]))
    for k in np.flatnonzero(c[:,4]):
        selected=(c[k,4]&np.array([1,2,4,8]))!=0
        boundaries.append(float(np.abs(side_dist[k,selected]).max()/local[k]))
    mib=[];group_mst=0.
    for g in np.unique(c[:,2]):
        if g<=0:continue
        ids=np.flatnonzero(c[:,2]==g)
        logs=np.log(p[ids,2:]);mib.extend(np.abs(logs-np.median(logs,axis=0)).mean(1).tolist())
    for g in np.unique(c[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(c[:,3]==g);group_mst+=mst_length(dist_matrix[ids][:,ids])
    ref_spread=np.linalg.norm(rc-rc.mean(0),axis=1)
    pred_spread=np.linalg.norm(pc-pc.mean(0),axis=1)
    out={
        'center_error_local_mean':mean(norm[movable]),'center_error_local_p95':percentile(norm[movable],.95),
        'center_error_local_max':percentile(norm[movable],1),
        'center_error_local_shift_aligned_mean':mean((np.linalg.norm(d-shift,axis=1)/local)[movable]),
        'center_within_5pct_local_fraction':mean(norm[movable]<=.05),
        'pair_relative_local_mean':mean(pair_error/pair_scale),
        'pair_relative_local_p95':percentile(pair_error/pair_scale,.95),
        'pair_relative_global_mean':mean(pair_error/scale),
        'group_relative_local_mean':mean((pair_error/pair_scale)[group_pair]),
        'direction_reference_pair_count':int(valid.sum()),
        'direction_any_center_order_retained_fraction':mean((allowed&center_order).any(1)[valid]),
        'direction_any_separation_retained_fraction':mean((allowed&(pred_margin>=-TOL)).any(1)[valid]),
        'direction_each_separation_retained_fraction':mean((pred_margin>=-TOL)[allowed]),
        'direction_best_violation_local_mean':mean(best_violation[valid]/pair_scale[valid]),
        'overlap_pair_count':int(overlaps.sum()),'overlap_block_fraction':mean(affected),
        'pair_overlap_area_fraction':float(oa.sum()/a.sum()),
        'block_overlap_area_fraction_mean':mean(per_block/a),
        'block_overlap_area_fraction_p95':percentile(per_block/a,.95),
        'max_block_overlap_area_fraction':percentile(per_block/a,1),
        'overlap_separation_local_p95':percentile((separation/pair_scale)[overlaps],.95),
        'overlap_separation_local_max':percentile((separation/pair_scale)[overlaps],1),
        'translation_radius_lb':None if locked_conflicts.any() else float(lb.max()) if lb.size else 0.,
        'translation_locked_pair_failure_count':int(locked_conflicts.sum()),
        'center_spread_ratio':float(np.sqrt(np.mean(pred_spread**2))/max(np.sqrt(np.mean(ref_spread**2)),1e-12)),
        'shape_log_ratio_rmse_free':float(np.sqrt(np.mean((np.log(p[free,2]/p[free,3])-np.log(r[free,2]/r[free,3]))**2))) if free.any() else None,
        'shape_wh_relative_mean_free':mean(np.abs(p[free,2:]/r[free,2:]-1)),
        'boundary_distance_local_mean':mean(boundaries),'boundary_distance_local_p95':percentile(boundaries,.95),
        'group_mst_gap_global_sum':float(group_mst/scale),
        'group_reference_contact_count':int(ref_contact.sum()),
        'group_reference_contact_strict_recall':mean(pred_contact[ref_contact]),
        'group_reference_contact_within_5pct_recall':mean(contact_distance[ref_contact]<=.05),
        'group_reference_contact_within_10pct_recall':mean(contact_distance[ref_contact]<=.1),
        'mib_log_wh_deviation_mean':mean(mib),
    }
    if b2b is not None:
        edges=np.asarray(b2b,dtype=float).reshape(-1,3)
        edges=edges[(edges[:,0]>=0)&(edges[:,1]>=0)&(edges[:,2]>0)&(edges[:,0]!=edges[:,1])]
        ei,ej=edges[:,:2].astype(int).T;w=edges[:,2]
        ep=pc[ej]-pc[ei];er=rc[ej]-rc[ei]
        out['edge_relative_local_weighted_mean']=weighted(np.linalg.norm(ep-er,axis=1)/((local[ei]+local[ej])/2),w)
        out['edge_distance_ratio_weighted_mean']=weighted(np.linalg.norm(ep,axis=1)/np.maximum(np.linalg.norm(er,axis=1),1e-12),w)
    if p2b is not None:
        edges=np.asarray(p2b,dtype=float).reshape(-1,3);edges=edges[(edges[:,0]>=0)&(edges[:,1]>=0)&(edges[:,2]>0)]
        ids=edges[:,1].astype(int)
        out['pin_center_error_global_weighted_mean']=weighted(np.linalg.norm(d[ids],axis=1)/scale,edges[:,2])
    return out


def definition_checks():
    checks=[]
    c=np.zeros((2,5),dtype=int);c[:,3]=1;a=np.ones(2)
    r=np.array([[0,0,1,1],[1,0,1,1]],dtype=float)
    x=evaluate_proposal(r,r,a,c)
    assert x['overlap_pair_count']==0 and x['translation_radius_lb']==0 and x['group_reference_contact_strict_recall']==1
    checks.append('edge_contact_is_legal_and_retained')
    p=r.copy();p[1,:2]=p[0,:2];x=evaluate_proposal(p,r,a,c)
    assert abs(x['translation_radius_lb']-.5)<1e-6 and x['center_spread_ratio']==0
    checks.append('coincident_unit_boxes_have_half_size_translation_lower_bound')
    p=np.array([[0,0,10,10],[4,4,2,2]],dtype=float);x=evaluate_proposal(p,p,np.array([100,4]),c)
    assert abs(x['translation_radius_lb']-.5)<1e-6 and abs(x['overlap_separation_local_max']-1)<1e-6
    checks.append('containment_uses_separation_distance_not_intersection_width')
    p=r.copy();p[:,:2]+=[1,2];x=evaluate_proposal(p,r,a,c)
    assert x['pair_relative_local_mean']==0 and x['center_error_local_mean']>0
    checks.append('translation_invariance_does_not_hide_absolute_error')
    p=np.array([[0,0,1,1],[1,1,1,1]],dtype=float);x=evaluate_proposal(p,p,a,c)
    assert x['group_reference_contact_count']==0 and x['group_mst_gap_global_sum']==0
    checks.append('corner_contact_is_not_positive_shared_edge')
    return checks
