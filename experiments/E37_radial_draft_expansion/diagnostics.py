"""Public-input geometry diagnostics, evaluated outside optimizer clocks."""
import numpy as np
from .backend.frozen.feasible import Objective
from .reference_fast_geometry import topology


def move(a,b,units):
    a=np.asarray(a);b=np.asarray(b);delta=(b[:,:2]+b[:,2:]/2)-(a[:,:2]+a[:,2:]/2)
    d=np.linalg.norm(delta,axis=1)/units
    ratio=np.abs(np.log((b[:,2]/b[:,3])/(a[:,2]/a[:,3])))
    return {'center_mean':float(d.mean()),'center_p95':float(np.percentile(d,95)),'center_max':float(d.max()),
        'log_AR_mean':float(ratio.mean()),'AR_change_ge2_fraction':float((ratio>=np.log(2)).mean())}


def geometry(x,p,original=None):
    x=np.asarray(x);units=np.sqrt(p.areas.copy());locked=p.fixed|p.preplaced;units[locked]=np.sqrt(p.specs[locked,2:].prod(1))
    i,j=np.triu_indices(p.n,1);overlap=np.minimum(x[i,:2]+x[i,2:],x[j,:2]+x[j,2:])-np.maximum(x[i,:2],x[j,:2])
    hit=(overlap>1e-6).all(1);depth=np.min(overlap[hit],axis=1)/np.minimum(units[i[hit]],units[j[hit]])
    degrees=np.bincount(np.r_[i[hit],j[hit]],minlength=p.n)
    lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0);centers=x[:,:2]+x[:,2:]/2
    wire=0.
    if len(p.b2b):
        a,b=p.b2b[:,:2].astype(int).T;wire+=float((np.abs(centers[a]-centers[b]).sum(1)*p.b2b[:,2]).sum())
    if len(p.p2b):
        a,b=p.p2b[:,:2].astype(int).T;wire+=float((np.abs(p.pins[a]-centers[b]).sum(1)*p.p2b[:,2]).sum())
    offset=np.linalg.norm(x[p.preplaced,:2]-p.specs[p.preplaced,:2],axis=1)/units[p.preplaced]
    full=topology(x,p.constraints,array=True);a,b,axis=full.T;tangent=1-axis
    gap=(x[b,axis]-x[a,axis]-x[a,axis+2])/np.minimum(units[a],units[b])
    shared=np.minimum(x[a,tangent]+x[a,tangent+2],x[b,tangent]+x[b,tangent+2])-np.maximum(x[a,tangent],x[b,tangent])
    near=(np.abs(gap)<=.02)&(shared>.02*np.minimum(units[a],units[b]))
    grouped=(p.constraints[a,3]>0)&(p.constraints[a,3]==p.constraints[b,3])
    boundary,grouping,mib=Objective(p).violations(x)
    result={'overlap_pairs':int(hit.sum()),'overlap_area_pair_sum_over_total':float(np.maximum(overlap,0).prod(1).sum()/sum(units**2)),
        'overlap_participating_fraction':float((degrees>0).mean()),'max_penetration':float(depth.max()) if len(depth) else 0.,
        'negative_coordinate_blocks':int((x[:,:2]<0).any(1).sum()),'most_negative_coordinate':float(min(0,x[:,:2].min())),
        'preplaced_offset_blocks':int((offset>1e-8).sum()),'preplaced_offset_max_block_units':float(offset.max()) if len(offset) else 0.,
        'bbox_area':float(np.prod(hi-lo)),'wire':wire,'boundary_proxy_violations':boundary,'grouping_proxy_violations':grouping,'mib_proxy_violations':mib,
        'near_contact_pairs':int(near.sum()),'near_group_contact_pairs':int((near&grouped).sum())}
    if original is not None:
        old=topology(original,p.constraints,array=True);result['topology_pair_flip_fraction']=float(np.any(full!=old,axis=1).mean())
        oa,ob,ox=old.T;ot=1-ox;oldgap=(original[ob,ox]-original[oa,ox]-original[oa,ox+2])/np.minimum(units[oa],units[ob])
        oldshared=np.minimum(original[oa,ot]+original[oa,ot+2],original[ob,ot]+original[ob,ot+2])-np.maximum(original[oa,ot],original[ob,ot])
        oldnear=(np.abs(oldgap)<=.02)&(oldshared>.02*np.minimum(units[oa],units[ob]))
        # Compare each original preferred relation, including after shapes change.
        newgap=(x[ob,ox]-x[oa,ox]-x[oa,ox+2])/np.minimum(units[oa],units[ob])
        newshared=np.minimum(x[oa,ot]+x[oa,ot+2],x[ob,ot]+x[ob,ot+2])-np.maximum(x[oa,ot],x[ob,ot])
        newnear=(np.abs(newgap)<=.02)&(newshared>.02*np.minimum(units[oa],units[ob]))
        oldgroup=(p.constraints[oa,3]>0)&(p.constraints[oa,3]==p.constraints[ob,3])
        result['original_near_contact_retention']=float((newnear&oldnear).sum()/oldnear.sum()) if oldnear.any() else None
        result['original_group_near_contact_retention']=float((newnear&oldnear&oldgroup).sum()/(oldnear&oldgroup).sum()) if (oldnear&oldgroup).any() else None
        result['movement_from_original']=move(original,x,units)
    return result
