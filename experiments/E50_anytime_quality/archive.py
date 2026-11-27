"""Quality-first, shared-deadline portfolio; public inputs and draft only.

The observer can record official scores but its return value is never read.
All online choices and stopping decisions are reference-free.
"""
import hashlib
import math
import time
import numpy as np
from .backend.frozen.packing import prepare_problem,_dimensions
from .backend.frozen.feasible import Objective,row_control
from .backend.frozen.solver import hard_valid
from .backend.guard import repair_tiny_negative
from .canonical import canonicalize
from .fast_geometry import topology,step,reduce_edges,contact_tree
from .projection import solve as project


def raw_measures(x,p):
    centers=x[:,:2]+x[:,2:]/2;wire=0.
    if len(p.b2b):
        i,j=p.b2b[:,:2].astype(int).T;wire+=float((np.abs(centers[i]-centers[j]).sum(1)*p.b2b[:,2]).sum())
    if len(p.p2b):
        i,j=p.p2b[:,:2].astype(int).T;wire+=float((np.abs(p.pins[i]-centers[j]).sum(1)*p.p2b[:,2]).sum())
    return float(np.prod((x[:,:2]+x[:,2:]).max(0)-x[:,:2].min(0))),wire


class Archive:
    def __init__(self,p,draft,start,observer):
        self.p=p;self.draft=draft.copy();self.start=start;self.observer=observer
        self.obj=Objective(p);self.rows=[];self.seen=set();self.rejected=0
        self.center=draft[:,:2]+draft[:,2:]/2;self.local=np.sqrt(p.areas);self.free=~p.preplaced;self.shape_free=~(p.preplaced|p.fixed)
        self.diagonal=float(np.linalg.norm((draft[:,:2]+draft[:,2:]).max(0)-draft[:,:2].min(0)))
        self.draft_wire=max(raw_measures(draft,p)[1],1e-8)
        self.original_relation=topology(draft,p.constraints,array=True)
        self.near=np.ones(len(self.original_relation),bool)
        if p.n>4:
            dist=np.linalg.norm(self.center[:,None]-self.center[None,:],axis=-1);np.fill_diagonal(dist,np.inf)
            neighbors=np.argsort(dist,axis=1)[:,:3];mask=np.zeros((p.n,p.n),bool)
            np.put_along_axis(mask,neighbors,True,axis=1);mask|=mask.T
            i,j=np.tril_indices(p.n,-1);self.near=mask[i,j]
    def distance(self,x):
        delta=np.linalg.norm(x[:,:2]+x[:,2:]/2-self.center,axis=1)
        local=delta/self.local;free=local[self.free]
        mean=float(free.mean()) if len(free) else 0.;p95=float(np.quantile(free,.95)) if len(free) else 0.
        shape=np.abs(np.log((x[:,2]/x[:,3])/(self.draft[:,2]/self.draft[:,3])))
        shape_mean=float(shape[self.shape_free].mean()) if self.shape_free.any() else 0.
        rel=topology(x,self.p.constraints,array=True)
        flips=float(np.any(rel[self.near]!=self.original_relation[self.near],axis=1).mean()) if self.near.any() else 0.
        score=.5*mean+.2*p95+.2*shape_mean+.1*flips
        return {'D':score,'mean_center_local':mean,'p95_center_local':p95,'max_center_local':float(local.max()),
            'mean_center_over_chip_diagonal':float(delta[self.free].mean()/max(self.diagonal,1e-8)) if self.free.any() else 0.,
            'max_center_over_chip_diagonal':float(delta.max()/max(self.diagonal,1e-8)),
            'mean_abs_log_aspect_change':shape_mean,'near_relation_change_fraction':flips}
    def accept(self,candidate,source):
        if candidate is None:return None
        try:
            raw=np.asarray(candidate,float)
            locked=self.p.fixed|self.p.preplaced
            if raw.shape!=(self.p.n,4) or not np.isfinite(raw).all() or (raw[:,2:]<=0).any():
                self.rejected+=1;return None
            if (np.abs(raw[locked,2:]-self.p.specs[locked,2:])>1e-8).any() or (np.abs(raw[self.p.preplaced]-self.p.specs[self.p.preplaced])>1e-8).any():
                self.rejected+=1;return None
            x=canonicalize(raw,self.p,tolerance=1e-4)
            x,_=repair_tiny_negative(x,self.p)
            if not hard_valid(x,self.p):self.rejected+=1;return None
            key=hashlib.sha256(x.tobytes()).hexdigest()
            if key in self.seen:return x
            self.seen.add(key)
            area,wire=raw_measures(x,self.p);violations=self.obj.violations(x)
            distance=self.distance(x)
            # Monotone public objective; no reference denominators or true Q.
            public=math.log(max(.5+.5*(area/self.obj.area_scale+wire/self.draft_wire),1e-12))+2*sum(violations)/self.obj.nsoft
            row={'id':len(self.rows),'source':source,'found_seconds':time.perf_counter()-self.start,
                'public_quality_cost':public,'distance':distance,'balanced_cost':public+.03*distance['D'],
                'area':area,'wire':wire,'soft_violations':list(map(int,violations)),
                'shelf_fallback':source=='shelf_fallback','positions':x.tolist()}
            if self.observer is not None:self.observer(x,row)  # deliberately ignore return value
            self.rows.append(row)
            return x
        except (ValueError,FloatingPointError):
            self.rejected+=1;return None
    def best(self,weight=.03,exclude_shelf=False):
        rows=[r for r in self.rows if not(exclude_shelf and r['shelf_fallback'])]
        if not rows:return None
        return min(rows,key=lambda r:r['public_quality_cost']+weight*r['distance']['D'])
    def frontier(self):
        return [r['id'] for r in self.rows if not any(
            other['public_quality_cost']<=r['public_quality_cost'] and other['distance']['D']<=r['distance']['D'] and
            (other['public_quality_cost']<r['public_quality_cost']-1e-10 or other['distance']['D']<r['distance']['D']-1e-10)
            for other in self.rows)]


