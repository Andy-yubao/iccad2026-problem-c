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


def optimize(draft,inputs,*,seconds=5.,observer=None,initial=None,run_native=True):
    start=time.perf_counter();deadline=start+max(.1,seconds)-.10
    p=prepare_problem(*inputs);draft=np.asarray(draft,float).copy()
    if draft.shape!=(p.n,4) or not np.isfinite(draft).all() or (draft[:,2:]<=0).any():raise ValueError('Invalid draft')
    archive=Archive(p,draft,start,observer);trace=[]
    archive.accept(row_control(p),'shelf_fallback');archive.accept(draft,'raw')
    if not archive.rows:raise RuntimeError('Public specifications have no verified fallback')
    widths,heights=_dimensions(p,draft[:,2]/draft[:,3],True)
    base=np.column_stack((np.maximum(0,draft[:,:2]+draft[:,2:]/2-np.column_stack((widths,heights))/2),widths,heights))
    base[p.preplaced]=p.specs[p.preplaced]
    states=[base.copy()];branches={k:base.copy() for k in range(6)};branch_best={}
    if initial is not None:
        archive.accept(initial,'external_protected');branches[1]=initial.copy();branches[3]=initial.copy()
    rng=np.random.default_rng(0);attempts=0;mip_done=False;packed=False
    anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale)
    def left():return deadline-time.perf_counter()
    # A protected unmodified-geometry trajectory supplies the first strong legal incumbent.
    if run_native and left()>.20:
        from .legacy_solver import optimize as legacy
        x,s=legacy(draft,inputs,iterations=128,seconds=min(1.55 if p.n>=90 else .85,left()-.04),projection=True,
            plateau=True,certified=False,robust_seeds=False,rescue=False,late_extension=False,
            quadratic_project=True,incumbent_project=True,snap=1e-4,
            observer=lambda x,label:archive.accept(x,'native/'+label))
        archive.accept(x,'native/return');states.append(x.copy());trace.append({'phase':'native_protected',**s})
        if not np.array_equal(x,row_control(p)):
            branches[1]=x.copy();branches[3]=x.copy()
    while left()>.09 and attempts<300:
        attempts+=1
        best=archive.best(exclude_shelf=True);quality=archive.best(0.,exclude_shelf=True)
        # Near-draft constructive rescue is budgeted and never silently replaces the draft.
        if best is None and not packed and attempts>=3 and left()>.35:
            from .backend.rescue import seed_pack
            t=time.perf_counter();x=seed_pack(draft,inputs,mode=0,seconds=min(.30,left()-.04));packed=True
            archive.accept(x,'draft_guided_pack');
            if x is not None:states.append(x.copy())
            trace.append({'phase':'draft_guided_pack','seconds':time.perf_counter()-t});continue
        # One bounded relation-MIP branch uses the current legal geometry as a hint.
        if best is not None and not mip_done and attempts>=7 and left()>.9:
            from .backend.frozen.topology_mip import optimize as mip
            seed=np.asarray(best['positions']);t=time.perf_counter()
            x,s=mip(p,seed,seconds=min(.55,left()-.16),freedom=.35,boundary_weight=16.,contact_weight=8.,
                    draft_reference=draft,distance_weight=.03)
            archive.accept(x,'relation_mip');mip_done=True
            if x is not None:states.append(x.copy())
            trace.append({'phase':'relation_mip','wall_seconds':time.perf_counter()-t,**s});continue
        slot=attempts%6;mode=int(slot in(4,5));weight={0:0.,1:0.,2:.08,3:.025,4:.04,5:.015}[slot]
        state=branches[slot].copy()
        if slot in (1,3) and quality is not None and slot not in branch_best:state=np.asarray(quality['positions'])
        full=topology(state,p.constraints,mode=mode,array=True,group_order=bool(slot==4))
        # Small reversible relation changes are restricted to nearby/ambiguous pairs,
        # never to a large global random restart.
        if slot==5 and len(full):
            i,j=full[:,:2].T;center=state[:,:2]+state[:,2:]/2
            margins=np.abs(center[i]-center[j])-(state[i,2:]+state[j,2:])/2
            priority=np.abs(margins[:,0]-margins[:,1])/np.maximum(np.sqrt(p.areas[i])+np.sqrt(p.areas[j]),1e-8)
            pool=np.argsort(priority)[:max(2,min(12,len(full)))]
            for k in rng.choice(pool,size=min(2,len(pool)),replace=False):
                i,j,axis=full[k];axis=1-axis
                a,b=(i,j) if center[i,axis]<=center[j,axis] else (j,i)
                full[k]=(a,b,axis)
        edges=reduce_edges(full,p.n)
        if slot==3 and best is not None:
            state=np.asarray(best['positions']);full=topology(state,p.constraints,mode=0,array=True);edges=reduce_edges(full,p.n)
            tree=contact_tree(state,p,full)
            x,s=project(p,state,edges,tree,seconds=min(.20,left()-.025),boundary=24.,contact=12.,anchors=anchors,
                quadratic=False,draft_reference=draft,distance_weight=weight)
            phase='fixed_shape_polish'
        else:
            x,s=step(p,state,edges,seconds=min(.24,left()-.025),boundary=24.,contact=12.,
                elastic=0. if best is not None and slot in(1,2) else 150000.,
                reweighted=bool(slot%2),conservative=True,full_edges=full,anchors=anchors,
                wire_exact=bool(slot==4),draft_reference=draft,distance_weight=weight,shape_weight=.4 if slot==2 else (.04 if weight else 0.))
            phase='shape_branch'
        clean=archive.accept(x,phase+f'/{slot}')
        trace.append({'phase':phase,'slot':slot,'distance_weight':weight,'accepted_legal':clean is not None,**s})
        if x is not None:
            states.append(clean if clean is not None else x);states=states[-4:]
            if clean is None:
                branches[slot]=x
            else:
                rank=archive.obj(clean)+weight*archive.distance(clean)['D']
                if slot not in branch_best or rank<branch_best[slot]:
                    branch_best[slot]=rank;branches[slot]=clean.copy()
        if clean is not None and left()>.08 and attempts%4==0:
            from .compact import polish
            y,s=polish(clean,p,draft,seconds=min(.035,left()-.025));archive.accept(y,'rigid_component_compact')
            trace.append({'phase':'rigid_component_compact',**s})
        best=archive.best(exclude_shelf=True)
        if best is not None and sum(best['soft_violations'])==0 and best['area']<=1.01*archive.obj.area_scale and best['wire']<=archive.draft_wire*.99 and best['distance']['D']<.5:
            trace.append({'phase':'public_proxy_stop','note':'reference-free; does not imply trueQ<1.03'});break
    selected=archive.best();quality=archive.best(0.)
    output=np.asarray(selected['positions']);assert hard_valid(output,p)
    stats={'seconds':time.perf_counter()-start,'budget_seconds':seconds,'budget_overrun':time.perf_counter()-start>seconds,
        'selected_id':selected['id'],'public_quality_best_id':quality['id'],'frontier_ids':archive.frontier(),
        'candidates':archive.rows,'trace':trace,'rejected_candidate_count':archive.rejected,
        'shelf_selected':selected['shelf_fallback'],'first_non_shelf_legal_seconds':next((r['found_seconds'] for r in archive.rows if not r['shelf_fallback']),None),
        'observer_return_used':False,'reference_geometry_used':False,'native_threads':4}
    return output,stats
