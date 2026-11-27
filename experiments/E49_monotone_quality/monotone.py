"""Public monotone-Q certificate, protected ordinary branch, constrained refinement."""
import time
import numpy as np
from .portfolio import Archive,raw_measures
from .backend.frozen.packing import prepare_problem,_dimensions
from .backend.frozen.solver import hard_valid
from .baseline.solver import optimize as ordinary
from .fast_geometry import step,topology,reduce_edges
from .projection import solve as project

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,
    robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)

def certificate(row,base,ns):
    k=max(1.,row['area']/base['area'],row['wire']/max(base['wire'],1e-12))
    return float(np.log(k)+2*(sum(row['soft_violations'])-sum(base['soft_violations']))/ns)

def optimize(draft,inputs,*,seconds=5.,observer=None,profile='v1'):
    begin=time.perf_counter();deadline=begin+seconds-.12;p=prepare_problem(*inputs)
    archive=Archive(p,np.asarray(draft,float),begin,observer);trace=[]
    requested_profile=profile
    ordinary_options=dict(STEADY)
    native_candidates=[]
    if profile in('v7','v8','v9','v10','v11','v12','v13','v14'):
        if p.n>=110:ordinary_options.update(adaptive=False,seconds=min(2.4,seconds*.5),iterations=96,late_extension=False)
        profile='v5' if profile in('v8','v9','v10','v11','v12','v13','v14') else 'v4'
    if requested_profile in('v8','v9','v10','v11','v12','v13','v14'):ordinary_options['observer']=lambda x,label:native_candidates.append((x.copy(),label))
    x,s=ordinary(draft,inputs,**ordinary_options)
    assert hard_valid(x,p)
    baseline=archive.accept(x,'protected_ordinary');base=archive.rows[-1]
    for candidate,label in native_candidates:archive.accept(candidate,'ordinary/'+label)
    base['safe_Q_log_bound']=0.;trace.append({'phase':'protected_ordinary',**s})
    base_id=base['id'];vbase=sum(base['soft_violations']);ns=archive.obj.nsoft
    rawbox=(draft[:,:2]+draft[:,2:]).max(0)-draft[:,:2].min(0)
    bbox=(baseline[:,:2]+baseline[:,2:]).max(0)-baseline[:,:2].min(0)
    draft_ratio=rawbox[0]/rawbox[1];base_ratio=bbox[0]/bbox[1]
    states={};attempt=0
    if requested_profile=='v14' and p.n>=90:
        from .refine import refine
        trace.extend(refine(archive,deadline,seconds=min(1.1,max(.1,(deadline-time.perf_counter())*.45))))
    if requested_profile in('v11','v12','v13'):
        from .repair import repair
        trace.extend(repair(archive,base,deadline,seconds=min(1.1,max(.1,(deadline-time.perf_counter())*.45)),shape=requested_profile=='v13'))
    if requested_profile in('v9','v10','v11','v12','v13','v14'):
        from .portfolio import optimize as search
        if deadline-time.perf_counter()>.15:
            _,extra=search(draft,inputs,seconds=deadline-time.perf_counter(),initial=baseline,run_native=False,
                observer=lambda x,row:archive.accept(x,'portfolio/'+row['source']))
            trace.extend(extra['trace'])
        deadline=time.perf_counter()
    while deadline-time.perf_counter()>.075 and attempt<300:
        count=8 if profile in('v4','v5','v6') else (6 if profile in('v2','v3') else 10)
        slot=attempt%count;cycle=attempt//count;attempt+=1
        safe=[r for r in archive.rows if certificate(r,base,ns)<=1e-10]
        best=min(safe,key=lambda r:r['public_quality_cost']+.02*r['distance']['D'])
        state=states.get(slot,np.asarray(best['positions']) if slot in (0,1,6) else baseline).copy()
        hard=slot in (1,3,5,7,9);expansion=np.exp(2*vbase/ns) if hard else 1.
        ratio=[base_ratio,draft_ratio,base_ratio*.82,base_ratio*1.22,draft_ratio*.88][(slot//2)%5]
        box=np.sqrt(base['area']*expansion*(1.-1e-7)*np.array([ratio,1/ratio]))
        full=topology(state,p.constraints,mode=int(slot>=4),array=True,group_order=slot>=8)
        left=deadline-time.perf_counter()
        if profile in('v2','v3','v4','v5','v6'):
            if profile in('v4','v5','v6') and slot in(3,5,6,7):
                state=np.asarray(min(archive.rows,key=lambda r:r['public_quality_cost']+.02*r['distance']['D'])['positions']).copy()
            if slot in (2,4) and slot not in states:state=np.asarray(draft).copy()
            if profile=='v3' or (profile in('v4','v5','v6') and slot in(2,4) and slot not in states):
                widths,heights=_dimensions(p,state[:,2]/state[:,3],True)
                wh=np.column_stack((widths,heights));state=np.column_stack((np.maximum(0,state[:,:2]+state[:,2:]/2-wh/2),wh))
                state[p.preplaced]=p.specs[p.preplaced]
            full=topology(state,p.constraints,mode=int(slot in(4,5)),array=True,group_order=slot==5)
            if profile in('v4','v5','v6') and slot==6:
                center=state[:,:2]+state[:,2:]/2;i,j,a=full.T;b=1-a
                other=np.abs(center[i,b]-center[j,b])-(state[i,b+2]+state[j,b+2])/2
                active=np.abs(state[j,a]-state[i,a]-state[i,a+2])/(np.sqrt(p.areas[i])+np.sqrt(p.areas[j]))
                priority=-other/(np.sqrt(p.areas[i])+np.sqrt(p.areas[j]))+active
                priority[p.preplaced[i]&p.preplaced[j]]=np.inf
                k=np.argsort(priority)[cycle%min(24,len(full))];i,j,a=full[k];b=1-a
                first,second=(i,j) if center[i,b]<=center[j,b] else(j,i);full[k]=(first,second,b)
            if (profile not in('v4','v5','v6') and slot==3) or (profile in('v4','v5','v6') and slot==7):
                from .fast_geometry import contact_tree
                y,info=project(p,state,reduce_edges(full,p.n),contact_tree(state,p,full),seconds=min(.3,left-.025),
                    boundary=32.,contact=16.,anchors=(base['area']/p.scale**2,base['wire']/p.scale),
                    draft_reference=draft,distance_weight=.003,engine='highs')
            else:
                y,info=step(p,state,reduce_edges(full,p.n),seconds=min(.22,left-.025),full_edges=full,
                    elastic=(150000. if slot==4 else 0.) if profile in('v4','v5','v6') else (100000. if slot in(2,4) else 0.),reweighted=slot%2==0,boundary=24.,contact=12.,wire_exact=profile=='v5' and slot==4,
                    wire_ordered=profile=='v6' and slot not in(2,4,6),
                    wire_limit=base['wire']*(np.exp(2*vbase/ns) if slot in(1,5) else 1.) if profile=='v6' and slot in(0,1,3,5) else None,
                    anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale) if profile in('v3','v4','v5','v6') else (base['area']/p.scale**2,base['wire']/p.scale),draft_reference=draft,distance_weight=.003,
                    shape_weight=(.4 if slot in(2,3) else .04) if profile in('v4','v5','v6') else ((.6 if slot in(2,4) else .04) if profile=='v3' else (.15 if slot in(2,4) else .01)),hard_boundary=slot in(1,5),hard_contact=slot in(1,5),
                    preserve_soft=slot==0)
        else:
            y,info=step(p,state,reduce_edges(full,p.n),seconds=min(.22,left-.025),full_edges=full,
            elastic=0.,reweighted=False,boundary=40.,contact=24.,wire_exact=True,
            anchors=(base['area']/p.scale**2,base['wire']/p.scale),
            draft_reference=draft,distance_weight=.005,shape_weight=.12 if slot in(2,3,4,5) else .01,
            box_limit=box,wire_limit=base['wire']*expansion*(1.-1e-7),
            preserve_soft=not hard,hard_boundary=hard,hard_contact=hard)
        clean=archive.accept(y,f'constrained/{slot}/{cycle}')
        if y is not None:states[slot]=clean if clean is not None else y
        trace.append({'phase':'constrained','slot':slot,'cycle':cycle,'hard_soft':hard,'accepted':clean is not None,**info})
    safe=[]
    for row in archive.rows:
        row['safe_Q_log_bound']=certificate(row,base,ns)
        if row['safe_Q_log_bound']<=1e-10:safe.append(row)
    choice=min(safe,key=lambda r:r['public_quality_cost']+.02*r['distance']['D'])
    assert hard_valid(np.asarray(choice['positions']),p)
    return np.asarray(choice['positions']),{'candidates':archive.rows,'selected_id':choice['id'],'baseline_id':base_id,
        'public_quality_best_id':min(archive.rows,key=lambda r:r['public_quality_cost'])['id'],
        'seconds':time.perf_counter()-begin,'safe_candidates':len(safe),'trace':trace,
        'shelf_selected':choice['shelf_fallback'],'public_Q_certificate':True,'reference_geometry_used':False,
        'observer_return_used':False,'configured_numerical_threads':4,'profile':requested_profile}
