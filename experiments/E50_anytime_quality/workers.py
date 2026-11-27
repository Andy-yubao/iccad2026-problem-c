"""Two numerical threads per worker; no references, denominators or Q feedback."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['NUMBA_NUM_THREADS']='1'
import time
import numpy as np
from .archive import Archive
from .backend.frozen.packing import prepare_problem,_dimensions
from .backend.frozen.feasible import Objective
from .fast_geometry import step,topology,reduce_edges,contact_tree
from .projection import solve as project

def explore(p,draft,deadline,archive,connection):
    widths,heights=_dimensions(p,draft[:,2]/draft[:,3],True);wh=np.column_stack((widths,heights))
    initial=np.column_stack((np.maximum(0,draft[:,:2]+draft[:,2:]/2-wh/2),wh));initial[p.preplaced]=p.specs[p.preplaced]
    states={i:initial.copy() for i in range(8)};overrides={i:{} for i in states};tried={i:set() for i in states};ranks={};trace=[];count=0
    anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale)
    while deadline-time.perf_counter()>.055 and count<512:
        while connection.poll():
            kind,hint=connection.recv()
            if kind=='hint':
                hint=np.asarray(hint,float);archive.accept(hint,'hint')
                for i in(4,5,6,7):states[i]=hint.copy();overrides[i]={};tried[i]=set();ranks.pop(i,None)
        slot=count%8;cycle=count//8;count+=1;state=states[slot]
        if slot in(6,7):
            if not archive.rows:continue
            leading=min(archive.rows,key=lambda r:r['public_quality_cost']+.025*r['distance']['D'])
            if not 0<sum(leading['soft_violations'])<=2:continue
        if slot in(4,5,6,7) and archive.rows:
            best=min(archive.rows,key=lambda r:r['public_quality_cost']+.025*r['distance']['D'])
            if slot not in ranks:state=np.asarray(best['positions'])
        full=topology(state,p.constraints,mode=int(slot in(1,3)),array=True,group_order=slot==3)
        for pair,edge in overrides[slot].items():
            i,j=pair;full[j*(j-1)//2+i]=edge
        edges=reduce_edges(full,p.n);left=deadline-time.perf_counter()
        if slot in(5,7):
            tree=contact_tree(state,p,full);clean_soft=sum(archive.obj.violations(state))==0
            y,info=project(p,state,edges,tree,seconds=min(.16,left-.025),engine='highs',anchors=anchors,
                draft_reference=draft,distance_weight=.005,boundary=24.,contact=16.,
                boundary_targets=np.flatnonzero(p.constraints[:,4]) if slot==7 or clean_soft else (),
                contact_targets=[t[:2] for t in tree] if slot==7 or clean_soft else (),preserve_satisfied=True)
        else:
            clean_soft=sum(archive.obj.violations(state))==0
            y,info=step(p,state,edges,seconds=min(.16,left-.025),full_edges=full,anchors=anchors,
                elastic=150000. if slot==2 else 0.,conservative=slot not in(4,6),reweighted=slot%2==0,
                boundary=24.,contact=12.,wire_exact=slot==1 and p.n<70,
                draft_reference=draft,distance_weight=.012 if slot in(1,3) else .003,
                shape_weight=.4 if slot in(1,3) else .015,
                hard_boundary=slot in(3,6) or(slot==4 and clean_soft),hard_contact=slot in(3,6) or(slot==4 and clean_soft))
        clean=archive.accept(y,f'explore/{slot}/{cycle}')
        trace.append({'phase':'explore','slot':slot,'cycle':cycle,'accepted':clean is not None,**info})
        if y is not None:
            if slot==6:states[7]=clean if clean is not None else y;ranks[7]=np.inf
            if clean is None:states[slot]=y
            else:
                rank=archive.obj(clean)+.025*archive.distance(clean)['D']
                if slot not in ranks or rank<ranks[slot]:states[slot]=clean;ranks[slot]=rank
        if y is None and info.get('status')=='PrimalInfeasible' and slot!=2:
            conflicts=info.get('conflicting_edges',[]);centers=state[:,:2]+state[:,2:]/2
            for i,j,axis,weight in conflicts:
                if p.preplaced[i] and p.preplaced[j]:continue
                other=1-axis;a,b=(i,j) if centers[i,other]<=centers[j,other] else(j,i)
                edge=(a,b,other)
                if edge in tried[slot]:continue
                overrides[slot][tuple(sorted((i,j)))]=edge;tried[slot].add(edge)
                trace.append({'phase':'infeasible_relation_repair','slot':slot,'old':[i,j,axis],'new':list(edge),'dual_weight':weight});break
    return {'trace':trace,'attempts':count}

def worker(connection,kind):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    from .canonical_native import warmup
    from .baseline.canonical_native import warmup as warm_baseline
    from .baseline.solver import optimize as ordinary
    warmup();warm_baseline();connection.send(('ready',{'kind':kind,'numerical_threads':2}))
    try:
        while True:
            job=connection.recv()
            if job is None:return
            if job[0]!='solve':continue
            _,inputs,draft,begin,deadline=job;p=prepare_problem(*inputs)
            def observe(x,row):
                row['worker']=kind;row['received_ordinal']=None
                row['found_seconds']=time.perf_counter()-begin
                row['native_proxy']=Objective(p)(x)
                connection.send(('candidate',row))
            archive=Archive(p,draft,begin,observe)
            if kind=='baseline':
                x,stats=ordinary(draft,inputs,iterations=96,seconds=min(2.3,max(.1,deadline-time.perf_counter()-.1)),
                    adaptive=False,late_extension=False,plateau=True,robust_seeds=True,certified=True,
                    quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4,
                    observer=lambda x,label:archive.accept(x,'ordinary/'+label))
                archive.accept(x,'ordinary/return')
            else:stats=explore(p,draft,deadline,archive,connection)
            connection.send(('done',stats))
    except (EOFError,BrokenPipeError):return
    except Exception as error:
        try:connection.send(('error',repr(error)))
        except (EOFError,BrokenPipeError):pass
    finally:connection.close()
