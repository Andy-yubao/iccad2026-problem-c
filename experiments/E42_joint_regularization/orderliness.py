"""Step 1/2: bounded simultaneous repair and matched feedback diagnostics."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from .feedback_geometry import ROOT,HERE,REVIEW,read,save,sha,geometry,pileup,official_metrics,ConeBuilder

OUT=HERE/'runs/20261024_orderliness_four_checks'
OLD=HERE/'runs/20261024_feedback_geometry23'


class Linear:
    def __init__(self):
        self.cost=[];self.low=[];self.high=[];self.integer=[];self.rows=[];self.rhs=[];self.lower=[]
    def var(self,lo=-np.inf,hi=np.inf,cost=0.,integer=False):
        k=len(self.cost);self.cost.append(cost);self.low.append(lo);self.high.append(hi);self.integer.append(int(integer));return k
    def le(self,row,rhs):
        self.rows.append(row);self.rhs.append(rhs);self.lower.append(-np.inf)
    def eq(self,row,rhs):
        self.rows.append(row);self.rhs.append(rhs);self.lower.append(rhs)
    def solve(self,seconds):
        ii=[];jj=[];vv=[]
        for i,row in enumerate(self.rows):
            for j,v in row.items():
                if v:ii.append(i);jj.append(j);vv.append(v)
        matrix=coo_matrix((vv,(ii,jj)),shape=(len(self.rows),len(self.cost))).tocsc()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            r=milp(np.asarray(self.cost),integrality=np.asarray(self.integer),bounds=Bounds(self.low,self.high),
                   constraints=LinearConstraint(matrix,self.lower,self.rhs),
                   options={'time_limit':seconds,'mip_rel_gap':.01,'threads':1,'random_seed':0})
        finite=lambda x:float(x) if x is not None and np.isfinite(x) else None
        info={'status':int(r.status),'message':r.message,'objective':finite(r.fun),
              'dual_bound':finite(getattr(r,'mip_dual_bound',None)),'gap':finite(getattr(r,'mip_gap',None)),
              'binary_variables':int(sum(self.integer)),'threads':1}
        return r.x,info


def edges(inputs):
    n=inputs[0]
    for rows,pin in ((inputs[2],False),(inputs[3],True)):
        for e in np.asarray(rows):
            i,j,w=int(e[0]),int(e[1]),float(e[2])
            if i==-1:continue
            if pin:
                if 0<=j<n and 0<=i<len(inputs[4]):yield i,j,w,pin
            elif 0<=i<n and 0<=j<n:yield i,j,w,pin


def fixed_shape(p,inputs,flexible=False,max_move=.5,span_gain=.01,wire_gain=.01,protect_boundary=True):
    """Minimize maximum L1 center movement; complete four-axis disjunctions."""
    n=len(p);s=np.sqrt(p[:,2:].prod(1).sum());q=p/s;wh=q[:,2:];center=q[:,:2]+wh/2
    local=np.sqrt(p[:,2:].prod(1))/s;c=np.asarray(inputs[5]);pre=c[:,1]!=0
    b=Linear();t=b.var(0,max_move,1.);xy=np.empty((n,2),int);distance=[]
    for i in range(n):
        for a in (0,1):
            lo=max(wh[i,a]/2,center[i,a]-max_move*local[i]);hi=center[i,a]+max_move*local[i]
            if pre[i]:lo=hi=center[i,a]
            xy[i,a]=b.var(lo,hi)
            d=b.var(0,max_move,1e-7/max(n,1));distance.append(d)
            b.le({xy[i,a]:1,d:-local[i]},center[i,a]);b.le({xy[i,a]:-1,d:-local[i]},-center[i,a])
        b.le({distance[-2]:1,distance[-1]:1,t:-1},0.)
    lo0=q[:,:2].min(0);hi0=(q[:,:2]+wh).max(0);span=hi0-lo0
    low=[b.var() for _ in (0,1)];high=[b.var() for _ in (0,1)]
    for a in (0,1):
        b.le({high[a]:1,low[a]:-1},span[a]*np.sqrt(1+span_gain))
        for i in range(n):
            b.le({low[a]:1,xy[i,a]:-1},-wh[i,a]/2)
            b.le({xy[i,a]:1,high[a]:-1},-wh[i,a]/2)
    # Protect all originally fully satisfied boundary blocks.
    for i in range(n):
        needed=[]
        for a,upper,bit in ((0,False,1),(0,True,2),(1,True,4),(1,False,8)):
            if int(c[i,4])&bit:
                edge=q[i,a]+(wh[i,a] if upper else 0.)
                needed.append((a,upper,abs(edge-(hi0[a] if upper else lo0[a]))*s<1e-6))
        if protect_boundary and needed and all(x[2] for x in needed):
            for a,upper,_ in needed:b.eq({xy[i,a]:1,(high[a] if upper else low[a]):-1},(-.5 if upper else .5)*wh[i,a])
    unresolved_pairs=0;forced_pairs=0
    for i in range(n):
        for j in range(i+1,n):
            options=[]
            for a in (0,1):
                half=(wh[i,a]+wh[j,a])/2
                for left,right in ((i,j),(j,i)):
                    if b.low[xy[left,a]]+half<=b.high[xy[right,a]]+1e-12:
                        options.append((left,right,a,half))
            if not options:
                return None,{'status':2,'message':'No separating axis is reachable for at least one pair within movement cap','pair':[i,j],
                             'certificate_scope':'fixed shapes, maximum L1 movement cap; independent of solver quality limits','binary_variables':0,'threads':1}
            if not flexible:
                options=[min(options,key=lambda r:(r[3]-(center[r[1],r[2]]-center[r[0],r[2]]))/(local[i]+local[j]))]
            if len(options)==1:
                l,r,a,half=options[0];b.le({xy[l,a]:1,xy[r,a]:-1},-half);forced_pairs+=1
            else:
                binaries=[];unresolved_pairs+=1
                for l,r,a,half in options:
                    big=max(0.,b.high[xy[l,a]]-b.low[xy[r,a]]+half)
                    binary=b.var(0,1,integer=True);binaries.append(binary)
                    b.le({xy[l,a]:1,xy[r,a]:-1,binary:big},big-half)
                b.le({v:-1 for v in binaries},-1.)
    old=official_metrics(p,inputs);wire={}
    for i,j,w,pin in edges(inputs):
        for a in (0,1):
            d=b.var(0,np.inf);wire[d]=w
            if pin:
                target=float(inputs[4][i,a])/s
                b.le({xy[j,a]:1,d:-1},target);b.le({xy[j,a]:-1,d:-1},-target)
            elif i!=j:
                b.le({xy[i,a]:1,xy[j,a]:-1,d:-1},0.);b.le({xy[i,a]:-1,xy[j,a]:1,d:-1},0.)
    b.le(wire,old['hpwl_total']*(1+wire_gain)/s)
    values,info=b.solve(2. if flexible else .5)
    info.update(flexible=flexible,max_move=max_move,span_gain=span_gain,wire_gain=wire_gain,protect_boundary=protect_boundary,choice_pairs=unresolved_pairs,forced_pairs=forced_pairs,
                certificate_scope='fixed shapes; movement cap, per-axis bbox span caps, HPWL cap, preserved successful boundary blocks')
    if values is None:return None,info
    out=np.column_stack((values[xy]*s-p[:,2:]/2,p[:,2:]));out[pre]=p[pre]
    return out,info


def shape_repair(p,inputs,max_log_shape=.05,max_move=.5):
    """Joint conic repair under original nearest-separation choices, exact areas."""
    n=len(p);s=np.sqrt(p[:,2:].prod(1).sum());q=p/s;centers=q[:,:2]+q[:,2:]/2
    wh0=q[:,2:];local=np.sqrt(p[:,2:].prod(1))/s;c=np.asarray(inputs[5]);locked=c[:,:2].any(1);pre=c[:,1]!=0
    b=ConeBuilder();xy=np.array([[b.var(),b.var()] for _ in range(n)]);wh=np.array([[b.var(),b.var()] for _ in range(n)])
    z=np.array([b.var() for _ in range(n)]);moves=np.array([[b.var(),b.var()] for _ in range(n)]);t=b.var();b.cost[t]=1.
    b.le({t:-1},0);b.le({t:1},max_move)
    low=np.array([b.var(),b.var()]);high=np.array([b.var(),b.var()]);lo0=q[:,:2].min(0);hi0=(q[:,:2]+wh0).max(0)
    for a in (0,1):
        b.le({high[a]:1,low[a]:-1},(hi0[a]-lo0[a])*np.sqrt(1.01))
    for i in range(n):
        for a in (0,1):
            b.le({low[a]:1,xy[i,a]:-1,wh[i,a]:.5},0.)
            b.le({xy[i,a]:1,wh[i,a]:.5,high[a]:-1},0.)
            b.le({xy[i,a]:-1,wh[i,a]:.5},0.)
            b.le({xy[i,a]:1,moves[i,a]:-local[i]},centers[i,a]);b.le({xy[i,a]:-1,moves[i,a]:-local[i]},-centers[i,a])
            b.le({moves[i,a]:-1},0.);b.cost[moves[i,a]]=1e-7/n
            if pre[i]:b.eq({xy[i,a]:1},centers[i,a])
        b.le({moves[i,0]:1,moves[i,1]:1,t:-1},0.)
        if locked[i]:
            b.eq({z[i]:1},0.)
            for a in (0,1):b.eq({wh[i,a]:1},wh0[i,a])
        else:
            b.le({z[i]:1},max_log_shape);b.le({z[i]:-1},max_log_shape)
            b.exp.extend([(z[i],1.,wh[i,0],wh0[i,0]),(z[i],-1.,wh[i,1],wh0[i,1])])
    for i in range(n):
        for j in range(i+1,n):
            options=[]
            for a in (0,1):
                l,r=(i,j) if centers[i,a]<=centers[j,a] else (j,i)
                deficit=(wh0[i,a]+wh0[j,a])/2-(centers[r,a]-centers[l,a])
                options.append((deficit/(local[i]+local[j]),l,r,a))
            _,l,r,a=min(options)
            b.le({xy[l,a]:1,xy[r,a]:-1,wh[l,a]:.5,wh[r,a]:.5},0.)
    wire={};old=official_metrics(p,inputs)
    for i,j,w,pin in edges(inputs):
        for a in (0,1):
            d=b.var();wire[d]=w;b.le({d:-1},0.)
            if pin:
                target=float(inputs[4][i,a])/s;b.le({xy[j,a]:1,d:-1},target);b.le({xy[j,a]:-1,d:-1},-target)
            elif i!=j:
                b.le({xy[i,a]:1,xy[j,a]:-1,d:-1},0.);b.le({xy[i,a]:-1,xy[j,a]:1,d:-1},0.)
    b.le(wire,old['hpwl_total']*1.01/s)
    values,info=b.solve();info.update(max_log_shape=max_log_shape,certificate_scope='original nearest-separation directions only')
    if values is None:return None,info
    zz=np.clip(values[z],-max_log_shape,max_log_shape);zz[locked]=0
    dims=p[:,2:]*np.exp(np.column_stack((zz,-zz)))
    out=np.column_stack((values[xy]*s-dims/2,dims));out[pre]=p[pre]
    return out,info


def verify(p,out,inputs):
    from scoring.checks import independent_hard_checks
    from datasets.reference_layouts import public_hard_reference
    _,areas,_,_,_,constraints,specs=inputs
    checks=independent_hard_checks(out.tolist(),areas,constraints,public_hard_reference(areas,constraints,specs))
    hard=bool(checks['is_feasible'] and (out[:,:2]>=0).all())
    m0=official_metrics(p,inputs);m=official_metrics(out,inputs)
    soft=[k for k in ('boundary_violations','grouping_violations','mib_violations') if m[k]>m0[k]]
    center=p[:,:2]+p[:,2:]/2;delta=out[:,:2]+out[:,2:]/2-center
    local=np.sqrt(p[:,2:].prod(1));d=np.abs(delta).sum(1)/local;free=np.asarray(constraints)[:,1]==0
    bbox0=geometry(p)['bbox'];bbox=geometry(out)['bbox']
    quality=not soft and bbox<=bbox0*1.010001 and m['hpwl_total']<=m0['hpwl_total']*1.010001
    return {'hard_legal':hard,'quality_guard':quality,'soft_regressions':soft,'metrics':m,
            'independent_hard_checks':checks,'max_move':float(d.max()),'mean_move_free':float(d[free].mean()) if free.any() else 0.,
            'worst10_mean_free':float(np.sort(d[free])[-max(1,int(np.ceil(free.sum()*.1))):].mean()) if free.any() else 0.,
            'fraction_free_moved_over_1pct':float(np.mean(d[free]>.01)) if free.any() else 0.,
            'bbox_relative_change':float(bbox/bbox0-1),'HPWL_relative_change':float(m['hpwl_total']/max(m0['hpwl_total'],1e-12)-1),
            'max_abs_log_width_change':float(np.abs(np.log(out[:,2]/p[:,2])).max())}


def case_task(task):
    row,record,output=task;output=Path(output);from datasets.public_member_inputs import public_member_inputs
    inputs=public_member_inputs(record,'local_test',authorize_local_test=True)
    p=np.asarray(read(ROOT/row['source_draft'])['raw_positions'],float);start=time.perf_counter()
    trials=[]
    for name,operation in [('fixed_relation',lambda:fixed_shape(p,inputs,False)),('flexible_relation',lambda:fixed_shape(p,inputs,True)),
                           ('joint_small_shape',lambda:shape_repair(p,inputs))]:
        out,info=operation();trial={'name':name,'solver':info}
        if out is not None:trial.update(verification=verify(p,out,inputs),positions=out.tolist())
        trials.append(trial)
    c=np.asarray(inputs[5]);r={'id':row['id'],'member':row['member'],'n':len(p),'annotation':row['annotation'],
        'source_draft':row['source_draft'],'source_draft_sha256':row['source_draft_sha256'],
        'constraints':c.tolist(),'input_features':{'preplaced_fraction':float(np.mean(c[:,1]!=0)),
        'fixed_fraction':float(np.mean(c[:,0]!=0)),'group_fraction':float(np.mean(c[:,3]>0)),
        'mib_fraction':float(np.mean(c[:,2]>0))},'pileup':row['pileup'],'trials':trials,'seconds':time.perf_counter()-start}
    save(output/'repair_local'/f"{row['id']}.json",r);return r['id'],r['seconds']


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'repair_local').mkdir(exist_ok=True)
    oldconfig=read(OLD/'config.json');manifest=read(REVIEW/'manifest.json');lookup={r['id']:r for r in manifest['cases']}
    source=read(HERE/'runs/20261024_final_test1000_seed20261024/config.json')
    rows=[read(OLD/'cases_local'/f'{key}.json') for key in oldconfig['cases']]
    config={'count':len(rows),'case_ids':[r['id'] for r in rows],'source_sha256':sha(Path(__file__)),
        'feedback_snapshot_sha256':sha(OLD/'feedback_snapshot.json'),'stage1_max_local_L1_move':.5,
        'HPWL_increase_cap':.01,'bbox_span_factor':float(np.sqrt(1.01)),'shape_log_cap':.05,
        'stage1_threads':'4 processes x 1 solver thread','stage1_budget_seconds':480,
        'authorisation':'2026-10-24 user requested sequential execution of all four diagnostic angles',
        'training':False,'new_sealed_test_access':False,'production_postprocessor_changed':False}
    if not (OUT/'config.json').exists():save(OUT/'config.json',config)
    elif read(OUT/'config.json')['source_sha256']!=config['source_sha256']:raise ValueError('Diagnostic source changed; preserve existing run before new invocation')
    tasks=[(r,source['records'][lookup[r['id']]['source_index']],str(OUT)) for r in rows if not (OUT/'repair_local'/f"{r['id']}.json").exists()]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(case_task,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            key,seconds=future.result();print(f'{i}/{len(tasks)} {key}: {seconds:.2f}s',flush=True)
            if time.perf_counter()-start>480:raise TimeoutError('Stage1 budget exceeded; saved results retained')
    save(OUT/'stage1_execution.json',{'status':'complete','count':len(rows),'seconds_this_invocation':time.perf_counter()-start})


if __name__=='__main__':prepare()
