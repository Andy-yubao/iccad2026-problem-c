"""Read-only shape-direction audit on every saved training-source operation."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import csv,datetime,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from numba import njit
from ..R16.replay import cached,read,save,sha,ROOT,RESEARCH,EXPERIMENT,process_lock

HERE=Path(__file__).resolve().parent
OUT=EXPERIMENT/'runs/research/20261103_R17_saved_shape_direction_audit'
PANELS=[('R15','budget_gate'),('R15','attraction_equal_slope'),('R16','longer_relaxation')]


def face_derivative(wi,hi,wj,hj,ux,uy):
    rx=(wi+wj)/(2*abs(ux)) if ux else np.inf
    ry=(hi+hj)/(2*abs(uy)) if uy else np.inf
    if np.isfinite(rx) and np.isfinite(ry) and abs(rx-ry)<=1e-11*max(rx,ry):
        return .5*(wi/(2*abs(ux))-hi/(2*abs(uy)))
    return wi/(2*abs(ux)) if rx<ry else -hi/(2*abs(uy))


def fixtures():
    rows=[]
    for wi,hi,wj,hj,ux,uy in [(1.,100.,1.,100.,.1,np.sqrt(.99)),(100.,1.,100.,1.,np.sqrt(.99),.1),
                              (2.,4.,3.,5.,1.,0.),(2.,4.,3.,5.,0.,1.),(2.,2.,2.,2.,np.sqrt(.5),np.sqrt(.5))]:
        def contact(q):
            return min((wi*np.exp(q)+wj)/(2*abs(ux)) if ux else np.inf,
                       (hi*np.exp(-q)+hj)/(2*abs(uy)) if uy else np.inf)
        derivative=face_derivative(wi,hi,wj,hj,ux,uy)
        finite=(contact(1e-6)-contact(-1e-6))/2e-6
        np.testing.assert_allclose(derivative,finite,rtol=2e-6,atol=1e-8)
        old=min(wi/(2*abs(ux)) if ux else np.inf,hi/(2*abs(uy)) if uy else np.inf)*(ux*ux-uy*uy)
        rows.append(dict(dimensions=[wi,hi,wj,hj],direction=[ux,uy],old_coefficient=old,face_derivative=derivative,finite_difference=finite))
    assert rows[0]['old_coefficient']*rows[0]['face_derivative']<0
    return rows


@njit(cache=True)
def inspect(history,mass,free,range_frac,ka,kr,area_source,rep_source):
    steps=history.shape[0]-1;n=history.shape[1]
    old=np.zeros((steps,n));exact=np.zeros((steps,n));totals=np.zeros(12)
    for t in range(steps):
        x=history[t];center=x[:,:2]+x[:,2:]/2
        for i in range(n):
            for j in range(i+1,n):
                dx=center[j,0]-center[i,0];dy=center[j,1]-center[i,1];d=np.hypot(dx,dy)
                if d>0:ux=dx/d;uy=dy/d
                else:
                    angle=(((i+1)*73856093 ^ (j+1)*19349663)%104729)*(2*np.pi/104729)
                    ux=np.cos(angle);uy=np.sin(angle)
                rx=(x[i,2]+x[j,2])/(2*abs(ux)) if ux else np.inf
                ry=(x[i,3]+x[j,3])/(2*abs(uy)) if uy else np.inf
                r=min(rx,ry);gap=d-r
                if abs(gap)<=32*np.finfo(np.float64).eps*max(d,r):gap=0.
                short=min(x[i,2],x[i,3],x[j,2],x[j,3]);cutoff=range_frac*short
                z=min(1.,max(0.,gap/cutoff))
                coupling=2*mass[i]*mass[j]/(mass[i]+mass[j]) if area_source or rep_source else 1.
                fa=ka*max(gap,0.)*(1-z)**2*(coupling if area_source else 1.)
                fr=-kr*short*np.tanh(max(-gap,0.)/short)*(coupling if rep_source else 1.)
                f=fa+fr
                if f==0:continue
                tie=np.isfinite(rx) and np.isfinite(ry) and abs(rx-ry)<=1e-11*max(rx,ry)
                for k in (i,j):
                    radial=min(x[k,2]/(2*abs(ux)) if ux else np.inf,x[k,3]/(2*abs(uy)) if uy else np.inf)
                    a=radial*(ux*ux-uy*uy)
                    if tie:b=.5*(x[k,2]/(2*abs(ux))-x[k,3]/(2*abs(uy)))
                    else:b=x[k,2]/(2*abs(ux)) if rx<ry else -x[k,3]/(2*abs(uy))
                    old[t,k]+=f*a;exact[t,k]+=f*b
                    if free[k]:
                        totals[0]+=1.;totals[1]+=1. if a*b<0 else 0.
                        totals[2]+=abs(f);totals[3]+=abs(f) if a*b<0 else 0.
                        totals[4]+=abs(f*b);totals[5]+=abs(f*b) if a*b<0 else 0.
                        totals[6]+=1. if tie else 0.;totals[7]+=1. if fa>0 and a*b<0 else 0.
                        totals[8]+=abs(fa);totals[9]+=abs(fa) if a*b<0 else 0.
                        totals[10]+=abs(fr);totals[11]+=abs(fr) if a*b<0 else 0.
    return old,exact,totals


def main():
    if OUT.exists():raise FileExistsError('Saved shape audit protected; inspect evidence before retry')
    inputs={}
    for version,arm in PANELS:
        run=EXPERIMENT/f'runs/research/20261103_{version}_train500_trajectory_audit'
        inputs[version+'/'+arm]=dict(run=run.relative_to(ROOT).as_posix(),
            registration_sha256=sha(run/'registration.json'),results_sha256=sha(run/'results.json'))
    identity=dict(plan=read(HERE/'plan.json'),inputs=inputs,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),HERE/'plan.json',RESEARCH/'R16/replay.py',RESEARCH/'R16/operator.py')})
    OUT.mkdir(parents=True)
    save(OUT/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in identity['sources']:z.write(ROOT/p,p)
    checked=fixtures();summary={};records=[];top=[];begin=time.perf_counter()
    with process_lock(RESEARCH/'.numerical.lock'):
        for label,source in inputs.items():
            run=ROOT/source['run'];arm=label.split('/')[1]
            registration=read(run/'registration.json');cfg=registration['identity']['configs'][arm]
            results=read(run/'results.json');assert results['status']=='complete'
            totals=np.zeros(12);module_ops=0;module_flip=0;stress_abs=0.;stress_flip_abs=0.;maximum_error=0.;case_count=0
            for row in results['records']:
                if row['arm']!=arm:continue
                if (RESEARCH/'STOP').exists() or (OUT/'STOP').exists():
                    save(OUT/'state.json',dict(status='stopped',completed=len(records),total=1500));return
                index=row['index'];path=run/'trajectories'/arm/f'{index:04d}.npz';assert sha(path)==row['trajectory_sha256']
                draft,public,control=cached(index);assert row['member']==control['member']
                free=~(np.asarray(public[5])[:,:2]!=0).any(1);case_count+=1
                area=draft[:,2:].prod(1);mass=area/np.median(area)
                with np.load(path) as a:
                    old,new,counts=inspect(a['history'],mass,free,cfg['attraction_range'],cfg['attraction_strength'],cfg['repulsion_strength'],cfg['area_attraction'],cfg.get('area_repulsion',False))
                    saved=a['module_log'][:,:,13]+a['module_log'][:,:,14]
                np.testing.assert_allclose(old,saved,rtol=2e-11,atol=2e-9)
                error=float(np.max(np.abs(old-saved)));maximum_error=max(maximum_error,error)
                # Relative threshold rejects cancellation noise; fixed-shape and preplaced excluded.
                eps=1e-10*np.maximum(1.,area)[None,:]
                active=free[None,:]&(np.abs(old)>eps)&(np.abs(new)>eps);flip=active&(old*new<0)
                module_ops+=int(active.sum());module_flip+=int(flip.sum());totals+=counts
                stress_abs+=float(np.abs(new)[active].sum());stress_flip_abs+=float(np.abs(new)[flip].sum())
                record=dict(panel_arm=label,index=index,member=row['member'],steps=len(old),
                    active_module_operations=int(active.sum()),opposite_module_operations=int(flip.sum()),
                    active_pair_module_operations=int(counts[0]),opposite_pair_module_operations=int(counts[1]),
                    force_abs_sum=float(counts[2]),opposite_force_abs_sum=float(counts[3]),
                    face_stress_abs_sum=float(counts[4]),opposite_face_stress_abs_sum=float(counts[5]),
                    old_stress_reproduction_max_abs_error=error,trajectory_sha256=row['trajectory_sha256'])
                records.append(record)
                ids=np.argwhere(flip)
                if len(ids):
                    rank=np.argsort(np.abs(new[flip]))[-3:]
                    for t,k in ids[rank]:top.append(dict(panel_arm=label,index=index,round=int(t),module=int(k),old_stress=float(old[t,k]),face_stress=float(new[t,k]),mass=float(mass[k])))
                save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=1500,elapsed_seconds=time.perf_counter()-begin))
                if (index+1)%100==0:print(label,index+1,'/500',flush=True)
            assert case_count==500
            summary[label]=dict(cases=case_count,active_pair_module_operations=int(totals[0]),opposite_pair_module_fraction=float(totals[1]/max(1.,totals[0])),
                opposite_force_weight_fraction=float(totals[3]/max(1e-30,totals[2])),opposite_face_stress_weight_fraction=float(totals[5]/max(1e-30,totals[4])),
                near_contact_face_ties=int(totals[6]),active_module_operations=module_ops,opposite_module_fraction=module_flip/max(1,module_ops),
                opposite_module_face_stress_weight_fraction=stress_flip_abs/max(1e-30,stress_abs),old_stress_reproduction_max_abs_error=maximum_error,
                opposite_attraction_force_weight_fraction=float(totals[9]/max(1e-30,totals[8])),opposite_repulsion_force_weight_fraction=float(totals[11]/max(1e-30,totals[10])))
    assert all(sha(ROOT/p)==digest for p,digest in identity['sources'].items())
    for source in inputs.values():
        assert sha(ROOT/source['run']/'registration.json')==source['registration_sha256']
        assert sha(ROOT/source['run']/'results.json')==source['results_sha256']
    top=sorted(top,key=lambda r:abs(r['face_stress']),reverse=True)[:100]
    result=dict(status='complete',summary=summary,fixtures=checked,records=records,top_mismatches=top,
                no_new_trajectory=True,no_planner_solve=True,test_read=False,elapsed_seconds=time.perf_counter()-begin)
    save(OUT/'results.json',result)
    with (OUT/'operations_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# Saved-operation rectangle shape-direction audit','',
           'Read-only train500 trajectories; no new layout, planner solve or test access. Finite differences check only the contact derivative with direction frozen.', '',
           '| Source | Pair contributions with opposite sign | Force-weighted opposite fraction | Aggregated module opposite fraction |',
           '|---|---:|---:|---:|']
    for arm,s in summary.items():lines.append(f"| {arm} | {s['opposite_pair_module_fraction']:.3%} | {s['opposite_force_weight_fraction']:.3%} | {s['opposite_module_fraction']:.3%} |")
    lines+=['','An active-face derivative can grow large near oblique thin rectangles. Existing shape and movement limits remain necessary. A mismatch does not establish a quality gain; a future shape-law arm requires matched final-Q controls with the original planner.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'state.json',dict(status='complete',completed=len(records),total=1500))
    print(json.dumps(dict(status='complete',summary=summary),indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():save(OUT/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(error),pid=os.getpid(),no_automatic_retry=True,test_read=False))
        raise
