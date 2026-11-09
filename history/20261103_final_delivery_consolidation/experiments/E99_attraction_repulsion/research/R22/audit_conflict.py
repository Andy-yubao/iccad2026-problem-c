"""Read saved force conflicts and reconstruct attraction endpoint participation."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import datetime,json,zipfile
from pathlib import Path
import numpy as np
from ..R16.replay import ROOT,EXPERIMENT,RESEARCH,read,save,sha,process_lock

HERE=Path(__file__).resolve().parent;SOURCE=EXPERIMENT/'runs/research/20261103_R15_train500_trajectory_audit'
OUT=EXPERIMENT/'runs/research/20261103_R22_saved_force_conflict_audit'


def main():
    if OUT.exists():raise FileExistsError('Audit evidence protected')
    identity=dict(plan=read(HERE/'audit_plan.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),HERE/'audit_plan.json')},source_registration_sha256=sha(SOURCE/'registration.json'),source_results_sha256=sha(SOURCE/'results.json'))
    OUT.mkdir(parents=True);save(OUT/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as f:
        for p in identity['sources']:f.write(ROOT/p,p)
    records=[];both_count=0;opposed_count=0;all_a=0.;blocked_a=0.;cosines=[];all_sum=0.;net_sum=0.
    cfg=read(SOURCE/'registration.json')['identity']['configs']['attraction_equal_slope']
    with process_lock(RESEARCH/'.numerical.lock'):
        results=read(SOURCE/'results.json');assert results['status']=='complete'
        for row in results['records']:
            if row['arm']!='attraction_equal_slope':continue
            if (RESEARCH/'STOP').exists() or (OUT/'STOP').exists():save(OUT/'state.json',dict(status='stopped',completed=len(records),total=500));return
            index=row['index'];p=SOURCE/'trajectories/attraction_equal_slope'/f'{index:04d}.npz';assert sha(p)==row['trajectory_sha256']
            with np.load(p) as f:x=f['history'][:-1];m=f['module_log']
            with np.load(EXPERIMENT/'runs/research/20261102_frozen_C3_train500_control/coordinates'/f'{index:04d}.npz') as f:movable=f['constraints'][:,1]==0
            a=m[:,:,1:3][:,movable];r=m[:,:,3:5][:,movable];an=np.linalg.norm(a,axis=2);rn=np.linalg.norm(r,axis=2);dot=(a*r).sum(2);both=(an>1e-12)&(rn>1e-12)
            both_count+=int(both.sum());opposed_count+=int((both&(dot<0)).sum());cosines.extend((dot[both]/(an[both]*rn[both])).tolist())
            all_sum+=float((an+rn).sum());net_sum+=float(np.linalg.norm(a+r,axis=2).sum())
            T,n=x.shape[:2];i,j=np.triu_indices(n,1);center=x[:,:,:2]+x[:,:,2:]/2;v=center[:,j]-center[:,i];d=np.hypot(v[:,:,0],v[:,:,1]);u=np.zeros_like(v);np.divide(v,d[:,:,None],out=u,where=d[:,:,None]>0)
            for t,k in np.argwhere(d==0):
                angle=(((int(i[k])+1)*73856093 ^ (int(j[k])+1)*19349663)%104729)*(2*np.pi/104729);u[t,k]=[np.cos(angle),np.sin(angle)]
            contact=np.full_like(u,np.inf);np.divide(x[:,i,2:]+x[:,j,2:],2*np.abs(u),out=contact,where=u!=0);contact=contact.min(2)
            g=d-contact;g[np.abs(g)<=32*np.finfo(float).eps*np.maximum(d,contact)]=0.;ell=np.minimum(x[:,i,2:].min(2),x[:,j,2:].min(2));z=np.clip(g/(cfg['attraction_range']*ell),0,1)
            fa=cfg['attraction_strength']*np.maximum(g,0)*(1-z)**2;np.testing.assert_allclose(2*fa.sum(1),m[:,:,20].sum(1),rtol=2e-11,atol=2e-9)
            blocked=(m[:,i,21]>0)|(m[:,j,21]>0);mass=float(fa.sum());gated=float(fa[blocked].sum());all_a+=mass;blocked_a+=gated
            records.append(dict(index=index,both_force_operations=int(both.sum()),opposed_operations=int((both&(dot<0)).sum()),attraction_pair_magnitude=mass,overlap_endpoint_attraction_magnitude=gated,trajectory_sha256=row['trajectory_sha256']))
            save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=500))
        assert len(records)==500 and all(sha(ROOT/p)==h for p,h in identity['sources'].items())
        summary=dict(cases=500,rounds=10000,both_force_operations=both_count,opposed_net_force_fraction=opposed_count/max(1,both_count),median_attraction_repulsion_cosine=float(np.median(cosines)),aggregate_net_over_separate_force_norm=net_sum/max(all_sum,1e-30),attraction_magnitude_incident_to_overlapped_endpoint_fraction=blocked_a/max(all_a,1e-30))
        save(OUT/'results.json',dict(status='complete',summary=summary,records=records,no_new_trajectory=True,no_planner_solves=True,test_read=False));save(OUT/'state.json',dict(status='complete',completed=500,total=500));print(json.dumps(dict(status='complete',summary=summary),indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),no_automatic_retry=True,test_read=False))
        raise
