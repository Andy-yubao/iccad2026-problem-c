"""Read-only all-snapshot pair-potential audit on current best training AR."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import datetime,json,zipfile
from pathlib import Path
import numpy as np
from ..R16.replay import ROOT,EXPERIMENT,RESEARCH,read,save,sha,process_lock

HERE=Path(__file__).resolve().parent;SOURCE=EXPERIMENT/'runs/research/20261103_R15_train500_trajectory_audit'
OUT=EXPERIMENT/'runs/research/20261103_R21_saved_energy_audit'


def energies(x,cfg):
    T,n=x.shape[:2];i,j=np.triu_indices(n,1);c=x[:,:,:2]+x[:,:,2:]/2;v=c[:,j]-c[:,i];d=np.hypot(v[:,:,0],v[:,:,1]);u=np.zeros_like(v)
    np.divide(v,d[:,:,None],out=u,where=d[:,:,None]>0)
    for t,k in np.argwhere(d==0):
        angle=(((int(i[k])+1)*73856093 ^ (int(j[k])+1)*19349663)%104729)*(2*np.pi/104729);u[t,k]=[np.cos(angle),np.sin(angle)]
    contact=np.full_like(u,np.inf);np.divide(x[:,i,2:]+x[:,j,2:],2*np.abs(u),out=contact,where=u!=0);contact=contact.min(2)
    g=d-contact;g[np.abs(g)<=32*np.finfo(float).eps*np.maximum(d,contact)]=0.
    ell=np.minimum(x[:,i,2:].min(2),x[:,j,2:].min(2));z=g/ell;r=cfg['attraction_range'];a=cfg['attraction_strength'];k=cfg['repulsion_strength']
    v=np.zeros_like(z);overlap=z<0;s=np.abs(z[overlap])
    v[overlap]=k*(s+np.log1p(np.exp(-2*s))-np.log(2.))-a*r*r/12
    near=(z>=0)&(z<r);s=z[near]
    v[near]=a*(s*s/2-2*s**3/(3*r)+s**4/(4*r*r)-r*r/12)
    return (ell*ell*v).sum(1)


def main():
    if OUT.exists():raise FileExistsError('Audit evidence protected')
    identity=dict(plan=read(HERE/'audit_plan.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),HERE/'audit_plan.json')},source_registration_sha256=sha(SOURCE/'registration.json'),source_results_sha256=sha(SOURCE/'results.json'))
    OUT.mkdir(parents=True);save(OUT/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as f:
        for p in identity['sources']:f.write(ROOT/p,p)
    cfg=read(SOURCE/'registration.json')['identity']['configs']['attraction_equal_slope'];records=[]
    with process_lock(RESEARCH/'.numerical.lock'):
        x=read(SOURCE/'results.json');assert x['status']=='complete'
        for row in x['records']:
            if row['arm']!='attraction_equal_slope':continue
            if (RESEARCH/'STOP').exists() or (OUT/'STOP').exists():save(OUT/'state.json',dict(status='stopped',completed=len(records),total=500));return
            index=row['index'];p=SOURCE/'trajectories/attraction_equal_slope'/f'{index:04d}.npz';assert sha(p)==row['trajectory_sha256']
            with np.load(p) as f:values=energies(f['history'],cfg);area=float(f['history'][0,:,2:].prod(1).sum())
            delta=np.diff(values);up=delta>1e-12*np.maximum(1,np.abs(values[:-1]))
            records.append(dict(index=index,energies=values.tolist(),ascending_rounds=int(up.sum()),max_normalized_ascent=float(delta.max()/area),net_normalized_change=float((values[-1]-values[0])/area),trajectory_sha256=row['trajectory_sha256']))
            save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=500))
        assert len(records)==500 and all(sha(ROOT/p)==h for p,h in identity['sources'].items())
        summary=dict(cases=500,rounds=10000,ascending_rounds=sum(r['ascending_rounds'] for r in records),cases_with_ascent=sum(r['ascending_rounds']>0 for r in records),
            mean_net_normalized_change=float(np.mean([r['net_normalized_change'] for r in records])),max_normalized_ascent=max(r['max_normalized_ascent'] for r in records))
        save(OUT/'results.json',dict(status='complete',summary=summary,records=records,no_new_trajectory=True,no_planner_solves=True,test_read=False));save(OUT/'state.json',dict(status='complete',completed=500,total=500))
        print(json.dumps(dict(status='complete',summary=summary),indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),no_automatic_retry=True,test_read=False))
        raise
