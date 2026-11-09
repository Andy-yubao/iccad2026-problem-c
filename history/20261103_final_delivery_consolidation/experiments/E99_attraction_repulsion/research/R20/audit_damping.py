"""Read-only independent reconstruction of current-best pair damping logs."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import datetime,json,zipfile
from pathlib import Path
import numpy as np
from ..R16.replay import ROOT,EXPERIMENT,RESEARCH,read,save,sha,process_lock

HERE=Path(__file__).resolve().parent
SOURCE=EXPERIMENT/'runs/research/20261103_R15_train500_trajectory_audit'
OUT=EXPERIMENT/'runs/research/20261103_R20_saved_degree_audit'


def main():
    if OUT.exists():raise FileExistsError('Audit evidence protected')
    identity=dict(plan=read(HERE/'audit_plan.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),HERE/'audit_plan.json')},
        original_registration_sha256=sha(SOURCE/'registration.json'),original_results_sha256=sha(SOURCE/'results.json'))
    OUT.mkdir(parents=True);save(OUT/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in identity['sources']:z.write(ROOT/p,p)
    records=[];all_amp=[];all_degree_ratio=[];force_mass=0.;amp_force=0.;double_force=0.;rowmax=0.;max_force_error=0.;suppressed=0;nonlimited=0
    with process_lock(RESEARCH/'.numerical.lock'):
        results=read(SOURCE/'results.json');assert results['status']=='complete'
        cfg=read(SOURCE/'registration.json')['identity']['configs']['attraction_equal_slope']
        for old in results['records']:
            if old['arm']!='attraction_equal_slope':continue
            if (RESEARCH/'STOP').exists() or (OUT/'STOP').exists():save(OUT/'state.json',dict(status='stopped',completed=len(records),total=500));return
            index=old['index'];p=SOURCE/'trajectories/attraction_equal_slope'/f'{index:04d}.npz';assert sha(p)==old['trajectory_sha256']
            with np.load(p) as a:x=a['history'][:-1];log=a['module_log'];rounds=a['round_log']
            T,n=x.shape[:2];i,j=np.triu_indices(n,1);centers=x[:,:,:2]+x[:,:,2:]/2;v=centers[:,j]-centers[:,i];d=np.linalg.norm(v,axis=2)
            u=np.zeros_like(v);np.divide(v,d[:,:,None],out=u,where=d[:,:,None]>0)
            for t,k in np.argwhere(d==0):
                angle=(((int(i[k])+1)*73856093 ^ (int(j[k])+1)*19349663)%104729)*(2*np.pi/104729)
                u[t,k]=[np.cos(angle),np.sin(angle)]
            radii=np.full_like(u,np.inf);np.divide(x[:,i,2:]+x[:,j,2:],2*np.abs(u),out=radii,where=u!=0)
            contact=radii.min(2);gap=d-contact;gap[np.abs(gap)<=32*np.finfo(float).eps*np.maximum(d,contact)]=0.
            ell=np.minimum(x[:,i,2:].min(2),x[:,j,2:].min(2));R=cfg['attraction_range']*ell;z=np.clip(gap/R,0,1)
            fa=cfg['attraction_strength']*np.maximum(gap,0)*(1-z)**2;fr=-cfg['repulsion_strength']*ell*np.tanh(np.maximum(-gap,0)/ell);f=fa+fr
            active=f!=0;degree=np.zeros((T,n),dtype=np.int64);ti=np.repeat(np.arange(T),len(i));ii=np.tile(i,T);jj=np.tile(j,T)
            np.add.at(degree,(ti,ii),active.ravel());np.add.at(degree,(ti,jj),active.ravel())
            np.testing.assert_array_equal(degree,log[:,:,0]);D=np.maximum(1,degree.max(1));np.testing.assert_array_equal(D,rounds[:,2])
            net=np.zeros((T,n,2))
            for axis in range(2):
                np.add.at(net[:,:,axis],(ti,ii),(f*u[:,:,axis]).ravel());np.add.at(net[:,:,axis],(ti,jj),(-f*u[:,:,axis]).ravel())
            saved=log[:,:,1:3]+log[:,:,3:5];np.testing.assert_allclose(net,saved,rtol=2e-10,atol=2e-9)
            error=float(np.abs(net-saved).max());max_force_error=max(max_force_error,error)
            denom=np.maximum(1,np.maximum(degree[:,i],degree[:,j]));amplification=D[:,None]/denom;weights=np.abs(f)
            all_amp.extend(amplification[active].tolist());all_degree_ratio.extend(np.broadcast_to(D[:,None],degree.shape)[degree>0]/degree[degree>0])
            w=float(weights.sum());force_mass+=w;amp_force+=float((weights*amplification).sum());double_force+=float(weights[amplification>=2].sum())
            row=np.zeros((T,n));pairweight=active/denom
            np.add.at(row,(ti,ii),pairweight.ravel());np.add.at(row,(ti,jj),pairweight.ravel());rowmax=max(rowmax,float(row.max()));assert row.max()<=1+1e-12
            raw=np.linalg.norm(log[:,:,6:8],axis=2);own=raw>log[:,:,8]+1e-10;valid=(raw>1e-10)&~own
            nonlimited+=int(valid.sum());suppressed+=int((valid&(rounds[:,4,None]<.999)).sum())
            records.append(dict(index=index,mean_max_degree=float(D.mean()),mean_active_degree=float(degree[degree>0].mean()),pair_amplification_median=float(np.median(amplification[active])),force_weighted_amplification=float((weights*amplification).sum()/max(w,1e-30)),reconstructed_force_max_abs_error=error,row_sum_max=float(row.max()),trajectory_sha256=old['trajectory_sha256']))
            save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=500))
        assert len(records)==500 and all(sha(ROOT/p)==h for p,h in identity['sources'].items())
        amp=np.array(all_amp);ratio=np.array(all_degree_ratio)
        summary=dict(cases=500,rounds=10000,pair_amplification_median=float(np.median(amp)),pair_amplification_p95=float(np.percentile(amp,95)),
            pair_fraction_amplification_at_least2=float((amp>=2).mean()),force_weighted_amplification=amp_force/max(force_mass,1e-30),force_fraction_amplification_at_least2=double_force/max(force_mass,1e-30),
            global_D_over_module_degree_median=float(np.median(ratio)),global_D_over_module_degree_p95=float(np.percentile(ratio,95)),proposed_max_row_sum=rowmax,
            reconstructed_force_max_abs_error=max_force_error,common_scale_collateral_fraction=suppressed/max(1,nonlimited))
        save(OUT/'results.json',dict(status='complete',summary=summary,records=records,no_new_trajectory=True,no_planner_solves=True,test_read=False))
        save(OUT/'state.json',dict(status='complete',completed=500,total=500))
        lines=['# Saved operation audit: global degree damping','',json.dumps(summary,indent=2),'','The amplification is hypothetical on original snapshots only. It is not a new trajectory or finalQ improvement. Symmetric pair degree weights preserve equal/opposite pair force; row-sum bound alone does not prove full-layout stability.']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='complete',summary=summary),indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),no_automatic_retry=True,test_read=False))
        raise
