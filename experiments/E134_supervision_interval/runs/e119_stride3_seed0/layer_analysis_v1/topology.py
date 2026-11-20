"""Pre-registered observable topology metrics, without changing any layout."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def finite(value):return float(value) if np.isfinite(value) else None
def ratio(numerator,denominator):return float(numerator/denominator) if denominator>0 else np.nan

def center_setup(reference_centers,areas,pre=None):
    ref=np.asarray(reference_centers,float);areas=np.asarray(areas,float);n=len(ref)
    scale=(np.sqrt(areas)[:,None]+np.sqrt(areas)[None,:])/2
    delta=(ref[None,:,:]-ref[:,None,:])/scale[...,None]
    dist=np.linalg.norm(delta,axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')
    neighbors={k:np.zeros((n,n),bool) for k in (4,6,8)}
    for k,mask in neighbors.items():mask[np.arange(n)[:,None],order[:,:min(k,n-1)]]=True
    pairs=np.triu(np.ones((n,n),bool),1)
    if pre is not None:pairs&=~(pre[:,None]&pre[None,:])
    local=pairs&(neighbors[6]|neighbors[6].T)
    return dict(ref=ref,areas=areas,scale=scale,delta=delta,order=order,neighbors=neighbors,pairs=pairs,local=local,
        axes=np.abs(delta)>=.10)

def center_metrics(centers,setup):
    p=np.asarray(centers,float);n=len(p);d=(p[None,:,:]-p[:,None,:])/setup['scale'][...,None]
    known=np.abs(d)>=.02;correct=(np.sign(d)==np.sign(setup['delta']))&known
    result={}
    for name,mask in (('global',setup['pairs']),('local',setup['local'])):
        eligible=mask[...,None]&setup['axes']
        result[name+'_axis_accuracy']=ratio((correct&eligible).sum(),eligible.sum())
        result[name+'_axis_resolved']=ratio((known&eligible).sum(),eligible.sum())
    dist=np.linalg.norm(d,axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')
    for k,reference in setup['neighbors'].items():
        actual=np.zeros((n,n),bool);actual[np.arange(n)[:,None],order[:,:min(k,n-1)]]=True
        result[f'knn{k}_recall']=float(((actual&reference).sum(1)/min(k,n-1)).mean())
    return result,d,correct

def intent(positions,setup):
    p=np.asarray(positions,float);center=p[:,:2]+p[:,2:]/2
    delta=center[None,:,:]-center[:,None,:]
    half=(p[:,None,2:]+p[None,:,2:])/2
    gaps=np.stack((delta[...,0]-half[...,0],-delta[...,0]-half[...,0],delta[...,1]-half[...,1],-delta[...,1]-half[...,1]),-1)/setup['scale'][...,None]
    choice=gaps.argmax(-1);top=np.sort(gaps,axis=-1)
    return gaps,choice,top[...,-1]-top[...,-2]>.02

_DS=None;_P=None;_MANIFEST=None
def initialize(manifest):
    global _DS,_P,_MANIFEST
    from datasets.random_reference import RandomReference
    _DS=RandomReference('dev_monitor500');_P=np.load(HERE/'positions.npy',mmap_mode='r');_MANIFEST=manifest

def worker(index):
    from scoring.draft_geometry import evaluate_geometry
    item=_DS[index];inputs=item['solver_inputs'];ref=np.asarray(item['supervision']['reference_xywh'],float)
    n=len(ref);areas=np.asarray(inputs[1],float);flags=np.asarray(inputs[5]);pre=flags[:,1]!=0
    reference_centers=ref[:,:2]+ref[:,2:]/2;setup=center_setup(reference_centers,areas,pre)
    refgap,_,_=intent(ref,setup);allowed=refgap>=-.01
    last=np.asarray(_P[index,12,:n],float);_,end_choice,end_known=intent(last,setup)
    final_pairs=setup['local']&end_known
    old=read(ROOT/f'experiments/E136_frontend_F_linear/runs/statistics_dev500_v1/geometry/{index:04d}.json')
    if old['member']!=item['member']:raise ValueError('Prior geometry members differ')
    if not np.array_equal(last,np.asarray(read(ROOT/_MANIFEST['evaluation']/f'cases/{index:04d}.json')['draft_positions'])):
        raise ValueError('S12 geometry differs')
    wire=np.zeros((n,n))
    for edge in np.asarray(inputs[2]):
        if np.all(edge==-1):continue
        a,b=int(edge[0]),int(edge[1]);wire[a,b]+=edge[2];wire[b,a]+=edge[2]
    wire=wire*np.triu(np.ones((n,n)),1);wire*=setup['pairs']
    result=[];previous=None;previous_delta=None
    nonpre=np.flatnonzero(~pre)
    nonpre_setup=center_setup(reference_centers[nonpre],areas[nonpre]) if len(nonpre)>1 else None
    for layer in range(13):
        p=np.asarray(_P[index,layer,:n],float);center=p[:,:2]+p[:,2:]/2
        top,d,correct=center_metrics(center,setup)
        _,choice,known=intent(p,setup)
        compatible=np.take_along_axis(allowed,choice[...,None],axis=-1)[...,0]
        eligible=setup['local']&allowed.any(-1)
        top.update(intent_reference_correct_clear=ratio((compatible&known&eligible).sum(),eligible.sum()),
            intent_resolved=ratio((known&eligible).sum(),eligible.sum()),
            intent_agreement_final=ratio(((choice==end_choice)&known&final_pairs).sum(),final_pairs.sum()))
        for name,mask in (('anchor',setup['local']&(pre[:,None]|pre[None,:])),('nonpre',setup['local']&~(pre[:,None]|pre[None,:]))):
            axes=mask[...,None]&setup['axes'];top[name+'_axis_accuracy']=ratio((correct&axes).sum(),axes.sum())
        weighted_axes=wire[...,None]*setup['axes']
        top['wire_axis_accuracy']=ratio((correct*weighted_axes).sum(),weighted_axes.sum())
        if nonpre_setup is not None:
            free_metrics,_,_=center_metrics(center[nonpre],nonpre_setup)
            top.update({f'nonpre_{key}':value for key,value in free_metrics.items() if key in ('local_axis_accuracy','knn6_recall')})
        if layer==12:geo=old['draft']['baseline']['geometry']
        else:geo,_,_=evaluate_geometry(p,inputs,details=layer in (0,3,6,9))
        geo.update(bbox_reference=geo['bbox_area']/old['reference']['geometry']['bbox_area'],
            hpwl_reference=geo['hpwl_total']/old['reference']['geometry']['hpwl_total'])
        change={}
        if previous is not None:
            before=previous[:,:2]+previous[:,2:]/2;movement=np.linalg.norm(center-before,axis=1)/np.sqrt(areas)
            axes=setup['local'][...,None]&setup['axes']
            code=np.where(np.abs(d)>=.02,np.sign(d),0);prevcode=np.where(np.abs(previous_delta)>=.02,np.sign(previous_delta),0)
            change=dict(center_move_local_mean=float(movement.mean()),center_move_local_p95=float(np.quantile(movement,.95)),
                logshape_change_mean=float(np.abs(np.log(p[:,2]/p[:,3])-np.log(previous[:,2]/previous[:,3])).mean()),
                local_axis_flip_rate=ratio(((code!=prevcode)&axes).sum(),axes.sum()))
        result.append(dict(layer=layer,topology={k:finite(v) for k,v in top.items()},geometry=geo,change=change))
        previous=p;previous_delta=d
    output=dict(index=index,member=item['member'],n=n,layers=result,
        reference_geometry=old['reference']['geometry'],final_geometry=old['final']['baseline']['geometry'],
        final_score=old['final']['baseline']['score'],original_final_Q=old['final']['baseline']['score']['cost'])
    path=HERE/f'cases/{index:04d}.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return index

def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    manifest=read(HERE/'manifest.json')
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(manifest,)) as pool:
        jobs=[pool.submit(worker,i) for i in range(500)]
        for j,future in enumerate(as_completed(jobs),1):
            future.result()
            if j%25==0:print('Topology/geometry',j,'/500',flush=True)

if __name__=='__main__':main()
