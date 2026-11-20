"""Supplementary saved-geometry/trace comparison and exchange sensitivity."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from helpers import pair_motion,center_setup,center_metrics
from topology_metric import target,similarity

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
AUX=ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0/layer_analysis_v1'
PRIOR=ROOT/'experiments/E139_delayed_overlap/runs/e119_delayed_overlap_seed0/comparative_audit_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    s=read(HERE/'summary.json');pp=read(HERE/'postprocessing_cases.json')
    rows=[read(HERE/f'cases/{i:04d}.json') for i in range(500)]
    reference=np.load(AUX/'reference.npy',mmap_mode='r');areas=np.load(AUX/'areas.npy',mmap_mode='r');flags=np.load(AUX/'flags.npy',mmap_mode='r')
    positions={'E139':np.load(PRIOR/'positions.npy',mmap_mode='r'),'E140':np.load(HERE/'positions.npy',mmap_mode='r')}
    losses={'E139':np.load(PRIOR/'losses.npy'),'E140':np.load(HERE/'losses.npy')}
    names=s['loss_names'];models=('E139','E140')
    shards=sorted({r['member'][0] for r in rows});cluster=np.array([shards.index(r['member'][0]) for r in rows])
    rng=np.random.default_rng(20261115);draws=rng.multinomial(len(shards),np.full(len(shards),1/len(shards)),size=2000)[:,cluster]
    bw=draws/draws.sum(1,keepdims=True)
    result=dict(loss_change={},cohort_uncertainty={},postprocessing_topology={},anchor_compatibility={},swap_sensitivity={},exchange_cases=[])
    delta=losses['E140'][:,12]-losses['E139'][:,12]
    net=-delta[:,0].mean()
    for j,key in enumerate(names):
        result['loss_change'][key]=dict(E139=float(losses['E139'][:,12,j].mean()),E140=float(losses['E140'][:,12,j].mean()),
            delta=float(delta[:,j].mean()),net_reduction_share=float(-delta[:,j].mean()/net))
    dq=np.array([r['Q_delta'] for r in pp]);dl=delta[:,0]
    result['loss_quality_quadrants']={
        'loss_lower_Q_better':int(((dl<0)&(dq < -1e-9)).sum()),'loss_lower_Q_worse':int(((dl<0)&(dq>1e-9)).sum()),
        'loss_lower_Q_same':int(((dl<0)&(np.abs(dq)<=1e-9)).sum()),'loss_higher_Q_better':int(((dl>=0)&(dq < -1e-9)).sum()),
        'loss_higher_Q_worse':int(((dl>=0)&(dq>1e-9)).sum()),'loss_higher_Q_same':int(((dl>=0)&(np.abs(dq)<=1e-9)).sum())}
    for cohort in ('E139_S3','E140_S3'):
        den=np.array([r['group_cohorts'][cohort]['count'] for r in rows]);result['cohort_uncertainty'][cohort]={}
        for key in ('unblocked','strict','path_strict'):
            values={m:np.array([sum(e[key][12] for e in r['group_cohorts'][cohort]['models'][m]) for r in rows]) for m in models}
            diff=values['E140']-values['E139'];boot=(draws@diff)/np.maximum(draws@den,1)
            result['cohort_uncertainty'][cohort][key]=dict(delta=float(diff.sum()/den.sum()),delta95=np.quantile(boot,[.025,.975]).tolist())
    post=[]
    for i,row in enumerate(rows):
        n=row['n'];ref=reference[i,:n];a=areas[i,:n];f=flags[i,:n];pre=f[:,1]!=0
        goal=target(ref[:,:2]+ref[:,2:]/2,a,preplaced=pre)
        setup=center_setup(ref[:,:2]+ref[:,2:]/2,a,pre)
        item=dict(index=i,member=row['member'],models={})
        for model,folder in [('E139',PRIOR.parent),('E140',HERE.parent)]:
            evaluation=read(folder/f'evaluation_no_mib_dev500/cases/{i:04d}.json');p=np.asarray(evaluation['positions']);centers=p[:,:2]+p[:,2:]/2
            top=center_metrics(centers,setup)[0]
            top['reference_relation_similarity']=similarity(centers,goal)['topology_similarity']
            item['models'][model]=top
        post.append(item)
    save(HERE/'postprocessing_topology_cases.json',post)
    for metric in ('local_axis_accuracy','knn6_recall','reference_relation_similarity'):
        a=np.array([r['models']['E139'][metric] for r in post]);b=np.array([r['models']['E140'][metric] for r in post])
        result['postprocessing_topology'][metric]=dict(E139=float(a.mean()),E140=float(b.mean()),delta=float((b-a).mean()),delta95=np.quantile(bw@(b-a),[.025,.975]).tolist())
    for model in models:
        records=[r['models'][model]['initial_anchor_compatibility'] for r in rows]
        result['anchor_compatibility'][model]=dict(cases=sum(r['any_conflict'] for r in records),
            indices=[i for i,r in enumerate(records) if r['any_conflict']],
            direct_PP_cases=sum(bool(r['direct_PP_conflicts']) for r in records),anchor_path_cases=sum(bool(r['anchor_path_conflicts']) for r in records),
            origin_path_cases=sum(bool(r['nonnegative_path_conflicts']) for r in records))
    code=inspect.getsource(pair_motion).replace('def pair_motion(p,areas,flags):','def with_margin(p,areas,flags,margin):').replace('.02*scale','margin*scale')
    scope={'np':np};exec(code,scope);motion=scope['with_margin']
    for margin in (.005,.02,.05):
        result['swap_sensitivity'][str(margin)]={}
        for model in models:
            count=0;cases=0
            for i,row in enumerate(rows):
                n=row['n'];_,events,_=motion(positions[model][i,3:,:n],areas[i,:n],flags[i,:n],margin);count+=len(events);cases+=bool(events)
            result['swap_sensitivity'][str(margin)][model]=dict(events=count,cases=cases)
    for row in rows:
        if row['models']['E140']['exchanges']:
            item=dict(index=row['index'],member=row['member'],events=row['models']['E140']['exchanges'],models={})
            for m in models:
                r=row['models'][m];item['models'][m]=dict(Q=r['final_metrics']['cost'],grouping=r['final_metrics']['grouping_violations'],boundary=r['final_metrics']['boundary_violations'],Tref=r['layers'][12]['topology']['reference_relation_similarity'])
            result['exchange_cases'].append(item)
    save(HERE/'deep_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('exchange_cases','loss_change')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
