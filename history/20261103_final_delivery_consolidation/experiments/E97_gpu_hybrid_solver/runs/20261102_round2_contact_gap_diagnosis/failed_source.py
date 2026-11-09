"""Saved final geometry/public constraints only; classify residual soft-gap sizes."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import json,math
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union
from .train_io.cache import load_rows
from .latency import HERE

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=HERE/'runs/20261102_round2_contact_gap_diagnosis';output.mkdir(exist_ok=False)
    rows=load_rows('dev_monitor500',list(range(500)));arms={}
    for arm in ['E85','C3','C6']:
        details=[];boundary=[];group=[]
        for index,row in enumerate(rows):
            path=HERE/f'runs/20261102_fullrepair_singlethread_dev500/{arm}/terminal/{index:04d}.json';saved=json.loads(path.read_text(encoding='utf-8'));assert tuple(saved['member'])==tuple(row['member'])
            p=np.array(saved['positions']);c=np.array(row['case'].solver_inputs[5]);lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0);weight=math.exp((len(p)-120)/12)
            gaps=np.c_[p[:,0]-lo[0],hi[0]-p[:,0]-p[:,2],hi[1]-p[:,1]-p[:,3],p[:,1]-lo[1]]
            missing=[]
            for i,code in enumerate(c[:,4].astype(int)):
                required=[j for j,bit in enumerate([1,2,4,8]) if code&bit]
                gap=max([abs(gaps[i,j]) for j in required],default=0)
                if required and gap>=1e-6:
                    item=dict(index=index,n=len(p),block=i,code=code,gap=gap,gap_local=gap/math.sqrt(p[i,2]*p[i,3]),weight=weight)
                    missing.append(item);boundary.append(item)
            disconnected=[]
            for label in sorted(set(c[:,3])-{0}):
                ids=np.flatnonzero(c[:,3]==label);u=unary_union([box(*[p[i,0],p[i,1],p[i,0]+p[i,2],p[i,1]+p[i,3]]) for i in ids])
                if u.geom_type!='MultiPolygon':continue
                components=list(u.geoms);reached={0};remaining=set(range(1,len(components)));mst=[]
                while remaining:
                    gap,a,b=min((components[a].distance(components[b]),a,b) for a in reached for b in remaining)
                    reached.add(b);remaining.remove(b);mst.append(float(gap))
                for gap in mst:group.append(dict(index=index,n=len(p),label=int(label),gap=gap,gap_local=gap/math.sqrt(float(np.mean(p[ids,2]*p[ids,3]))),weight=weight))
                disconnected.append(dict(label=int(label),components=len(components),mst_gaps=mst))
            assert len(missing)==saved['metrics']['boundary_violations'],(arm,index,len(missing),saved['metrics']['boundary_violations'])
            assert sum(d['components']-1 for d in disconnected)==saved['metrics']['grouping_violations']
            if missing or disconnected:details.append(dict(index=index,n=len(p),member=list(row['member']),boundary=missing,group=disconnected))
        summary={}
        for name,items in [('boundary',boundary),('group',group)]:
            mass=sum(x['weight'] for x in items)
            summary[name]=dict(count=len(items),physical_gap_weighted_mean=sum(x['weight']*x['gap'] for x in items)/max(mass,1e-30),physical_gap_max=max([x['gap'] for x in items],default=0),weighted_fraction_gap_le={str(t):sum(x['weight'] for x in items if x['gap']<=t)/max(mass,1e-30) for t in [1e-6,1e-4,1e-3,.01,.1,1.]},weighted_fraction_local_gap_le={str(t):sum(x['weight'] for x in items if x['gap_local']<=t)/max(mass,1e-30) for t in [1e-6,1e-4,1e-3,.01,.1]},largest10=sorted(items,key=lambda x:x['weight']*x['gap_local'],reverse=True)[:10])
        arms[arm]=dict(summary=summary,details=details)
    result=dict(status='complete',scope='Saved final geometry/public constraints only; no network, solver, scorer, training or threshold scan. Fixed reporting bins only.',group_gap='MST distance between exact Shapely union components; zero may mean point contact and is not an edge-contact feasibility certificate',arms=arms,test_read=False)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({a:{n:{k:v for k,v in t.items() if k!='largest10'} for n,t in d['summary'].items()} for a,d in arms.items()},indent=2))

if __name__=='__main__':main()
