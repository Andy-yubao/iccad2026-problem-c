"""New planning metrics from saved coordinates only; no model/solver/scoring."""
from pathlib import Path
import sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
HERE=ROOT/'experiments/E85_cooperative_space_refinement'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def union(rectangles):
    if not len(rectangles):return 0.
    rect=np.asarray(rectangles);xs=np.unique(np.r_[rect[:,0],rect[:,0]+rect[:,2]])
    area=0.
    for a,b in zip(xs[:-1],xs[1:]):
        active=rect[(rect[:,0]<b)&(rect[:,0]+rect[:,2]>a)]
        if not len(active):continue
        intervals=sorted(zip(active[:,1],active[:,1]+active[:,3]));lo,hi=intervals[0];height=0.
        for c,d in intervals[1:]:
            if c>hi:height+=hi-lo;lo,hi=c,d
            else:hi=max(hi,d)
        area+=(b-a)*(height+hi-lo)
    return area

def module_counts(positions):
    p=np.asarray(positions,dtype=float);lo=p[:,:2];hi=lo+p[:,2:];area=p[:,2:].prod(-1)
    low=np.maximum(lo[:,None],lo[None,:]);high=np.minimum(hi[:,None],hi[None,:]);length=np.maximum(high-low,0)
    inter=length.prod(-1);np.fill_diagonal(inter,0)
    pair_load=inter.sum(1)/area;coverage=[]
    for i in range(len(p)):
        ids=np.flatnonzero(inter[i]>0)
        rect=np.c_[low[i,ids],length[i,ids]]
        coverage.append(union(rect)/area[i])
    coverage=np.asarray(coverage)
    assert np.all(coverage<=1+1e-8) and np.all(coverage<=pair_load+1e-8)
    return dict(union_N5=int((coverage>.05).sum()),pair_sum_N5=int((pair_load>.05).sum()),
                max_union_fraction=float(coverage.max()),max_pair_load=float(pair_load.max()),
                over05_ids=np.flatnonzero(coverage>.05).tolist())

def summarize(rows,key):
    counts=np.array([r[key] for r in rows])
    return dict(cases=len(counts),mean_count=float(counts.mean()),median_count=float(np.median(counts)),
                p95_count=float(np.percentile(counts,95)),max_count=int(counts.max()),total_modules=int(counts.sum()),
                zero_cases=int((counts==0).sum()),zero_case_fraction=float((counts==0).mean()))

def main():
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    nmap={tuple(r['member']):r['n'] for r in features}
    report=dict(scope='saved exposed dev only; no new inference or scoring',threshold=.05,operator='strict greater than',
                union_definition='area(R_i intersect union_{j!=i} R_j)/area(R_i); all valid modules including anchors',
                legacy_pair_definition='sum_j area(R_i intersect R_j)/area(R_i); same pixels can count repeatedly',
                test_read=False,models={})
    paths={'E74':ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0'}
    paths.update({arm:HERE/('runs/20261030_P74_F81_'+arm+'_seed0') for arm in ('control','cooperative')})
    for arm,folder in paths.items():
        records=read(folder/'terminal/results.json')['records'];rows=[]
        weights=np.exp((np.array([r['n'] for r in records])-max(r['n'] for r in records))/12)
        components=[]
        for r in records:
            raw=module_counts(r['draft_positions']);final=module_counts(r['positions']);m=r['metrics']
            assert final['union_N5']==0
            h=.5*max(0,m['hpwl_gap']);a=.5*max(0,m['area_gap']);soft=(1+h+a)*(math.exp(2*m['violations_relative'])-1)
            assert abs(1+h+a+soft-m['cost'])<1e-9
            components.append([h,a,soft])
            rows.append(dict(index=r['index'],member=r['member'],n=r['n'],Q=m['cost'],rescue_used=r['rescue_used'],
                draft=raw,postprocessed=final,Q_excess_components=dict(wire=h,area=a,soft=soft)))
        raw_rows=[r['draft'] for r in rows];c=np.average(np.asarray(components),weights=weights,axis=0)
        wQ=float(np.average([r['Q'] for r in rows],weights=weights))
        largest=sorted(rows,key=lambda r:weights[r['index']]*(r['Q']-1),reverse=True)
        model=dict(draft_union=summarize(raw_rows,'union_N5'),draft_pair_sum=summarize(raw_rows,'pair_sum_N5'),
            weighted_Q=wQ,Q_excess=dict(zip(('wire','area','soft'),c.tolist())),
            cases_Q_at_least_101=sum(r['Q']>=1.01 for r in rows),
            postprocessed_N5_total=sum(r['postprocessed']['union_N5'] for r in rows),
            largest_weighted_Q_excess=[dict(index=r['index'],n=r['n'],Q=r['Q'],rescue=r['rescue_used']) for r in largest[:10]])
        if arm!='E74':
            dev=read(folder/'full_dev.json')['records'];counts=[]
            for r in dev:
                x=r['module_load_gt05_fraction']*nmap[tuple(r['member'])]
                assert abs(x-round(x))<1e-7;counts.append(dict(pair_sum_N5=round(x)))
            model['dev2000_legacy_pair_sum']=summarize(counts,'pair_sum_N5')
        model['records']=rows
        report['models'][arm]=model
        print(arm,json.dumps({k:v for k,v in model.items() if k!='records'},ensure_ascii=False),flush=True)
    (OUT/'baseline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
