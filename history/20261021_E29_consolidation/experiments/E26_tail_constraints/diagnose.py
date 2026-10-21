"""Analyze existing output records only; no inference, optimizer or test reads."""
import json,math
from collections import Counter
import numpy as np
from .io import ROOT,HERE,save,sha

SOURCE=ROOT/'experiments/E25_integrated_baseline/runs/20261020_controls_steady_monitor500/results.json'
def main():
    result=json.loads(SOURCE.read_text(encoding='utf-8'));rows=result['records']
    q=np.array([r['metrics']['cost'] for r in rows]);n=np.array([r['n'] for r in rows]);w=np.exp((n-n.max())/12);w/=w.sum()
    geometry=np.array([1+.5*(max(0,r['metrics']['area_gap'])+max(0,r['metrics']['hpwl_gap'])) for r in rows])
    factor=np.array([math.exp(2*r['metrics']['violations_relative']) for r in rows])
    assert np.allclose(q,np.minimum(geometry*factor,9.999999),rtol=0,atol=1e-12)
    detail=[]
    for i,r in enumerate(rows):
        m=r['metrics'];detail.append({'index':r['index'],'n':r['n'],'Q':q[i],'seconds':r['seconds'],
            'weighted_excess':w[i]*(q[i]-1),'weighted_geometry_excess':w[i]*(geometry[i]-1),
            'weighted_soft_increment':w[i]*geometry[i]*(factor[i]-1),'soft':[m[k] for k in ['boundary_violations','grouping_violations','mib_violations']],
            'phase_counts':dict(Counter(t['phase'] for t in r['trace']))})
    summary={'count':len(rows),'weighted_Q':float(w@q),'weighted_geometry_excess':float(w@(geometry-1)),
        'weighted_soft_increment':float(w@(geometry*(factor-1))),'max_Q':float(q.max()),
        'above_threshold':{str(t):int((q>t).sum()) for t in [1.1,1.2,1.4,1.8,2]},
        'soft_bad':sum(sum(r['soft'])>0 for r in detail),'rescue_cases':sum('full_packed_refine' in r['phase_counts'] for r in detail)}
    save(HERE/'runs/20261020_saved_tail_diagnosis/results.json',{'source_sha256':sha(SOURCE),'summary':summary,
        'worst_Q':sorted(detail,key=lambda r:r['Q'],reverse=True)[:12],
        'weighted_contributors':sorted(detail,key=lambda r:r['weighted_excess'],reverse=True)[:12],
        'test_read':False,'new_inference_or_optimization':False})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
