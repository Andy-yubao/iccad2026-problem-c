"""Paired development comparisons using saved outputs only."""
from .resources import configure
configure()
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .io import HERE,save,sha

def compare(base_path,new_path):
    base=json.loads(base_path.read_text());new=json.loads(new_path.read_text());lookup={tuple(r['member']):r for r in base['records']}
    rows=new['records'];old=[lookup[tuple(r['member'])] for r in rows]
    a=np.array([r['metrics']['cost'] for r in old]);b=np.array([r['metrics']['cost'] for r in rows]);d=b-a
    n=np.array([r['n'] for r in rows]);w=np.exp((n-n.max())/12)
    before=np.array([r['seconds'] for r in old]);after=np.array([r['seconds'] for r in rows])
    shards=sorted({r['member'][0] for r in rows});keys={key:i for i,key in enumerate(shards)}
    group=np.array([keys[r['member'][0]] for r in rows]);weights=np.bincount(group,weights=w)
    effect=np.bincount(group,weights=w*d);rng=np.random.default_rng(20261022)
    samples=rng.integers(0,len(shards),(2000,len(shards)));boot=effect[samples].sum(1)/weights[samples].sum(1)
    k=max(1,int(np.ceil(.05*len(rows))))
    out={'baseline':str(base_path.relative_to(HERE)), 'candidate':str(new_path.relative_to(HERE)),
        'baseline_sha256':sha(base_path),'candidate_sha256':sha(new_path),'count':len(rows),
        'weighted_Q_before':float(np.average(a,weights=w)),'weighted_Q_after':float(np.average(b,weights=w)),
        'weighted_Q_delta':float(np.average(d,weights=w)),
        'paired_shard_bootstrap_weighted_delta_95CI':np.percentile(boot,[2.5,97.5]).tolist(),
        'bootstrap_shards':len(shards),'bootstrap_replicates':2000,
        'mean_seconds_before':float(before.mean()),'mean_seconds_after':float(after.mean()),
        'relative_seconds_change':float(after.mean()/before.mean()-1),
        'p95_Q_before':float(np.percentile(a,95)),'p95_Q_after':float(np.percentile(b,95)),
        'cvar95_Q_before':float(np.sort(a)[-k:].mean()),'cvar95_Q_after':float(np.sort(b)[-k:].mean()),
        'max_Q_before':float(a.max()),'max_Q_after':float(b.max()),
        'improved':int((d<-1e-10).sum()),'regressed':int((d>1e-10).sum()),'max_regression':float(d.max()),
        'hard_legal':sum(r['metrics']['is_feasible'] for r in rows),
        'phases':dict(Counter(t['phase'] for r in rows for t in r.get('trace',[]))),
        'scope':'Adaptive development, MIB derived task, neutral runtime Q; not contest score; confidence interval does not correct adaptive selection',
        'test_read':False}
    out['groups']={}
    for label,mask in [('lt60',n<60),('60to89',(n>=60)&(n<90)),('90to109',(n>=90)&(n<110)),('ge110',n>=110)]:
        if mask.any():out['groups'][label]={'count':int(mask.sum()),'weighted_Q_before':float(np.average(a[mask],weights=w[mask])),
            'weighted_Q_after':float(np.average(b[mask],weights=w[mask])),
            'mean_seconds_before':float(before[mask].mean()),'mean_seconds_after':float(after[mask].mean()),
            'max_Q_before':float(a[mask].max()),'max_Q_after':float(b[mask].max())}
    out['largest_regressions']=[{'index':rows[i]['index'],'member':rows[i]['member'],'n':int(n[i]),
            'Q_before':float(a[i]),'Q_after':float(b[i]),'delta':float(d[i])} for i in np.argsort(-d)[:10]]
    out['worst_candidates']=[{'index':rows[i]['index'],'member':rows[i]['member'],'n':int(n[i]),
        'Q_before':float(a[i]),'Q_after':float(b[i]),'seconds':float(after[i]),'metrics':rows[i]['metrics']} for i in np.argsort(-b)[:10]]
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',type=Path,nargs='+')
    ap.add_argument('--baseline',type=Path,default=HERE/'runs/20261022_matched_steady_monitor500/results.json');args=ap.parse_args()
    for path in args.candidate:
        if path.is_dir():path=path/'results.json'
        result=compare(args.baseline.resolve(),path.resolve());save(path.parent/'paired_analysis.json',result)
        print(json.dumps({key:value for key,value in result.items() if key not in ('worst_candidates','largest_regressions')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
