"""Re-score saved outputs and report fixed-panel acceptance and paired effects."""
import argparse
import json
from collections import Counter
import numpy as np
from .cases import prepare_cache
from .io import HERE,ROOT,save,sha
from scoring.evaluation import evaluate_case


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder');parser.add_argument('--panel',choices=['monitor','confirmation'],default='monitor');args=parser.parse_args()
    from pathlib import Path
    folder=Path(args.folder);result=json.loads((folder/'results.json').read_text());config=json.loads((folder/'config.json').read_text())
    records=result['records'];cached=prepare_cache(args.panel);public=cached['rows']
    if len(records)!=500 or {r['index'] for r in records}!=set(range(500)):raise ValueError('Acceptance requires all 500 prescribed cases')
    scores=[];legal=[]
    for record in records:
        row=public[record['index']]
        if row['member']!=record['member']:raise ValueError('Panel member mismatch')
        if 'positions' not in record:raise ValueError('Missing failed output; cannot pass acceptance')
        value=evaluate_case(row['case'],record['positions'],record['seconds'])['metrics']
        if abs(value['cost']-record['metrics']['cost'])>1e-10:raise ValueError('Stored score differs from independent re-score')
        scores.append(value['cost']);legal.append(value['is_feasible'])
    scores=np.array(scores);n=np.array([r['n'] for r in records]);w=np.exp(n/12-n.max()/12);seconds=np.array([r['seconds'] for r in records])
    weighted=float(np.average(scores,weights=w));mean_seconds=float(seconds.mean())
    analysis={'source_result_sha256':sha(folder/'results.json'),'panel':args.panel,'count':500,'hard_legal':sum(legal),
        'weighted_Q':weighted,'mean_Q':float(scores.mean()),'p95_Q':float(np.percentile(scores,95)),'max_Q':float(scores.max()),
        'Q_gt_1_8':int((scores>1.8).sum()),'Q_gt_2':int((scores>2).sum()),'seconds_mean':mean_seconds,
        'seconds_p95':float(np.percentile(seconds,95)),'seconds_max':float(seconds.max()),'direct':bool(config['direct']),
        'target_passed':bool(sum(legal)==500 and config['direct'] and weighted<1.04 and mean_seconds<.25),
        'metrics_scope':'adaptive development, MIB derived task, neutral runtime Q; not official contest score','test_read':False}
    analysis['groups']={}
    for name,mask in [('n_lt_60',n<60),('n60_89',(n>=60)&(n<90)),('n90_109',(n>=90)&(n<110)),('n_ge_110',n>=110)]:
        if mask.any():analysis['groups'][name]={'count':int(mask.sum()),'weighted_Q':float(np.average(scores[mask],weights=w[mask])),
            'mean_seconds':float(seconds[mask].mean()),'max_Q':float(scores[mask].max())}
    analysis['mean_violations']={key:float(np.mean([r['metrics'][key] for r in records])) for key in ['boundary_violations','grouping_violations','mib_violations']}
    analysis['phases']=dict(Counter(t['phase'] for r in records for t in r['trace']))
    rng=np.random.default_rng(20261020);samples=rng.integers(0,500,(2000,500));ws=w[samples]
    analysis['bootstrap_weighted_Q_95CI']=np.percentile((scores[samples]*ws).sum(1)/ws.sum(1),[2.5,97.5]).tolist()
    analysis['bootstrap_mean_seconds_95CI']=np.percentile(seconds[samples].mean(1),[2.5,97.5]).tolist()
    if args.panel=='monitor':
        baseline=json.loads((ROOT/'experiments/E17_e10_geometry_v5/runs/20261020_e10_geometry_v5_seed0/final_evaluation/results.json').read_text())['records']
        mapping={tuple(r['member']):r for r in baseline};old=np.array([mapping[tuple(r['member'])]['metrics']['cost'] for r in records]);delta=scores-old
        analysis['paired_E17']={'weighted_Q_delta':float(np.average(delta,weights=w)),'improved':int((delta<-1e-10).sum()),
            'regressed':int((delta>1e-10).sum()),'max_Q_regression':float(delta.max()),
            'weighted_delta_95CI':np.percentile((delta[samples]*ws).sum(1)/ws.sum(1),[2.5,97.5]).tolist()}
    save(folder/'audit.json',analysis);print(json.dumps(analysis,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
