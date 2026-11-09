"""Offline paired reports; never invokes a model, optimizer or data reader."""
import argparse
import json
import math
import csv
import numpy as np
from .io import HERE,save,sha


def summarize(records,policy):
    rows=[r for r in records if policy in r['variants']]
    if not rows:return {}
    v=[r['variants'][policy] for r in rows];q=np.array([x['metrics']['cost'] for x in v]);t=np.array([x['complete_seconds'] for x in v]);n=np.array([r['n'] for r in rows]);w=np.exp((n-n.max())/12)
    return {'count':len(rows),'hard_legal':sum(x['metrics']['is_feasible'] and x['hard_check']['is_feasible'] for x in v),
        'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),
        'CVaR95_Q':float(np.sort(q)[-max(1,math.ceil(.05*len(q))):].mean()),'max_Q':float(q.max()),
        'complete_seconds_mean':float(t.mean()),'complete_seconds_p95':float(np.percentile(t,95)),
        'complete_seconds_max':float(t.max()),'over_2s':int((t>2).sum()),
        'area_ratio_mean':float(np.mean([x['geometry']['area_ratio'] for x in v])),
        'hpwl_gap_mean':float(np.mean([x['metrics']['hpwl_gap'] for x in v])),
        'area_gap_mean':float(np.mean([x['metrics']['area_gap'] for x in v])),
        'soft_violations_total':sum(x['metrics']['total_soft_violations'] for x in v),
        'postprocess_seconds_mean':float(np.mean([x['postprocess_seconds'] for x in v]))}


def report(folder):
    if (folder/'view.json').exists():
        view=json.loads((folder/'view.json').read_text(encoding='utf-8'));records=[];seen=set();elapsed=0.;finished=True;accounting=set()
        for entry in view['parents']:
            path=HERE/'runs'/entry['tag']/'results_local.json'
            if sha(path)!=entry['sha256']:raise ValueError('Parent results changed')
            parent=json.loads(path.read_text(encoding='utf-8'));elapsed+=parent['cumulative_pipeline_seconds'];finished &= parent['finished']
            accounting.add(parent['identity']['config'].get('time_accounting','v1: recorded solve-stage only; shared public problem preparation and offline witness generation/loading excluded; not learned deployment latency'))
            for r in parent['records']:
                if r['index'] in seen:raise ValueError('Duplicate view member')
                seen.add(r['index']);records.append({**r,'variants':{p:r['variants'][p] for p in view['policies']}})
        if len(accounting)!=1:raise ValueError('Mixed timing contracts')
        s={'identity':{'config':{'policies':view['policies'],'stage':'combined_development','time_accounting':next(iter(accounting))}},'records':records,'finished':finished,'cumulative_pipeline_seconds':elapsed}
    else:
        s=json.loads((folder/'results_local.json').read_text(encoding='utf-8'))
        if 'case_files' in s:s['records']=[json.loads((folder/name).read_text(encoding='utf-8')) for name in s['case_files'] if (folder/name).exists()]
    policies=s['identity']['config']['policies']
    requested_budget=s['identity']['config'].get('quality_case_seconds')
    summaries={p:summarize(s['records'],p) for p in policies};comparisons={};rows=[]
    for policy in policies:
        if policy=='control':continue
        pairs=[r for r in s['records'] if {'control',policy}<=set(r['variants'])]
        if not pairs:continue
        n=np.array([r['n'] for r in pairs]);w=np.exp((n-n.max())/12)
        bq,cq,bt,ct=[np.array([r['variants'][p][kind] if kind=='complete_seconds' else r['variants'][p]['metrics']['cost'] for r in pairs]) for p,kind in [('control','Q'),(policy,'Q'),('control','complete_seconds'),(policy,'complete_seconds')]]
        wb,wc=float(np.average(bq,weights=w)),float(np.average(cq,weights=w));tail=max(1,math.ceil(.05*len(pairs)))
        severe=(cq-bq>.05)&(cq>bq*1.05)
        c={'count':len(pairs),'weighted_Q_control':wb,'weighted_Q_candidate':wc,'weighted_Q_delta':wc-wb,
            'relative_excess_Q_reduction':(wb-wc)/max(wb-1,1e-12),'mean_Q_delta':float((cq-bq).mean()),
            'complete_time_delta':float((ct-bt).mean()),'time_ratio':float(ct.mean()/bt.mean()),
            'wins_ties_losses':[int((cq<bq-1e-6).sum()),int((abs(cq-bq)<=1e-6).sum()),int((cq>bq+1e-6).sum())],
            'CVaR95_control':float(np.sort(bq)[-tail:].mean()),'CVaR95_candidate':float(np.sort(cq)[-tail:].mean()),
            'max_Q_control':float(bq.max()),'max_Q_candidate':float(cq.max()),
            'severe_regression_indices':[r['index'] for r,flag in zip(pairs,severe) if flag],
            'all_hard_legal':all(r['variants'][p]['metrics']['is_feasible'] and r['variants'][p]['hard_check']['is_feasible'] for r in pairs for p in ('control',policy)),
            'candidate_over_2s':int((ct>2).sum())}
        if requested_budget is not None:c['requested_case_budget_met']=bool(np.all(ct<=requested_budget))
        c['quality_and_time_gate_passed']=bool(c['all_hard_legal'] and c['relative_excess_Q_reduction']>=.05 and c['time_ratio']<=1.03 and c['CVaR95_candidate']<=c['CVaR95_control']+1e-8 and c['max_Q_candidate']<=c['max_Q_control']+1e-8 and not severe.any())
        if requested_budget is not None:c['quality_and_time_gate_passed'] &= c['requested_case_budget_met']
        if len(pairs)>=100:
            rng=np.random.default_rng(20261024);intervals={}
            for cluster in ('shard','worker'):
                labels=np.array([r['member'][0] if cluster=='shard' else r['member'][0].split('/')[0] for r in pairs]);groups=[np.flatnonzero(labels==label) for label in sorted(set(labels))];draw=[]
                for _ in range(2000):
                    ids=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))]);draw.append([np.average((cq-bq)[ids],weights=w[ids]),(ct-bt)[ids].mean()])
                intervals[cluster]={'weighted_Q_delta_95':np.percentile(np.array(draw)[:,0],[2.5,97.5]).tolist(),'time_delta_95':np.percentile(np.array(draw)[:,1],[2.5,97.5]).tolist()}
            c['paired_cluster_bootstrap']=intervals
        comparisons[policy]=c
        for r in pairs:
            b,cand=r['variants']['control'],r['variants'][policy]
            rows.append({'index':r['index'],'n':r['n'],'candidate':policy,'control_Q':b['metrics']['cost'],'candidate_Q':cand['metrics']['cost'],
                'Q_delta':cand['metrics']['cost']-b['metrics']['cost'],'control_seconds':b['complete_seconds'],'candidate_seconds':cand['complete_seconds'],
                'control_area_ratio':b['geometry']['area_ratio'],'candidate_area_ratio':cand['geometry']['area_ratio'],
                'control_relative_distortion':b['geometry']['relative_distortion'],'candidate_relative_distortion':cand['geometry']['relative_distortion']})
    out={'finished':s['finished'],'stage':s['identity']['config']['stage'],'variants':summaries,'comparisons':comparisons,'cumulative_pipeline_seconds':s['cumulative_pipeline_seconds'],
        'time_accounting':s['identity']['config'].get('time_accounting','v1: recorded solve-stage only; shared public problem preparation and offline witness generation/loading excluded; not learned deployment latency'),'test_read':False}
    save(folder/'summary.json',out)
    if rows:
        with (folder/'paired_cases.csv').open('w',encoding='utf-8-sig',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    path=(HERE/'runs'/a.tag).resolve()
    if not path.is_relative_to(HERE/'runs'):raise ValueError('Bad path')
    print(json.dumps(report(path),indent=2))
