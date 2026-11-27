"""Combine completed, frozen 100+400 evidence; never solve or infer."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from experiments.E46_global_draft_refinement.io import HERE,ROOT,save,sources
from experiments.E46_global_draft_refinement.report import report

POLICIES=['control','large_area_fast']


def main():
    folder=Path(__file__).resolve().parent
    tags=['20261024_coupled_screen100','20261024_frozen_confirm400']
    states=[json.loads((HERE/'runs'/tag/'results_local.json').read_text(encoding='utf-8')) for tag in tags]
    frozen=json.loads((HERE/'runs/frozen_final.json').read_text(encoding='utf-8'))
    assert all(s['finished'] for s in states)
    assert all(s['identity']['sources']==frozen['sources'] for s in states)
    assert sources()==frozen['sources']
    records=[]
    for tag,s in zip(tags,states):
        for r in s['records']:
            v={p:{k:x for k,x in r['variants'][p].items() if k not in ('positions','stats')} for p in POLICIES}
            records.append({'index':r['index'],'member':r['member'],'n':r['n'],'variants':v,'source_run':tag})
    records.sort(key=lambda r:r['index'])
    assert [r['index'] for r in records]==list(range(500))
    assert len(set(tuple(r['member']) for r in records))==500
    identity={'sources':frozen['sources'],'config':{**states[1]['identity']['config'],'stage':'combined500','policies':POLICIES,
        'panel':list(range(500)),'members':[r['member'] for r in records],'derived_from_runs':tags}}
    save(folder/'results_local.json',{'identity':identity,'records':records,'finished':True,
        'cumulative_pipeline_seconds':sum(s['cumulative_pipeline_seconds'] for s in states),'coordinates':'See original source runs; no duplicate geometry here','test_read':False})
    combined=report(folder)
    original_records=[r for s in states for r in s['records']]
    byindex={r['index']:r for r in original_records}
    paired=[]
    for r in records:
        b,c=[r['variants'][p] for p in POLICIES]
        paired.append({'index':r['index'],'n':r['n'],'control_Q':b['metrics']['cost'],'candidate_Q':c['metrics']['cost'],
            'Q_delta':c['metrics']['cost']-b['metrics']['cost'],'time_delta':c['complete_seconds']-b['complete_seconds'],
            'control_seconds':b['complete_seconds'],'candidate_seconds':c['complete_seconds'],
            'area_gap_delta':c['metrics']['area_gap']-b['metrics']['area_gap'],'hpwl_gap_delta':c['metrics']['hpwl_gap']-b['metrics']['hpwl_gap'],
            'soft_delta':c['metrics']['total_soft_violations']-b['metrics']['total_soft_violations'],
            'control_area_ratio':b['geometry']['area_ratio'],'candidate_area_ratio':c['geometry']['area_ratio']})
    strata={}
    for label,use in [('small_medium',lambda r:r['n']<90),('large',lambda r:r['n']>=90)]:
        subset=[r for r in records if use(r)]
        q=[]
        for r in subset:
            b,c=[r['variants'][p] for p in POLICIES];q.append(c['metrics']['cost']-b['metrics']['cost'])
        strata[label]={'count':len(subset),'wins_ties_losses':[sum(x<-1e-6 for x in q),sum(abs(x)<=1e-6 for x in q),sum(x>1e-6 for x in q)],
            'public_early_stop_count':sum(byindex[r['index']]['variants']['large_area_fast']['stats']['public_stopped'] for r in subset),
            'mean_time_delta':float(np.mean([r['variants']['large_area_fast']['complete_seconds']-r['variants']['control']['complete_seconds'] for r in subset])),
            'max_Q_regression':float(max(q)),
            'area_ratio_control_mean':float(np.mean([r['variants']['control']['geometry']['area_ratio'] for r in subset])),
            'area_ratio_candidate_mean':float(np.mean([r['variants']['large_area_fast']['geometry']['area_ratio'] for r in subset]))}
    origins=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    unchanged={name:hashlib.sha256((ROOT/v['source']).read_bytes()).hexdigest()==v['sha256'] for name,v in origins.items()}
    checks={'sources_match_frozen':True,'copied_origins_unchanged':all(unchanged.values()),'origin_files':len(unchanged),
        'exact_500_members':True,'all_1000_hard_legal':all(v['metrics']['is_feasible'] and v['hard_check']['is_feasible'] for r in records for v in r['variants'].values()),
        'small_medium_same_Q':all(r['control_Q']==r['candidate_Q'] for r in paired if r['n']<90),
        'test_read':False,'new_training':False,'baseline_replaced':False}
    ledgers=[]
    for directory in sorted((HERE/'runs').iterdir()):
        f=directory/'results_local.json'
        if not f.exists() or directory==folder:continue
        s=json.loads(f.read_text(encoding='utf-8'))
        if 'records' not in s:continue
        variants=[v for r in s['records'] for v in r['variants'].values()]
        ledgers.append({'run':directory.name,'cases':len(s['records']),'logical_outputs':len(variants),'top_level_optimizer_calls':sum(v.get('shared_execution_source') is None for v in variants),
            'finished':s['finished'],'cumulative_pipeline_seconds':s['cumulative_pipeline_seconds']})
    confirm=json.loads((HERE/'runs'/tags[1]/'summary.json').read_text(encoding='utf-8'))
    detail={'checks':checks,'strata':strata,'best_improvements':sorted(paired,key=lambda r:r['Q_delta'])[:10],
        'largest_regressions':sorted(paired,key=lambda r:r['Q_delta'],reverse=True)[:10],
        'over_2s':[r for r in paired if r['candidate_seconds']>2],
        'run_ledger':ledgers,'research_pipeline_seconds':sum(r['cumulative_pipeline_seconds'] for r in ledgers),
        'quality_time_confirmation_passed':confirm['comparisons']['large_area_fast']['quality_and_time_gate_passed'],
        'full500_quality_time_passed':combined['comparisons']['large_area_fast']['quality_and_time_gate_passed'],
        'analysis_completed_utc':datetime.now(timezone.utc).isoformat(),'test_read':False}
    save(folder/'diagnostics.json',detail)
    print(json.dumps({'summary':combined,'diagnostics':detail},ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    n=np.array([r['n'] for r in paired]);dq=np.array([r['Q_delta'] for r in paired]);dt=np.array([r['time_delta'] for r in paired])
    colors=np.where(n>=90,'#0f766e','#94a3b8')
    for ax,y,title,label in [(axes[0],dq,'Quality change (lower is better)','Candidate Q - control Q'),(axes[1],dt,'Complete-time change','Candidate seconds - control seconds')]:
        ax.scatter(n,y,s=18,c=colors,alpha=.75);ax.axhline(0,c='#334155',lw=.8);ax.axvline(90,c='#94a3b8',lw=.8,ls='--');ax.set(xlabel='Block count',ylabel=label,title=title);ax.grid(alpha=.15)
    fig.savefig(folder/'paired_effects.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
