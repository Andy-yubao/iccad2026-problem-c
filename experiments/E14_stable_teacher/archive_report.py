"""Read-only completed-archive audit, Q statistics and distribution figure."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import numpy as np
from .data import HERE,ROOT,VERSION,save,sha


def distribution(rows):
    q=np.asarray([r['metrics']['cost'] for r in rows if math.isfinite(float(r.get('metrics',{}).get('cost',float('nan'))))])
    good=[r for r in rows if math.isfinite(float(r.get('metrics',{}).get('cost',float('nan'))))]
    if not len(q):return {'count':len(rows),'finite_q':0}
    w=np.exp(np.asarray([r['n'] for r in good])/12.)
    return {'count':len(rows),'finite_q':len(q),'mean':float(q.mean()),'median':float(np.median(q)),
            'p90':float(np.quantile(q,.9)),'p95':float(np.quantile(q,.95)),
            'p99':float(np.quantile(q,.99)),'max':float(q.max()),'min':float(q.min()),
            'module_weighted_q':float(np.average(q,weights=w)),
            'thresholds':{str(t):{'count':int((q<=t).sum()),'fraction':float((q<=t).mean()),'strict_lt_count':int((q<t).sum())} for t in (1.02,1.04,1.06,1.1,1.2,1.5,1.8,2.)}}


def main():
    start=time.perf_counter();source=HERE/'runs/teacher_archive25000_v1';output=HERE/'runs/20261019_archive25000_results'
    db=sqlite3.connect((source/'bank.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    db.execute('BEGIN')
    candidates=[dict(r) for r in db.execute('SELECT * FROM candidates')]
    attempts=[dict(r) for r in db.execute('SELECT * FROM attempts')]
    historical=[dict(r) for r in db.execute('SELECT * FROM historical_records')]
    meta={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
    db.rollback();db.close()
    rows=[dict(json.loads(c['record']),candidate_id=c['id']) for c in candidates if c['status']=='archived']
    teacher=json.loads((source/'teacher.json').read_text())
    selected=teacher['records'];plan=json.loads((ROOT/'protocol/teacher_archive25000_v1.json').read_text())
    progress=json.loads((source/'progress.json').read_text())
    referenced=set();issues=[];total_bytes=0;allowed=set(json.loads((ROOT/'protocol/split_v1.json').read_text())['shards']['train'])
    def check_file(r):
        found=[];path=source/r['archive_file']
        if r['member'][0] not in allowed:found.append({'member':r['member'],'error':'outside_train'})
        if not path.exists():return 0,[{'member':r['member'],'error':'missing_file'}]
        size=path.stat().st_size
        if sha(path)!=r['archive_sha256']:found.append({'member':r['member'],'error':'sha_mismatch'})
        try:
            with np.load(path) as z:
                if z['positions'].shape!=(r['n'],4) or len(z['input_areas'])!=r['n'] or str(z['task_version'])!=VERSION:
                    found.append({'member':r['member'],'error':'shape_or_version'})
        except Exception as exc:found.append({'member':r['member'],'error':repr(exc)})
        return size,found
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,(size,found) in enumerate(pool.map(check_file,rows)):
            total_bytes+=size;issues.extend(found)
            if (i+1)%5000==0:print(f'Audited {i+1}/{len(rows)} layout files',flush=True)
    by_member={tuple(r['member']):r for r in rows}
    selected_members=[tuple(r['member']) for r in selected]
    if len(set(selected_members))!=len(selected_members):issues.append({'error':'duplicate_teacher_member'})
    for r in selected:
        saved=by_member.get(tuple(r['member']))
        if saved is None or saved['archive_sha256']!=r['archive_sha256'] or not r.get('hard_legal',r.get('metrics',{}).get('is_feasible',False)) or float(r['metrics']['cost'])>1.06:
            issues.append({'error':'teacher_selection_mismatch','member':r['member']})
    strata=[]
    for b,(lo,hi) in enumerate(((1,40),(41,60),(61,80),(81,100),(101,110),(111,120))):
        full=[r for r in rows if r['stratum']==b];chosen=[r for r in selected if r['stratum']==b]
        strata.append({'min_n':lo,'max_n':hi,'archive_target':plan['archive_strata_quotas'][b],
                       'teacher_target':plan['teacher_strata_quotas'][b],'all':distribution(full),'teacher':distribution(chosen)})
    paired=[]
    for h in historical:
        if h['status']!='rejected':continue
        before=json.loads(h['record']);member=tuple(before['member']);after=by_member.get(member)
        old=before.get('metrics',before.get('rejected_metrics'))
        if after is None:issues.append({'error':'historical_drop_not_archived','member':member});continue
        paired.append({'member':list(member),'before_q':old['cost'],'after_q':after.get('metrics',{}).get('cost'),
                       'after_hard_legal':after.get('hard_legal',False),'recomputed':after.get('historical_drop_recomputed',False)})
    new=[r for r in rows if r.get('execution_version')=='E14.archive25000.v1']
    solve=np.asarray([r['postprocess_seconds'] for r in new]);elapsed=meta.get('archive_elapsed',progress['elapsed_seconds'])
    stages={s:sum(a['stage']==s for a in attempts) for s in sorted({a['stage'] for a in attempts})}
    finite_pairs=[r for r in paired if r['after_q'] is not None and math.isfinite(float(r['after_q']))]
    historical_summary={'count':len(paired),'all_recomputed':all(r['recomputed'] for r in paired),
       'Q_le_1_04':sum(r['after_hard_legal'] and r['after_q']<=1.04 for r in finite_pairs),
       'Q_le_1_06':sum(r['after_hard_legal'] and r['after_q']<=1.06 for r in finite_pairs),
       'mean_before_q':float(np.mean([r['before_q'] for r in finite_pairs])),
       'mean_after_q':float(np.mean([r['after_q'] for r in finite_pairs])),
       'improved':sum(r['after_q']<r['before_q']-1e-9 for r in finite_pairs),
       'worsened':sum(r['after_q']>r['before_q']+1e-9 for r in finite_pairs)}
    summary={'status':progress['status'],'distinct_layouts':len(rows),'imported_layouts':len(rows)-len(new),'new_results':len(new),
             'attempt_count':len(attempts),'attempt_stages':stages,'all_hard_legal':sum(r.get('hard_legal',False) if r.get('execution_version')=='E14.archive25000.v1' else bool(r['metrics']['is_feasible'] and r['independent_hard_checks']['is_feasible']) for r in rows),
             'evaluation_or_encoding_errors':sum('evaluation_or_encoding_error' in r for r in rows),
             'all':distribution(rows),'teacher':distribution(selected),'new_results_distribution':distribution(new),
             'teacher_quality_eligible':sum(c['status']=='archived' and c['hard_legal'] and c['q'] is not None and c['q']<=1.06 for c in candidates),
             'teacher_composition':{'historical_drop_recomputed':sum(r.get('historical_drop_recomputed',False) for r in selected),
                                    'imported_accepted':sum(r.get('archive_origin')=='imported_accepted' for r in selected),
                                    'new_archive_results':sum(r.get('execution_version')=='E14.archive25000.v1' for r in selected)},
             'historical_drops':historical_summary,'strata':strata,
             'runtime':{'new_archive_wall_seconds':elapsed,'new_results_per_hour':len(new)/elapsed*3600,
                        'postprocess_mean_seconds':float(solve.mean()),'postprocess_p95_seconds':float(np.quantile(solve,.95)),
                        'postprocess_p99_seconds':float(np.quantile(solve,.99)),'postprocess_max_seconds':float(solve.max())},
             'integrity':{'issues':issues,'files_checked':len(rows),'referenced_layout_bytes':total_bytes,
                          'layout_files_present':len(list((source/'layouts').glob('*.npz'))),
                          'unique_members':len(by_member),'teacher_unique_members':len(set(selected_members)),
                          'all_members_train_only':all(r['member'][0] in allowed for r in rows)},
             'local_test_read':False,'Q_scope':'Neutral-runtime Q on MIB-derived inputs with original reference denominators; selected teacher labels, not student/hidden-test performance',
             'worst10':sorted(rows,key=lambda r:r.get('metrics',{}).get('cost',-1),reverse=True)[:10]}
    save(output/'summary.json',summary)
    save(output/'historical_drop_pairs.json',paired)
    with (output/'all_q.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['shard','row','n','Q','hard_legal','teacher_selected','file'])
        membership=set(selected_members)
        for r in rows:writer.writerow([*r['member'],r['n'],r.get('metrics',{}).get('cost'),r.get('hard_legal',r['metrics']['is_feasible']),tuple(r['member']) in membership,r['archive_file']])
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allq=np.asarray([r['metrics']['cost'] for r in rows]);tq=np.asarray([r['metrics']['cost'] for r in selected])
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    bins=np.linspace(.999,1.2,75)
    axes[0].hist(allq,bins=bins,label='All archived layouts',alpha=.6,color='#2563eb')
    axes[0].hist(tq,bins=bins,label='Selected teacher layouts',alpha=.75,color='#10b981')
    axes[0].set(xlabel='Q (lower is better)',ylabel='Count',title=f'Q distribution (Q>1.2: {(allq>1.2).sum()} archived)')
    for threshold in (1.04,1.06):axes[0].axvline(threshold,color='#dc2626',linestyle='--',linewidth=1)
    for q,label,color in [(allq,'All archived','#2563eb'),(tq,'Selected teacher','#10b981')]:
        q=np.sort(q);axes[1].plot(q,np.arange(1,len(q)+1)/len(q),label=label,color=color)
    axes[1].set(xlim=(.999,max(1.2,float(allq.max()))),ylim=(0,1.01),xlabel='Q',ylabel='Cumulative fraction',title='Full-range cumulative distribution')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'q_distribution.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('strata','worst10')},indent=2),flush=True)
    print(f'Audit seconds: {time.perf_counter()-start:.1f}',flush=True)


if __name__=='__main__':main()
