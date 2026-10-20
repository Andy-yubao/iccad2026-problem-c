"""Read-only committed-result snapshot and strict-Q threshold comparison."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from .data import HERE, save


def main():
    source=HERE/'runs/teacher_q104_parallel_v1'
    output=HERE/'runs/20261019_q106_threshold_audit'
    db=sqlite3.connect((source/'bank.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    db.execute('BEGIN')
    rows=[dict(r) for r in db.execute('SELECT * FROM candidates ORDER BY id')]
    metadata={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
    db.rollback();db.close()
    old=HERE/'runs/teacher_train16000_v1'
    old_labels={tuple(r['member']):old/r['label_file'] for line in (old/'records.jsonl').read_text().splitlines()
                if (r:=json.loads(line)).get('label_file')}
    scored=[];priority=[];statuses={}
    for c in rows:
        statuses[c['status']]=statuses.get(c['status'],0)+1
        if not c['record']:continue
        r=json.loads(c['record']);m=r.get('metrics',r.get('rejected_metrics'))
        if m is None or not math.isfinite(float(m['cost'])):continue
        q=float(m['cost']);checks=r.get('independent_hard_checks')
        item={'id':c['id'],'member':[c['shard'],c['row']],'n':c['n'],'stratum':c['bin'],
              'status':c['status'],'Q':q,'official_hard_legal':bool(m['is_feasible']),
              'independent_hard_legal':bool(checks['is_feasible']) if checks is not None else None,
              'execution_version':r.get('execution_version','inherited')}
        scored.append(item)
        if c['status']=='rejected' and 1.04<q<1.06:
            path=old_labels.get(tuple(item['member']))
            item=dict(item,saved_old_layout=str(path) if path is not None and path.exists() else None)
            priority.append(item)
    def counts(items):
        n=len(items)
        return {'scored':n,**{key:{'count':sum(predicate(r['Q']) for r in items),
                                  'fraction':sum(predicate(r['Q']) for r in items)/n if n else None}
                for key,predicate in [('Q_lt_1_04',lambda q:q<1.04),('Q_le_1_04',lambda q:q<=1.04),
                                      ('Q_lt_1_06',lambda q:q<1.06),('Q_le_1_06',lambda q:q<=1.06),
                                      ('Q_eq_1_04',lambda q:q==1.04),('Q_eq_1_06',lambda q:q==1.06),
                                      ('Q_between_strict',lambda q:1.04<q<1.06)]}}
    summary={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'bank':str(source),'statuses':statuses,
             'overall':counts(scored),'parallel_only':counts([r for r in scored if r['execution_version']=='E14.parallel_teacher.v1']),
             'by_stratum':[{'stratum':i,**counts([r for r in scored if r['stratum']==i])} for i in range(6)],
             'priority_rejected_count':len(priority),'priority_with_saved_old_layout':sum(r['saved_old_layout'] is not None for r in priority),
             'priority_requiring_recomputation':sum(r['saved_old_layout'] is None for r in priority),
             'official_hard_legal_scored':sum(r['official_hard_legal'] for r in scored),
             'independent_hard_check_unknown':sum(r['independent_hard_legal'] is None for r in scored),
             'attempt_budget_counter':metadata.get('solves'),'local_test_read':False,
             'scope':'Distinct members with finite committed Q; pending/incompatible/failed-dispatch-only candidates excluded. No Q rounding. No generation or threshold mutation.'}
    save(output/'summary.json',summary)
    save(output/'priority_recompute_candidates.json',{'interval':'1.04 < Q < 1.06','snapshot_utc':summary['snapshot_utc'],
              'note':'Existing old layouts are only existence-checked here; verify hash/legality before reusing. Not accepted until revalidated under a new versioned policy.',
              'members':priority})
    with (output/'scored_members.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(scored[0]));writer.writeheader();writer.writerows(scored)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
