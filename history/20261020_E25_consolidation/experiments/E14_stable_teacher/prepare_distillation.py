"""Freeze best-Q 20000 grouped student splits and worst-Q 1000 stress cases."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix
from .data import HERE,ROOT,VERSION,save,sha

VERSION_DATA='E14.distillation_top20000.v1'
OUT=HERE/'runs/distillation_top20000_v1'
REGISTRY=ROOT/'protocol/distillation_workset_v4.json'
SOURCE=HERE/'runs/teacher_archive25000_v1'
SEED=20261019
FROZEN=('protocol/split_v1.json','protocol/workset_v2.json','protocol/derived_workset_v3.json')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def rank(tag,value):
    return hashlib.sha256(f'{VERSION_DATA}:{SEED}:{tag}:{value}'.encode()).hexdigest()


def bucket(n):
    for b,hi in enumerate((40,60,80,100,110,120)):
        if n<=hi:return b
    raise ValueError(n)


def qbucket(q):
    for b,hi in enumerate((1.,1.02,1.04,1.06,float('inf'))):
        if q<=hi:return b


def group_split(rows):
    """Exact 2000/2000 holdouts, disjoint shards, balanced size/Q/worker marginals."""
    groups=collections.defaultdict(list)
    for r in rows:groups[r['member'][0]].append(r)
    names=sorted(groups,key=lambda s:(rank('group',s),s));g=len(names)
    features=np.zeros((111,g))
    for j,name in enumerate(names):
        for r in groups[name]:
            features[bucket(r['n']),j]+=1
            features[6+qbucket(r['Q']),j]+=1
            features[11+int(name.split('/')[0].split('_')[1]),j]+=1
    sizes=features[:6].sum(0);target=features.sum(1)*.1
    # x_dev[g],x_test[g], absolute marginal deviations for both sets.
    k=len(target);nv=2*g+2*k
    objective=np.zeros(nv)
    weights=np.r_[np.full(11,1.),np.full(100,.15)]/np.maximum(target,1.)
    objective[2*g:2*g+k]=weights;objective[2*g+k:]=weights
    for j,name in enumerate(names):
        objective[j]=1e-8*int(rank('dev-tie',name)[:8],16)/2**32
        objective[g+j]=1e-8*int(rank('test-tie',name)[:8],16)/2**32
    matrix=lil_matrix((g+2+4*k+100,nv));lo=[];hi=[];i=0
    for j in range(g):matrix[i,j]=1;matrix[i,g+j]=1;lo.append(-np.inf);hi.append(1);i+=1
    for split in range(2):
        matrix[i,split*g:(split+1)*g]=sizes;lo.append(2000);hi.append(2000);i+=1
    for split in range(2):
        for f in range(k):
            slack=2*g+split*k+f
            matrix[i,split*g:(split+1)*g]=features[f];matrix[i,slack]=-1;lo.append(-np.inf);hi.append(target[f]);i+=1
            matrix[i,split*g:(split+1)*g]=-features[f];matrix[i,slack]=-1;lo.append(-np.inf);hi.append(-target[f]);i+=1
    for worker in range(100):
        js=[j for j,s in enumerate(names) if s.startswith(f'worker_{worker}/')]
        for j in js:matrix[i,j]=1;matrix[i,g+j]=1
        lo.append(-np.inf);hi.append(max(0,len(js)-1));i+=1
    result=milp(objective,integrality=np.r_[np.ones(2*g),np.zeros(2*k)],
                bounds=Bounds(np.zeros(nv),np.r_[np.ones(2*g),np.full(2*k,np.inf)]),
                constraints=LinearConstraint(matrix.tocsr(),np.asarray(lo),np.asarray(hi)),
                options={'time_limit':45.,'mip_rel_gap':.02})
    if result.x is None:raise RuntimeError(f'Grouped allocation infeasible: {result.message}')
    x=np.rint(result.x[:2*g]).astype(int);assignment={}
    for j,name in enumerate(names):assignment[name]='dev' if x[j] else ('test' if x[g+j] else 'train')
    splits={s:[r for r in rows if assignment[r['member'][0]]==s] for s in ('train','dev','test')}
    if [len(splits[s]) for s in ('train','dev','test')]!=[16000,2000,2000]:raise ValueError('Exact grouped counts failed')
    return splits,assignment,{'solver':'scipy.optimize.milp/HiGHS','time_limit_seconds':45,'status':int(result.status),
                             'message':result.message,'objective':float(result.fun),'mip_gap':float(result.mip_gap),
                             'features':'6 module bins + 5 teacher-Q bins + 100 worker counts; at least one training shard per represented worker',
                             'note':'Membership frozen from this feasible solution; re-use manifests rather than re-optimize.'}


def monitor(rows,count,tag):
    grouped=[[r for r in rows if bucket(r['n'])==b] for b in range(6)]
    exact=np.asarray([len(g) for g in grouped])*count/len(rows);quota=np.floor(exact).astype(int)
    for b in sorted(range(6),key=lambda b:(-(exact[b]-quota[b]),b))[:count-int(quota.sum())]:quota[b]+=1
    out=[]
    for b,g in enumerate(grouped):out.extend(sorted(g,key=lambda r:rank(tag,r['member']))[:quota[b]])
    assert len(out)==count
    return out


def statistics(rows):
    q=np.asarray([r['Q'] for r in rows]);ns=np.asarray([r['n'] for r in rows])
    return {'count':len(rows),'shards':len({r['member'][0] for r in rows}),
            'workers':len({r['member'][0].split('/')[0] for r in rows}),
            'Q_mean':float(q.mean()),'Q_min':float(q.min()),'Q_max':float(q.max()),
            'Q_median':float(np.median(q)),'module_min':int(ns.min()),'module_max':int(ns.max()),
            'module_bins':np.bincount([bucket(int(n)) for n in ns],minlength=6).tolist()}


def main():
    if REGISTRY.exists() or OUT.exists():raise FileExistsError('Frozen distillation set already exists; do not repartition')
    before={p:sha(ROOT/p) for p in FROZEN}
    db=sqlite3.connect((SOURCE/'bank.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    status=json.loads((SOURCE/'progress.json').read_text())
    if status['status']!='completed' or status['distinct_layouts']!=25000:raise ValueError('Archive not completed')
    rows=[]
    for c in db.execute('SELECT id,shard,row,n,q,hard_legal,record FROM candidates WHERE status="archived"'):
        r=json.loads(c['record'])
        if not c['hard_legal'] or not math.isfinite(c['q']):raise ValueError('Archive contains unresolved invalid teacher')
        rows.append({'candidate_id':c['id'],'member':[c['shard'],c['row']],'case_id':r['case_id'],'n':c['n'],'Q':c['q'],
                     'archive_file':(SOURCE/r['archive_file']).relative_to(ROOT).as_posix(),'archive_sha256':r['archive_sha256'],
                     'task_version':VERSION})
    db.close()
    if len(rows)!=25000 or len({tuple(r['member']) for r in rows})!=25000:raise ValueError('Archive uniqueness mismatch')
    rows.sort(key=lambda r:(r['Q'],rank('member-tie',r['member']),r['member']))
    for i,r in enumerate(rows):r['quality_rank_ascending']=i+1
    top=rows[:20000];tail=sorted(rows,key=lambda r:(-r['Q'],rank('member-tie',r['member']),r['member']))[:1000]
    assert not {tuple(r['member']) for r in top}&{tuple(r['member']) for r in tail}
    splits,assignment,allocation=group_split(top)
    sets={**splits,'tail_test':tail,'train_monitor500':monitor(splits['train'],500,'train-monitor'),
          'dev_monitor500':monitor(splits['dev'],500,'dev-monitor')}
    member_sets={s:{tuple(r['member']) for r in v} for s,v in sets.items()}
    shard_sets={s:{r['member'][0] for r in v} for s,v in sets.items()}
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        assert not member_sets[a]&member_sets[b] and not shard_sets[a]&shard_sets[b]
    for s in ('train','dev','test'):assert not member_sets[s]&member_sets['tail_test']
    source_members={tuple(r['member']) for r in rows};allowed=set(json.loads((ROOT/'protocol/split_v1.json').read_text())['shards']['train'])
    assert all(r['member'][0] in allowed and tuple(r['member']) in source_members for r in rows)
    for r in top+tail:
        if not (ROOT/r['archive_file']).is_file():raise FileNotFoundError(r['archive_file'])
    OUT.mkdir(parents=True)
    manifests={}
    for name,records in sets.items():
        payload={'version':VERSION_DATA,'role':name,'source_partition':'train','source_archive':SOURCE.relative_to(ROOT).as_posix(),
                 'task_version':VERSION,'count':len(records),'sealed_for_student_evaluation':name in ('test','tail_test'),
                 'records':records}
        save(OUT/f'{name}.json',payload)
        manifests[name]={'path':(OUT/f'{name}.json').relative_to(ROOT).as_posix(),'sha256':sha(OUT/f'{name}.json'),**statistics(records)}
    save(OUT/'group_assignment.json',assignment)
    source_digest=digest([(r['member'],r['Q'],r['archive_sha256']) for r in rows])
    overlap={s:len(shard_sets[s]&shard_sets['tail_test']) for s in ('train','dev','test')}
    audit={'version':VERSION_DATA,'status':'prepared','selection':'globally best 20000 by unrounded Q, deterministic hash tie-break',
           'top20000':statistics(top),'tail1000':statistics(tail),'sets':manifests,'allocation':allocation,
           'source_archive_records_sha256':source_digest,'primary_member_and_shard_disjoint':True,
           'tail_member_disjoint_from_all_primary_sets':True,'tail_shared_shards_by_primary_role':overlap,
           'tail_shared_shards_with_top20000':len(shard_sets['tail_test']&set(assignment)),
           'tail_scope':'Preselected teacher-quality tail stress test; shares source shards with primary splits, not independent generalization test',
           'teacher_exposure':'All candidates from original split_v1 train; pretrained teacher may have seen them. Holdout separation is for the new student.',
           'original_local_test_read':False,'training_started':False,'frozen_original_hashes':before,
           'file_integrity_basis':'Archive full 25000-file SHA/schema audit already passed; per-file SHA rechecked lazily by loader; preparation checks member/file coverage without repeat full hashing'}
    save(OUT/'summary.json',audit)
    registry={'version':4,'dataset_version':VERSION_DATA,'seed':SEED,'task_version':VERSION,
              'supersedes_for_next_distillation_only':'derived_workset_v3 train/dev/test roles; original membership files remain unchanged',
              'student_split_counts':{'train':16000,'dev':2000,'test':2000,'tail_test':1000},
              'source_partition':'original split_v1 train only','source_archive':SOURCE.relative_to(ROOT).as_posix(),
              'source_archive_records_sha256':source_digest,'sets':manifests,
              'summary':(OUT/'summary.json').relative_to(ROOT).as_posix(),'summary_sha256':sha(OUT/'summary.json'),
              'allocation_sha256':sha(OUT/'group_assignment.json'),'frozen_original_hashes':before,
              'test_use':'Only after student model/checkpoint/solver frozen; never for loss, gradients, checkpoint choice, early stopping or tuning',
              'tail_scope':audit['tail_scope'],'old_local_test_stays_sealed':True}
    assert before=={p:sha(ROOT/p) for p in FROZEN}
    save(REGISTRY,registry)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
