"""Q-gated, module-stratified teacher bank; SQLite is the resume authority."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np
import torch
from datasets.paths import dataset_paths
from .data import HERE, ROOT, VERSION, reader, derived_case, save, sha

PLAN = ROOT / 'protocol/teacher_selection_q104_v1.json'
DEFAULT = HERE / 'runs/teacher_q104_stratified_v1'
OLD = HERE / 'runs/teacher_train16000_v1'
BINS = [(1,40),(41,60),(61,80),(81,100),(101,110),(111,120)]


def stratum(n):
    for i,(lo,hi) in enumerate(BINS):
        if lo <= n <= hi:return i
    raise ValueError(f'Unregistered module count {n}')


def eligible(metrics, checks):
    q = float(metrics['cost'])
    return bool(metrics['is_feasible'] and checks['is_feasible'] and math.isfinite(q) and q <= 1.04)


def connect(out):
    db = sqlite3.connect(out/'bank.sqlite3')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS candidates (id INTEGER PRIMARY KEY, shard TEXT, row INTEGER, n INTEGER, bin INTEGER, origin TEXT, status TEXT DEFAULT "pending", record TEXT, UNIQUE(shard,row))')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS candidate_lookup ON candidates(bin,status,id)')
    return db


def meta(db,key,default):
    r=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
    return json.loads(r[0]) if r else default


def put(db,key,value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value)))


def fingerprint():
    files = [Path(__file__), HERE/'pipeline.py', HERE/'stable.py', HERE/'targets.py', HERE/'data.py']
    files += [HERE/f'{name}.py' for name in ('optimizer','robust','conic_geometry','guard','rescue','selection','rebuild','reduction','certificate')]
    files += sorted((HERE/'frozen').rglob('*.py'))
    files += [ROOT/p for p in ('datasets/worksets.py','datasets/cases.py','datasets/paths.py','scoring/evaluation.py','scoring/checks.py','scoring/official.py','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py','protocol/split_v1.json','protocol/derived_workset_v3.json')]
    from .pipeline import CHECKPOINT_SHA256
    return {'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
            'checkpoint_sha256':CHECKPOINT_SHA256,'plan_sha256':sha(PLAN)}


def export(db,out,status):
    quotas=json.loads(PLAN.read_text())['quotas']
    accepted=[json.loads(r[0]) for r in db.execute('SELECT record FROM candidates WHERE status="accepted" ORDER BY id')]
    counts=[db.execute('SELECT COUNT(*) FROM candidates WHERE status="accepted" AND bin=?',(i,)).fetchone()[0] for i in range(6)]
    progress={'status':status,'accepted':len(accepted),'target':sum(quotas),'elapsed_seconds':meta(db,'elapsed',0),
              'new_solves':meta(db,'solves',0),'processed':db.execute('SELECT COUNT(*) FROM candidates WHERE status!="pending"').fetchone()[0],
              'strata':[{'min_n':lo,'max_n':hi,'quota':quotas[i],'accepted':counts[i],'remaining':quotas[i]-counts[i]} for i,(lo,hi) in enumerate(BINS)],
              'local_test_read':False,'pid':os.getpid()}
    save(out/'accepted.json',{'task_version':VERSION,'quality_max':1.04,'records':accepted})
    save(out/'progress.json',progress)
    return progress


def prepare(out):
    if out.exists():raise FileExistsError(f'Bank already exists: {out}; use generate/status')
    parent=json.loads((ROOT/'protocol/derived_workset_v3.json').read_text())['sets']['train']['members']
    allowed=set(json.loads((ROOT/'protocol/split_v1.json').read_text())['shards']['train'])
    grouped={}
    for shard,row in parent:
        assert shard in allowed
        grouped.setdefault(shard,[]).append(row)
    sizes={};root=dataset_paths()['training_root']
    for shard,rows in grouped.items():
        raw=torch.load(root/shard,map_location='cpu',weights_only=False)
        for row in rows:sizes[(shard,row)]=len(raw[0][row])
    quotas=[sum(stratum(n)==i for n in sizes.values()) for i in range(6)]
    expected=json.loads(PLAN.read_text())['quotas']
    if quotas!=expected:raise ValueError('Frozen parent quota mismatch')
    out.mkdir(parents=True);(out/'labels').mkdir();db=connect(out)
    with db:
        for shard,row in parent:
            n=sizes[(shard,row)]
            db.execute('INSERT INTO candidates(shard,row,n,bin,origin) VALUES (?,?,?,?,?)',(shard,row,n,stratum(n),'v3_parent'))
        for line in (OLD/'records.jsonl').read_text().splitlines():
            rec=json.loads(line);shard,row=rec['member']
            found=db.execute('SELECT * FROM candidates WHERE shard=? AND row=?',(shard,row)).fetchone()
            if found is None or found['status']!='pending':raise ValueError('Unknown/duplicate old record')
            m=rec.get('metrics',rec.get('rejected_metrics'))
            if m is None:raise ValueError('Unresolved old error')
            keep=eligible(m,rec.get('independent_hard_checks',{'is_feasible':False}))
            record=dict(rec,n=found['n'],stratum=found['bin'],origin='imported',source_record=rec['index'])
            if keep:
                src=OLD/rec['label_file']
                if sha(src)!=rec['label_sha256']:raise ValueError('Old label checksum mismatch')
                with np.load(src) as z:
                    assert str(z['task_version'])==VERSION and len(z['input_areas'])==found['n']
                name=f"labels/{found['id']:06d}.npz"
                # Independent copy: future writes cannot mutate the original evidence.
                shutil.copyfile(src,out/name)
                if sha(out/name)!=rec['label_sha256']:raise ValueError('Copy checksum mismatch')
                record['label_file']=name
            else:
                record.pop('label_file',None);record.pop('label_sha256',None)
                record['rejection_reason']='Q>1.04 or hard legality'
            db.execute('UPDATE candidates SET status=?,record=? WHERE id=?',('accepted' if keep else 'rejected',json.dumps(record),found['id']))
        put(db,'fingerprint',fingerprint());put(db,'elapsed',0);put(db,'solves',0)
    save(out/'config.json',{'identity':fingerprint(),'source_bank':str(OLD),'historical_attempts':3376,'seed':0,'threads':2})
    print(json.dumps(export(db,out,'prepared'),indent=2));db.close()


def expand(db):
    """Register a whole deterministic train-only shard atomically before advancing."""
    shards=json.loads((ROOT/'protocol/split_v1.json').read_text())['shards']['train']
    rank=lambda s:hashlib.sha256(('E14.q104.v1:'+s).encode()).hexdigest()
    workers=[sorted((s for s in shards if s.startswith(f'worker_{w}/')),key=rank) for w in range(100)]
    order=[workers[w][j] for j in range(72) for w in range(100)]
    cursor=meta(db,'scan_cursor',0)
    if cursor==len(order):return False
    shard=order[cursor];raw=torch.load(dataset_paths()['training_root']/shard,map_location='cpu',weights_only=False)
    rows=sorted(range(112),key=lambda r:rank(f'{shard}:{r}'))
    with db:
        for row in rows:
            n=len(raw[0][row])
            db.execute('INSERT OR IGNORE INTO candidates(shard,row,n,bin,origin) VALUES (?,?,?,?,?)',(shard,row,n,stratum(n),'expanded_train'))
        put(db,'scan_cursor',cursor+1)
    return True


def generate(out,max_seconds,max_solves,smoke_count):
    from .pipeline import Baseline
    from .targets import SCHEMA,encode,validate
    from .frozen.packing import prepare_problem
    from .frozen.solver import hard_valid
    from scoring.evaluation import evaluate_case
    if not (out/'config.json').exists():raise ValueError('Run prepare first')
    import msvcrt
    lock=(out/'generation.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock.close();raise RuntimeError('Another generator holds this bank')
    db=connect(out)
    if meta(db,'fingerprint',None)!=fingerprint():raise ValueError('Solver/data/plan/source identity changed; explicit new version required')
    # Verify all retained files once per launch, including after interruption.
    for (payload,) in db.execute('SELECT record FROM candidates WHERE status="accepted"'):
        r=json.loads(payload)
        if sha(out/r['label_file'])!=r['label_sha256']:raise ValueError('Accepted label integrity failure')
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    elapsed=meta(db,'elapsed',0);solves=meta(db,'solves',0);start=time.perf_counter();solver=None;steps=0;status='running'
    quotas=json.loads(PLAN.read_text())['quotas']
    try:
        while True:
            counts=[db.execute('SELECT COUNT(*) FROM candidates WHERE status="accepted" AND bin=?',(i,)).fetchone()[0] for i in range(6)]
            if counts==quotas:status='completed';break
            if (out/'STOP').exists() or elapsed+time.perf_counter()-start>=max_seconds or solves>=max_solves:
                status='budget_or_stop';break
            if smoke_count is not None and steps>=smoke_count:status='smoke_stopped';break
            b=min((i for i in range(6) if counts[i]<quotas[i]),key=lambda i:(counts[i]/quotas[i],i))
            c=db.execute('SELECT * FROM candidates WHERE bin=? AND status="pending" ORDER BY id LIMIT 1',(b,)).fetchone()
            if c is None:
                if not expand(db):status='pool_exhausted';break
                continue
            member=[c['shard'],c['row']];raw=reader([member],'train').case(0)
            try:case,_=derived_case(raw)
            except ValueError as exc:
                if str(exc) not in ('locked_dimensions_disagree','mean_area_incompatible_with_locked_shape'):raise
                with db:db.execute('UPDATE candidates SET status="incompatible",record=? WHERE id=?',(json.dumps({'member':member,'reason':str(exc)}),c['id']))
                continue
            if solver is None:solver=Baseline('cpu')
            p=prepare_problem(*case.solver_inputs);x,stats=solver.solve(case.solver_inputs);solves+=1;steps+=1
            if not hard_valid(x,p):raise ValueError(f'Hard validity failure: {member}')
            result=evaluate_case(case,x,stats['end_to_end_seconds']);result.pop('positions')
            if not result['metrics']['is_feasible'] or not result['independent_hard_checks']['is_feasible']:
                raise ValueError(f'Independent hard validity failure: {member}')
            keep=eligible(result['metrics'],result['independent_hard_checks'])
            rec={'member':member,'case_id':case.case_id,'n':c['n'],'stratum':b,'origin':c['origin'],**result,'solver_seconds':stats['end_to_end_seconds']}
            if keep:
                target=encode(x,p);validate(target,p)
                target.update(input_areas=np.asarray(case.solver_inputs[1]).copy(),task_version=np.asarray(VERSION),schema=np.asarray(SCHEMA))
                name=f"labels/{c['id']:06d}.npz";path=out/name;tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.savez_compressed(f,**target);f.flush();os.fsync(f.fileno())
                os.replace(tmp,path);rec.update(label_file=name,label_sha256=sha(path))
            else:rec['rejection_reason']='Q>1.04'
            with db:
                db.execute('UPDATE candidates SET status=?,record=? WHERE id=?',('accepted' if keep else 'rejected',json.dumps(rec),c['id']))
                put(db,'solves',solves);put(db,'elapsed',elapsed+time.perf_counter()-start)
            if steps%25==0:export(db,out,'running')
            print(f"accepted={sum(counts)+int(keep)}/16000 Q={rec['metrics']['cost']:.6f} {'KEEP' if keep else 'DROP'} {member}",flush=True)
    except KeyboardInterrupt:status='interrupted'
    except Exception as exc:
        status='failed';save(out/'last_error.json',{'error':repr(exc),'member':locals().get('member'),'retry_pending_on_resume':True});raise
    finally:
        with db:put(db,'elapsed',elapsed+time.perf_counter()-start);put(db,'solves',solves)
        print(json.dumps(export(db,out,status),indent=2));db.close();lock.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','generate','status'])
    ap.add_argument('--output',type=Path,default=DEFAULT)
    ap.add_argument('--max-seconds',type=float,default=172800,help='Cumulative new-generation wall budget')
    ap.add_argument('--max-solves',type=int,default=60000,help='Cumulative new solve budget')
    ap.add_argument('--smoke-count',type=int,choices=range(1,6))
    args=ap.parse_args()
    if args.max_seconds<=0 or args.max_solves<=0:raise ValueError('Positive budgets required')
    if args.action=='prepare':prepare(args.output)
    elif args.action=='generate':generate(args.output,args.max_seconds,args.max_solves,args.smoke_count)
    else:
        if not (args.output/'bank.sqlite3').exists():raise FileNotFoundError(args.output)
        print((args.output/'progress.json').read_text())


if __name__=='__main__':main()
