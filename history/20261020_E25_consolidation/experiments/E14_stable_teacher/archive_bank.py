"""Preserve all layouts for 25000 distinct train cases; teacher selection is a view."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import time
import zipfile
import numpy as np
import torch
from . import selective_bank as bank
from . import parallel_bank as parallel
from .data import HERE,ROOT,VERSION,derived_case,save,sha

DEFAULT=HERE/'runs/teacher_archive25000_v1'
PLAN=ROOT/'protocol/teacher_archive25000_v1.json'


def identity():
    return {'version':'E14.archive25000.v1','base':parallel.identity(),
            'source_sha256':sha(Path(__file__)),'plan_sha256':sha(PLAN)}


def counts(db):
    return [db.execute('SELECT COUNT(*) FROM candidates WHERE status="archived" AND bin=?',(b,)).fetchone()[0] for b in range(6)]


def export(db,out,quality=1.06,manifests=True,status=None):
    plan=json.loads(PLAN.read_text());n=counts(db)
    chosen=[];eligible=[]
    for b,quota in enumerate(plan['teacher_strata_quotas']):
        eligible.append(db.execute('SELECT COUNT(*) FROM candidates WHERE status="archived" AND hard_legal=1 AND q<=? AND bin=?',(quality,b)).fetchone()[0])
        chosen.extend(json.loads(r[0]) for r in db.execute('SELECT record FROM candidates WHERE status="archived" AND hard_legal=1 AND q<=? AND bin=? ORDER BY q,id LIMIT ?',(quality,b,quota)))
    if manifests:
        save(out/'teacher.json',{'quality_max_inclusive':quality,'teacher_target':16000,'selected_count':len(chosen),'quotas':plan['teacher_strata_quotas'],'eligible_by_stratum':eligible,'records':chosen})
        tmp=out/'archive_manifest.tmp'
        with tmp.open('w',encoding='utf-8') as f:
            for (r,) in db.execute('SELECT record FROM candidates WHERE status="archived" ORDER BY id'):f.write(r+'\n')
        os.replace(tmp,out/'archive_manifest.jsonl')
    progress={'status':status or bank.meta(db,'status','prepared'),'distinct_layouts':sum(n),'distinct_target':25000,
              'new_attempts':db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],
              'historical_drops_remaining':db.execute('SELECT COUNT(*) FROM candidates WHERE priority=0 AND status="pending"').fetchone()[0],
              'elapsed_seconds':bank.meta(db,'archive_elapsed',0),'teacher_quality_max':quality,'teacher_selected':len(chosen),
              'teacher_eligible':sum(eligible),'teacher_shortfall_by_stratum':[max(0,q-v) for q,v in zip(plan['teacher_strata_quotas'],eligible)],
              'archive_strata_counts':n,'archive_strata_targets':plan['archive_strata_quotas'],'local_test_read':False,'pid':os.getpid()}
    save(out/'progress.json',progress)
    return progress


def prepare(out,source):
    if out.exists():raise FileExistsError(out)
    # SQLite online backup is a consistent snapshot even if a paused old process
    # retains its application mutex. Committed source labels are immutable.
    src=sqlite3.connect((source/'bank.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    src.row_factory=sqlite3.Row
    staging=out.with_name(out.name+'_preparing');staging.mkdir(parents=True,exist_ok=True)
    (staging/'layouts').mkdir(exist_ok=True)
    dst=bank.connect(staging)
    try:
        if bank.meta(src,'parallel_identity',None)!=parallel.identity():raise ValueError('Source solver/execution identity mismatch')
        src.backup(dst)
        dst.execute('ALTER TABLE candidates ADD COLUMN priority INTEGER DEFAULT 1')
        dst.execute('ALTER TABLE candidates ADD COLUMN q REAL')
        dst.execute('ALTER TABLE candidates ADD COLUMN hard_legal INTEGER DEFAULT 0')
        dst.execute('CREATE TABLE historical_records(id INTEGER PRIMARY KEY, status TEXT, record TEXT)')
        dst.execute('CREATE TABLE attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,candidate_id INTEGER,stage TEXT,file TEXT,record TEXT)')
        rows=list(dst.execute('SELECT * FROM candidates WHERE status!="pending"'))
        with dst:
            for c in rows:
                dst.execute('INSERT INTO historical_records VALUES (?,?,?)',(c['id'],c['status'],c['record']))
                r=json.loads(c['record'])
                if c['status']=='accepted':
                    p=source/r['label_file']
                    if sha(p)!=r['label_sha256']:raise ValueError('Source label hash mismatch')
                    name=f"layouts/{c['id']:06d}_import.npz";target=staging/name
                    if not target.exists() or sha(target)!=r['label_sha256']:shutil.copyfile(p,target)
                    if sha(target)!=r['label_sha256']:raise ValueError('Copied layout hash mismatch')
                    r.update(archive_file=name,archive_sha256=r['label_sha256'],label_file=name,
                             input_storage='Frozen source member plus input_areas/task_version in NPZ',archive_origin='imported_accepted')
                    dst.execute('UPDATE candidates SET status="archived",q=?,hard_legal=1,record=? WHERE id=?',(r['metrics']['cost'],json.dumps(r),c['id']))
                elif c['status']=='rejected':
                    dst.execute('UPDATE candidates SET status="pending",priority=0,record=NULL WHERE id=?',(c['id'],))
                elif c['status']!='incompatible':raise ValueError(f"Unresolved source status: {c['status']}")
            bank.put(dst,'archive_identity',identity());bank.put(dst,'archive_elapsed',0)
            bank.put(dst,'prior_attempt_counter',bank.meta(src,'solves',0))
            bank.put(dst,'status','prepared')
        save(staging/'config.json',{'identity':identity(),'source_bank':str(source),'source_config':json.loads((source/'config.json').read_text()),
              'target_distinct':25000,'default_quality_max':1.06,'retention':'all returned layouts; no Q gate on storage'})
        with zipfile.ZipFile(staging/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for relative in identity()['base']['base']['sources']:z.write(ROOT/relative,relative)
            for p in (Path(__file__),HERE/'parallel_bank.py',PLAN,bank.PLAN):z.write(p,p.relative_to(ROOT))
        progress=export(dst,staging,status='prepared');dst.close();src.close()
        # Windows indexers can briefly hold a just-written file in the directory.
        for attempt in range(30):
            try:staging.rename(out);break
            except PermissionError:
                if attempt==29:raise
                time.sleep(.2)
        print(json.dumps(progress,indent=2))
    finally:
        src.close()


def postprocess(inputs,draft):
    from .optimizer import optimize
    from .stable import polish
    inputs=(int(inputs[0]),*(torch.from_numpy(v) for v in inputs[1:]))
    start=time.perf_counter();x,stats=optimize(draft,inputs,seconds=5.)
    x,stable=polish(draft,x,inputs,stats,seconds=16.)
    return x,{'postprocess_seconds':time.perf_counter()-start,'worker_pid':os.getpid(),'stability':stable}


def commit(db,attempt,record):
    q=record.get('metrics',{}).get('cost');legal=record.get('hard_legal',False)
    if q is not None and not math.isfinite(float(q)):q=None
    with db:
        db.execute('UPDATE candidates SET status="archived",q=?,hard_legal=?,record=? WHERE id=?',(q,int(legal),json.dumps(record),attempt['candidate_id']))
        db.execute('UPDATE attempts SET stage="committed",record=? WHERE id=?',(json.dumps(record),attempt['id']))


def recover(db,out):
    for a in list(db.execute('SELECT * FROM attempts WHERE stage="dispatched"')):
        path=out/a['file']
        if path.exists():
            with np.load(path) as z:r=json.loads(str(z['record_json']))
            r.update(archive_file=a['file'],archive_sha256=sha(path),label_file=a['file'],label_sha256=sha(path))
            commit(db,a,r)
        else:
            with db:db.execute('UPDATE attempts SET stage="interrupted" WHERE id=?',(a['id'],))


def store(db,out,c,case,x,stats,inference,attempt):
    from scoring.evaluation import evaluate_case
    from .targets import encode,validate,SCHEMA
    from .frozen.packing import prepare_problem
    record={'member':[c['shard'],c['row']],'case_id':case.case_id,'n':c['n'],'stratum':c['bin'],'origin':c['origin'],
            'execution_version':'E14.archive25000.v1','attempt_id':attempt['id'],'worker_pid':stats['worker_pid'],
            'postprocess_seconds':stats['postprocess_seconds'],'inference_seconds_amortized':inference,
            'historical_drop_recomputed':c['priority']==0,'hard_legal':False}
    values={'positions':np.asarray(x),'block_count':np.asarray(case.block_count),'input_areas':np.asarray(case.solver_inputs[1]),
            **{k:np.asarray(v) for k,v in zip(('b2b','p2b','pins','constraints','specs'),case.solver_inputs[2:])},
            'task_version':np.asarray(VERSION),'schema':np.asarray('E14.full_layout_archive.v1')}
    # Neither Q nor legality can veto the durable coordinate/input archive.
    try:
        result=evaluate_case(case,x,stats['postprocess_seconds']+inference);result.pop('positions')
        record.update(result);record['hard_legal']=bool(result['metrics']['is_feasible'] and result['independent_hard_checks']['is_feasible'])
        if record['hard_legal']:
            target=encode(x,prepare_problem(*case.solver_inputs));validate(target,prepare_problem(*case.solver_inputs))
            values.update(target);values['schema']=np.asarray(SCHEMA)
    except Exception as exc:record['evaluation_or_encoding_error']=repr(exc);record['hard_legal']=False
    values['record_json']=np.asarray(json.dumps(record))
    path=out/attempt['file'];tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    record.update(archive_file=attempt['file'],archive_sha256=sha(path),label_file=attempt['file'],label_sha256=sha(path))
    commit(db,attempt,record)
    return record


def generate(args):
    from .pipeline import Baseline
    out=args.output;lock=parallel.lock_bank(out);db=bank.connect(out)
    pool=None;queue=deque();current=None;status='running';start=time.perf_counter();elapsed=bank.meta(db,'archive_elapsed',0)
    try:
        if bank.meta(db,'archive_identity',None)!=identity():raise ValueError('Archive source/plan identity mismatch')
        execution={'workers':4,'draft_device':'cuda','batch_size':8,'max_inflight':8}
        recover(db,out)
        for (payload,) in db.execute('SELECT record FROM candidates WHERE status="archived"'):
            r=json.loads(payload)
            if sha(out/r['archive_file'])!=r['archive_sha256']:raise ValueError('Archived layout hash mismatch')
        if (out/'STOP').exists():status='budget_or_stop';return
        if not torch.cuda.is_available():raise ValueError('CUDA required')
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        solver=Baseline('cuda');pool=ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=parallel.initialize_worker)
        list(pool.map(parallel.worker_ready,range(4)))
        targets=json.loads(PLAN.read_text())['archive_strata_quotas'];submitted=0;completed=0
        with (out/'sessions.jsonl').open('a') as f:f.write(json.dumps({'started_unix':time.time(),'command':__import__('sys').argv,'execution':execution,'max_seconds':args.max_seconds,'max_attempts':args.max_attempts})+'\n')
        while True:
            n=counts(db);attempted=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
            stop=(out/'STOP').exists() or elapsed+time.perf_counter()-start>=args.max_seconds or attempted>=args.max_attempts
            if args.smoke_count is not None and submitted>=args.smoke_count:stop=True
            selected=[];busy={item[0]['id'] for item in queue};reserved=[sum(item[0]['bin']==b for item in queue) for b in range(6)]
            slots=min(8-len(queue),args.max_attempts-attempted)
            # Refill in useful GPU microbatches while CPU workers consume the previous batch.
            if len(queue)>4:slots=0
            if args.smoke_count is not None:slots=min(slots,args.smoke_count-submitted)
            while not stop and len(selected)<slots:
                suffix=' AND id NOT IN ('+','.join('?' for _ in busy)+')' if busy else ''
                row=db.execute('SELECT * FROM candidates WHERE status="pending" AND priority=0'+suffix+' ORDER BY id LIMIT 1',tuple(sorted(busy))).fetchone()
                if row is None:
                    # Do not dispatch unseen members until every historical DROP has committed.
                    if any(item[0]['priority']==0 for item in queue) or any(c['priority']==0 for c,_ in selected):break
                    bins=[b for b in range(6) if n[b]+reserved[b]<targets[b]]
                    if not bins:break
                    b=min(bins,key=lambda b:((n[b]+reserved[b])/targets[b],b))
                    row=db.execute('SELECT * FROM candidates WHERE status="pending" AND bin=?'+suffix+' ORDER BY id LIMIT 1',(b,*sorted(busy))).fetchone()
                    if row is None:
                        if bank.expand(db):continue
                        status='pool_exhausted';break
                c=dict(row);current=[c['shard'],c['row']]
                try:case,_=derived_case(parallel.shard_reader(c['shard']).case(c['row']))
                except ValueError as exc:
                    if str(exc) not in ('locked_dimensions_disagree','mean_area_incompatible_with_locked_shape'):raise
                    if c['priority']==0:raise ValueError('Historical DROP became incompatible; investigate input identity')
                    with db:db.execute('UPDATE candidates SET status="incompatible",record=? WHERE id=?',(json.dumps({'member':current,'reason':str(exc)}),c['id']))
                    continue
                selected.append((c,case));busy.add(c['id']);reserved[c['bin']]+=1
            if selected:
                drafts,inference=parallel.drafts(solver,[case for _,case in selected],'cuda')
                for (c,case),draft in zip(selected,drafts):
                    with db:
                        cur=db.execute('INSERT INTO attempts(candidate_id,stage) VALUES (?,"dispatched")',(c['id'],));aid=cur.lastrowid
                        filename=f"layouts/{c['id']:06d}_{aid:07d}.npz"
                        db.execute('UPDATE attempts SET file=? WHERE id=?',(filename,aid))
                    attempt={'id':aid,'candidate_id':c['id'],'file':filename}
                    payload=(case.block_count,*(np.asarray(v).copy() for v in case.solver_inputs[1:]))
                    queue.append((c,case,pool.submit(postprocess,payload,draft),inference,attempt));submitted+=1
            if not queue:
                if sum(n)==25000 and db.execute('SELECT COUNT(*) FROM candidates WHERE priority=0 AND status="pending"').fetchone()[0]==0:status='completed'
                elif status=='pool_exhausted':pass
                elif args.smoke_count is not None and submitted>=args.smoke_count:status='smoke_stopped'
                elif stop:status='budget_or_stop'
                else:raise RuntimeError('Cannot fill archive quotas')
                break
            c,case,future,inference,attempt=queue[0];current=[c['shard'],c['row']]
            x,stats=future.result();r=store(db,out,c,case,x,stats,inference,attempt);queue.popleft();completed+=1
            with db:bank.put(db,'archive_elapsed',elapsed+time.perf_counter()-start)
            q=r.get('metrics',{}).get('cost');print(f"archived={sum(n)+1}/25000 Q={q} SAVED {current}",flush=True)
            if completed%25==0:export(db,out,manifests=completed%250==0,status='running')
    except KeyboardInterrupt:status='interrupted'
    except Exception as exc:
        status='failed';save(out/'last_error.json',{'error':repr(exc),'member':current,'uncommitted_remain_pending':True});raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)
            # Even on Ctrl+C or another worker's failure, preserve every completed
            # in-flight result. Unstarted/cancelled attempts remain retryable.
            for c,case,future,inference,attempt in queue:
                if future.cancelled():continue
                try:
                    x,stats=future.result();store(db,out,c,case,x,stats,inference,attempt)
                except Exception as exc:
                    status='failed'
                    save(out/f"attempt_error_{attempt['id']:07d}.json",{'member':[c['shard'],c['row']],'error':repr(exc)})
        with db:bank.put(db,'archive_elapsed',elapsed+time.perf_counter()-start);bank.put(db,'status',status)
        print(json.dumps(export(db,out,status=status),indent=2));db.close();lock.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','generate','select','status'])
    ap.add_argument('--output',type=Path,default=DEFAULT)
    ap.add_argument('--source',type=Path,default=parallel.DEFAULT)
    ap.add_argument('--quality-max',type=float,default=1.06)
    ap.add_argument('--max-seconds',type=float,default=172800)
    ap.add_argument('--max-attempts',type=int,default=60000)
    ap.add_argument('--smoke-count',type=int,choices=range(1,6))
    args=ap.parse_args()
    if args.max_seconds<=0 or args.max_attempts<1 or not math.isfinite(args.quality_max):raise ValueError('Invalid budget/threshold')
    if args.action=='prepare':prepare(args.output,args.source)
    elif args.action=='status':print((args.output/'progress.json').read_text())
    elif args.action=='select':
        lock=parallel.lock_bank(args.output);db=bank.connect(args.output)
        try:print(json.dumps(export(db,args.output,args.quality_max),indent=2))
        finally:db.close();lock.close()
    else:
        if not args.output.exists():prepare(args.output,args.source)
        generate(args)


if __name__=='__main__':main()
