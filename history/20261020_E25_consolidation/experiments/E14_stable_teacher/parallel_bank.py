"""Bounded draft producer, persistent CPU process pool, single ordered committer."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import torch
from . import selective_bank as bank
from .data import HERE, ROOT, VERSION, reader, derived_case, save, sha

DEFAULT=HERE/'runs/teacher_q104_parallel_v1'


def lock_bank(out):
    import msvcrt
    handle=(out/'generation.lock').open('a+b')
    if (out/'generation.lock').stat().st_size==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        handle.close();raise RuntimeError(f'Generator already active: {out}')
    return handle


def identity():
    return {'execution_version':'E14.parallel_teacher.v1','base':bank.fingerprint(),
            'parallel_source_sha256':sha(Path(__file__))}


def prepare(out,source):
    if out.exists():raise FileExistsError(out)
    source_lock=lock_bank(source)
    try:
        src=bank.connect(source)
        if bank.meta(src,'fingerprint',None)!=bank.fingerprint():raise ValueError('Source bank identity mismatch')
        out.mkdir(parents=True);(out/'labels').mkdir()
        dst=bank.connect(out);src.backup(dst)
        for (payload,) in dst.execute('SELECT record FROM candidates WHERE status="accepted"'):
            r=json.loads(payload);p=source/r['label_file']
            if sha(p)!=r['label_sha256']:raise ValueError('Source label integrity failure')
            shutil.copyfile(p,out/r['label_file'])
            if sha(out/r['label_file'])!=r['label_sha256']:raise ValueError('Copied label integrity failure')
        with dst:
            bank.put(dst,'parallel_identity',identity())
            bank.put(dst,'migrated_from',str(source))
        save(out/'config.json',{'identity':identity(),'source_bank':str(source),'source_config':json.loads((source/'config.json').read_text()),
                              'quality_max':1.04,'seed':0,'cpu_worker_torch_threads':1,'clarabel_max_threads':2})
        with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for relative in identity()['base']['sources']:
                archive.write(ROOT/relative,relative)
            archive.write(Path(__file__),Path(__file__).relative_to(ROOT))
            archive.write(bank.PLAN,bank.PLAN.relative_to(ROOT))
        print(json.dumps(bank.export(dst,out,'prepared'),indent=2));dst.close();src.close()
    finally:source_lock.close()


@lru_cache(maxsize=8)
def shard_reader(shard):
    return reader([(shard,r) for r in range(112)],'train')


def initialize_worker():
    # BLAS/OpenMP environment limits were set before spawn/import; no new dependency.
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(0);np.random.seed(0)


def worker_ready(_):
    from .optimizer import optimize
    from .stable import polish
    return os.getpid()


def postprocess(inputs,draft):
    from .optimizer import optimize
    from .stable import polish
    from .frozen.packing import prepare_problem
    from .frozen.solver import hard_valid
    # Official scoring inside polish expects Torch's numel/item API.
    # Reconstruct private CPU tensors only after NumPy IPC has completed.
    inputs=(int(inputs[0]),*(torch.from_numpy(v) for v in inputs[1:]))
    start=time.perf_counter()
    x,stats=optimize(draft,inputs,seconds=5.)
    x,stable=polish(draft,x,inputs,stats,seconds=16.)
    if not hard_valid(x,prepare_problem(*inputs)):raise ValueError('Worker hard validity failure')
    return x,{'postprocess_seconds':time.perf_counter()-start,'worker_pid':os.getpid(),'stability':stable,
              'rescue_used':stats.get('rescue_used',False)}


def drafts(solver,cases,device):
    from .frozen.inference import prepare_inference
    from .frozen.batches import decode
    start=time.perf_counter()
    if device=='cpu':return [solver.draft(c.solver_inputs) for c in cases],(time.perf_counter()-start)/len(cases)
    singles=[prepare_inference(c.solver_inputs,'cpu') for c in cases]
    n=max(c.block_count for c in cases);public={}
    for key,value in singles[0].items():
        if key=='scale':public[key]=torch.cat([s[key] for s in singles]);continue
        pair=key in ('adjacency','mib','group')
        public[key]=torch.zeros((len(cases),n,n) if pair else (len(cases),n,*value.shape[2:]),dtype=value.dtype)
        for i,c in enumerate(cases):
            ni=c.block_count
            if pair:public[key][i,:ni,:ni]=singles[i][key][0]
            else:public[key][i,:ni]=singles[i][key][0]
    public={k:v.to(device) for k,v in public.items()}
    with torch.inference_mode():
        x=(decode(solver.model(public),public)*public['scale'][:,None,None]).cpu().numpy().astype(float)
    out=[]
    for i,c in enumerate(cases):
        v=x[i,:c.block_count].copy();constraints=np.asarray(c.solver_inputs[5]);specs=np.asarray(c.solver_inputs[6])
        locked=constraints[:,:2].any(1);v[locked,2:]=specs[locked,2:];v[constraints[:,1]!=0]=specs[constraints[:,1]!=0]
        if not np.isfinite(v).all():raise ValueError('Nonfinite draft')
        out.append(v)
    return out,(time.perf_counter()-start)/len(cases)


def generate(args):
    from .pipeline import Baseline
    from .targets import SCHEMA,encode,validate
    from .frozen.packing import prepare_problem
    from scoring.evaluation import evaluate_case
    out=args.output;handle=lock_bank(out);db=bank.connect(out)
    try:
        if bank.meta(db,'parallel_identity',None)!=identity():raise ValueError('Parallel/base source identity changed; use a new version')
        execution={'workers':args.workers,'draft_device':args.draft_device,'batch_size':args.batch_size,'max_inflight':args.workers*2}
        old=bank.meta(db,'execution',None)
        if old is not None and old!=execution:raise ValueError('Resume execution mismatch; preserve workers/device/batch size')
        for (payload,) in db.execute('SELECT record FROM candidates WHERE status="accepted"'):
            r=json.loads(payload)
            if sha(out/r['label_file'])!=r['label_sha256']:raise ValueError('Accepted label hash mismatch')
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        if args.draft_device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA required; select --draft-device cpu for a separate run')
        with db:bank.put(db,'execution',execution)
        # Set before spawn so worker libraries see limits at import time.
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        solver=Baseline(args.draft_device)
        pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker)
        elapsed=bank.meta(db,'elapsed',0);solves=bank.meta(db,'solves',0);start=time.perf_counter()
        quotas=json.loads(bank.PLAN.read_text())['quotas'];queue=deque();submitted=0;committed=0;status='running';current=None
        session={'started_unix':time.time(),'command':__import__('sys').argv,'execution':execution,'max_seconds':args.max_seconds,'max_solves':args.max_solves}
        with (out/'sessions.jsonl').open('a') as f:f.write(json.dumps(session)+'\n')
        def persist(state):
            with db:bank.put(db,'elapsed',elapsed+time.perf_counter()-start)
            progress=bank.export(db,out,state)
            progress.update(execution=execution,inflight=len(queue),session_committed=committed,
                            session_seconds=time.perf_counter()-start,shard_cache=shard_reader.cache_info()._asdict())
            save(out/'progress.json',progress)
        persist('running')
        try:
            # Complete dependency/initializer checks before handing workers any case.
            list(pool.map(worker_ready,range(args.workers)))
            while True:
                stopped=(out/'STOP').exists() or elapsed+time.perf_counter()-start>=args.max_seconds or solves>=args.max_solves
                smoke_done=args.smoke_count is not None and submitted>=args.smoke_count
                counts=[db.execute('SELECT COUNT(*) FROM candidates WHERE status="accepted" AND bin=?',(i,)).fetchone()[0] for i in range(6)]
                available=min(args.batch_size,args.workers*2-len(queue),args.max_solves-solves)
                if args.smoke_count is not None:available=min(available,args.smoke_count-submitted)
                selected=[];reserved=[sum(item[0]['bin']==i for item in queue) for i in range(6)]
                busy={item[0]['id'] for item in queue}
                if not stopped and not smoke_done:
                    while len(selected)<available:
                        bins=[i for i in range(6) if counts[i]+reserved[i]<quotas[i]]
                        if not bins:break
                        b=min(bins,key=lambda i:((counts[i]+reserved[i])/quotas[i],i))
                        exclusion=' AND id NOT IN ('+','.join('?' for _ in busy)+')' if busy else ''
                        row=db.execute('SELECT * FROM candidates WHERE bin=? AND status="pending"'+exclusion+' ORDER BY id LIMIT 1',(b,*sorted(busy))).fetchone()
                        if row is None:
                            if bank.expand(db):continue
                            status='pool_exhausted';break
                        c=dict(row);member=[c['shard'],c['row']]
                        try:case,_=derived_case(shard_reader(c['shard']).case(c['row']))
                        except ValueError as exc:
                            if str(exc) not in ('locked_dimensions_disagree','mean_area_incompatible_with_locked_shape'):raise
                            with db:db.execute('UPDATE candidates SET status="incompatible",record=? WHERE id=?',(json.dumps({'member':member,'reason':str(exc)}),c['id']))
                            continue
                        selected.append((c,case));busy.add(c['id']);reserved[b]+=1
                if selected:
                    positions,inference=drafts(solver,[case for _,case in selected],args.draft_device)
                    for (c,case),draft in zip(selected,positions):
                        # Charge attempted work before dispatch; interruption cannot reset the solve budget.
                        solves+=1;submitted+=1
                        with db:bank.put(db,'solves',solves)
                        # Plain arrays avoid PyTorch shared-storage feeder handshakes on Windows,
                        # including executor failure during process initialization.
                        public_inputs=(case.block_count,*(np.asarray(v).copy() for v in case.solver_inputs[1:]))
                        future=pool.submit(postprocess,public_inputs,draft)
                        queue.append((c,case,future,inference,time.perf_counter()))
                if not queue:
                    if counts==quotas:status='completed'
                    elif status=='pool_exhausted':pass
                    elif stopped:status='budget_or_stop'
                    elif smoke_done:status='smoke_stopped'
                    else:raise RuntimeError('No work available before quotas met')
                    break
                c,case,future,inference,dispatched=queue[0];current=[c['shard'],c['row']]
                # FIFO commits prevent selecting merely the fastest finished samples.
                x,stats=future.result();queue.popleft()
                result=evaluate_case(case,x,stats['postprocess_seconds']+inference);result.pop('positions')
                if not result['metrics']['is_feasible'] or not result['independent_hard_checks']['is_feasible']:raise ValueError('Official/independent hard validity failure')
                keep=bank.eligible(result['metrics'],result['independent_hard_checks'])
                rec={'member':current,'case_id':case.case_id,'n':c['n'],'stratum':c['bin'],'origin':c['origin'],**result,
                     'execution_version':'E14.parallel_teacher.v1','draft_device':args.draft_device,
                     'worker_pid':stats['worker_pid'],
                     'inference_seconds_amortized':inference,'solver_seconds':stats['postprocess_seconds']+inference,
                     'dispatch_to_commit_seconds':time.perf_counter()-dispatched,'rescue_used':stats['rescue_used'],
                     'stability_triggered':stats['stability']['triggered']}
                if keep:
                    p=prepare_problem(*case.solver_inputs);target=encode(x,p);validate(target,p)
                    target.update(input_areas=np.asarray(case.solver_inputs[1]).copy(),task_version=np.asarray(VERSION),schema=np.asarray(SCHEMA))
                    name=f"labels/{c['id']:06d}.npz";path=out/name;tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,**target);f.flush();os.fsync(f.fileno())
                    os.replace(tmp,path);rec.update(label_file=name,label_sha256=sha(path))
                else:rec['rejection_reason']='Q>1.04'
                with db:
                    db.execute('UPDATE candidates SET status=?,record=? WHERE id=?',('accepted' if keep else 'rejected',json.dumps(rec),c['id']))
                    bank.put(db,'elapsed',elapsed+time.perf_counter()-start)
                committed+=1
                print(f"accepted={sum(counts)+int(keep)}/16000 Q={rec['metrics']['cost']:.6f} {'KEEP' if keep else 'DROP'} {current}",flush=True)
                if committed%25==0:persist('running')
        except KeyboardInterrupt:status='interrupted'
        except Exception as exc:
            status='failed';save(out/'last_error.json',{'error':repr(exc),'member':current,'uncommitted_remain_pending':True});raise
        finally:
            # STOP drains the bounded queue through the loop; Ctrl+C cancels unstarted work.
            pool.shutdown(wait=True,cancel_futures=True)
            persist(status)
            print(json.dumps(json.loads((out/'progress.json').read_text()),indent=2))
    finally:db.close();handle.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','generate','status'])
    ap.add_argument('--output',type=Path,default=DEFAULT)
    ap.add_argument('--source',type=Path,default=bank.DEFAULT)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--draft-device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-seconds',type=float,default=172800)
    ap.add_argument('--max-solves',type=int,default=60000)
    ap.add_argument('--smoke-count',type=int,choices=range(1,6))
    args=ap.parse_args()
    if args.workers<1 or args.workers>8 or args.batch_size<1 or args.max_seconds<=0 or args.max_solves<1:raise ValueError('Invalid worker/batch/budget settings')
    if args.action=='prepare':prepare(args.output,args.source)
    elif args.action=='status':print((args.output/'progress.json').read_text())
    else:
        if not args.output.exists():prepare(args.output,args.source)
        generate(args)


if __name__=='__main__':main()
