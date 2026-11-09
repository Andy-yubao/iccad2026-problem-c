"""Four independent one-thread workers, with matched within-case strategy runs."""
# Set library limits before importing NumPy/Torch, but do not pin the parent:
# spawned workers must inherit its full permitted affinity mask.
from . import resources
import argparse
import ctypes
import concurrent.futures as futures
import multiprocessing as mp
from pathlib import Path
import json
import sqlite3
import sys
import time
import zlib
import numpy as np
from .io import ROOT,HERE,save,sha,sources,snapshot,exclusive

ARMS=('reference','control','chain_repair','pivot_repair','progress_escape')
STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
DATABASE=ROOT/'experiments/E35_optimizer_attribution/runs/20261022_attribution100/outcomes.sqlite3'
STATE=None


def packed(value):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    return zlib.compress(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=default).encode(),1)


def unpack(value):return json.loads(zlib.decompress(value))


def cpu_slots():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    current=ctypes.c_size_t();system=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(current),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
    allowed=[i for i in range(64) if current.value&(1<<i)]
    chosen=allowed[::2][:4]
    if len(chosen)<4:raise RuntimeError('Four separated CPU slots are unavailable')
    return chosen


def initialize(queue):
    global STATE
    import os
    os.environ['E36_CPU_SLOT']=str(queue.get(timeout=30))
    resource=resources.configure()
    from .cases import prepare_cache
    from .canonical_native import warmup
    con=sqlite3.connect('file:'+DATABASE.resolve().as_posix()+'?mode=ro',uri=True)
    saved={r['index']:r for (blob,) in con.execute("SELECT payload FROM records WHERE arm='full'") for r in [unpack(blob)]};con.close()
    rows=prepare_cache()['rows'];warmup()
    STATE=(resource,saved,rows)


def execute(index,arms,deadline):
    from . import solver,reference_solver
    from .run import geometry
    from scoring.evaluation import evaluate_case
    resource,saved,rows=STATE;r=saved[index];case=rows[index]['case']
    initial=next(e for e in r['events'] if e['kind']=='initial_state' and e['depth']==1)
    draft=np.array(r['layouts'][initial['draft']],dtype=float)
    units=np.sqrt(np.asarray(case.solver_inputs[1],dtype=float));c=np.asarray(case.solver_inputs[5]);specs=np.asarray(case.solver_inputs[6]);locked=c[:,:2].any(1);units[locked]=np.sqrt(specs[locked,2:].prod(1))
    records=[]
    for arm in arms:
        if time.monotonic()>=deadline or (HERE/'runs/STOP').exists():break
        started=time.perf_counter();cpu_start=time.process_time()
        try:
            if arm=='reference':positions,stats=reference_solver.optimize(draft,case.solver_inputs,**STEADY)
            else:positions,stats=solver.optimize(draft,case.solver_inputs,policy=arm,**STEADY)
            seconds=time.perf_counter()-started;cpu_seconds=time.process_time()-cpu_start
            metrics=evaluate_case(case,positions,seconds)['metrics'];assert metrics['is_feasible']
            telemetry=stats.get('telemetry',[]);root=[t for t in telemetry if t['level']==0];trace=stats['trace']
            record={'index':index,'member':r['member'],'n':r['n'],'arm':arm,'seconds':seconds,'cpu_seconds':cpu_seconds,
                'metrics':metrics,'positions':positions,'stats':stats,'draft_to_final':geometry(draft,positions,units),
                'root_shape_calls':len(root) if telemetry else sum(t['phase']=='shape' for t in trace),
                'all_shape_calls':len(telemetry) if telemetry else None,
                'root_cumulative_center_move':sum(t.get('mean_center_move',0.) for t in root) if telemetry else None,
                'repair_calls':sum(t['phase']=='chain_repair' for t in trace),
                'repair_moves':sum(len(t['moves']) for t in trace if t['phase']=='chain_repair'),
                'pivot_calls':sum(t['phase']=='conflict_pivot' for t in trace),
                'conflict_escape':any(t['phase']=='conflict_escape' for t in trace),
                'rescue_calls':sum(t['phase']=='full_packed_refine' for t in trace),
                'resource':resource,'cached_inference_seconds':r['inference_seconds']}
        except Exception as error:
            record={'index':index,'member':r['member'],'n':r['n'],'arm':arm,'seconds':time.perf_counter()-started,
                'cpu_seconds':time.process_time()-cpu_start,'error':repr(error),'metrics':{'is_feasible':False,'cost':10.},'positions':None,'resource':resource}
        records.append(record)
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=('gate','main'),default='gate');args=ap.parse_args()
    start=time.perf_counter();slots=cpu_slots();folder=HERE/'runs'/('20261022_stage2_gate5' if args.stage=='gate' else '20261022_stage2_paired100');folder.mkdir(parents=True,exist_ok=True)
    budget_path=HERE/'runs/budget.json';prior=json.loads(budget_path.read_text());limit=prior['max_minutes']*60
    con=sqlite3.connect('file:'+DATABASE.resolve().as_posix()+'?mode=ro',uri=True)
    saved={r['index']:r for (blob,) in con.execute("SELECT payload FROM records WHERE arm='full'") for r in [unpack(blob)]};con.close()
    panel=sorted(saved,key=lambda i:(saved[i]['n'],i)) if args.stage=='main' else [329,496,368,127,332]
    identity={'sources':sources(),'source_database_sha256':sha(DATABASE),'panel':panel,'members':[saved[i]['member'] for i in panel],
        'arms':list(ARMS),'steady':STEADY,'workers':4,'threads_per_worker':1,'cpu_slots':slots,'stage':args.stage,
        'max_minutes_cumulative':prior['max_minutes'],'command':[sys.executable,*sys.argv],
        'timing':'measured postprocessor wall and process CPU seconds per arm; initialization/scoring excluded; batch makespan separate',
        'model_training':False,'test_read':False}
    if args.stage=='main':
        gate=json.loads((HERE/'runs/20261022_stage2_gate5/summary.json').read_text());gate_identity=json.loads((HERE/'runs/20261022_stage2_gate5/config.json').read_text())
        if not gate['gate_passed'] or identity['sources']!=gate_identity['sources']:raise ValueError('Five-case gate/source identity failed')
    with exclusive(HERE/'runs/research.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Run identity changed')
        else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
        if (folder/'summary.json').exists():print('Already complete');return
        elapsed=lambda:prior['elapsed_seconds']+time.perf_counter()-start
        remaining=limit-elapsed()
        if remaining<=0:print('Cumulative budget exhausted');return
        deadline=time.monotonic()+remaining
        db=sqlite3.connect(folder/'outcomes.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS records(idx INTEGER NOT NULL,arm TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(idx,arm))');db.commit()
        complete={(i,a) for i,a in db.execute('SELECT idx,arm FROM records')};rng=np.random.default_rng(20261023)
        permutations=[rng.permutation(len(ARMS)).tolist() for _ in range((len(panel)+len(ARMS)-1)//len(ARMS))]
        jobs=[]
        for ordinal,index in enumerate(panel):
            perm=permutations[ordinal//len(ARMS)];shift=ordinal%len(ARMS);order=perm[shift:]+perm[:shift]
            arms=[ARMS[i] for i in order if (index,ARMS[i]) not in complete]
            if arms:jobs.append((index,arms))
        context=mp.get_context('spawn');queue=context.Queue()
        for slot in slots:queue.put(slot)
        completed=0
        with futures.ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
            pending={};cursor=0
            def dispatch():
                nonlocal cursor
                while len(pending)<4 and cursor<len(jobs) and time.monotonic()<deadline and not (HERE/'runs/STOP').exists():
                    index,arms=jobs[cursor];cursor+=1;pending[pool.submit(execute,index,arms,deadline)]=index
            dispatch()
            while pending:
                finished,_=futures.wait(pending,timeout=1,return_when=futures.FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future);records=future.result()
                    for r in records:db.execute('INSERT INTO records VALUES(?,?,?)',(r['index'],r['arm'],packed(r)))
                    db.commit();completed+=1
                    save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':prior['max_minutes']})
                    save(folder/'progress.json',{'completed_jobs':completed,'scheduled_jobs':cursor,'total_jobs':len(jobs),'records':db.execute('SELECT COUNT(*) FROM records').fetchone()[0],'cumulative_elapsed_seconds':elapsed()})
                    if args.stage=='gate' or completed%10==0:print('stage2 cases',completed,'/',len(jobs),'cumulative seconds',round(elapsed(),1),flush=True)
                dispatch()
        records=[unpack(row[0]) for row in db.execute('SELECT payload FROM records')];summaries={}
        if len(records)!=len(panel)*len(ARMS):
            save(folder/'stop.json',{'status':'budget_or_user_stop','records':len(records),'cumulative_elapsed_seconds':elapsed()});save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':prior['max_minutes']});db.close();return
        for arm in ARMS:
            rr=[r for r in records if r['arm']==arm];q=np.array([r['metrics']['cost'] for r in rr]);n=np.array([r['n'] for r in rr]);w=np.exp((n-n.max())/12);tt=np.array([r['seconds'] for r in rr])
            summaries[arm]={'count':len(rr),'hard_legal':sum(r['metrics']['is_feasible'] for r in rr),'errors':sum('error' in r for r in rr),
                'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),
                'cvar95_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),'max_Q':float(q.max()),
                'seconds_mean':float(tt.mean()),'seconds_p95':float(np.percentile(tt,95)),'cpu_seconds_mean':float(np.mean([r['cpu_seconds'] for r in rr])),
                'root_shapes_mean':float(np.mean([r.get('root_shape_calls',0) for r in rr])),
                'final_mean_center_move':float(np.mean([r['draft_to_final']['mean_center_move'] for r in rr if 'draft_to_final' in r])),
                'repair_cases':sum(r.get('repair_calls',0)>0 for r in rr),'repair_moves':sum(r.get('repair_moves',0) for r in rr),
                'pivot_cases':sum(r.get('pivot_calls',0)>0 for r in rr),'pivot_attempts':sum(r.get('pivot_calls',0) for r in rr),
                'escape_cases':sum(r.get('conflict_escape',False) for r in rr),'rescue_cases':sum(r.get('rescue_calls',0)>0 for r in rr),
                'wall_above_2x_cpu_cases':sum(r['seconds']>max(.05,2*r['cpu_seconds']) for r in rr)}
        valid=all(s['hard_legal']==len(panel) and s['errors']==0 for s in summaries.values())
        result={'status':'complete','stage':args.stage,'gate_passed':valid,'summaries':summaries,'cumulative_elapsed_seconds':elapsed(),
            'batch_wall_seconds':time.perf_counter()-start,'record_count':len(records),'workers':4,'threads_per_worker':1,'test_read':False,'model_training':False}
        save(folder/'summary.json',result);save(folder/'progress.json',{'status':'complete','cases':len(panel),'records':len(records),'cumulative_elapsed_seconds':elapsed()})
        save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':prior['max_minutes']});db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
