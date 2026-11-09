"""Fixed five-percent draft preprocessing; four one-thread matched workers."""
from . import resources
import argparse
import concurrent.futures as futures
import multiprocessing as mp
import ctypes
from pathlib import Path
import sqlite3
import zlib
import json
import time
import sys
import numpy as np
from .io import ROOT,HERE,save,sha,sources,snapshot,exclusive
from .expansion import ARMS,expand

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
DISCOVERY=ROOT/'experiments/E35_optimizer_attribution/runs/20261022_attribution100/outcomes.sqlite3'
CONFIRMATION=ROOT/'experiments/E36_persistent_conflict_repair/runs/20261022_confirmation500/outcomes.sqlite3'
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
    a=ctypes.c_size_t();b=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(a),ctypes.byref(b)):raise ctypes.WinError(ctypes.get_last_error())
    slots=[i for i in range(64) if a.value&(1<<i)][::2][:4]
    if len(slots)!=4:raise RuntimeError('Four separated CPU slots unavailable')
    return slots


def saved_drafts(stage):
    database=CONFIRMATION if stage=='confirmation' else DISCOVERY
    con=sqlite3.connect('file:'+database.resolve().as_posix()+'?mode=ro',uri=True)
    if stage=='confirmation':
        rows={i:unpack(blob) for i,blob in con.execute('SELECT idx,payload FROM drafts')}
    else:
        rows={}
        for (blob,) in con.execute("SELECT payload FROM records WHERE arm='full'"):
            r=unpack(blob);initial=next(e for e in r['events'] if e['kind']=='initial_state' and e['depth']==1)
            rows[r['index']]={'draft':r['layouts'][initial['draft']],'member':r['member'],'n':r['n'],'inference_seconds':r['inference_seconds']}
    con.close();return rows,database


def initialize(queue,stage):
    global STATE
    import os
    os.environ['E37_CPU_SLOT']=str(queue.get(timeout=30));resource=resources.configure()
    from .cases import public_cases
    from .canonical_native import warmup
    rows,metadata,_=public_cases('confirmation' if stage=='confirmation' else 'monitor');drafts,_=saved_drafts(stage)
    for i,r in drafts.items():assert list(metadata[i]['member'])==r['member']
    warmup();STATE=(resource,rows,drafts)


def execute(index,arms,deadline):
    from . import solver
    from .backend.frozen.packing import prepare_problem
    from .diagnostics import geometry,move
    from scoring.evaluation import evaluate_case
    resource,rows,drafts=STATE;case=rows[index];old=drafts[index];draft=np.asarray(old['draft'],dtype=float);p=prepare_problem(*case.solver_inputs)
    units=np.sqrt(p.areas.copy());locked=p.fixed|p.preplaced;units[locked]=np.sqrt(p.specs[locked,2:].prod(1))
    original_diagnostic=geometry(draft,p);original_pairs=original_diagnostic['overlap_pairs'];records=[]
    for arm in arms:
        if time.monotonic()>=deadline or (HERE/'runs/STOP').exists():break
        calls=[];native_step=solver.step
        def collect(*args,**kwargs):
            result=native_step(*args,**kwargs);calls.append((args[1],result[0],result[1]));return result
        solver.step=collect;started=time.perf_counter();cpu_start=time.process_time()
        try:
            transformed,raw,preprocessing=expand(draft,case.solver_inputs,arm);operator_seconds=time.perf_counter()-started
            positions,stats=solver.optimize(transformed,case.solver_inputs,**STEADY)
            seconds=time.perf_counter()-started;cpu_seconds=time.process_time()-cpu_start
            # All scoring, detailed geometry and serialization are deferred.
            metrics=evaluate_case(case,positions,seconds)['metrics'];assert metrics['is_feasible']
            raw_geometry=geometry(raw,p,draft);input_geometry=geometry(transformed,p,draft);final_geometry=geometry(positions,p,draft)
            if arm!='reference':
                assert raw_geometry['overlap_pairs']<=original_pairs
                assert raw_geometry['overlap_area_pair_sum_over_total']<=original_diagnostic['overlap_area_pair_sum_over_total']+1e-9
            movements=[move(a,b,units) for a,b,s in calls if b is not None]
            first=stats['trace'][0] if stats['trace'] else {}
            records.append({'index':index,'member':old['member'],'n':p.n,'arm':arm,'seconds':seconds,'cpu_seconds':cpu_seconds,
                'operator_seconds':operator_seconds,'resource':resource,'metrics':metrics,'positions':positions,
                'input_positions':transformed,'raw_expanded_positions':raw,'preprocessing':preprocessing,
                'original_geometry':original_diagnostic,'raw_expansion_geometry':raw_geometry,'input_geometry':input_geometry,'final_geometry':final_geometry,
                'stats':stats,'all_shape_calls':len(calls),'root_shape_calls':sum(t['phase']=='shape' for t in stats['trace']),
                'shape_kernel_statuses':[s['status'] for a,b,s in calls],'all_shape_iterations':sum(s['iterations'] for a,b,s in calls),
                'all_shape_cumulative_center':sum(m['center_mean'] for m in movements),'first_phase':first.get('phase'),'first_status':first.get('status'),
                'rescue_calls':sum(t['phase']=='full_packed_refine' for t in stats['trace']),'cached_inference_seconds':old['inference_seconds']})
        except Exception as error:
            records.append({'index':index,'member':old['member'],'n':p.n,'arm':arm,'seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu_start,
                'metrics':{'is_feasible':False,'cost':10.},'error':repr(error),'positions':None,'resource':resource})
        finally:solver.step=native_step
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=('gate','main','confirmation'),default='gate');ap.add_argument('--max-minutes',type=float,default=20.);args=ap.parse_args()
    start=time.perf_counter();slots=cpu_slots();run={'gate':'20261022_gate5','main':'20261022_paired100','confirmation':'20261022_confirmation500'}[args.stage]
    folder=HERE/'runs'/run;folder.mkdir(parents=True,exist_ok=True);drafts,database=saved_drafts(args.stage)
    from .cases import WEIGHTS,BASE
    assert sha(WEIGHTS)=='3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    assert sha(HERE/'model.py')==sha(BASE/'model.py')
    panel=[329,496,368,127,332] if args.stage=='gate' else sorted(drafts)
    arms=list(ARMS)
    if args.stage=='main':assert json.loads((HERE/'runs/20261022_gate5/summary.json').read_text())['gate_passed']
    if args.stage=='confirmation':
        selection=json.loads((HERE/'runs/selection.json').read_text());assert selection['eligible'];arms=['reference',selection['arm']]
    identity={'sources':sources(),'weights_sha256':sha(WEIGHTS),'upstream_database':str(database),'upstream_database_sha256':sha(database),
        'panel':panel,'members':[drafts[i]['member'] for i in panel],'arms':arms,'factor':1.05,'stage':args.stage,'steady':STEADY,
        'workers':4,'threads_per_worker':1,'cpu_slots':slots,'max_minutes':args.max_minutes,'command':[sys.executable,*sys.argv],
        'timing':'actual radial preprocessing + original-logic postprocessor wall time, shared O(1) shape collector included, detailed metrics/scoring excluded',
        'model_training':False,'test_read':False,'panel_scope':'adaptive development; frozen cached drafts; confirmation reuses E36 developer-exposed panel' if args.stage=='confirmation' else 'E35 fixed adaptive dev100 (five historical stress cases for gate)'}
    if args.stage!='gate':
        gate_config=json.loads((HERE/'runs/20261022_gate5/config.json').read_text())
        assert identity['sources']==gate_config['sources'],'Sources changed after gate'
    budget_path=HERE/'runs/budget.json'
    with exclusive(HERE/'runs/research.lock'):
        if (folder/'config.json').exists():assert json.loads((folder/'config.json').read_text())==identity,'Run identity changed'
        else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
        if (folder/'summary.json').exists():print('Already complete');return
        prior=json.loads(budget_path.read_text()) if budget_path.exists() else {'elapsed_seconds':0.,'max_minutes':args.max_minutes}
        assert prior['max_minutes']==args.max_minutes
        elapsed=lambda:prior['elapsed_seconds']+time.perf_counter()-start
        if elapsed()>=args.max_minutes*60:print('Cumulative budget exhausted');return
        deadline=time.monotonic()+args.max_minutes*60-elapsed()
        db=sqlite3.connect(folder/'outcomes.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS records(idx INTEGER NOT NULL,arm TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(idx,arm))');db.commit()
        existing={(i,a) for i,a in db.execute('SELECT idx,arm FROM records')};rng=np.random.default_rng(20261025);jobs=[]
        permutations=[rng.permutation(len(arms)).tolist() for _ in range((len(panel)+len(arms)-1)//len(arms))]
        for ordinal,index in enumerate(panel):
            order=permutations[ordinal//len(arms)];shift=ordinal%len(arms);order=order[shift:]+order[:shift]
            todo=[arms[j] for j in order if (index,arms[j]) not in existing]
            if todo:jobs.append((index,todo))
        context=mp.get_context('spawn');queue=context.Queue()
        for slot in slots:queue.put(slot)
        completed=0
        with futures.ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize,initargs=(queue,args.stage)) as pool:
            pending={};cursor=0
            def dispatch():
                nonlocal cursor
                while len(pending)<4 and cursor<len(jobs) and time.monotonic()<deadline and not (HERE/'runs/STOP').exists():
                    i,todo=jobs[cursor];cursor+=1;pending[pool.submit(execute,i,todo,deadline)]=i
            dispatch()
            while pending:
                done,_=futures.wait(pending,timeout=1,return_when=futures.FIRST_COMPLETED)
                for f in done:
                    pending.pop(f);rr=f.result()
                    for r in rr:db.execute('INSERT INTO records VALUES(?,?,?)',(r['index'],r['arm'],packed(r)))
                    db.commit();completed+=1;save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':args.max_minutes})
                    save(folder/'progress.json',{'completed_cases':completed,'count':len(panel),'records':db.execute('SELECT COUNT(*) FROM records').fetchone()[0],'cumulative_elapsed_seconds':elapsed()})
                    if args.stage=='gate' or completed%10==0:print('radial cases',completed,'/',len(jobs),'cumulative seconds',round(elapsed(),1),flush=True)
                dispatch()
        rr=[unpack(row[0]) for row in db.execute('SELECT payload FROM records')]
        if len(rr)!=len(panel)*len(arms):save(folder/'stop.json',{'status':'budget_or_user_stop','records':len(rr),'elapsed_seconds':elapsed()});db.close();return
        summaries={}
        for arm in arms:
            selected=[r for r in rr if r['arm']==arm];q=np.array([r['metrics']['cost'] for r in selected]);n=np.array([r['n'] for r in selected]);w=np.exp((n-n.max())/12);tt=np.array([r['seconds'] for r in selected]);valid=[r for r in selected if 'error' not in r]
            s={'count':len(selected),'hard_legal':sum(r['metrics']['is_feasible'] for r in selected),'errors':sum('error' in r for r in selected),
                'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),
                'cvar95_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),'max_Q':float(q.max()),
                'seconds_mean':float(tt.mean()),'seconds_p95':float(np.percentile(tt,95)),
                'operator_seconds_mean':float(np.mean([r['operator_seconds'] for r in valid])) if valid else None,
                'all_shape_calls_mean':float(np.mean([r['all_shape_calls'] for r in valid])) if valid else None,
                'rescue_cases':sum(r.get('rescue_calls',0)>0 for r in selected),
                'first_PrimalInfeasible':sum(r.get('first_status')=='PrimalInfeasible' for r in selected),
                'input_overlap_pairs_mean':float(np.mean([r['input_geometry']['overlap_pairs'] for r in valid])) if valid else None,
                'input_overlap_area_ratio_mean':float(np.mean([r['input_geometry']['overlap_area_pair_sum_over_total'] for r in valid])) if valid else None,
                'input_negative_blocks_mean':float(np.mean([r['input_geometry']['negative_coordinate_blocks'] for r in valid])) if valid else None,
                'input_preplaced_offset_blocks_mean':float(np.mean([r['input_geometry']['preplaced_offset_blocks'] for r in valid])) if valid else None,
                'final_mean_center_move':float(np.mean([r['final_geometry']['movement_from_original']['center_mean'] for r in valid])) if valid else None,
                'final_mean_log_AR_change':float(np.mean([r['final_geometry']['movement_from_original']['log_AR_mean'] for r in valid])) if valid else None}
            summaries[arm]=s
        passed=all(s['hard_legal']==len(panel) and s['errors']==0 for s in summaries.values())
        result={'status':'complete','stage':args.stage,'gate_passed':passed,'summaries':summaries,'record_count':len(rr),
            'cumulative_elapsed_seconds':elapsed(),'batch_wall_seconds':time.perf_counter()-start,'test_read':False,'model_training':False}
        save(folder/'summary.json',result);save(folder/'progress.json',{'status':'complete','cases':len(panel),'records':len(rr),'cumulative_elapsed_seconds':elapsed()});save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':args.max_minutes});db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
