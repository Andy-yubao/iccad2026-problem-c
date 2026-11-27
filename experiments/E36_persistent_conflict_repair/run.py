"""Frozen-draft paired postprocessor research; one process, two CPU slots."""
from .resources import configure
RESOURCE=configure()
import argparse
from pathlib import Path
import json
import sqlite3
import zlib
import time
import sys
import numpy as np
from .cases import prepare_cache,WEIGHTS
from .canonical_native import warmup
from .io import ROOT,HERE,save,sha,sources,snapshot,exclusive
from . import solver,reference_solver
from scoring.evaluation import evaluate_case

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
ARMS=('reference','control','hard_after_legal','local_repair','early_rescue','local_then_escape')
DATABASE=ROOT/'experiments/E35_optimizer_attribution/runs/20261022_attribution100/outcomes.sqlite3'


def packed(value):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    return zlib.compress(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=default).encode(),1)


def unpack(value):return json.loads(zlib.decompress(value))


def geometry(a,b,units):
    a=np.asarray(a);b=np.asarray(b);d=np.linalg.norm((b[:,:2]+b[:,2:]/2)-(a[:,:2]+a[:,2:]/2),axis=1)/units
    ar=np.abs(np.log((b[:,2]/b[:,3])/(a[:,2]/a[:,3])))
    return {'mean_center_move':float(d.mean()),'p95_center_move':float(np.percentile(d,95)),'max_center_move':float(d.max()),
        'mean_log_AR_change':float(ar.mean()),'fraction_AR_change_ge_2':float((ar>=np.log(2)).mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('gate','main'),default='gate');parser.add_argument('--max-minutes',type=float,default=20.)
    args=parser.parse_args();start=time.perf_counter();folder=HERE/'runs'/('20261022_gate5' if args.stage=='gate' else '20261022_paired100')
    folder.mkdir(parents=True,exist_ok=True)
    # All formal inputs are selected by the already frozen E35 panel, not Q.
    db=sqlite3.connect('file:'+DATABASE.resolve().as_posix()+'?mode=ro',uri=True)
    saved={r['index']:r for (blob,) in db.execute("SELECT payload FROM records WHERE arm='full'") for r in [unpack(blob)]};db.close();assert len(saved)==100
    panel=sorted(saved,key=lambda i:(saved[i]['n'],i)) if args.stage=='main' else [329,496,368,127,332]
    if args.stage=='main':
        gate=json.loads((HERE/'runs/20261022_gate5/summary.json').read_text())
        if not gate['gate_passed']:raise ValueError('Five-case mechanism gate did not pass')
    inputs=prepare_cache()['rows']; warm_start=time.perf_counter();warmup();warm_seconds=time.perf_counter()-warm_start
    identity={'sources':sources(),'weights_sha256':sha(WEIGHTS),'source_database_sha256':sha(DATABASE),
        'panel':panel,'members':[saved[i]['member'] for i in panel],'arms':ARMS,'steady':STEADY,
        'resources':{k:v for k,v in RESOURCE.items() if k!='pid'},'max_minutes':args.max_minutes,
        'stage':args.stage,'command':[sys.executable,*sys.argv],'timing':'measured postprocessor only; cached shared drafts, no inference here',
        'model_training':False,'test_read':False}
    identity=json.loads(json.dumps(identity))
    if args.stage=='main':
        previous=json.loads((HERE/'runs/20261022_gate5/config.json').read_text())
        if identity['sources']!=previous['sources']:raise ValueError('Sources changed after five-case gate; repeat gate explicitly in a new directory')
    budget_path=HERE/'runs/budget.json'
    with exclusive(HERE/'runs/research.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Run identity changed')
        else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
        if (folder/'summary.json').exists():print('Already complete');return
        prior=json.loads(budget_path.read_text()) if budget_path.exists() else {'elapsed_seconds':0.,'max_minutes':args.max_minutes}
        if prior['max_minutes']!=args.max_minutes:raise ValueError('Do not reset the cumulative budget')
        elapsed=lambda:prior['elapsed_seconds']+time.perf_counter()-start
        save(folder/'initialization.json',{'native_warmup_seconds':warm_seconds,'resource':RESOURCE})
        con=sqlite3.connect(folder/'outcomes.sqlite3');con.execute('PRAGMA journal_mode=WAL')
        con.execute('CREATE TABLE IF NOT EXISTS records (idx INTEGER NOT NULL,arm TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(idx,arm))');con.commit()
        rng=np.random.default_rng(20261022);blocks=[rng.permutation(len(ARMS)).tolist() for _ in range((len(panel)+len(ARMS)-1)//len(ARMS))]
        for ordinal,index in enumerate(panel):
            perm=blocks[ordinal//len(ARMS)];shift=ordinal%len(ARMS);order=perm[shift:]+perm[:shift]
            old=saved[index];initial=next(e for e in old['events'] if e['kind']=='initial_state' and e['depth']==1)
            draft=np.array(old['layouts'][initial['draft']],dtype=float);case=inputs[index]['case']
            units=np.sqrt(np.asarray(case.solver_inputs[1],dtype=float));constraints=np.asarray(case.solver_inputs[5]);specs=np.asarray(case.solver_inputs[6]);locked=constraints[:,:2].any(1);units[locked]=np.sqrt(specs[locked,2:].prod(1))
            for method in order:
                arm=ARMS[method]
                if con.execute('SELECT 1 FROM records WHERE idx=? AND arm=?',(index,arm)).fetchone():continue
                if (folder/'STOP').exists() or (HERE/'runs/STOP').exists() or elapsed()>=args.max_minutes*60:
                    save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':args.max_minutes});save(folder/'progress.json',{'status':'stopped','rows':con.execute('SELECT COUNT(*) FROM records').fetchone()[0]});con.close();return
                begin=time.perf_counter()
                try:
                    if arm=='reference':positions,stats=reference_solver.optimize(draft,case.solver_inputs,**STEADY)
                    else:positions,stats=solver.optimize(draft,case.solver_inputs,policy=arm,**STEADY)
                    seconds=time.perf_counter()-begin
                    # Scoring and final-coordinate diagnostics are outside solver time.
                    metrics=evaluate_case(case,positions,seconds)['metrics'];movement=geometry(draft,positions,units)
                    traces=stats['trace'];telemetry=stats.get('telemetry',[])
                    root=[t for t in telemetry if t['level']==0]
                    record={'index':index,'member':old['member'],'n':old['n'],'arm':arm,'seconds':seconds,
                        'metrics':metrics,'positions':positions,'stats':stats,'draft_to_final':movement,
                        'cached_original_inference_seconds':old['inference_seconds'],
                        'root_shape_calls':len(root) if telemetry else sum(t['phase']=='shape' for t in traces),
                        'all_shape_calls':len(telemetry) if telemetry else None,
                        'root_cumulative_center_move':sum(t.get('mean_center_move',0.) for t in root) if telemetry else None,
                        'local_repair_calls':sum(t['phase']=='local_repair' for t in traces),
                        'local_moves':sum(len(t.get('moves',[])) for t in traces if t['phase']=='local_repair'),
                        'conflict_escape':any(t['phase']=='conflict_escape' for t in traces),
                        'robust_refine_calls':sum(t['phase']=='full_packed_refine' for t in traces),
                        'first_legal_shape_step':next((k+1 for k,t in enumerate(root) if t.get('hard_legal')),None)}
                    assert metrics['is_feasible']
                except Exception as error:
                    record={'index':index,'member':old['member'],'n':old['n'],'arm':arm,'seconds':time.perf_counter()-begin,
                        'error':repr(error),'metrics':{'is_feasible':False,'cost':10.},'positions':None}
                    if args.stage=='gate':raise
                con.execute('INSERT INTO records VALUES (?,?,?)',(index,arm,packed(record)));con.commit()
                save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':args.max_minutes})
            save(folder/'progress.json',{'status':'running','complete_cases':ordinal+1,'count':len(panel),'elapsed_seconds':elapsed()})
            if args.stage=='gate' or (ordinal+1)%10==0:print('paired cases',ordinal+1,'/',len(panel),'cumulative seconds',round(elapsed(),1),flush=True)
        records=[unpack(row[0]) for row in con.execute('SELECT payload FROM records')];summaries={}
        for arm in ARMS:
            rr=[r for r in records if r['arm']==arm];assert len(rr)==len(panel)
            q=np.array([r['metrics']['cost'] for r in rr]);n=np.array([r['n'] for r in rr]);w=np.exp((n-n.max())/12);tt=[r['seconds'] for r in rr]
            s={'count':len(rr),'hard_legal':sum(r['metrics']['is_feasible'] for r in rr),'errors':sum('error' in r for r in rr),
                'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),
                'cvar95_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),'max_Q':float(q.max()),
                'seconds_mean':float(np.mean(tt)),'seconds_p95':float(np.percentile(tt,95)),
                'root_shape_calls_mean':float(np.mean([r.get('root_shape_calls',0) for r in rr])),
                'mean_final_center_move':float(np.mean([r['draft_to_final']['mean_center_move'] for r in rr if 'draft_to_final' in r])),
                'local_repair_cases':sum(r.get('local_repair_calls',0)>0 for r in rr),'local_moves':sum(r.get('local_moves',0) for r in rr),
                'escape_cases':sum(r.get('conflict_escape',False) for r in rr),'rescue_cases':sum(r.get('robust_refine_calls',0)>0 for r in rr)}
            for key in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations'):
                s[key+'_mean']=float(np.mean([r['metrics'][key] for r in rr if key in r['metrics']]))
            summaries[arm]=s
        valid=all(s['hard_legal']==len(panel) and s['errors']==0 for s in summaries.values())
        result={'status':'complete','stage':args.stage,'gate_passed':valid,'summaries':summaries,'cumulative_elapsed_seconds':elapsed(),
            'solver_calls':len(records),'test_read':False,'model_training':False}
        save(folder/'summary.json',result);save(budget_path,{'elapsed_seconds':elapsed(),'max_minutes':args.max_minutes})
        con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close();print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
