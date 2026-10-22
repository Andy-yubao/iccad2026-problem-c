"""User-authorized 100-case optimizer audit, one numerical worker/two CPUs."""
from .resources import configure
RESOURCE=configure()
import argparse
import json
import sqlite3
import sys
import time
import zlib
import numpy as np
import torch
from .cases import prepare_cache,predict,seed_all,WEIGHTS
from .model import GeometryModel
from . import solver,reference_solver
from .trace import Observer,ARM
from .canonical_native import warmup
from .io import HERE,ROOT,save,sha,sources,snapshot,exclusive
from scoring.evaluation import evaluate_case

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
ARMS=('reference','full','no_shape','no_projection','no_quick_project','no_robust_seeds','no_certified','no_plateau','no_late_extension','no_incumbent_reset','linear_projection','no_contact_penalty','no_boundary_penalty','no_reweight')
OLD=ROOT/'experiments/E25_integrated_baseline/runs/20261020_controls_steady_monitor500/results.json'


def packed(value):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    return zlib.compress(json.dumps(value,ensure_ascii=False,separators=(',',':'),default=default).encode(),level=1)


def unpack(value):return json.loads(zlib.decompress(value))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');ap.add_argument('--max-minutes',type=float,default=20.);args=ap.parse_args()
    started=time.perf_counter();folder=HERE/'runs'/('20261022_check5' if args.check else '20261022_attribution100');folder.mkdir(parents=True,exist_ok=True)
    rows=prepare_cache()['rows'];old=json.loads(OLD.read_text());assert len(old['records'])==len(rows)==500
    for i,row in enumerate(rows):assert row['member']==old['records'][i]['member']
    order=sorted(range(500),key=lambda i:rows[i]['case'].block_count);panel=[order[i] for i in range(2,500,5)]
    if args.check:panel=[panel[i] for i in (0,24,49,74,99)]
    elif not (HERE/'runs/20261022_check5/summary.json').exists():raise ValueError('Five-case audit must finish first')
    assert sha(WEIGHTS)=='3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    assert sha(HERE/'model.py')==sha(ROOT/'experiments/E25_integrated_baseline/model.py')
    warm_started=time.perf_counter();warmup();warm_seconds=time.perf_counter()-warm_started
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=False);model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    identity={'sources':sources(),'weights_sha256':sha(WEIGHTS),'original500_sha256':sha(OLD),'panel':panel,'members':[rows[i]['member'] for i in panel],
        'arms':list(ARMS),'config':STEADY,'resource':{k:v for k,v in RESOURCE.items() if k!='pid'},'max_minutes':args.max_minutes,'command':[sys.executable,*sys.argv],'test_read':False,
        'clock':'Audited controller/polish subtract measured recorder overhead; solver kernels retain original clocks. Reference uses original E25 clocks.'}
    with exclusive(folder/'run.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Run identity changed')
        else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
        if (folder/'summary.json').exists():print('Already complete',flush=True);return
        con=sqlite3.connect(folder/'outcomes.sqlite3');con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=FULL')
        con.execute('CREATE TABLE IF NOT EXISTS records (idx INTEGER NOT NULL, arm TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(idx,arm))')
        con.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)');con.commit()
        found=con.execute("SELECT value FROM metadata WHERE key='elapsed'").fetchone();prior=float(found[0]) if found else 0.
        elapsed=lambda:prior+time.perf_counter()-started
        save(folder/'initialization.json',{'native_warmup_seconds':warm_seconds,'seconds_to_initialized':time.perf_counter()-started,'resource':RESOURCE})
        random=np.random.default_rng(20261022);blocks=[random.permutation(len(ARMS)).tolist() for _ in range((len(panel)+len(ARMS)-1)//len(ARMS))]
        for ordinal,index in enumerate(panel):
            base_order=blocks[ordinal//len(ARMS)];shift=ordinal%len(ARMS);methods=base_order[shift:]+base_order[:shift]
            for method in methods:
                arm=ARMS[method]
                if con.execute('SELECT 1 FROM records WHERE idx=? AND arm=?',(index,arm)).fetchone():continue
                if (folder/'STOP').exists() or elapsed()>=args.max_minutes*60:
                    save(folder/'progress.json',{'status':'paused','elapsed_seconds':elapsed(),'rows':con.execute('SELECT COUNT(*) FROM records').fetchone()[0]});con.close();return
                row=rows[index];case=row['case'];seed_all(index,2);begin=time.perf_counter();obs=Observer() if arm=='full' else None;token=ARM.set('full' if arm=='reference' else arm)
                positions=None
                try:
                    draft,inference=predict(model,case.solver_inputs)
                    if arm=='reference':positions,stats=reference_solver.optimize(draft,case.solver_inputs,**STEADY)
                    elif obs:
                        with obs.activate():positions,stats=solver.optimize(draft,case.solver_inputs,**STEADY)
                    else:positions,stats=solver.optimize(draft,case.solver_inputs,**STEADY)
                    seconds=time.perf_counter()-begin
                    metrics=evaluate_case(case,positions,seconds)['metrics']
                    result={'index':index,'member':row['member'],'n':case.block_count,'arm':arm,'seconds':seconds,'inference_seconds':inference,
                        'metrics':metrics,'positions':positions,'trace':stats['trace'],'row_retained':stats['row_retained'],'proxy':stats['proxy'],
                        'observer_seconds':obs.overhead if obs else 0.,'events':obs.events if obs else [],'layouts':obs.layouts if obs else {}}
                except Exception as error:
                    result={'index':index,'member':row['member'],'n':case.block_count,'arm':arm,'seconds':time.perf_counter()-begin,
                        'metrics':{'is_feasible':False,'cost':10.},'positions':positions,'error':repr(error),'events':obs.events if obs else [],'layouts':obs.layouts if obs else {}}
                    if args.check:raise
                finally:ARM.reset(token)
                con.execute('INSERT INTO records VALUES (?,?,?)',(index,arm,packed(result)));con.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('elapsed',str(elapsed())));con.commit()
            save(folder/'progress.json',{'status':'running','complete_cases':ordinal+1,'count':len(panel),'elapsed_seconds':elapsed()})
            if args.check or (ordinal+1)%10==0:print('audit cases',ordinal+1,'/',len(panel),'seconds',round(elapsed(),1),flush=True)
        records=[unpack(r[0]) for r in con.execute('SELECT payload FROM records')]
        summaries={}
        for arm in ARMS:
            selected=[r for r in records if r['arm']==arm];assert len(selected)==len(panel)
            q=np.asarray([r['metrics']['cost'] for r in selected]);t=np.asarray([r['seconds'] for r in selected]);n=np.asarray([r['n'] for r in selected]);w=np.exp((n-n.max())/12)
            summaries[arm]={'count':len(q),'hard_legal':sum(r['metrics']['is_feasible'] for r in selected),'errors':sum('error' in r for r in selected),
                'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'cvar95_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),'max_Q':float(q.max()),
                'seconds_mean':float(t.mean()),'seconds_p95':float(np.percentile(t,95)),'row_retained':sum(r.get('row_retained',False) for r in selected)}
        if args.check:
            assert all(s['hard_legal']==5 and s['errors']==0 for s in summaries.values())
            for r in records:
                if r['arm']=='full':
                    assert any(e['kind']=='acceptance' for e in r['events'])
                    assert any(e['kind']=='final_state' and e['depth']==1 for e in r['events'])
                    assert all(e['after_proxy']<=e['before_proxy'] for e in r['events'] if e['kind']=='acceptance')
        save(folder/'summary.json',{'status':'complete','summaries':summaries,'elapsed_seconds':elapsed(),'panel_size':len(panel),'test_read':False})
        con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close();print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
