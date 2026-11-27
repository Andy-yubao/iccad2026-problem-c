"""One frozen candidate on preselected, non-monitor development500; four workers."""
from . import resources
import concurrent.futures as futures
import multiprocessing as mp
from pathlib import Path
import json
import sqlite3
import sys
import time
import hashlib
import numpy as np
from .io import ROOT,HERE,save,sha,sources,snapshot,exclusive
from .run_stage2 import cpu_slots,packed,unpack
from .cases import panel_metadata,WEIGHTS,BASE

ARMS=('reference','control','pivot_repair')
MODEL=None


def initialize(queue):
    global MODEL
    import os
    os.environ['E36_CPU_SLOT']=str(queue.get(timeout=30));resource=resources.configure()
    import torch
    from .cases import public_cases,seed_all
    from .model import GeometryModel
    from .canonical_native import warmup
    from . import run_stage2
    seed_all(0,1);cases,metadata,_=public_cases('confirmation');warmup()
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=False);MODEL=GeometryModel(**state['model_config']).eval();MODEL.load_state_dict(state['model'])
    run_stage2.STATE=(resource,{},[{'case':case,'member':r['member'],'n':case.block_count} for case,r in zip(cases,metadata)])


def execute(index,arms,deadline):
    from . import run_stage2
    from .cases import predict
    if time.monotonic()>=deadline or (HERE/'runs/STOP').exists():return {'records':[]}
    resource,saved,rows=run_stage2.STATE;case=rows[index]['case'];draft,inference=predict(MODEL,case.solver_inputs)
    digest=hashlib.sha256(draft.tobytes()).hexdigest()
    saved[index]={'index':index,'member':rows[index]['member'],'n':case.block_count,
        'events':[{'kind':'initial_state','depth':1,'draft':'draft'}],'layouts':{'draft':draft},'inference_seconds':inference}
    records=run_stage2.execute(index,arms,deadline)
    assert hashlib.sha256(draft.tobytes()).hexdigest()==digest
    for r in records:r['draft_sha256']=digest
    saved.pop(index,None)
    return {'records':records,'draft':draft,'inference_seconds':inference,'member':rows[index]['member'],'draft_sha256':digest}


def main():
    start=time.perf_counter();folder=HERE/'runs/20261022_confirmation500';folder.mkdir(parents=True,exist_ok=True)
    prior=json.loads((HERE/'runs/budget.json').read_text());previous=json.loads((HERE/'runs/20261022_stage2_paired100/config.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in previous['sources'].items()),'Frozen numerical pipeline changed after100'
    assert sha(WEIGHTS)=='3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    assert sha(HERE/'model.py')==sha(BASE/'model.py')
    metadata=panel_metadata('confirmation');monitor=panel_metadata('monitor');assert len(metadata)==500
    assert not ({tuple(r['member']) for r in metadata}&{tuple(r['member']) for r in monitor})
    slots=cpu_slots()
    identity={'sources':sources(),'panel':list(range(500)),'members':[r['member'] for r in metadata],'arms':list(ARMS),
        'weights_sha256':sha(WEIGHTS),'input_cache_manifest_sha256':sha(BASE/'runs/prepared_v5_e10_cache_v1/dev/manifest.json'),
        'workers':4,'threads_per_worker':1,'cpu_slots':slots,'max_minutes_cumulative':prior['max_minutes'],
        'command':[sys.executable,*sys.argv],'panel_scope':'preselected E20-rule non-monitor development500; other historical studies may have used it; not unseen test',
        'nonoverlap_with_discovery':True,'model_training':False,'test_read':False,
        'timing':'measured postprocessor wall/process CPU seconds; one shared model inference per case reported separately',
        'candidate_frozen_after':'20261022_stage2_paired100','numerical_sources_match_frozen100':True}
    with exclusive(HERE/'runs/research.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Run identity changed')
        else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
        if (folder/'summary.json').exists():print('Already complete');return
        elapsed=lambda:prior['elapsed_seconds']+time.perf_counter()-start
        if elapsed()>=prior['max_minutes']*60:print('Cumulative budget exhausted');return
        deadline=time.monotonic()+prior['max_minutes']*60-elapsed()
        db=sqlite3.connect(folder/'outcomes.sqlite3');db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS records(idx INTEGER NOT NULL,arm TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(idx,arm))')
        db.execute('CREATE TABLE IF NOT EXISTS drafts(idx INTEGER PRIMARY KEY,payload BLOB NOT NULL)');db.commit()
        done={(i,a) for i,a in db.execute('SELECT idx,arm FROM records')};rng=np.random.default_rng(20261024)
        orders=[rng.permutation(3).tolist() for _ in range(167)];jobs=[]
        for index in range(500):
            order=orders[index//3];shift=index%3;order=order[shift:]+order[:shift]
            arms=[ARMS[j] for j in order if (index,ARMS[j]) not in done]
            if arms:jobs.append((index,arms))
        context=mp.get_context('spawn');queue=context.Queue()
        for slot in slots:queue.put(slot)
        completed=0
        with futures.ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
            cursor=0;pending={}
            def dispatch():
                nonlocal cursor
                while len(pending)<4 and cursor<len(jobs) and time.monotonic()<deadline and not (HERE/'runs/STOP').exists():
                    i,arms=jobs[cursor];cursor+=1;pending[pool.submit(execute,i,arms,deadline)]=i
            dispatch()
            while pending:
                finished,_=futures.wait(pending,timeout=1,return_when=futures.FIRST_COMPLETED)
                for f in finished:
                    i=pending.pop(f);result=f.result()
                    for r in result['records']:db.execute('INSERT INTO records VALUES(?,?,?)',(i,r['arm'],packed(r)))
                    if 'draft' in result:
                        old=db.execute('SELECT payload FROM drafts WHERE idx=?',(i,)).fetchone()
                        if old:assert unpack(old[0])['draft_sha256']==result['draft_sha256']
                        else:db.execute('INSERT INTO drafts VALUES(?,?)',(i,packed({k:result[k] for k in ('draft','member','inference_seconds','draft_sha256')})))
                    db.commit();completed+=1
                    save(HERE/'runs/budget.json',{'elapsed_seconds':elapsed(),'max_minutes':prior['max_minutes']})
                    save(folder/'progress.json',{'completed_cases':completed,'scheduled_cases':cursor,'count':500,'records':db.execute('SELECT COUNT(*) FROM records').fetchone()[0],'cumulative_elapsed_seconds':elapsed()})
                    if completed%50==0:print('confirmation cases',completed,'/500 cumulative seconds',round(elapsed(),1),flush=True)
                dispatch()
        records=[unpack(row[0]) for row in db.execute('SELECT payload FROM records')]
        if len(records)!=1500:
            save(folder/'stop.json',{'status':'budget_or_user_stop','records':len(records),'cumulative_elapsed_seconds':elapsed()});save(HERE/'runs/budget.json',{'elapsed_seconds':elapsed(),'max_minutes':prior['max_minutes']});db.close();return
        summaries={}
        for arm in ARMS:
            rr=[r for r in records if r['arm']==arm];q=np.array([r['metrics']['cost'] for r in rr]);n=np.array([r['n'] for r in rr]);w=np.exp((n-n.max())/12);tt=np.array([r['seconds'] for r in rr])
            summaries[arm]={'count':500,'hard_legal':sum(r['metrics']['is_feasible'] for r in rr),'errors':sum('error' in r for r in rr),
                'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),
                'cvar95_Q':float(np.sort(q)[-25:].mean()),'max_Q':float(q.max()),'seconds_mean':float(tt.mean()),'seconds_p95':float(np.percentile(tt,95)),
                'cpu_seconds_mean':float(np.mean([r['cpu_seconds'] for r in rr])),
                'shared_inference_seconds_mean':float(np.mean([r['cached_inference_seconds'] for r in rr])),
                'estimated_inference_plus_post_seconds_mean':float(np.mean([r['seconds']+r['cached_inference_seconds'] for r in rr])),
                'root_shapes_mean':float(np.mean([r.get('root_shape_calls',0) for r in rr])),
                'final_mean_center_move':float(np.mean([r['draft_to_final']['mean_center_move'] for r in rr if 'draft_to_final' in r])),
                'pivot_cases':sum(r.get('pivot_calls',0)>0 for r in rr),'pivot_attempts':sum(r.get('pivot_calls',0) for r in rr),
                'rescue_cases':sum(r.get('rescue_calls',0)>0 for r in rr),
                'wall_above_2x_cpu_cases':sum(r['seconds']>max(.05,2*r['cpu_seconds']) for r in rr)}
        result={'status':'complete','gate_passed':all(s['hard_legal']==500 and s['errors']==0 for s in summaries.values()),'summaries':summaries,
            'record_count':1500,'draft_count':db.execute('SELECT COUNT(*) FROM drafts').fetchone()[0],
            'batch_wall_seconds':time.perf_counter()-start,'cumulative_elapsed_seconds':elapsed(),'workers':4,'threads_per_worker':1,'test_read':False,'model_training':False}
        save(folder/'summary.json',result);save(folder/'progress.json',{'status':'complete','cases':500,'records':1500,'cumulative_elapsed_seconds':elapsed()})
        save(HERE/'runs/budget.json',{'elapsed_seconds':elapsed(),'max_minutes':prior['max_minutes']});db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
