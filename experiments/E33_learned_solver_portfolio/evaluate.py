"""Single frozen gate endpoint, all comparisons and runtime included."""
from .resources import configure
RESOURCE=configure()
import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib
import numpy as np
import torch
from .cases import prepare_cache,seed_all
from .routing import Gate,prefix,statistics
from .inference import load_gate
from .prepare import expert,load_experts,WEIGHTS,GEOMETRY,RELATION,ACTION_COUNT
from .calibrate import select
from .data import load_rows
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .io import HERE,ROOT,save,sha,sources,snapshot,exclusive
from .inspect_labels import metrics as compact_metrics
from scoring.evaluation import evaluate_case

METHODS=('e25','fast','geometry','relation','n_rule','gate')

def summary(records):
    q=np.asarray([r['metrics']['cost'] for r in records]);t=np.asarray([r['seconds'] for r in records]);n=np.asarray([r['n'] for r in records]);out=compact_metrics(q,t,n)
    out.update(hard_legal=sum(r['metrics']['is_feasible'] for r in records),p95_Q=float(np.percentile(q,95)),seconds_p95=float(np.percentile(t,95)),errors=sum('error' in r for r in records),
        Q_gt_2=int((q>2).sum()),row_retained=sum(r.get('row_retained',False) for r in records),actions=np.bincount([r.get('action',0) for r in records],minlength=ACTION_COUNT).tolist(),
        rescued_cases=sum(any(t['phase']=='full_packed_refine' for t in r.get('trace',[])) for r in records))
    for key in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations'):
        values=[r['metrics'][key] for r in records if key in r['metrics']];out[key+'_mean']=float(np.mean(values)) if values else None
    return out

def paired(records,method):
    a=[r['methods']['e25'] for r in records];b=[r['methods'][method] for r in records];n=np.asarray([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    d=np.asarray([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(a,b)]);td=np.asarray([y['seconds']-x['seconds'] for x,y in zip(a,b)])
    groups=sorted({r['member'][0] for r in records});lookup={s:i for i,s in enumerate(groups)};cluster=np.asarray([lookup[r['member'][0]] for r in records]);rng=np.random.default_rng(20261022);sample=rng.integers(0,len(groups),(2000,len(groups)))
    effects=np.bincount(cluster,weights=w*d);weights=np.bincount(cluster,weights=w);te=np.bincount(cluster,weights=td);counts=np.bincount(cluster)
    boot=effects[sample].sum(1)/weights[sample].sum(1);tb=te[sample].sum(1)/counts[sample].sum(1)
    return {'weighted_Q_delta':float(np.average(d,weights=w)),'weighted_Q_delta_95CI':np.percentile(boot,[2.5,97.5]).tolist(),'seconds_delta':float(td.mean()),'seconds_delta_95CI':np.percentile(tb,[2.5,97.5]).tolist(),
        'improved':int((d<-1e-10).sum()),'regressed':int((d>1e-10).sum()),'max_Q_regression':float(d.max()),'bootstrap_shards':len(groups),
        'largest_regressions':[{'index':records[i]['index'],'member':records[i]['member'],'n':records[i]['n'],'delta':float(d[i])} for i in np.argsort(-d)[:10]]}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--tag',required=True);ap.add_argument('--panel',choices=('monitor','new_confirmation'),default='monitor');args=ap.parse_args()
    started=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();seed_all(0);warmup();reduce_warmup()
    contract_path=args.checkpoint.parents[1]/'frozen.json';contract=json.loads(contract_path.read_text())
    if sha(args.checkpoint)!=contract['checkpoint_sha256']:raise ValueError('Gate checkpoint changed')
    if sha(WEIGHTS)!=contract['expert_weights']['E25'] or sha(GEOMETRY)!=contract['expert_weights']['E30'] or sha(RELATION)!=contract['expert_weights']['E32_relation']:raise ValueError('Expert weights changed')
    for name,digest in contract['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Gate method source changed: '+name)
    if args.panel=='new_confirmation':
        panel_path=ROOT/'experiments/E32_learned_relation_feedback/runs/research_contract.json';members=json.loads(panel_path.read_text())['E33_reserved_confirmation_members']
        if sha(panel_path)!=contract['confirmation_reservation_sha256']:raise ValueError('Confirmation reservation changed')
        allrows=load_rows('dev');mapping={tuple(r['member']):r for r in allrows};rows=[mapping[tuple(m)] for m in members];methods=['e25',contract.get('deployment_method','gate')]
    else:rows=prepare_cache()['rows'];methods=list(METHODS)
    base,geometry,relation=load_experts(rows[0]['case'].solver_inputs);gate,state=load_gate(args.checkpoint)
    folder=HERE/'runs'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    identity={'sources':sources(),'frozen_contract_sha256':sha(contract_path),'members':[r['member'] for r in rows],'panel':args.panel,'methods':methods,'command':[sys.executable,*sys.argv],
        'resources':{k:v for k,v in RESOURCE.items() if k!='pid'},'test_read':False}
    with exclusive(folder/'evaluation.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Endpoint identity changed')
        else:
            save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);save(folder/'initialization.json',{'seconds':time.perf_counter()-started,'resource':RESOURCE})
        if (folder/'results.json').exists():print('Completed endpoint reused',flush=True);return
        path=folder/'partial.sqlite3';progress_db=sqlite3.connect(path);progress_db.execute('PRAGMA journal_mode=WAL');progress_db.execute('PRAGMA synchronous=FULL')
        progress_db.execute('CREATE TABLE IF NOT EXISTS cases (idx INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        records=[json.loads(zlib.decompress(row[0])) for row in progress_db.execute('SELECT payload FROM cases ORDER BY idx')];existing={r['index']:r for r in records};rng=np.random.default_rng(20261022);permutation=rng.permutation(500)
        for index,row in enumerate(rows):
            if (folder/'STOP').exists() or time.time()>=deadline:progress_db.close();print('Paused at case boundary',flush=True);return
            case=row['case'];shift=int(permutation[index]%len(methods));order=methods[shift:]+methods[:shift]
            record=existing.get(index,{'index':index,'member':row['member'],'n':case.block_count,'order':order,'methods':{}})
            if index in existing and all(method in record['methods'] for method in methods):continue
            if index not in existing:records.append(record);existing[index]=record
            for method in record['order']:
                if method in record['methods']:continue
                seed_all(index);begin=time.perf_counter();draft=None;confidence=None
                try:
                    if method=='gate' and contract['threshold']>1.:
                        action=0
                    elif method=='gate':
                        draft,hidden,_=prefix(base,case.solver_inputs);features=statistics(case.solver_inputs,draft,hidden)
                        with torch.inference_mode():probability=gate(torch.from_numpy(features)[None])[0].softmax(-1).numpy()
                        action=int(select(probability,contract['threshold'])[0]);confidence=float(probability.max())
                        if action==2:draft=None
                    elif method=='n_rule':
                        rule=next(v for v in contract['simple_n_rule'] if v['lo']<=case.block_count<v['hi']);action=rule['action']
                    else:action={'e25':0,'fast':1,'geometry':2,'relation':3}[method]
                    x,stats,_=expert(action,case,base,geometry,relation,draft=draft);elapsed=time.perf_counter()-begin
                    result={'n':case.block_count,'seconds':elapsed,'action':action,'confidence':confidence,'positions':x.tolist(),'metrics':evaluate_case(case,x.tolist(),elapsed)['metrics'],'trace':stats['trace'],'row_retained':stats['row_retained'],
                        'ensemble_agreement':bool(gate.last_consensus[0]) if method=='gate' and getattr(gate,'last_consensus',None) is not None else None}
                except Exception as error:result={'n':case.block_count,'seconds':time.perf_counter()-begin,'metrics':{'cost':10.,'is_feasible':False},'error':repr(error)}
                record['methods'][method]=result
            payload=zlib.compress(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),level=1)
            progress_db.execute('INSERT OR REPLACE INTO cases VALUES (?,?)',(index,payload));progress_db.commit()
            if (index+1)%25==0:print('complete',index+1,'/',len(rows),{m:round(record['methods'][m]['metrics']['cost'],4) for m in methods},flush=True)
        result={'status':'complete','summaries':{m:summary([r['methods'][m] for r in records]) for m in methods},'paired':{m:paired(records,m) for m in methods if m!='e25'},'records':records,'test_read':False}
        save(folder/'results.json',result);save(folder/'summary.json',{k:v for k,v in result.items() if k!='records'});progress_db.close();path.unlink();print(json.dumps(result['summaries']),flush=True)

if __name__=='__main__':main()
