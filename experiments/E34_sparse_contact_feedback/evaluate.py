"""One sparse endpoint; reuse immutable dense controls, confirm one family method."""
from .resources import configure
RESOURCE=configure()
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .cases import WEIGHTS,prepare_cache,predict,graph,seed_all
from .model import GeometryModel
from .sparse import SparseContacts
from .check import load_models,CHECKPOINT,PARENT,STEADY
from .solver import optimize
from .data import load_rows
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .io import HERE,ROOT,save,sha,sources,snapshot,exclusive
from scoring.evaluation import evaluate_case

METHODS=('sparse_static','sparse_feedback')

def summary(records):
    q=np.asarray([r['metrics']['cost'] for r in records]);t=np.asarray([r['seconds'] for r in records]);n=np.asarray([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    out={'count':len(q),'hard_legal':sum(r['metrics']['is_feasible'] for r in records),'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),'p95_Q':float(np.percentile(q,95)),
        'cvar95_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),'max_Q':float(q.max()),'seconds_mean':float(t.mean()),'seconds_p95':float(np.percentile(t,95)),'seconds_max':float(t.max()),
        'errors':sum('error' in r for r in records),'row_retained':sum(r.get('row_retained',False) for r in records),'dense_fallbacks':sum(r.get('dense_fallbacks',0) for r in records),
        'policy_seconds_mean':float(np.mean([r.get('policy_seconds',0) for r in records])),'rescued_cases':sum(any(t['phase']=='full_packed_refine' for t in r.get('trace',[])) for r in records)}
    for key in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations'):
        values=[r['metrics'][key] for r in records if key in r['metrics']];out[key+'_mean']=float(np.mean(values)) if values else None
    return out

def paired(records,method,controls,control_method):
    before=[controls[tuple(r['member'])]['methods'][control_method] for r in records];after=[r['methods'][method] for r in records];n=np.asarray([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    delta=np.asarray([b['metrics']['cost']-a['metrics']['cost'] for a,b in zip(before,after)]);td=np.asarray([b['seconds']-a['seconds'] for a,b in zip(before,after)])
    groups=sorted({r['member'][0] for r in records});lookup={g:i for i,g in enumerate(groups)};cluster=np.asarray([lookup[r['member'][0]] for r in records]);rng=np.random.default_rng(20261022);samples=rng.integers(0,len(groups),(2000,len(groups)))
    num=np.bincount(cluster,weights=w*delta);den=np.bincount(cluster,weights=w);time_sum=np.bincount(cluster,weights=td);counts=np.bincount(cluster)
    boot=num[samples].sum(1)/den[samples].sum(1);tb=time_sum[samples].sum(1)/counts[samples].sum(1)
    return {'weighted_Q_delta':float(np.average(delta,weights=w)),'weighted_Q_delta_95CI':np.percentile(boot,[2.5,97.5]).tolist(),'seconds_delta':float(td.mean()),'seconds_delta_95CI':np.percentile(tb,[2.5,97.5]).tolist(),
        'max_Q_regression':float(delta.max()),'improved':int((delta<-1e-10).sum()),'regressed':int((delta>1e-10).sum()),'bootstrap_shards':len(groups),
        'largest_regressions':[{'member':records[i]['member'],'index':records[i]['index'],'n':records[i]['n'],'delta':float(delta[i])} for i in np.argsort(-delta)[:10]]}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',required=True);ap.add_argument('--panel',choices=('monitor','new_confirmation'),default='monitor');ap.add_argument('--method',choices=METHODS);args=ap.parse_args()
    begin=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();seed_all(0);warmup();reduce_warmup()
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if state['identity']['arm']!='replay' or state['identity']['seed']!=0 or state['identity']['base_checkpoint_sha256']!=sha(WEIGHTS):raise ValueError('Registered parent identity changed')
    if args.panel=='new_confirmation':
        if args.method is None:raise ValueError('Confirm only the frozen family method')
        contract=json.loads((HERE/'runs/frozen_candidate.json').read_text())
        if contract['checkpoint_sha256']!=sha(CHECKPOINT) or contract['method']!=args.method:raise ValueError('Family method contract changed')
        for name,digest in contract['sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen sparse source changed: '+name)
        reservation=PARENT/'runs/research_contract.json'
        if sha(reservation)!=contract['reservation_sha256']:raise ValueError('Family panel reservation changed')
        members=json.loads(reservation.read_text())['E32_confirmation_members'];allrows=load_rows('dev');mapping={tuple(r['member']):r for r in allrows};rows=[mapping[tuple(m)] for m in members];methods=['e25',args.method]
        controls=None;control_path=None
    else:
        gate=json.loads((HERE/'runs/20261022_sparse_check5/results.json').read_text())
        if not gate['continue_to_terminal'] or gate['checkpoint_sha256']!=sha(CHECKPOINT):raise ValueError('Sparse mechanism gate failed or weights changed')
        rows=prepare_cache()['rows'];methods=[args.method] if args.method else list(METHODS)
        control_path=PARENT/'runs/20261022_replay_seed0_terminal500/results.json';old=json.loads(control_path.read_text());controls={tuple(r['member']):r for r in old['records']}
        config=json.loads((control_path.parent/'config.json').read_text())
        if config['checkpoint_sha256']!=sha(CHECKPOINT):raise ValueError('Dense controls use different weights')
        if set(controls)!={tuple(r['member']) for r in rows}:raise ValueError('Sparse/dense panels differ')
    parent=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**parent['model_config']).eval();base.load_state_dict(parent['model'])
    example=graph(rows[0]['case'].solver_inputs)
    with torch.inference_mode():fastbase=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(base,(example,),strict=False)))
    draft,_=predict(fastbase,rows[0]['case'].solver_inputs);dense,sparse=load_models(CHECKPOINT,rows[0]['case'].solver_inputs,draft)
    folder=HERE/'runs'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    identity={'sources':sources(),'checkpoint_sha256':sha(CHECKPOINT),'parent_checkpoint_sha256':sha(WEIGHTS),'members':[r['member'] for r in rows],'methods':methods,'panel':args.panel,
        'control_result':None if control_path is None else str(control_path.relative_to(ROOT)),'control_result_sha256':None if control_path is None else sha(control_path),'control_timing':'same hardware/resources but earlier E32 run; not contemporaneous alternating controls' if control_path else 'contemporaneous alternating E25/candidate',
        'command':[sys.executable,*sys.argv],'resource':{k:v for k,v in RESOURCE.items() if k!='pid'},'test_read':False}
    with exclusive(folder/'evaluation.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Sparse endpoint identity changed')
        else:
            save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity);save(folder/'initialization.json',{'seconds':time.perf_counter()-begin,'resource':RESOURCE})
        if (folder/'results.json').exists():print('Completed sparse endpoint reused',flush=True);return
        path=folder/'partial.json';records=json.loads(path.read_text()) if path.exists() else [];existing={r['index']:r for r in records};rng=np.random.default_rng(20261022);permutation=rng.permutation(500)
        for index,row in enumerate(rows):
            if (folder/'STOP').exists() or time.time()>=deadline:print('Paused at case boundary',flush=True);return
            case=row['case'];shift=int(permutation[index]%len(methods));order=methods[shift:]+methods[:shift];record=existing.get(index,{'index':index,'member':row['member'],'n':case.block_count,'order':order,'methods':{}})
            if index not in existing:records.append(record);existing[index]=record
            for method in record['order']:
                if method in record['methods']:continue
                seed_all(index);started=time.perf_counter()
                try:
                    draft,_=predict(base if method=='e25' else fastbase,case.solver_inputs)
                    callback=None if method=='e25' else SparseContacts(sparse,dense,case.solver_inputs,draft,feedback=method=='sparse_feedback')
                    policy=dict(STEADY)
                    if method=='e25':policy['native_reduce']=False
                    x,stats=optimize(draft,case.solver_inputs,**policy,relation_policy=callback);elapsed=time.perf_counter()-started
                    result={'n':case.block_count,'seconds':elapsed,'metrics':evaluate_case(case,x.tolist(),elapsed)['metrics'],'positions':x.tolist(),'trace':stats['trace'],'row_retained':stats['row_retained'],
                        'dense_fallbacks':0 if callback is None else callback.dense_fallbacks,'policy_seconds':sum(t.get('seconds',0) for t in stats['trace'] if t['phase']=='learned_relations')}
                except Exception as error:result={'n':case.block_count,'seconds':time.perf_counter()-started,'metrics':{'cost':10.,'is_feasible':False},'error':repr(error)}
                record['methods'][method]=result
            save(path,records)
            if (index+1)%25==0:print('complete',index+1,'/',len(rows),{m:round(record['methods'][m]['metrics']['cost'],4) for m in methods},flush=True)
        reference=controls if controls is not None else {tuple(r['member']):r for r in records}
        comparisons={}
        for method in methods:
            if method=='e25':continue
            comparisons[method]={'against_e25':paired(records,method,reference,'e25')}
            if controls is not None:comparisons[method]['against_dense']=paired(records,method,reference,'contact_feedback' if method=='sparse_feedback' else 'contact_static')
        result={'status':'complete','summaries':{m:summary([r['methods'][m] for r in records]) for m in methods},'paired':comparisons,'records':records,'test_read':False}
        save(folder/'results.json',result);save(folder/'summary.json',{k:v for k,v in result.items() if k!='records'});path.unlink();print(json.dumps(result['summaries']),flush=True)

if __name__=='__main__':main()
