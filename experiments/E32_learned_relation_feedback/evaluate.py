"""One joint terminal assessment per frozen best, balanced method order."""
from .resources import configure
RESOURCE=configure()
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from datasets.reference_layouts import registry,REGISTRY
from .cases import WEIGHTS,prepare_cache,predict,graph,panel_metadata,seed_all
from .model import GeometryModel
from .relations import RelationPolicy,encode_positions
from .policy import LearnedRelations
from .solver import optimize
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .data import load_rows
from .io import HERE,ROOT,sha,save,sources,snapshot,exclusive
from scoring.evaluation import evaluate_case

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
METHODS=('e25','implementation_control','contact_static','contact_feedback','full_feedback')
RULES={'contact_static':dict(feedback=False,axis=False,contact=True),'contact_feedback':dict(feedback=True,axis=False,contact=True),'full_feedback':dict(feedback=True,axis=True,contact=True)}

def new_confirmation_metadata():
    excluded={tuple(r['member']) for role in ('monitor','confirmation') for r in panel_metadata(role)}
    grouped={}
    for row in registry()['sets']['dev']['records']:
        if tuple(row['member']) not in excluded:grouped.setdefault(row['n'],[]).append(row)
    for rows in grouped.values():rows.sort(key=lambda r:hashlib.sha256(('E32.confirmation500.v1:'+json.dumps(r['member'])).encode()).hexdigest())
    chosen=[]
    for depth in range(max(map(len,grouped.values()))):
        for n in sorted(grouped):
            if depth<len(grouped[n]):chosen.append(grouped[n][depth])
            if len(chosen)==500:return chosen
    raise ValueError('Insufficient extra development members')

def summary(records):
    q=np.asarray([r['metrics']['cost'] for r in records]);n=np.asarray([r['n'] for r in records]);t=np.asarray([r['seconds'] for r in records]);w=np.exp((n-n.max())/12)
    out={'count':len(records),'hard_legal':sum(r['metrics']['is_feasible'] for r in records),'weighted_Q':float(np.average(q,weights=w)),'mean_Q':float(q.mean()),
        'p95_Q':float(np.percentile(q,95)),'cvar95_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),'max_Q':float(q.max()),'Q_gt_1_8':int((q>1.8).sum()),'Q_gt_2':int((q>2).sum()),
        'seconds_mean':float(t.mean()),'seconds_p95':float(np.percentile(t,95)),'seconds_max':float(t.max()),'errors':sum('error' in r for r in records),
        'row_retained':sum(r.get('row_retained',False) for r in records),'inference_seconds_mean':float(np.mean([r.get('inference_seconds',0) for r in records])),
        'policy_seconds_mean':float(np.mean([r.get('policy_seconds',0) for r in records]))}
    for key in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations'):
        values=[r['metrics'][key] for r in records if key in r['metrics']];out[key+'_mean']=float(np.mean(values)) if values else None
    out['rescued_cases']=sum(any(t['phase']=='full_packed_refine' for t in r.get('trace',[])) for r in records)
    return out

def paired(records,method):
    baseline=[r['methods']['e25'] for r in records];after=[r['methods'][method] for r in records]
    n=np.asarray([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    delta=np.asarray([b['metrics']['cost']-a['metrics']['cost'] for a,b in zip(baseline,after)])
    td=np.asarray([b['seconds']-a['seconds'] for a,b in zip(baseline,after)])
    groups=sorted({r['member'][0] for r in records});lookup={g:i for i,g in enumerate(groups)};cluster=np.asarray([lookup[r['member'][0]] for r in records])
    rng=np.random.default_rng(20261022);sample=rng.integers(0,len(groups),(2000,len(groups)))
    numerator=np.bincount(cluster,weights=w*delta);denominator=np.bincount(cluster,weights=w);counts=np.bincount(cluster);times=np.bincount(cluster,weights=td)
    boots=numerator[sample].sum(1)/denominator[sample].sum(1);tb=times[sample].sum(1)/counts[sample].sum(1)
    return {'weighted_Q_delta':float(np.average(delta,weights=w)),'weighted_Q_delta_95CI':np.percentile(boots,[2.5,97.5]).tolist(),
        'seconds_delta':float(td.mean()),'seconds_delta_95CI':np.percentile(tb,[2.5,97.5]).tolist(),'improved':int((delta<-1e-10).sum()),'regressed':int((delta>1e-10).sum()),
        'max_Q_regression':float(delta.max()),'bootstrap_shards':len(groups),'largest_regressions':[{'index':records[i]['index'],'member':records[i]['member'],'n':records[i]['n'],'delta':float(delta[i])} for i in np.argsort(-delta)[:10]]}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--tag',required=True)
    ap.add_argument('--panel',choices=('monitor','new_confirmation','smoke'),default='monitor');ap.add_argument('--method',choices=METHODS);args=ap.parse_args()
    begin=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();seed_all(0);warmup();reduce_warmup()
    if args.panel=='new_confirmation':
        if args.method not in RULES:raise ValueError('Confirm one already frozen learned method')
        contract=json.loads((HERE/'runs/frozen_candidate.json').read_text())
        if contract['checkpoint_sha256']!=sha(args.checkpoint) or contract['method']!=args.method:raise ValueError('Confirmation contract mismatch')
        for name,digest in contract['sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
        metadata=new_confirmation_metadata()
        reserved=json.loads((HERE/'runs/research_contract.json').read_text())
        if [list(r['member']) for r in metadata]!=reserved['E32_confirmation_members']:raise ValueError('Reserved confirmation membership changed')
        allrows=load_rows('dev');mapping={tuple(r['member']):r for r in allrows};rows=[mapping[tuple(r['member'])] for r in metadata]
    elif args.panel=='smoke':
        members=json.loads((HERE/'runs/20261022_interface_check5/config.json').read_text())['members']
        allrows=load_rows('train');mapping={tuple(r['member']):r for r in allrows};rows=[mapping[tuple(member)] for member in members]
    else:rows=prepare_cache()['rows']
    indices=list(range(500)) if args.panel!='smoke' else list(range(5))
    methods=['e25',args.method] if args.method else list(METHODS)
    base_state=torch.load(WEIGHTS,map_location='cpu',weights_only=False);base=GeometryModel(**base_state['model_config']).eval();base.load_state_dict(base_state['model'])
    example=graph(rows[0]['case'].solver_inputs)
    with torch.inference_mode():fastbase=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(base,(example,),strict=False)))
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state.get('implementation_check',False):
        if args.panel!='smoke':raise ValueError('Implementation-check weights cannot be a formal candidate')
    elif state['identity']['base_checkpoint_sha256']!=sha(WEIGHTS):raise ValueError('Frozen draft checkpoint changed')
    policy=RelationPolicy(**state['model_config']).eval();policy.load_state_dict(state['model'])
    d,_=predict(fastbase,rows[0]['case'].solver_inputs);d=encode_positions(torch.tensor(d,dtype=torch.float32)[None]/example['scale'][:,None,None],example)
    with torch.inference_mode():policy=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(policy,(example,d,d),strict=False)))
    folder=HERE/'runs'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    identity={'checkpoint_sha256':sha(args.checkpoint),'checkpoint':str(args.checkpoint.resolve()),'base_checkpoint_sha256':sha(WEIGHTS),'sources':sources(),
        'methods':methods,'rules':RULES,'policy':STEADY,'panel':args.panel,'members':[rows[i]['member'] for i in indices],'registry_sha256':sha(REGISTRY),
        'command':[sys.executable,*sys.argv],'resource':{k:v for k,v in RESOURCE.items() if k!='pid'},'selection_state':state['progress']['best_epoch'],'test_read':False}
    with exclusive(folder/'evaluation.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Evaluation identity changed')
        else:
            save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
            save(folder/'initialization.json',{'seconds':time.perf_counter()-begin,'resource':RESOURCE,'test_read':False})
        if (folder/'results.json').exists():print('Completed assessment reused',flush=True);return
        path=folder/'partial.json';records=json.loads(path.read_text()) if path.exists() else [];existing={r['index']:r for r in records}
        rng=np.random.default_rng(20261022);permutation=rng.permutation(len(indices))
        for ordinal,index in enumerate(indices):
            if (folder/'STOP').exists() or time.time()>=deadline:print('Paused at case boundary',flush=True);return
            row=rows[index];case=row['case']
            # Cyclic balanced schedule, each method occupies every position equally.
            shift=int(permutation[ordinal]%len(methods));order=methods[shift:]+methods[:shift]
            record=existing.get(index,{'index':index,'member':row['member'],'n':case.block_count,'order':order,'methods':{}})
            if index not in existing:records.append(record);existing[index]=record
            for method in record['order']:
                if method in record['methods']:continue
                seed_all(index);started=time.perf_counter();inference=0.
                try:
                    draft,inference=predict(base if method=='e25' else fastbase,case.solver_inputs)
                    callback=LearnedRelations(policy,case.solver_inputs,draft,**RULES[method]) if method in RULES else None
                    x,stats=optimize(draft,case.solver_inputs,**STEADY,native_reduce=method!='e25',relation_policy=callback);elapsed=time.perf_counter()-started
                    metrics=evaluate_case(case,x.tolist(),elapsed)['metrics'];trace=stats['trace']
                    result={'n':case.block_count,'metrics':metrics,'seconds':elapsed,'positions':x.tolist(),'trace':trace,'inference_seconds':inference,
                        'policy_seconds':sum(t.get('seconds',0) for t in trace if t['phase']=='learned_relations'),'policy_inferences':0 if callback is None else callback.inferences,'row_retained':stats['row_retained']}
                except Exception as error:result={'n':case.block_count,'metrics':{'cost':10.,'is_feasible':False},'seconds':time.perf_counter()-started,'error':repr(error)}
                record['methods'][method]=result
            save(path,records)
            if (ordinal+1)%25==0 or args.panel=='smoke':print('complete',ordinal+1,'/',len(indices),{m:round(record['methods'][m]['metrics']['cost'],4) for m in methods},flush=True)
        result={'status':'complete','summaries':{m:summary([r['methods'][m] for r in records]) for m in methods},'paired':{m:paired(records,m) for m in methods if m!='e25'},'records':records,'test_read':False}
        save(folder/'results.json',result);save(folder/'summary.json',{k:v for k,v in result.items() if k!='records'});path.unlink();print(json.dumps(result['summaries']),flush=True)

if __name__=='__main__':main()
