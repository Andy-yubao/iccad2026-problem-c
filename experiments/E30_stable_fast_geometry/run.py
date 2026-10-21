"""Sequential, resumable direct CPU evaluation under two execution slots."""
from .resources import configure
RESOURCE=configure()
import argparse
import json
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import torch
from .cases import prepare_cache,seed_all,predict,WEIGHTS
from .model import GeometryModel
from .io import HERE,ROOT,save,sources,snapshot,sha,exclusive
from .solver import optimize
from .canonical_native import warmup
from scoring.evaluation import evaluate_case

STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
CONFIGS={'steady':STEADY,
    'native':dict(STEADY,native_reduce=True),
    'push':dict(STEADY,native_reduce=True,certified_push=True),
    'tail':dict(STEADY,native_reduce=True,tail_refine=True),
    'budget':dict(STEADY,native_reduce=True,fast_budget=True),
    'warm':dict(STEADY,native_reduce=True,fast_budget=True,warm_project=True),
    'progress':dict(STEADY,native_reduce=True,fast_budget=True,early_projection_steps=2,shape_limit=8,plateau_tol=2e-4,plateau_min_legal=2)}

def summary(records):
    q=np.array([r['metrics']['cost'] for r in records]);n=np.array([r['n'] for r in records]);w=np.exp((n-n.max())/12);t=np.array([r['seconds'] for r in records])
    count=max(1,int(np.ceil(.05*len(q))))
    out={'count':len(q),'hard_legal':sum(r['metrics']['is_feasible'] for r in records),
         'mean_Q':float(q.mean()),'weighted_Q':float(np.average(q,weights=w)),
         'p95_Q':float(np.percentile(q,95)),'cvar95_Q':float(np.sort(q)[-count:].mean()),'max_Q':float(q.max()),
         'Q_gt_1_8':int((q>1.8).sum()),'Q_gt_2':int((q>2).sum()),
         'seconds_mean':float(t.mean()),'seconds_p95':float(np.percentile(t,95)),'seconds_max':float(t.max()),
         'inference_seconds_mean':float(np.mean([r.get('inference_seconds',0) for r in records])),
         'row_fallbacks':sum(r.get('row',False) for r in records),'errors':sum('error' in r for r in records)}
    for key in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations'):
        values=[r['metrics'][key] for r in records if key in r['metrics']]
        out[key+'_mean']=float(np.mean(values)) if values else None
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--configs',nargs='+',choices=list(CONFIGS),default=['steady'])
    ap.add_argument('--count',type=int,choices=[5,100,500],default=500);ap.add_argument('--tag',required=True)
    ap.add_argument('--panel',choices=['monitor','confirmation'],default='monitor');ap.add_argument('--checkpoint',type=Path,default=WEIGHTS)
    ap.add_argument('--jit',action='store_true',help='Freeze and optimize the same inference graph before resident timing')
    args=ap.parse_args();seed_all();init=time.perf_counter();warmup()
    from .native_reduce import warmup as reduce_warmup
    reduce_warmup();cached=prepare_cache(args.panel)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    if args.jit:
        from .cases import graph
        with torch.inference_mode():
            example=graph(cached['rows'][0]['case'].solver_inputs)
            model=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(model,(example,),strict=False)))
    initialization=time.perf_counter()-init;rows=cached['rows'];order=sorted(range(500),key=lambda i:rows[i]['case'].block_count)
    panel=list(range(500)) if args.count==500 else ([order[i] for i in range(2,500,5)] if args.count==100 else [order[i] for i in (5,125,250,375,495)])
    for name in args.configs:
        folder=HERE/'runs'/f'{args.tag}_{name}_{args.panel}{args.count}';folder.mkdir(parents=True,exist_ok=True)
        identity={'config':CONFIGS[name],'panel':panel,'members':[rows[i]['member'] for i in panel],
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint),'sources':sources(),
            'registry_sha256':cached['registry_sha256'],'workers':1,'threads_per_worker':2,'direct':True,'jit':args.jit,
            'command':sys.argv,'python':sys.executable,'cpu_affinity':RESOURCE['affinity'],
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()}
        with exclusive(folder/'run.lock'):
            if (folder/'config.json').exists():
                if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Run identity changed; use a new tag')
            else:
                save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
                save(folder/'initialization.json',{'seconds':initialization,'resource':RESOURCE,'test_read':False})
            if (folder/'results.json').exists():continue
            path=folder/'partial.json';done=json.loads(path.read_text()) if path.exists() else [];completed={r['index'] for r in done}
            for index in panel:
                if index in completed:continue
                if (folder/'STOP').exists():print('Paused:',folder,flush=True);return
                row=rows[index];case=row['case'];seed_all(index);begin=time.perf_counter();inference=0.
                try:
                    draft,inference=predict(model,case.solver_inputs)
                    x,stats=optimize(draft,case.solver_inputs,**CONFIGS[name]);elapsed=time.perf_counter()-begin
                    metrics=evaluate_case(case,x.tolist(),elapsed)['metrics']
                    record={'index':index,'member':row['member'],'n':case.block_count,'seconds':elapsed,
                        'metrics':metrics,'row':stats['row_retained'],'positions':x.tolist(),'trace':stats['trace'],
                        'inference_seconds':inference,'postprocess_seconds':stats['seconds'],'proxy':stats['proxy']}
                except Exception as error:
                    record={'index':index,'member':row['member'],'n':case.block_count,'seconds':time.perf_counter()-begin,
                        'metrics':{'is_feasible':False,'cost':10.},'error':repr(error),'inference_seconds':inference}
                done.append(record);save(path,done)
                if len(done)%25==0 or args.count==5:print(name,len(done),'last Q',record['metrics']['cost'],flush=True)
            result={'summary':summary(done),'records':sorted(done,key=lambda r:r['index']),'test_read':False}
            save(folder/'results.json',result);path.unlink();print(name,json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
