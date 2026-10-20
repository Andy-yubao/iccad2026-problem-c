"""Resumable fixed-panel candidate evaluation, 2 processes x 2 threads."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing as mp
import os
import sys
import platform
import time
import numpy as np
from .cases import prepare_cache,seed_all,predict,WEIGHTS
from .model import GeometryModel
from .io import HERE,save,sources,snapshot,exclusive
from .solver import optimize
from .canonical_native import warmup
from scoring.evaluation import evaluate_case

CONFIGS={
 'fast':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,robust_seeds=True,certified=True,snap=1e-4),
 'lp_gate':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,snap=1e-4),
 'aligned':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,draft_proxy=True,snap=1e-4),
 'qp_gate':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)}

CONFIGS['qp_first']={**CONFIGS['qp_gate'],'warm_project':True}
CONFIGS['public_stop']={**CONFIGS['qp_gate'],'early_public':True}
CONFIGS['steady']={**CONFIGS['qp_gate'],'fast_budget':False}
CONFIGS['selection']={**CONFIGS['qp_gate'],'selection_draft':True}
CONFIGS['steady_select']={**CONFIGS['steady'],'selection_draft':True}


def initialize(direct=False,panel='monitor'):
    global rows,model,resident_initialization
    import torch
    started=time.perf_counter();warmup()
    seed_all(0,2);rows=prepare_cache(panel)['rows'];model=None
    if direct:
        state=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
        model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
    resident_initialization=time.perf_counter()-started


def evaluate(index,config):
    row=rows[index];case=row['case'];seed_all(index,2);begin=time.perf_counter()
    try:
        if model is not None:
            draft,inference_seconds=predict(model,case.solver_inputs)
        else:draft,inference_seconds=row['draft'],row['inference_seconds']
        x,stats=optimize(draft,case.solver_inputs,**CONFIGS[config]);elapsed=time.perf_counter()-begin+(inference_seconds if model is None else 0.)
        metrics=evaluate_case(case,x.tolist(),elapsed)['metrics']
        record={'index':index,'member':row['member'],'n':case.block_count,'seconds':elapsed,
            'metrics':metrics,'row':stats['row_retained'],'positions':x.tolist(),'trace':stats['trace'],
            'inference_seconds':inference_seconds,'postprocess_seconds':stats['seconds'],
            'worker_pid':os.getpid(),'resident_initialization_seconds':resident_initialization}
    except Exception as error:
        record={'index':index,'member':row['member'],'n':case.block_count,'seconds':time.perf_counter()-begin+(row['inference_seconds'] if model is None else 0.),
            'metrics':{'is_feasible':False,'cost':10.},'error':repr(error)}
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--configs',nargs='+',choices=list(CONFIGS),default=list(CONFIGS));ap.add_argument('--count',type=int,choices=[100,500],default=100)
    ap.add_argument('--tag',required=True);ap.add_argument('--direct',action='store_true',help='Recompute public features, network and decode inside each timed call')
    ap.add_argument('--panel',choices=['monitor','confirmation'],default='monitor');args=ap.parse_args()
    if args.panel=='confirmation' and args.count!=500:raise ValueError('Confirmation uses the entire prespecified panel')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='2'
    compile_started=time.perf_counter();warmup();compile_seconds=time.perf_counter()-compile_started
    cached=prepare_cache(args.panel);order=sorted(range(500),key=lambda i:cached['rows'][i]['case'].block_count)
    panel=list(range(500)) if args.count==500 else [order[i] for i in range(2,500,5)]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(args.direct,args.panel)) as pool:
        for config in args.configs:
            folder=HERE/'runs'/f'{args.tag}_{config}_{args.panel}{args.count}';folder.mkdir(parents=True,exist_ok=True)
            save(folder/'initialization.json',{'native_warmup_seconds_main':compile_seconds,
                'scope':'separate resident initialization, not included in steady per-request time',
                'first_JIT_cold_evidence':'runs/20261020_native_parity5_r2/results.json'})
            identity={'config':CONFIGS[config],'panel':panel,'members':[cached['rows'][i]['member'] for i in panel],
                'checkpoint_sha256':cached['checkpoint_sha256'],'sources':sources(),'workers':2,'threads_per_worker':2,'direct':args.direct,
                'command':sys.argv,'python':sys.executable,'platform':platform.platform(),'processor':platform.processor()}
            if (folder/'config.json').exists():
                if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Run identity changed; use a new tag')
            else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
            if (folder/'results.json').exists():continue
            path=folder/'partial.json';done=json.loads(path.read_text()) if path.exists() else [];completed={r['index'] for r in done};todo=iter(i for i in panel if i not in completed);pending={}
            def submit():
                if (folder/'STOP').exists():return
                index=next(todo,None)
                if index is not None:pending[pool.submit(evaluate,index,config)]=index
            for _ in range(2):submit()
            while pending:
                finished,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future);done.append(future.result());save(path,done);submit()
                    if len(done)%25==0:print(config,len(done),flush=True)
            if len(done)!=len(panel):print('Paused:',folder,flush=True);return
            q=np.array([r['metrics']['cost'] for r in done]);n=np.array([r['n'] for r in done]);w=np.exp(n/12-(n/12).max());times=np.array([r['seconds'] for r in done])
            summary={'count':len(done),'hard_legal':sum(r['metrics']['is_feasible'] for r in done),'mean_Q':float(q.mean()),'weighted_Q':float(np.average(q,weights=w)),
                'p95_Q':float(np.percentile(q,95)),'max_Q':float(q.max()),'seconds_mean':float(times.mean()),'seconds_p95':float(np.percentile(times,95)),
                'row_fallbacks':sum(r.get('row',False) for r in done),'errors':sum('error' in r for r in done)}
            save(folder/'results.json',{'summary':summary,'records':sorted(done,key=lambda r:r['index']),'test_read':False});path.unlink();print(config,json.dumps(summary),flush=True)


if __name__=='__main__':main()
