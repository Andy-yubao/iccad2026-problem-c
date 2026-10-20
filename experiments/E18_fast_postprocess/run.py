"""Resumable fixed-panel candidate evaluation, 2 processes x 2 threads."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing as mp
import os
import time
import numpy as np
from .cases import prepare_cache,seed_all
from .io import HERE,save,sources,snapshot,exclusive
from .solver import optimize
from scoring.evaluation import evaluate_case

CONFIGS={
 'vector4':dict(iterations=4,seconds=.20),
 'boundary6':dict(iterations=6,seconds=.20,mode=1,reweighted=False,boundary=40.,contact=20.),
 'relaxed6':dict(iterations=6,seconds=.20,relaxed=True),
 'boundary4':dict(iterations=4,seconds=.20,mode=1,reweighted=False,boundary=20.,contact=10.),
 'hard_relaxed8':dict(iterations=8,seconds=.20,hard=True,relaxed=True,reserve=.04,snap=1e-5),
 'hard_conservative8':dict(iterations=8,seconds=.20,hard=True,reserve=.04,snap=1e-5),
 'hard_boundary8':dict(iterations=8,seconds=.20,hard=True,relaxed=True,mode=1,reweighted=False,boundary=40.,contact=20.,reserve=.04,snap=1e-5),
 'compact_project':dict(iterations=6,seconds=.20,hard=True,relaxed=True,mode=1,reweighted=False,boundary=40.,contact=20.,reserve=.075,snap=1e-4),
 'soft_project':dict(iterations=6,seconds=.20,relaxed=True,reserve=.075,snap=1e-4),
 'hard_project':dict(iterations=6,seconds=.20,hard=True,relaxed=True,reserve=.075,snap=1e-4),
 'near_boundary':dict(iterations=8,seconds=.20,hard=True,relaxed=True,mode=1,reserve=.06,snap=1e-4),
 'group_order':dict(iterations=8,seconds=.20,hard=True,relaxed=True,mode=1,group_order=True,reserve=.06,snap=1e-4),
 'group_soft':dict(iterations=8,seconds=.20,relaxed=True,mode=1,group_order=True,elastic_strength=1000.,reserve=.06,snap=1e-4),
 'adaptive_near':dict(hard=True,relaxed=True,mode=1,adaptive=True,snap=1e-4),
 'draft_aligned':dict(hard=True,relaxed=True,mode=1,adaptive=True,draft_proxy=True,snap=1e-4),
 'draft_uniform':dict(hard=True,relaxed=True,mode=1,adaptive=True,draft_proxy=True,reweighted=False,boundary=40.,contact=20.,snap=1e-4),
 'adaptive_rescue':dict(hard=True,relaxed=True,mode=1,adaptive=True,rescue=True,snap=1e-4),
 'conflict_hybrid':dict(hard=True,relaxed=True,mode=1,adaptive=True,repair_edges=True,hybrid=True,rescue=True,snap=1e-4),
 'conflict_relaxed':dict(hard=True,relaxed=True,mode=1,adaptive=True,repair_edges=True,rescue=True,snap=1e-4),
 'near_legal':dict(hard=True,relaxed=False,near_seed=True,safe_mode=True,mode=1,adaptive=True,snap=1e-4),
 'near_shape':dict(hard=True,relaxed=True,hybrid=True,near_seed=True,safe_mode=True,mode=1,adaptive=True,snap=1e-4),
 'near_scan':dict(hard=True,relaxed=False,near_seed=True,seed_order=1,safe_mode=True,mode=1,adaptive=True,snap=1e-4),
 'faithful_adaptive':dict(relaxed=False,mode=0,adaptive=True,plateau=True,rescue=True,snap=1e-5),
 'faithful_boundary':dict(relaxed=False,mode=1,adaptive=True,plateau=True,rescue=True,snap=1e-5),
 'center_aligned':dict(relaxed=False,mode=0,adaptive=True,plateau=True,preserve_centers=True,snap=1e-5),
 'anchor_route':dict(relaxed=False,mode=0,adaptive=True,plateau=True,preserve_centers=True,anchor_route=True,snap=1e-5),
 'anchor_relaxed':dict(relaxed=True,mode=0,adaptive=True,plateau=True,preserve_centers=True,anchor_route=True,snap=1e-5),
 'anchor_square':dict(relaxed=False,mode=0,adaptive=True,plateau=True,preserve_centers=True,anchor_route=True,shape_square=True,snap=1e-5),
 'selective_extension':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,rescue=True,snap=1e-4),
 'robust_fast':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,robust_seeds=True,certified=True,snap=1e-4),
 'robust_aligned':dict(relaxed=False,mode=0,adaptive=True,fast_budget=True,late_extension=True,plateau=True,robust_seeds=True,certified=True,draft_proxy=True,snap=1e-4)}


def initialize():
    global rows
    seed_all(0,2);rows=prepare_cache()['rows']


def evaluate(index,config):
    row=rows[index];case=row['case'];seed_all(index,2);begin=time.perf_counter()
    try:
        x,stats=optimize(row['draft'],case.solver_inputs,**CONFIGS[config]);elapsed=time.perf_counter()-begin+row['inference_seconds']
        metrics=evaluate_case(case,x.tolist(),elapsed)['metrics']
        record={'index':index,'member':row['member'],'n':case.block_count,'seconds':elapsed,
            'metrics':metrics,'row':stats['row_retained'],'positions':x.tolist(),'trace':stats['trace']}
    except Exception as error:
        record={'index':index,'member':row['member'],'n':case.block_count,'seconds':time.perf_counter()-begin+row['inference_seconds'],
            'metrics':{'is_feasible':False,'cost':10.},'error':repr(error)}
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--configs',nargs='+',choices=list(CONFIGS),default=list(CONFIGS));ap.add_argument('--count',type=int,choices=[100,500],default=100)
    ap.add_argument('--tag',required=True);args=ap.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='2'
    cached=prepare_cache();order=sorted(range(500),key=lambda i:cached['rows'][i]['case'].block_count)
    panel=list(range(500)) if args.count==500 else [order[i] for i in range(2,500,5)]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for config in args.configs:
            folder=HERE/'runs'/f'{args.tag}_{config}_dev{args.count}';folder.mkdir(parents=True,exist_ok=True)
            identity={'config':CONFIGS[config],'panel':panel,'checkpoint_sha256':cached['checkpoint_sha256'],'sources':sources(),'workers':2,'threads_per_worker':2}
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
