"""One frozen, alternating-order, paired development confirmation."""
from .resources import configure
RESOURCE=configure()
import argparse
import datetime as dt
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .cases import prepare_cache,seed_all,predict,graph,WEIGHTS,BASE
from datasets.paths import resolved_data_paths
from .model import GeometryModel
from .run import STEADY,CONFIGS,summary
from .solver import optimize
from .canonical_native import warmup
from .native_reduce import warmup as reduce_warmup
from .io import HERE,ROOT,save,sha,sources,snapshot,exclusive
from scoring.evaluation import evaluate_case


def analyze(records):
    groups=sorted({r['member'][0] for r in records});lookup={g:i for i,g in enumerate(groups)}
    cluster=np.array([lookup[r['member'][0]] for r in records]);counts=np.bincount(cluster)
    n=np.array([r['n'] for r in records]);w=np.exp((n-n.max())/12)
    a=np.array([r['baseline']['metrics']['cost'] for r in records]);b=np.array([r['candidate']['metrics']['cost'] for r in records]);delta=b-a
    t0=np.array([r['baseline']['seconds'] for r in records]);t1=np.array([r['candidate']['seconds'] for r in records])
    rng=np.random.default_rng(20261022);samples=rng.integers(0,len(groups),(2000,len(groups)))
    weights=np.bincount(cluster,weights=w);effects=np.bincount(cluster,weights=w*delta)
    boot=effects[samples].sum(1)/weights[samples].sum(1)
    time_effects=np.bincount(cluster,weights=t1-t0)
    time_boot=time_effects[samples].sum(1)/counts[samples].sum(1)
    out={'weighted_Q_delta':float(np.average(delta,weights=w)),
        'weighted_delta_95CI':np.percentile(boot,[2.5,97.5]).tolist(),
        'mean_seconds_delta':float((t1-t0).mean()),'mean_seconds_delta_95CI':np.percentile(time_boot,[2.5,97.5]).tolist(),
        'improved':int((delta<-1e-10).sum()),'regressed':int((delta>1e-10).sum()),
        'max_Q_regression':float(delta.max()),'bootstrap_shards':len(groups),'bootstrap_replicates':2000,
        'baseline_first_count':sum(r['order'][0]=='baseline' for r in records),
        'candidate_first_count':sum(r['order'][0]=='candidate' for r in records)}
    out['largest_regressions']=[{'index':records[i]['index'],'member':records[i]['member'],'n':int(n[i]),
        'baseline_Q':float(a[i]),'candidate_Q':float(b[i]),'delta':float(delta[i])} for i in np.argsort(-delta)[:10]]
    out['groups']={}
    for label,mask in [('lt60',n<60),('60to89',(n>=60)&(n<90)),('90to109',(n>=90)&(n<110)),('ge110',n>=110)]:
        if mask.any():out['groups'][label]={'count':int(mask.sum()),'baseline_weighted_Q':float(np.average(a[mask],weights=w[mask])),
            'candidate_weighted_Q':float(np.average(b[mask],weights=w[mask])),
            'baseline_mean_seconds':float(t0[mask].mean()),'candidate_mean_seconds':float(t1[mask].mean())}
    return out


def render(result,rows,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualization.layouts import plot_case,geometry_details
    records=result['records'];qa=np.array([r['baseline']['metrics']['cost'] for r in records]);qb=np.array([r['candidate']['metrics']['cost'] for r in records])
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for arm,label,color in [('baseline','E25 matched','#d97706'),('candidate','Frozen E30 seed0','#4f46e5')]:
        q=np.array([r[arm]['metrics']['cost'] for r in records]);t=np.array([r[arm]['seconds'] for r in records]);y=np.arange(1,len(q)+1)/len(q)
        axes[0].step(np.sort(q),y,where='post',label=label,color=color)
        axes[1].step(np.sort(t),y,where='post',label=label,color=color)
    axes[0].set(xlabel='Neutral-runtime Q',ylabel='Empirical cumulative probability',ylim=(.9,1.002),title='Quality tail: paired development confirmation')
    axes[1].set(xlabel='Complete solve seconds',ylabel='Empirical cumulative probability',title='Runtime: alternating order')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(folder/'quality_runtime.png',dpi=180);plt.close(fig)
    chosen=[]
    for i in [int(qa.argmax()),int(qb.argmax()),int((qb-qa).argmax())]:
        if i not in chosen:chosen.append(i)
    plots=[]
    for rank,i in enumerate(chosen,1):
        record=records[i];case=rows[record['index']]['case'];c=np.asarray(case.solver_inputs[5]).tolist();variants={}
        for arm in ('baseline','candidate'):
            r=record[arm];variants[arm]={'metrics':r['metrics']}
            if 'positions' in r:
                variants[arm].update(positions=r['positions'],soft_details=geometry_details(r['positions'],c))
            else:variants[arm]['unavailable']='No returned layout'
        filename=f'layout_{rank:02d}.png'
        plot_case({'panel':'paired confirmation','case_id':str(record['member']),'constraints':c,'variants':variants},folder/filename,
            stages=(('baseline',f'E25 Q={qa[i]:.4f}'),('candidate',f'E30 Q={qb[i]:.4f}')))
        plots.append({'index':record['index'],'member':record['member'],'file':filename})
    save(folder/'figures.json',{'plots':plots,'test_read':False})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--smoke-monitor',action='store_true')
    ap.add_argument('--deadline',default='2026-10-22T08:01:00+08:00');args=ap.parse_args()
    started=time.perf_counter();deadline=dt.datetime.fromisoformat(args.deadline).timestamp()
    contract_path=HERE/'runs/frozen_candidate.json';contract=json.loads(contract_path.read_text())
    for name,digest in contract['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen method source changed: '+name)
    candidate_path=ROOT/contract['checkpoint']
    if sha(candidate_path)!=contract['checkpoint_sha256'] or sha(WEIGHTS)!=contract['parent_checkpoint_sha256']:raise ValueError('Frozen weights changed')
    panel='monitor' if args.smoke_monitor else 'confirmation';cached=prepare_cache(panel);rows=cached['rows']
    if cached['registry_sha256']!=contract['registry_sha256']:raise ValueError('Frozen registry changed')
    manifest_path=BASE/'runs/prepared_v5_e10_cache_v1/dev/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['identity']['paths']!=sha(ROOT/'protocol/paths.json'):raise ValueError('Frozen cache paths changed')
    order=sorted(range(500),key=lambda i:rows[i]['case'].block_count)
    indices=[order[i] for i in (5,125,250,375,495)] if args.smoke_monitor else list(range(500))
    warmup();reduce_warmup();models={}
    for arm,path in [('baseline',WEIGHTS),('candidate',candidate_path)]:
        state=torch.load(path,map_location='cpu',weights_only=False);model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
        if arm=='candidate':
            with torch.inference_mode():
                example=graph(rows[indices[0]]['case'].solver_inputs)
                model=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(model,(example,),strict=False)))
        models[arm]=model
    for model in models.values():predict(model,rows[indices[0]]['case'].solver_inputs)
    initialization=time.perf_counter()-started
    permutation=np.random.default_rng(20261022).permutation(len(indices));candidate_first={indices[j] for j in permutation[:len(indices)//2]}
    name='20261022_pairing_smoke5' if args.smoke_monitor else '20261022_frozen_confirmation500'
    folder=HERE/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    identity={'sources':sources(),'frozen_candidate_sha256':sha(contract_path),'candidate_checkpoint_sha256':sha(candidate_path),
        'baseline_checkpoint_sha256':sha(WEIGHTS),'indices':indices,'members':[rows[i]['member'] for i in indices],
        'candidate_first':sorted(candidate_first),'baseline_config':STEADY,'candidate_config':CONFIGS[contract['postprocessor']],
        'panel':panel,'smoke':args.smoke_monitor,'workers':1,'threads':2,'affinity':RESOURCE['affinity'],
        'registry_sha256':cached['registry_sha256'],'split_sha256':sha(ROOT/'protocol/split_v1.json'),
        'paths_sha256':sha(ROOT/'protocol/paths.json'),'cache_manifest_sha256':sha(manifest_path),'data_paths':resolved_data_paths(),
        'candidate_jit':True,'baseline_jit':False,'command':[sys.executable,*sys.argv],
        'versions':{name:metadata.version(name) for name in ('torch','numpy','scipy','clarabel','highspy','numba','shapely','matplotlib')},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'timing':'Public features + CPU inference + decode + all repair + guard; resident loading/JIT and scoring excluded',
        'order':'Exactly balanced randomized first-arm assignment, then same-case sequential calls; smoke has 3/2 split',
        'test_read':False}
    with exclusive(folder/'run.lock'):
        if (folder/'config.json').exists():
            if json.loads((folder/'config.json').read_text())!=identity:raise ValueError('Resume identity changed')
        else:save(folder/'config.json',identity);snapshot(folder/'source_snapshot.zip',identity)
        if (folder/'results.json').exists():
            print('Reuse completed paired confirmation',flush=True);return
        path=folder/'partial.json';partial=json.loads(path.read_text()) if path.exists() else {'records':[],'elapsed_seconds':0.}
        records=partial['records'];by_index={r['index']:r for r in records};prior=partial['elapsed_seconds'];run_start=time.perf_counter()
        if len(by_index)!=len(records) or not set(by_index).issubset(set(indices)):raise ValueError('Duplicate or foreign partial indices')
        if not (folder/'initialization.json').exists():save(folder/'initialization.json',{'resident_seconds':initialization,
            'all_setup_seconds':run_start-started,'resource':RESOURCE,'test_read':False})
        calls_path=folder/'invocations.json';calls=json.loads(calls_path.read_text()) if calls_path.exists() else []
        calls.append({'argv':sys.argv,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'initialization_seconds':initialization,'prior_seconds':prior})
        save(calls_path,calls)
        def store():save(path,{'records':records,'elapsed_seconds':prior+time.perf_counter()-run_start,'test_read':False})
        for index in indices:
            row=rows[index];case=row['case'];arms=['candidate','baseline'] if index in candidate_first else ['baseline','candidate']
            record=by_index.get(index)
            if record is None:
                record={'index':index,'member':row['member'],'n':case.block_count,'order':arms};records.append(record);by_index[index]=record
            if record['member']!=row['member'] or record['order']!=arms:raise ValueError('Partial membership/order changed')
            for arm in arms:
                if arm in record:continue
                if (folder/'STOP').exists() or time.time()>=deadline:store();print('Paused paired evaluation',flush=True);return
                seed_all(index);begin=time.perf_counter();inference=0.
                try:
                    draft,inference=predict(models[arm],case.solver_inputs)
                    config=STEADY if arm=='baseline' else CONFIGS[contract['postprocessor']]
                    x,stats=optimize(draft,case.solver_inputs,**config);seconds=time.perf_counter()-begin
                    metric=evaluate_case(case,x.tolist(),seconds)['metrics']
                    value={'metrics':metric,'seconds':seconds,'positions':x.tolist(),'trace':stats['trace'],
                           'inference_seconds':inference,'row':stats['row_retained'],'postprocess_seconds':stats['seconds']}
                except Exception as error:
                    value={'metrics':{'cost':10.,'is_feasible':False},'seconds':time.perf_counter()-begin,'error':repr(error),'inference_seconds':inference}
                record[arm]=value;store()
            if len(records)%25==0 or args.smoke_monitor:print('Paired',len(records),'of',len(indices),flush=True)
        if len(records)!=len(indices) or any(not all(arm in r for arm in ('baseline','candidate')) for r in records):raise ValueError('Incomplete paired panel')
        summaries={arm:summary([{**r[arm],'n':r['n']} for r in records]) for arm in ('baseline','candidate')}
        result={'status':'complete','summary':summaries,'paired':analyze(records),'records':records,
                'elapsed_seconds':prior+time.perf_counter()-run_start,'wall_this_invocation_seconds':time.perf_counter()-started,
                'panel':panel,'test_read':False,
                'scope':('Five known monitor cases, harness implementation check only' if args.smoke_monitor else
                    'Prespecified additional development confirmation; not sealed test or contest score; candidate fixed before this evaluation')}
        save(folder/'results.json',result);path.unlink();print(json.dumps({'summary':summaries,'paired':result['paired']},ensure_ascii=False),flush=True)
        if not args.smoke_monitor:render(result,rows,folder)


if __name__=='__main__':main()
