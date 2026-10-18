"""Paired draft/frozen-repair evaluation, with explicit resumable panel scope."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from datasets.paths import ROOT
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .model import GeometryModel
from .batches import decode
from .inference import prepare_inference
from .occupancy import exact_diagnostics
from .solver import optimize
from .records import HERE,save,append,sha,provenance,snapshot,run_path
from .panels import read as read_panels
from .train import VARIANTS

E6_CHECKPOINT=ROOT/'experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0/attention_geometry/checkpoints/best.pt'


def load_model(path,device):
    state=torch.load(path,map_location=device,weights_only=False)
    model=GeometryModel(**state['model_config']).to(device)
    model.load_state_dict(state['model']);return model.eval()


def checkpoint_paths(primary,additional,names):
    """Join retained controls by reference; never copy or rewrite old weights."""
    paths={}
    for name in names:
        path=primary/name/'checkpoints/best.pt'
        if not path.exists():
            candidates=[r/name/'checkpoints/best.pt' for r in additional if (r/name/'checkpoints/best.pt').exists()]
            if len(candidates)!=1:raise ValueError(f'Need exactly one additional checkpoint for {name}: {candidates}')
            path=candidates[0]
        paths[name]=path
    return paths


def summarize(rows,names):
    out={}
    for name in names:
        records=[r['variants'][name] for r in rows];metrics=[r['final']['metrics'] for r in records]
        if not metrics:continue
        q=np.array([v['cost'] for v in metrics]);ok=[r for r in records if 'error' not in r]
        def mean(key):
            vals=[v[key] for v in metrics if v.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        out[name]={'cases':len(rows),'Q_mean':float(q.mean()),'Q_weighted':official_evaluator().compute_total_score(q.tolist(),[r['block_count'] for r in rows]),
                   'Q_median':float(np.median(q)),'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),
                   'Q_below1_2':int((q<1.2).sum()),'Q_below1_15':int((q<1.15).sum()),
                   'feasible':sum(v['is_feasible'] for v in metrics),'execution_errors':len(records)-len(ok),
                   **{k+'_mean':mean(k) for k in ('hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations')}}
        for key in ('feature_seconds','forward_seconds','postprocess_seconds','end_to_end_seconds','shape_iterations','trace_subproblems','guard_fallback'):
            vals=[r[key] for r in ok]
            out[name][key]={'mean':float(np.mean(vals)),'median':float(np.median(vals)),'p95':float(np.quantile(vals,.95))} if vals else None
        for stage in ('raw_diagnostics','final_diagnostics'):
            out[name][stage]={k:float(np.mean([r[stage][k] for r in ok])) for k in ok[0][stage]} if ok else {}
        vals=[r['center_displacement_local_mean'] for r in ok]
        out[name]['center_displacement_local_mean']=float(np.mean(vals)) if vals else None
    paired={}
    if 'frozen_e6' in out:
        for name in names:
            if name=='frozen_e6':continue
            good=[r for r in rows if 'error' not in r['variants'][name] and 'error' not in r['variants']['frozen_e6']]
            qdiff=[r['variants'][name]['final']['metrics']['cost']-r['variants']['frozen_e6']['final']['metrics']['cost'] for r in rows]
            times=[r['variants'][name]['postprocess_seconds']-r['variants']['frozen_e6']['postprocess_seconds'] for r in good]
            paired[name]={'Q_difference_all_cases':float(np.mean(qdiff)),'wins':sum(d< -1e-8 for d in qdiff),
                          'postprocess_seconds_difference_paired_successes':float(np.mean(times)) if times else None,
                          'paired_timing_count':len(times),'timing_missing_on_error':len(rows)-len(times),
                          'postprocess_time_nonincrease':bool(len(times)==len(rows) and np.mean(times)<=0),
                          'end_to_end_time_nonincrease':bool(len(good)==len(rows) and np.mean([
                              r['variants'][name]['end_to_end_seconds']-r['variants']['frozen_e6']['end_to_end_seconds'] for r in good])<=0)}
    return {'variants':out,'paired_vs_frozen_e6':paired}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--additional-train-run',type=Path,action='append',default=[],help='Retained variants missing from the primary run, e.g. stopped attention_geometry control')
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=list(VARIANTS))
    p.add_argument('--panel',choices=('dev5','dev100','test50'),default='dev100')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true',help='Bounded developer check: first dev5 member and at most one new model, no scientific result')
    args=p.parse_args();begin=time.perf_counter();run=run_path(args.train_run);output=run_path(args.output)
    additional=[run_path(path) for path in args.additional_train_run]
    if len(set([run,*additional]))!=1+len(additional):raise ValueError('Duplicate training run')
    training_config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    training_configs={str(run):training_config,**{str(path):json.loads((path/'config.json').read_text(encoding='utf-8')) for path in additional}}
    if any(c.get('smoke') for c in training_configs.values()) and not args.smoke:raise ValueError('Smoke checkpoints cannot be used for scientific evaluation')
    if args.smoke and (args.panel!='dev5' or len(args.variants)!=1 or 'smoke' not in output.name):
        raise ValueError('Smoke requires --panel dev5, exactly one variant and a smoke output name')
    if output==run or output.is_relative_to(run):raise ValueError('Use a separate evaluation run to keep training resume provenance stable')
    output.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    m=workset_manifest();panels=read_panels();test=args.panel=='test50'
    if any(c['workset_sha256']!=m['manifest_sha256'] for c in training_configs.values()):raise ValueError('Training workset mismatch')
    reader=LayoutReader(m,'local_test' if test else 'dev_quality',allow_local_test=test)
    selected=reader.members if args.panel=='dev100' else tuple((r['shard'],r['sample_index']) for r in panels[args.panel])
    if args.smoke:selected=selected[:1]
    # Reader enforces canonical members. Selection happens before any tensor load.
    indices=[reader.members.index(pair) for pair in selected]
    paths=checkpoint_paths(run,additional,args.variants);paths['frozen_e6']=E6_CHECKPOINT
    if len(set(args.variants))!=len(args.variants):raise ValueError('Duplicate variants')
    names=['frozen_e6',*args.variants]
    config={'panel':args.panel,'smoke':args.smoke,'scientific_result':not args.smoke,
            'members':[list(p) for p in selected],'workset_sha256':m['manifest_sha256'],
            'checkpoint_sha256':{n:sha(v) for n,v in paths.items()},'checkpoint_paths':{n:str(v) for n,v in paths.items()},
            'training_sources':{path:{'config_sha256':sha(Path(path)/'config.json'),'schedule':c.get('schedule','original'),
                 'scheduler':c['scheduler'],'early_stop':c['early_stop']} for path,c in training_configs.items()},
            'variants':names,'device':args.device,'threads':2,'postprocess':{'seconds':120.,'shape_steps':24,'reweighted':True,'mip_seconds':0},
            'test_feedback_disclosure':('one developer fixture only; not a scientific result' if args.smoke else
                'test50 may be used for diagnosis under user authorization; reuse for changes must be disclosed' if test else 'dev development evaluation'),
            'timing_order':'case-index-rotated variant order; all model CUDA contexts warmed before timing',
            **provenance()}
    if args.resume:
        old=json.loads((output/'config.json').read_text(encoding='utf-8'))
        for key,value in config.items():
            if key not in ('command','created_utc','git_head','git_status') and value!=old.get(key):raise ValueError('Evaluation resume mismatch:'+key)
    else:save(output/'config.json',config);snapshot(output,config['source_sha256'])
    rows=[json.loads(s) for s in (output/'cases.jsonl').read_text(encoding='utf-8').splitlines()] if args.resume and (output/'cases.jsonl').exists() else []
    done={r['case_id'] for r in rows};stopped=[False]
    signal.signal(signal.SIGINT,lambda *unused:stopped.__setitem__(0,True))
    models={name:load_model(path,args.device) for name,path in paths.items()}
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    read_count=0;warm=False
    try:
        for case_index,idx in enumerate(indices):
            case_id=f"{'local_test' if test else 'dev'}:{selected[case_index][0]}:{selected[case_index][1]}"
            if case_id in done:continue
            if stopped[0] or (output/'STOP').exists():break
            raw=reader.case(idx);read_count+=1
            scoring=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            if not warm:
                warm_g=prepare_inference(raw.solver_inputs,args.device)
                with torch.no_grad():
                    for model in models.values():model(warm_g)
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                warm=True
            row={'case_id':raw.case_id,'source':raw.source,'block_count':raw.block_count,
                 'reference_positions':raw.reference_positions,'constraints':np.asarray(raw.solver_inputs[5]).tolist(),'variants':{}}
            order=names[case_index%len(names):]+names[:case_index%len(names)]
            for name in order:
                try:
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                    start=time.perf_counter();g=prepare_inference(raw.solver_inputs,args.device)
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                    prep=time.perf_counter()-start;start=time.perf_counter()
                    with torch.no_grad():pos=(decode(models[name](g),g)[0]*g['scale'][0]).cpu().numpy().astype(float)
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                    forward=time.perf_counter()-start
                    # Restore public immutable physical specs exactly; normalized
                    # float32 roundtrips must not alter the solver input contract.
                    c=np.asarray(raw.solver_inputs[5]);spec=np.asarray(raw.solver_inputs[6]);locked=c[:,:2].any(1)
                    pos[locked,2:]=spec[locked,2:];pos[c[:,1]!=0]=spec[c[:,1]!=0]
                    raw_di=exact_diagnostics(pos)
                    start=time.perf_counter();final,stats=optimize(pos,raw.solver_inputs,seconds=120.,shape_steps=24,reweighted=True,mip_seconds=0.)
                    post=time.perf_counter()-start
                    evaluated=evaluate_case(scoring,final,prep+forward+post)
                    effective=np.asarray(raw.solver_inputs[1]).copy();effective[locked]=spec[locked,2:].prod(1)
                    displacement=np.linalg.norm(final[:,:2]+final[:,2:]/2-pos[:,:2]-pos[:,2:]/2,axis=1)/np.sqrt(effective)
                    row['variants'][name]={'raw_positions':pos.tolist(),'raw_diagnostics':raw_di,'final':evaluated,
                        'final_diagnostics':exact_diagnostics(final),'feature_seconds':prep,'forward_seconds':forward,
                        'postprocess_seconds':post,'end_to_end_seconds':prep+forward+post,'trace':stats,
                        'shape_iterations':sum(s.get('phase')=='shape' for s in stats['trace']),
                        'trace_subproblems':len(stats['trace']),'guard_fallback':int(stats['guard_fallback']),
                        'center_displacement_local_mean':float(displacement.mean())}
                except Exception as exc:
                    # Error cases retain Q=10 and remain in the main denominator.
                    row['variants'][name]={'error':{'type':type(exc).__name__,'message':str(exc)},
                        'final':{'metrics':{'cost':10.,'is_feasible':False}}}
            append(output/'cases.jsonl',row);rows.append(row)
            print(json.dumps({'complete':len(rows),'panel':args.panel,'case_id':raw.case_id,
                              'Q':{n:row['variants'][n]['final']['metrics']['cost'] for n in names}}),flush=True)
            save(output/'results.json',{'status':'running','completed_cases':len(rows),'expected_cases':len(selected),**summarize(rows,names)})
    finally:
        save(output/'results.json',{'status':'completed' if len(rows)==len(selected) else 'interrupted',
             'completed_cases':len(rows),'expected_cases':len(selected),'test_payload_reads_this_invocation':read_count if test else 0,
             'wall_seconds_this_invocation':time.perf_counter()-begin,'scope':config['test_feedback_disclosure'],
             'Q_target_achieved_on_complete_panel':bool(test and len(rows)==50 and any(
                 v['Q_mean']<1.2 and v['feasible']==50 for n,v in summarize(rows,names)['variants'].items() if n!='frozen_e6')),
             **summarize(rows,names)})


if __name__=='__main__':main()
