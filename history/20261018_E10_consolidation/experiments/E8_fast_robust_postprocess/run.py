"""E8 frozen-draft paired study; dev only, resumable, total wall budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
PACKAGE='experiments.E8_fast_robust_postprocess'
CHECKPOINT=ROOT/'experiments/repair_friendly_draft/runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'
CHECKPOINT_HASH='47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28'
VARIANTS=('control120','control_short','sparse_short','dense_short','shape_first','shape_first_3s','linear_area','linear_area_3s','certified120','certified_short','topology_short','topology_3s','safe120','beam_short','beam_3s','joint_independent','joint_shared','joint_fixed','joint_shared_3s','joint_shared_10s')
VARIANTS+=('conic_short','conic_3s','conic_10s','conic_relaxed','conic_relaxed_3s','conic_dense','conic_dense_3s')
VARIANTS+=('conic_quadratic','conic_quadratic_3s','conic_quadratic_relaxed','conic_quadratic_relaxed_3s')
VARIANTS+=('conic_diagonal','conic_diagonal_3s','conic_diagonal_relaxed','conic_diagonal_relaxed_3s')
VARIANTS+=('safe_plain120',)
VARIANTS+=('conic_diagonal_10s',)
VARIANTS+=('conic_robust',)


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def append(path,value):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows,variants):
    from scoring.official import official_evaluator
    result={}
    for name in variants:
        group=[r for r in rows if name in r['variants']]
        if not group:
            continue
        records=[r['variants'][name] for r in group]
        metrics=[r['final']['metrics'] for r in records]
        q=np.asarray([m['cost'] for m in metrics])
        times=np.asarray([r['solve_seconds'] for r in records])
        worker=np.asarray([r['worker_wall_seconds'] for r in records])
        result[name]={'cases':len(group),'Q_mean':float(q.mean()),'Q_weighted':float(official_evaluator().compute_total_score(q.tolist(),[r['block_count'] for r in group])),
                      'Q_p95':float(np.quantile(q,.95)),'Q_max':float(q.max()),
                      'worst10percent_mean':float(np.sort(q)[-max(1,math.ceil(len(q)*.1)):].mean()),
                      'Q_above1_5':int((q>1.5).sum()),'Q_above2':int((q>2).sum()),
                      'feasible':sum(m['is_feasible'] for m in metrics),
                      'errors':sum('error' in r for r in records),
                      'postprocess_seconds_mean':float(times.mean()),'postprocess_seconds_p95':float(np.quantile(times,.95)),
                      'postprocess_seconds_max':float(times.max()),'postprocess_seconds_total':float(times.sum()),
                      'worker_wall_seconds_total':float(worker.sum()),
                      'end_to_end_seconds_total':float(sum(r['inference_seconds']+v['solve_seconds'] for r,v in zip(group,records))),
                      'guard_fallback_count':sum(bool(r.get('stats',{}).get('guard_fallback')) for r in records)}
    paired={};base_name='control120' if 'control120' in variants else 'safe_plain120'
    for name in variants:
        if name==base_name:
            continue
        group=[r for r in rows if name in r['variants'] and base_name in r['variants']]
        if not group:
            continue
        delta=[r['variants'][name]['final']['metrics']['cost']-r['variants'][base_name]['final']['metrics']['cost'] for r in group]
        base=sum(r['variants'][base_name]['solve_seconds'] for r in group)
        new=sum(r['variants'][name]['solve_seconds'] for r in group)
        paired[name]={'cases':len(group),'baseline_variant':base_name,'Q_mean_difference':float(np.mean(delta)),
                      'wins':sum(x < -1e-8 for x in delta),'ties':sum(abs(x)<=1e-8 for x in delta),
                      'losses':sum(x>1e-8 for x in delta),'postprocess_speedup':base/max(new,1e-12),
                      'end_to_end_speedup':sum(r['inference_seconds']+r['variants'][base_name]['solve_seconds'] for r in group)/max(sum(r['inference_seconds']+r['variants'][name]['solve_seconds'] for r in group),1e-12),
                      'inference_and_worker_wall_speedup':sum(r['inference_seconds']+r['variants'][base_name]['worker_wall_seconds'] for r in group)/max(sum(r['inference_seconds']+r['variants'][name]['worker_wall_seconds'] for r in group),1e-12)}
    return {'variants':result,'paired_vs_control120' if base_name=='control120' else 'paired_vs_safe_plain120':paired}


def report(output,rows,variants,status,expected,wall):
    config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    result={'status':status,'completed_cases':sum(all(v in r['variants'] for v in variants) for r in rows),
            'expected_cases':expected,'scope':config['panel']+' development panel; five-case probes are mechanism evidence only',
            'test_payload_reads':0,'wall_seconds_this_invocation':wall,**summary(rows,variants)}
    save(output/'results.json',result)
    lines=['# E8 paired postprocessor results','',f'Status: {status}; complete {result["completed_cases"]}/{expected}.',
           '', 'Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.',
           '', '| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in result['variants'].items():
        speed=result.get('paired_vs_control120',result.get('paired_vs_safe_plain120',{})).get(name,{}).get('postprocess_speedup',1.)
        lines.append(f'| {name} | {s["cases"]} | {s["feasible"]} | {s["Q_mean"]:.6f} | {s["Q_max"]:.6f} | {s["postprocess_seconds_mean"]:.4f} | {speed:.2f} |')
    lines+=['','Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.','']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def worker_call(output,inputs,draft,name,seconds,persistent=None):
    start=time.perf_counter()
    payload={'inputs':[int(inputs[0]),*[np.asarray(x).tolist() for x in inputs[1:]]],
             'draft':draft,'variant':name,'seconds':seconds}
    if persistent is not None:return persistent.call(payload)
    with tempfile.TemporaryDirectory(prefix='worker_',dir=output) as temp:
        request,response=Path(temp)/'request.json',Path(temp)/'response.json'
        request.write_text(json.dumps(payload),encoding='utf-8')
        try:
            process=subprocess.run([sys.executable,'-B','-m',PACKAGE+'.worker',str(request),str(response)],
                                   cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=seconds+10)
            if process.returncode or not response.exists():
                raise RuntimeError(process.stderr[-2500:] or process.stdout[-2500:] or f'Worker exit {process.returncode}')
            value=json.loads(response.read_text(encoding='utf-8'))
        except Exception as exc:
            value={'error':{'type':type(exc).__name__,'message':str(exc)},'solve_seconds':time.perf_counter()-start}
    value['worker_wall_seconds']=time.perf_counter()-start
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panel',choices=('pilot5','diagnostic5','dev100','confirm100'),default='pilot5')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=['control120'])
    parser.add_argument('--short-seconds',type=float,default=1.5)
    parser.add_argument('--budget-seconds',type=float,default=900)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reuse-run',type=Path,help='Reuse identical frozen drafts and control120 metrics by reference, without copying coordinates')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--persistent-worker',action='store_true')
    parser.add_argument('--final-only',action='store_true',help='Score only terminal output; omit offline improvement-curve scoring')
    parser.add_argument('--workers',type=int,nargs='+',help='At most five retrospective mechanism cases from the selected panel')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(HERE/'runs'):
        raise ValueError('Output must be within this experiment runs/')
    if args.report_only:
        config=json.loads((output/'config.json').read_text(encoding='utf-8'))
        previous=json.loads((output/'results.json').read_text(encoding='utf-8'))
        report(output,read_rows(output/'cases.jsonl'),config['variants'],previous['status'],len(config['members']),previous.get('wall_seconds_this_invocation',0.))
        return
    if args.short_seconds<=0 or args.budget_seconds<=0 or len(set(args.variants))!=len(args.variants):
        raise ValueError('Invalid budgets or duplicate variants')
    from datasets.worksets import LayoutReader,workset_manifest
    from datasets.cases import _build_case
    from datasets.paths import dataset_paths
    from scoring.evaluation import evaluate_case
    import torch
    from experiments.E8_fast_robust_postprocess.frozen.model import GeometryModel
    from experiments.E8_fast_robust_postprocess.frozen.inference import prepare_inference
    from experiments.E8_fast_robust_postprocess.frozen.batches import decode
    begin=time.perf_counter()
    manifest=workset_manifest()
    reader=LayoutReader(manifest,'dev' if args.panel=='confirm100' else 'dev_quality')
    selected_workers=(37,45,59,65,73) if args.panel=='diagnostic5' else (0,24,49,74,99)
    if args.panel=='confirm100':
        from datasets.worksets import member_pairs
        quality=set(member_pairs(manifest,'dev_quality'))
        quality_shards={pair[0] for pair in quality}
        indices=[]
        for worker in range(100):
            # Select another shard where possible, solely by frozen membership.
            candidates=[i for i,pair in enumerate(reader.members)
                        if pair[0].startswith(f'worker_{worker}/') and pair[0] not in quality_shards]
            indices.append(min(candidates,key=lambda i:hashlib.sha256(
                f'E8_confirm100_v1:{reader.members[i][0]}:{reader.members[i][1]}'.encode()).hexdigest()))
        if len(indices)!=100 or quality.intersection(reader.members[i] for i in indices):
            raise AssertionError('Confirmation membership mismatch')
    else:
        indices=list(range(len(reader))) if args.panel=='dev100' else [i for i,pair in enumerate(reader.members) if int(pair[0].split('/')[0].split('_')[1]) in selected_workers]
    if args.panel not in ('dev100','confirm100') and len(indices)!=5:
        raise AssertionError('Pilot membership mismatch')
    if args.workers is not None:
        if not 1<=len(set(args.workers))<=5 or any(w<0 or w>=100 for w in args.workers):
            raise ValueError('Worker subset requires 1..5 unique public worker IDs')
        indices=[i for i in indices if int(reader.members[i][0].split('/')[0].split('_')[1]) in args.workers]
        if len(indices)!=len(set(args.workers)):raise ValueError('Requested worker absent from parent panel')
    actual_checkpoint_hash=sha(CHECKPOINT)
    if actual_checkpoint_hash!=CHECKPOINT_HASH:
        raise ValueError('Frozen E7 checkpoint changed')
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(HERE.rglob('*.py')) if 'runs' not in p.relative_to(HERE).parts}
    reuse=args.reuse_run.resolve() if args.reuse_run else None
    if reuse is not None:
        if not reuse.is_relative_to(HERE/'runs') or reuse==output:raise ValueError('Invalid reuse run')
        reuse_config=json.loads((reuse/'config.json').read_text(encoding='utf-8'))
        if not set(reader.members[i] for i in indices).issubset(set(map(tuple,reuse_config['members']))) or reuse_config['checkpoint_sha256']!=actual_checkpoint_hash:
            raise ValueError('Reuse member/checkpoint mismatch')
    config={'panel':args.panel,'members':[list(reader.members[i]) for i in indices],'variants':args.variants,
            'persistent_worker':args.persistent_worker,'final_only':args.final_only,
            'workers_subset':args.workers,
            'seed':0,'threads':2,'device':args.device,'short_seconds':args.short_seconds,
            'control_seconds':120.,'control_shape_steps':24,'control_reweighted':True,'MIP':False,
            'workset_sha256':manifest['manifest_sha256'],'checkpoint':str(CHECKPOINT),'checkpoint_sha256':actual_checkpoint_hash,
            'source_sha256':sources,'paths':{k:str(v) for k,v in dataset_paths().items()},
            'split_file_sha256':sha(ROOT/'protocol/split_v1.json'),
            'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
                'scipy':importlib.metadata.version('scipy'),'highspy':importlib.metadata.version('highspy'),
                'clarabel':importlib.metadata.version('clarabel'),
                'cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
                'thread_scope':'Torch=2, shape HiGHS=2; fixed LP retains original SciPy HiGHS defaults'},
            'variant_order':'case-index rotated; all variants receive the same cached E7 draft',
            'supervisor_timeout':'solver budget + 10 seconds for interpreter imports/I/O; actual wall time recorded',
            'selection':('diagnostic5 retrospectively selected workers37,45,59,65,73 after baseline: two guard failures and three small soft-tail cases' if args.panel=='diagnostic5' else 'pilot workers 0,24,49,74,99 chosen before outcomes; full panel canonical dev_quality100'),
            'test_disclosure':'Previously saved E7 partial test feedback motivated design. This run reads dev only.',
            'reuse_run':str(reuse) if reuse else None,
            'reuse_fingerprints':{name:sha(reuse/name) for name in ('config.json','cases.jsonl','drafts.jsonl')} if reuse else {}}
    if args.panel=='confirm100':
        config['selection']='One per worker from workset_v2 dev, excluding all dev_quality shards; minimum SHA256 E8_confirm100_v1:<shard>:<row>; fixed before reading outputs. Additional development confirmation, not a sealed test.'
    if args.workers is not None:config['selection']+=' Retrospective mechanism-only worker subset: '+str(args.workers)
    if args.resume:
        old=json.loads((output/'config.json').read_text(encoding='utf-8'))
        if any(old.get(k)!=v for k,v in config.items()):
            raise ValueError('Resume configuration/source mismatch: use a new run for changed code')
    else:
        output.mkdir(parents=True,exist_ok=False)
        config.update(created_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
        save(output/'config.json',config)
        with zipfile.ZipFile(output/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for rel in sources:
                archive.write(ROOT/rel,rel)
            archive.write(HERE/'inherited_sources.json',str((HERE/'inherited_sources.json').relative_to(ROOT)))
    previous_wall=sum(event.get('wall_seconds',0.) for event in read_rows(output/'invocations.jsonl') if event.get('kind')=='invocation_end')
    if previous_wall>=args.budget_seconds:
        raise ValueError('Cumulative wall budget exhausted; explicitly provide a larger budget to continue')
    append(output/'invocations.jsonl',{'kind':'invocation_start','utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'cumulative_wall_budget_seconds':args.budget_seconds})
    rows=read_rows(output/'cases.jsonl')
    drafts={r['case_id']:r for r in read_rows(output/'drafts.jsonl')}
    reused_rows={}
    if reuse:
        drafts.update({r['case_id']:r for r in read_rows(reuse/'drafts.jsonl')})
        reused_rows={r['case_id']:r for r in read_rows(reuse/'cases.jsonl')}
    by_id={r['case_id']:r for r in rows}
    stopped=[False]
    signal.signal(signal.SIGINT,lambda *unused:stopped.__setitem__(0,True))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model=None
    warm=False
    status='complete'
    if args.persistent_worker:
        from experiments.E8_fast_robust_postprocess.persistent import PersistentWorker
        persistent=PersistentWorker(output,ROOT,PACKAGE)
    else:persistent=None
    for case_index,idx in enumerate(indices):
        shard,index=reader.members[idx]
        case_id=f'dev:{shard}:{index}'
        row=by_id.get(case_id)
        if row and all(name in row['variants'] for name in args.variants):
            continue
        if stopped[0] or (output/'STOP').exists():
            status='user_stop';break
        raw=reader.case(idx)
        case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
        if case_id not in drafts:
            if model is None:
                state=torch.load(CHECKPOINT,map_location=args.device,weights_only=False)
                model=GeometryModel(**state['model_config']).to(args.device)
                model.load_state_dict(state['model']);model.eval()
            public=prepare_inference(raw.solver_inputs,args.device)
            if not warm:
                with torch.no_grad():
                    model(public)
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                warm=True
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter()
            public=prepare_inference(raw.solver_inputs,args.device)
            with torch.no_grad():
                position=(decode(model(public),public)[0]*public['scale'][0]).cpu().numpy().astype(float)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            inference=time.perf_counter()-start
            c=np.asarray(raw.solver_inputs[5]);spec=np.asarray(raw.solver_inputs[6]);locked=c[:,:2].any(1)
            position[locked,2:]=spec[locked,2:];position[c[:,1]!=0]=spec[c[:,1]!=0]
            draft={'case_id':case_id,'source':raw.source,'block_count':raw.block_count,'draft':position.tolist(),'inference_seconds':inference}
            append(output/'drafts.jsonl',draft);drafts[case_id]=draft
        draft=drafts[case_id]
        if row is None:
            row={'case_id':case_id,'source':raw.source,'block_count':raw.block_count,
                 'inference_seconds':draft['inference_seconds'],'variants':{}}
            rows.append(row);by_id[case_id]=row
        for reference in ('control120','safe_plain120'):
            if reference in args.variants and reference not in row['variants'] and case_id in reused_rows and reference in reused_rows[case_id]['variants']:
                source=reused_rows[case_id]['variants'][reference]
                row['variants'][reference]={k:source[k] for k in ('solve_seconds','worker_wall_seconds')}
                row['variants'][reference].update(final={'metrics':source['final']['metrics']},
                      reused_from={'run':str(reuse),'case_id':case_id,'variant':reference},stats={'guard_fallback':source['stats'].get('guard_fallback',False)})
        order=args.variants[case_index%len(args.variants):]+args.variants[:case_index%len(args.variants)]
        for name in order:
            if name in row['variants']:
                continue
            seconds=120. if name in ('control120','certified120','safe120','safe_plain120') else 5. if name=='conic_robust' else 10. if name.endswith('_10s') else 3. if name.endswith('_3s') else args.short_seconds
            if previous_wall+time.perf_counter()-begin+seconds+12 > args.budget_seconds:
                status='resource_budget';break
            if stopped[0] or (output/'STOP').exists():
                status='user_stop';break
            value=worker_call(output,raw.solver_inputs,draft['draft'],name,seconds,persistent=persistent)
            if 'error' not in value:
                try:
                    value['final']=evaluate_case(case,value.pop('positions'),draft['inference_seconds']+value['solve_seconds'])
                except Exception as exc:
                    value['error']={'type':type(exc).__name__,'message':str(exc),'phase':'official_and_independent_evaluation'}
                    value['final']={'metrics':{'cost':10.,'is_feasible':False}}
                events=value.get('stats',{}).pop('improvements',[])
                if args.final_only:events=[]
                curve=[]
                for event in events:
                    try:
                        metrics=evaluate_case(case,event.pop('positions'),event['elapsed'])['metrics']
                        curve.append({**event,'Q':metrics['cost'],'feasible':metrics['is_feasible']})
                    except Exception as exc:
                        curve.append({k:v for k,v in event.items() if k!='positions'}|{'evaluation_error':str(exc)})
                value['quality_curve_offline']=curve
            else:
                value['final']={'metrics':{'cost':10.,'is_feasible':False}}
            row['variants'][name]=value
            # One canonical record per case; atomic replace enables per-variant resume.
            temp=output/'cases.jsonl.tmp'
            temp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
            temp.replace(output/'cases.jsonl')
            report(output,rows,args.variants,'running',len(indices),time.perf_counter()-begin)
            print(json.dumps({'case':case_id,'variant':name,'Q':value['final']['metrics']['cost'],
                              'seconds':value['solve_seconds'],'error':value.get('error')},ensure_ascii=False),flush=True)
        if status!='complete':
            break
    if persistent is not None:persistent.close()
    elapsed=time.perf_counter()-begin
    final_report=report(output,rows,args.variants,status,len(indices),elapsed)
    final_report['cumulative_wall_seconds_completed_invocations']=previous_wall+elapsed
    save(output/'results.json',final_report)
    append(output/'invocations.jsonl',{'kind':'invocation_end','utc':datetime.now(timezone.utc).isoformat(),'status':status,'wall_seconds':elapsed})
    print(json.dumps({'status':status,'output':str(output)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
