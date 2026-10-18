"""Frozen train/dev evaluation; raw geometry and unchanged E1 postprocessing."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
import shapely
from datasets.paths import ROOT,resolved_data_paths
from datasets.worksets import workset_manifest,LayoutReader
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .model import GeometryModel
from .data import decode_layout
from .inference import prepare_inference
from .train import evaluate,read_samples,save,sha,VARIANTS
from .postprocess import finalize
from .proposal_metrics import evaluate_proposal,movement_bound_with_shape_relaxation,joint_bound_checks


LEGACY=ROOT/'experiments/graph_proposal/runs/20261016_deterministic_workset_v2_seed0/checkpoints/best.pt'


def load(path,legacy=False):
    state=torch.load(path,map_location='cuda',weights_only=False)
    if legacy:
        model=GeometryModel('baseline')
        # Exact parameter mapping, no imports from E1 private algorithms.
        model.load_state_dict({k.removeprefix('encoder.'):v for k,v in state['model'].items()})
    else:
        model=GeometryModel(**state['model_config']);model.load_state_dict(state['model'])
    return model.cuda().eval(),state


def aggregate(rows,names):
    evaluator=official_evaluator();out={}
    for name in names:
        records=[r['variants'][name] for r in rows if name in r['variants']]
        ids=[r['block_count'] for r in rows if name in r['variants']]
        stages={}
        for stage in ('raw','final'):
            metrics=[r[stage]['metrics'] for r in records]
            def mean(k):return float(np.mean([r[k] for r in metrics])) if metrics else None
            stages[stage]={'count':len(metrics),'feasible_count':sum(r['is_feasible'] for r in metrics),
                           'weighted_Q':evaluator.compute_total_score([r['cost'] for r in metrics],ids),
                           **{k+'_mean':mean(k) for k in ('cost','hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations','violations_relative')}}
        keys=sorted(records[0]['raw']['diagnostics']) if records else []
        di={}
        for k in keys:
            vals=[r['raw']['diagnostics'][k] for r in records if r['raw']['diagnostics'][k] is not None]
            if vals:di[k]={'mean':float(np.mean(vals)),'median':float(np.median(vals)),'p10':float(np.quantile(vals,.1)),'p90':float(np.quantile(vals,.9))}
        stages['raw']['diagnostics']=di
        stages['final']['fallback_count']=sum(r['postprocess']['fallback'] for r in records)
        stages['final']['solve_seconds_mean']=float(np.mean([r['solve_seconds'] for r in records]))
        stages['final']['budget_overrun_count']=sum(r['solve_seconds']>15 for r in records)
        stages['final']['center_displacement_local_mean']=float(np.mean([r['center_displacement_local_mean'] for r in records]))
        stages['raw']['fixed_shape_LB_at_most5pct_count']=sum((r['raw']['diagnostics']['translation_radius_lb'] is not None and r['raw']['diagnostics']['translation_radius_lb']<=.05) for r in records)
        stages['raw']['optimistic_joint5pct_not_excluded_count']=sum((r['raw']['diagnostics']['translation_radius_lb_shape5pct'] is not None and r['raw']['diagnostics']['translation_radius_lb_shape5pct']<=.05) for r in records)
        out[name]=stages
    # Paired factorial effects, all on exactly the same fixed dev cases.
    pairs={}
    for a,b in [('baseline_geometry','baseline_mse'),('attention_mse','baseline_mse'),('attention_geometry','attention_mse'),('attention_geometry','baseline_geometry'),('baseline_local','baseline_mse'),('baseline_geometry','baseline_local')]:
        if a in names and b in names:
            differences=[r['variants'][a]['final']['metrics']['cost']-r['variants'][b]['final']['metrics']['cost'] for r in rows]
            pairs[a+'_vs_'+b]={'wins':sum(d< -1e-8 for d in differences),'ties':sum(abs(d)<=1e-8 for d in differences),
                              'losses':sum(d>1e-8 for d in differences),'mean_Q_difference':float(np.mean(differences)),
                              'weighted_Q_difference':out[a]['final']['weighted_Q']-out[b]['final']['weighted_Q']}
    reference=[r['reference_metrics'] for r in rows]
    return {'variants':out,'paired_final':pairs,'case_count':len(rows),'test_payload_reads':0,
            'reference':{'feasible_count':sum(m['is_feasible'] for m in reference),
                         'weighted_Q':evaluator.compute_total_score([m['cost'] for m in reference],[r['block_count'] for r in rows]),
                         **{k+'_mean':float(np.mean([m[k] for m in reference])) for k in ('boundary_violations','grouping_violations','mib_violations','violations_relative')}}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--extra-run',type=Path,action='append',default=[])
    args=p.parse_args();begin=time.monotonic();run=args.run_dir.resolve();output=run/'evaluation'
    if not run.is_relative_to((Path(__file__).parent/'runs').resolve()):raise ValueError('Wrong experiment run')
    if output.exists() and not args.resume:raise ValueError('Evaluation exists; explicit resume')
    output.mkdir(exist_ok=args.resume)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest=workset_manifest();models={};states={};checkpoints={}
    for name,_,_ in VARIANTS:
        if not (run/name/'results.json').exists():raise ValueError('Training incomplete:'+name)
        path=run/name/'checkpoints/best.pt';models[name],states[name]=load(path);checkpoints[name]={'path':str(path),'sha256':sha(path)}
    for extra in args.extra_run:
        extra=extra.resolve()
        if not extra.is_relative_to((Path(__file__).parent/'runs').resolve()):raise ValueError('Extra run outside experiment')
        for path in sorted(extra.glob('*/checkpoints/best.pt')):
            name=path.parents[1].name
            if name in models:raise ValueError('Duplicate variant:'+name)
            if not (path.parents[1]/'results.json').exists():raise ValueError('Incomplete extra training')
            models[name],states[name]=load(path)
            checkpoints[name]={'path':str(path),'sha256':sha(path),'note':'adaptive mechanism ablation specified before dev_quality evaluation'}
    models['legacy_frozen'],states['legacy_frozen']=load(LEGACY,legacy=True)
    checkpoints['legacy_frozen']={'path':str(LEGACY),'sha256':sha(LEGACY),'note':'historical differently trained checkpoint; not a samebudget control'}
    source=[*sorted(Path(__file__).parent.glob('*.py')),*sorted((ROOT/'scoring').glob('*.py')),
            *sorted((ROOT/'datasets').glob('*.py')),ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
            ROOT/'protocol/split_v1.json',ROOT/'protocol/workset_v2.json',ROOT/'protocol/paths.json',ROOT/'environment.yml']
    config={'command':[__import__('sys').executable,*__import__('sys').argv],'created_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoints':checkpoints,'workset_sha256':manifest['manifest_sha256'],'paths':resolved_data_paths(),
            'panels':'fulltrain20000/dev2500 supervised;dev_quality100 raw and frozenE1 correct+pack',
            'source_sha256':{s.relative_to(ROOT).as_posix():sha(s) for s in source},'test_payload_reads':0,
            'device':torch.cuda.get_device_name(0),'cpu_threads':2,'torch':torch.__version__,
            'numpy':np.__version__,'shapely':shapely.__version__,'precision':'float32; deterministic algorithms; no AMP',
            'postprocess':'independently copied E1 code, unchanged correct12/order/pack/rowfallback; no new repair',
            'selection':'best fixed500dev geometry monitor; evaluation does not change checkpoint or hyperparameters',
            'timing':'CUDA synchronize percase, modelpreloaded; public feature prep+transfer charged to every variant; forward+decode+postprocess included15s; references/scoring outside',
            'scope':'single seed; monitor selected training checkpoints, not independent test; raw Q10 and illegalHPWL not feasible quality',
            'solver_label_isolation':'target/reference removed from model input dictionary; references used only in external diagnostics/scoring',
            'joint_bound_definition_checks':joint_bound_checks(),
            'joint_bound':'optimistic independent freeWH shrink exp(-eta), area dropped; fixedWH/preplaced locked. Necessary position budget only; <=5pct does not prove joint repair.'}
    if not args.resume:
        save(output/'config.json',config)
        with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for s in source:z.write(s,s.relative_to(ROOT))
    else:
        old=json.loads((output/'config.json').read_text(encoding='utf-8'))
        if old['checkpoints']!=checkpoints:raise ValueError('Frozen checkpoint mismatch')
    load_seconds=time.monotonic()-begin
    if not (output/'supervised.json').exists():
        train=read_samples(LayoutReader(manifest,'train'));dev=read_samples(LayoutReader(manifest,'dev'))
        audit={}
        for panel,samples in [('train',train),('dev',dev)]:
            ns=[s['n'] for s in samples];outside=free_count=0;logmin=float('inf');logmax=-float('inf')
            for s in samples:
                z=s['target'][:,2][s['mask'][:,2].bool()]
                if len(z):
                    outside+=int((z.abs()>5).sum());free_count+=len(z);logmin=min(logmin,float(z.min()));logmax=max(logmax,float(z.max()))
            audit[panel]={'layouts':len(samples),'block_count_min':min(ns),'block_count_max':max(ns),
                          'block_count_mean':float(np.mean(ns)),'free_shape_target_count':free_count,
                          'target_log_ratio_min':logmin,'target_log_ratio_max':logmax,'targets_outside_decoder_log_ratio_range':outside}
        save(output/'dataset_characteristics.json',audit)
        sup={}
        for name,model in models.items():
            sup[name]={'train':evaluate(model,train,list(range(len(train))),32,'cuda'),
                       'dev':evaluate(model,dev,list(range(len(dev))),32,'cuda'),
                       'best_step':states[name].get('step',states[name].get('total_steps')),
                       'train_seen_at_best':int((states[name].get('seen',states[name].get('seen_counts'))>0).sum())}
            save(output/'supervised_partial.json',sup)
            print(json.dumps({'phase':'supervised','variant':name,'train':sup[name]['train'],'dev':sup[name]['dev']}),flush=True)
        save(output/'supervised.json',sup)
        # Partial contains no unique information once the complete file is durable.
        (output/'supervised_partial.json').unlink()
        del train,dev
    rows=[]
    if args.resume and (output/'cases.jsonl').exists():
        with (output/'cases.jsonl').open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    done={r['case_id'] for r in rows};reader=LayoutReader(manifest,'dev_quality')
    with (output/'cases.jsonl').open('a',encoding='utf-8') as stream,torch.inference_mode():
        for i in range(len(reader)):
            raw=reader.case(i)
            if raw.case_id in done:continue
            inp=raw.solver_inputs
            scoring_case=_build_case(raw.case_id,raw.source,inp[1:6],raw.reference_positions,raw.stored_metrics)
            item={'case_id':raw.case_id,'source':raw.source,'block_count':raw.block_count,
                  'reference':raw.reference_positions,'public_constraints':torch.as_tensor(inp[5]).tolist(),
                  'area_targets':torch.as_tensor(inp[1]).tolist(),
                  'reference_metrics':evaluate_case(scoring_case,raw.reference_positions,0)['metrics'],'variants':{}}
            torch.cuda.synchronize();prep_start=time.monotonic()
            g,decode_graph=prepare_inference(inp,'cuda')
            torch.cuda.synchronize();prep_seconds=time.monotonic()-prep_start
            for name,model in models.items():
                torch.cuda.synchronize();start=time.monotonic()
                positions=decode_layout(model(g)[0,:raw.block_count],decode_graph).cpu().numpy().astype(float)
                torch.cuda.synchronize();forward=time.monotonic()-start
                final,stats=finalize(positions,inp,max(.05,15-prep_seconds-forward));seconds=prep_seconds+time.monotonic()-start
                raw_evaluation=evaluate_case(scoring_case,positions,prep_seconds+forward)
                final_evaluation=evaluate_case(scoring_case,final,seconds)
                effective=np.asarray(inp[1],dtype=float).copy();locked=np.asarray(inp[5])[:,:2].any(1)
                effective[locked]=np.asarray(inp[6])[locked,2:].prod(1)
                diagnostics=evaluate_proposal(positions,raw.reference_positions,effective,inp[5],inp[2],inp[3])
                for eta,label in [(.05,'5pct'),(.1,'10pct')]:
                    bound=movement_bound_with_shape_relaxation(positions,effective,inp[5],eta)
                    diagnostics['translation_radius_lb_shape'+label]=bound['lower_bound']
                    diagnostics['translation_shape'+label+'_locked_pair_failures']=bound['locked_pair_failures']
                displacement=np.linalg.norm(final[:,:2]+final[:,2:]/2-positions[:,:2]-positions[:,2:]/2,axis=1)/np.sqrt(effective)
                item['variants'][name]={'raw':{'positions':positions.tolist(),'metrics':raw_evaluation['metrics'],'diagnostics':diagnostics},
                                        'final':{'positions':final.tolist(),'metrics':final_evaluation['metrics'],
                                                 'hard_checks':final_evaluation['independent_hard_checks']},
                                        'feature_prep_seconds':prep_seconds,'forward_seconds':forward,'solve_seconds':seconds,'postprocess':stats,
                                        'center_displacement_local_mean':float(displacement.mean()),
                                        'center_displacement_local_max':float(displacement.max())}
            stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush();rows.append(item)
            if len(rows)%10==0:print(json.dumps({'phase':'dev_quality','complete':len(rows),'seconds':time.monotonic()-begin}),flush=True)
    results=aggregate(rows,list(models))
    bins={'21_to50':[r for r in rows if r['block_count']<=50],
          '51_to90':[r for r in rows if 50<r['block_count']<=90],
          '91_to120':[r for r in rows if r['block_count']>90]}
    results['by_size']={name:aggregate(selected,list(models)) for name,selected in bins.items() if selected}
    mib_summary={'free_mib_groups_at_least2':0,'different_exact_areas':0,'different_areas_with_intersecting_1pct_intervals':0}
    for row in rows:
        c=np.asarray(row['public_constraints']);areas=np.asarray(row['area_targets'])
        for group_id in np.unique(c[:,2]):
            if group_id<=0:continue
            ids=np.flatnonzero((c[:,2]==group_id)&~c[:,:2].any(1))
            if len(ids)<2:continue
            mib_summary['free_mib_groups_at_least2']+=1
            if np.ptp(areas[ids])>1e-8:
                mib_summary['different_exact_areas']+=1
                if (areas[ids]*.99).max()<=(areas[ids]*1.01).min():mib_summary['different_areas_with_intersecting_1pct_intervals']+=1
    results['area_representation_diagnostic']=mib_summary
    results.update(status='completed',load_seconds=load_seconds,wall_seconds=time.monotonic()-begin)
    save(output/'results.json',results);print(json.dumps({'phase':'finished','case_count':len(rows),'wall_seconds':results['wall_seconds']}),flush=True)


if __name__=='__main__':main()
