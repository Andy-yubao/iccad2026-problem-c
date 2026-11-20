"""Paired descriptive tables only: frozen reference / S12 draft / saved repair.

No training, optimization, checkpoint selection or model-Q reevaluation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import ast
import concurrent.futures
import csv
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
import traceback
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
MODELS=(
    ('baseline','每三层基线','E134_supervision_interval','e119_stride3'),
    ('A','A 完整/均分','E135_initial_supervision','e119_A'),
    ('B','B 完整/线性','E135_initial_supervision','e119_B'),
    ('C','C F/均分','E135_initial_supervision','e119_C'),
    ('D','D F/线性','E136_frontend_F_linear','e119_D'),
)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def register():
    from datasets.random_reference import RandomReference
    ds=RandomReference('dev_monitor500');models=[];case_hashes={}
    for key,label,experiment,arm in MODELS:
        owner=ROOT/'experiments'/experiment;run=owner/'runs'/f'{arm}_seed0';evaluation=run/'evaluation_no_mib_dev500'
        result=read(evaluation/'results.json');checkpoint=run/'checkpoints/best.pt'
        if result['status']!='complete' or result['count']!=500 or sha(checkpoint)!=result['checkpoint_sha256']:
            raise ValueError('Frozen completed evaluation required: '+key)
        gate=read(owner/'preflight.json')
        if any(sha(ROOT/name)!=digest for name,digest in gate['sources'].items()):raise ValueError('Frozen sources changed: '+key)
        hashes=[]
        for i in range(500):
            path=evaluation/f'cases/{i:04d}.json';row=read(path)
            if row['index']!=i or row['member']!=ds.records[ds.indices[i]]['member'] or row['checkpoint_sha256']!=result['checkpoint_sha256']:
                raise ValueError('Paired member/checkpoint mismatch')
            hashes.append(sha(path))
        models.append(dict(key=key,label=label,experiment=experiment,arm=arm,checkpoint=checkpoint.relative_to(ROOT).as_posix(),
            checkpoint_sha256=result['checkpoint_sha256'],evaluation=evaluation.relative_to(ROOT).as_posix(),
            result=result,result_sha256=sha(evaluation/'results.json'),case_sha256=hashes))
    common=None
    for _,_,experiment,_ in MODELS:
        path=ROOT/'experiments'/experiment/'loss.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='base_per_case')
        signature=ast.dump(node,include_attributes=False)
        if common is not None and signature!=common:raise ValueError('Different common Loss formula')
        common=signature
    ledger=ds.folder/'checksums.jsonl'
    expected=read(ROOT/'experiments/E136_frontend_F_linear/plan.json')['payload_checksums_sha256']
    if sha(ledger)!=expected:raise ValueError('Payload ledger changed')
    checksums={r['id']:r['payload_sha256'] for line in ledger.read_text(encoding='utf-8').splitlines() for r in json.loads(line)['records']}
    payloads=[]
    for index in ds.indices:
        record=ds.records[index];payloads.append(dict(path=(ds.folder/record['payload']).relative_to(ROOT).as_posix(),sha256=checksums[record['id']]))
    return dict(role='dev_monitor500',count=500,members=[ds.records[i]['member'] for i in ds.indices],models=models,payloads=payloads,
        sources={name:sha(ROOT/name) for name in ('scoring/draft_geometry.py','scoring/layout_geometry.py','scoring/checks.py','scoring/research_no_mib.py',
            'scoring/evaluation.py','experiments/E136_frontend_F_linear/loss.py','experiments/E136_frontend_F_linear/simple_loss.py')},
        statistics_source_sha256=sha(__file__),training=False,postprocessing_calls=0,model_Q_calls=0,reference_Q_calls=500,test_read=False,
        loss_scope='Common full public-priority-v1, S12 raw head for draft; reference target state; final XYWH reencoded. No supervision-depth weights.',
        geometry_scope='Actual reference / saved decoded draft / saved final coordinates; original areas, no MIB. Geometry details=True, original registered metric implementation.')

_DATASET=None
_MANIFEST=None

def initialize_geometry(manifest):
    global _DATASET,_MANIFEST
    from datasets.random_reference import RandomReference
    _DATASET=RandomReference('dev_monitor500');_MANIFEST=manifest

def geometry_job(index):
    import dataclasses
    import numpy as np
    import torch
    from datasets.cases import ResearchCase
    from scoring.draft_geometry import evaluate_geometry,repair_movement
    from scoring.research_no_mib import evaluate_case
    manifest=_MANIFEST;path=HERE/f'geometry/{index:04d}.json'
    if path.exists():
        existing=read(path)
        if existing['member']!=manifest['members'][index]:raise ValueError('Existing geometric member differs')
        return index
    payload=manifest['payloads'][index]
    if sha(ROOT/payload['path'])!=payload['sha256']:raise ValueError('Reference payload changed')
    item=_DATASET[index];inputs=item['solver_inputs'];reference=np.asarray(item['supervision']['reference_xywh'],float)
    n=int(inputs[0]);c=np.asarray(inputs[5]);areas=np.asarray(inputs[1]);specs=np.asarray(inputs[6])
    rows={m['key']:read(ROOT/m['evaluation']/f'cases/{index:04d}.json') for m in manifest['models']}
    counts=dict(n=n,fixed_or_preplaced=int(((c[:,0]!=0)|(c[:,1]!=0)).sum()),preplaced=int((c[:,1]!=0).sum()),
        groups=int(sum((c[:,3]==g).sum()>1 for g in set(c[:,3])-{0})),boundary_modules=int((c[:,4]!=0).sum()))
    output=dict(index=index,member=item['member'],counts=counts,reference={},draft={},final={},movement={})
    ref_geo,groups,boundaries=evaluate_geometry(reference,inputs,details=True)
    def add_ratios(geo):
        geo.update(bbox_reference=geo['bbox_area']/ref_geo['bbox_area'],
            hpwl_reference=geo['hpwl_total']/ref_geo['hpwl_total'] if ref_geo['hpwl_total'] else None,
            union_over_public_area=geo['actual_over_public_area']*(1-geo['duplicate_coverage']))
        return geo
    sample=next(iter(rows.values()))
    tensor_inputs=(n,*(torch.as_tensor(v) for v in inputs[1:]))
    baseline=dict(hpwl_baseline=sample['metrics']['hpwl_baseline'],area_baseline=sample['metrics']['bbox_area_baseline'])
    case=ResearchCase(f'dev{index:04d}',dict(member=item['member']),tensor_inputs,reference.tolist(),baseline)
    ref_score=evaluate_case(case,reference,0.)['metrics']
    pre=c[:,1]!=0
    if pre.any():
        delta=(reference[pre,:2]+reference[pre,2:]/2)-(specs[pre,:2]+specs[pre,2:]/2)
        ref_geo['preplaced_center_local_mean']=float((np.linalg.norm(delta,axis=1)/np.sqrt(specs[pre,2:].prod(1))).mean())
    output['reference']=dict(geometry=add_ratios(ref_geo),score=ref_score)
    for model in manifest['models']:
        key=model['key'];row=rows[key]
        if row['member']!=item['member'] or sha(ROOT/model['evaluation']/f'cases/{index:04d}.json')!=model['case_sha256'][index]:
            raise ValueError('Saved result source changed')
        for stage,positions,metrics in (('draft',row['draft_positions'],row['raw_metrics']),('final',row['positions'],row['metrics'])):
            geo,_,_=evaluate_geometry(np.asarray(positions,float),inputs,details=True)
            if not np.isclose(geo['hpwl_total'],metrics['hpwl_total'],rtol=2e-7,atol=1e-6):raise ValueError('HPWL implementation differs')
            geo=add_ratios(geo)
            if stage=='draft':
                if not np.isclose(geo['duplicate_coverage'],row['raw_geometry']['repeated_coverage_fraction'],rtol=0,atol=1e-10):
                    raise ValueError('Overlap geometry differs from saved record')
            # Public P-center drift accompanies the historical lower-left drift metric.
            pre=c[:,1]!=0
            if pre.any():
                p=np.asarray(positions,float)
                delta=(p[pre,:2]+p[pre,2:]/2)-(specs[pre,:2]+specs[pre,2:]/2)
                geo['preplaced_center_local_mean']=float((np.linalg.norm(delta,axis=1)/np.sqrt(specs[pre,2:].prod(1))).mean())
            output[stage][key]=dict(geometry=geo,score=metrics,complete_ms=row['complete_seconds']*1000 if stage=='final' else row['draft_seconds']*1000)
        output['movement'][key]=repair_movement(row['draft_positions'],row['positions'],areas)
    save(path,output)
    return index

def loss_values(values):
    from experiments.E136_frontend_F_linear.loss import A,B
    out={name:float(value[0]) for name,value in values.items()}
    out.update(reference_position=out['position'],reference_shape=.25*out['shape'],
        collision_penetration=A*out['penetration'],collision_mean_load=A*out['module_load_mean'],
        collision_max_excess=A*out['module_max_excess10'],wirelength_component=.5*B*out['wirelength'],compactness_component=.5*B*out['compactness'])
    leaves=('reference_position','reference_shape','collision_penetration','collision_mean_load','collision_max_excess',
        'wirelength_component','compactness_component','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted')
    if abs(sum(out[k] for k in leaves)-out['total'])>max(2e-6,abs(out['total'])*3e-6):raise ValueError('Loss leaf terms do not sum')
    return out

def capture_losses(manifest,beat):
    import numpy as np
    import torch
    from datasets.random_reference import RandomReference
    from experiments.E136_frontend_F_linear.data import load_rows
    from experiments.E136_frontend_F_linear.resident import ResidentRows
    from experiments.E136_frontend_F_linear.batches import inputs_only
    from experiments.E136_frontend_F_linear.loss import base_per_case
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=RandomReference('dev_monitor500');rows=load_rows('dev_monitor500')
    pack=ResidentRows(rows,'cuda',max_nodes=120)
    for model in manifest['models']:
        key=model['key'];destination=HERE/f'losses/{key}.json'
        if destination.exists():continue
        Predictor=importlib.import_module(f"experiments.{model['experiment']}.evaluation.run").Predictor
        result=[];raw_states=[]
        with torch.inference_mode():
            predictor=Predictor(ROOT/model['checkpoint'],dataset[0]['solver_inputs'])
            for i in range(500):
                item=dataset[i];record=read(ROOT/model['evaluation']/f'cases/{i:04d}.json');n=record['n']
                actual=predictor.predict(item['solver_inputs'])
                np.testing.assert_array_equal(actual,np.asarray(record['draft_positions']))
                batch=pack.batch([i])
                for name,value in inputs_only(batch).items():
                    if not torch.equal(value,predictor.g[name]):raise ValueError('Cached Loss/public model inputs differ: '+name)
                raw=loss_values(base_per_case(predictor.states[-1],batch))
                final=torch.as_tensor(record['positions'],dtype=batch['target'].dtype,device='cuda')
                state=torch.zeros_like(batch['target'])
                state[0,:n,:2]=(final[:,:2]+final[:,2:]/2)/batch['scale'][0]
                state[0,:n,2]=(final[:,2]/final[:,3]).log()
                repaired=loss_values(base_per_case(state,batch))
                entry=dict(index=i,member=record['member'],draft=raw,final=repaired)
                if key=='baseline':entry['reference']=loss_values(base_per_case(batch['target'],batch))
                result.append(entry);raw_states.append(predictor.states[-1][0].cpu().numpy().copy())
                if (i+1)%25==0:
                    beat(activity='collecting common Loss',model=key,completed=i+1,total=500)
                    print('Loss',key,i+1,'/500',flush=True)
            mean=float(np.mean([row['draft']['total'] for row in result]))
            if abs(mean-model['result']['common_final_new_dev_loss'])>5e-5:raise ValueError('Terminal Loss disagrees with saved dev monitoring')
            save(destination,dict(model=key,checkpoint_sha256=model['checkpoint_sha256'],rows=result,
                mean_draft_loss=mean,original_monitor_loss=model['result']['common_final_new_dev_loss'],terminal_drafts_bitwise_equal=True))
            np.save(HERE/f'losses/{key}_raw_states.npy',np.stack(raw_states),allow_pickle=False)
        del predictor,result,raw_states
        gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    started=time.perf_counter()
    def beat(**fields):save(HERE/'state.json',dict(status='running',elapsed_seconds=time.perf_counter()-started,**fields))
    if (HERE/'manifest.json').exists():
        if not args.resume:raise FileExistsError('Existing statistics; explicit --resume required')
        manifest=read(HERE/'manifest.json')
        if manifest['statistics_source_sha256']!=sha(__file__):raise ValueError('Statistics source changed')
    else:
        manifest=register();save(HERE/'manifest.json',manifest)
    beat(activity='starting paired geometry and Loss statistics',cases=500,models=5)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=initialize_geometry,initargs=(manifest,)) as executor:
            futures=[executor.submit(geometry_job,i) for i in range(500)]
            capture_losses(manifest,beat)
            completed=0
            for future in concurrent.futures.as_completed(futures):
                future.result();completed+=1
                if completed%25==0:
                    beat(activity='collecting exact paired geometry',completed=completed,total=500)
                    print('Geometry',completed,'/500',flush=True)
        # Reporting is kept separate from the numerical collector.
        from .report import build_report
        build_report(manifest)
        if any(sha(ROOT/name)!=digest for name,digest in manifest['sources'].items()):raise ValueError('Statistics formula changed during collection')
        for model in manifest['models']:
            if sha(ROOT/model['checkpoint'])!=model['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        save(HERE/'verification.json',dict(status='passed',paired_cases=500,models=5,geometry_stages=5500,
            model_terminal_drafts_bitwise_equal=True,loss_terms_sum_verified=True,HPWL_matches_original_records=True,
            original_overlap_metrics_match=True,reference_metrics_computed_not_forced_to_zero=True,
            original_weights_results_unchanged=True,training=False,postprocessing_calls=0,model_Q_calls=0,reference_Q_calls=500,test_read=False))
        save(HERE/'state.json',dict(status='complete',elapsed_seconds=time.perf_counter()-started,cases=500,models=5))
        print('Complete: paired reference/draft/final statistical tables.',flush=True)
    except BaseException as error:
        save(HERE/'state.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started));raise

if __name__=='__main__':main()
