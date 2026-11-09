"""Requested train100 gallery; frozen bests, no training/repair/Q/test access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from datasets.expanded_reference import registry,REGISTRY,ExpandedReferenceLayouts
from experiments.E61_explosion_replay.cache import load_rows
from experiments.E61_explosion_replay.batches import collate,inputs_only,decode
from experiments.E61_explosion_replay.training_model import GeometryModel
from experiments.E61_explosion_replay.losses import components,WEIGHTS
from experiments.E61_explosion_replay.diagnostics import geometry_checks
from experiments.E61_explosion_replay.inference import graph as public_graph
from visualization.layouts import plot_case,geometry_details

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def bad_blocks(positions,areas,constraints):
    p=np.asarray(positions);a=np.asarray(areas);c=np.asarray(constraints)
    lo=np.maximum(p[:,None,:2],p[None,:,:2]);hi=np.minimum((p[:,:2]+p[:,2:])[:,None],(p[:,:2]+p[:,2:])[None])
    intersection=np.maximum(hi-lo,0).prod(-1);np.fill_diagonal(intersection,0)
    load=intersection.sum(1)/np.maximum(a,1e-12)
    return [{'block':int(i),'overlap_load':float(load[i]),'mib':int(c[i,2]),'group':int(c[i,3])}
            for i in np.argsort(-load,kind='stable')[:10]]

def init_render():
    global RECORDS
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    RECORDS={r['train_index']:r for r in read(HERE/'drafts.json')['records']}

def render(index,force=False):
    r=RECORDS[index];target=HERE/'images'/f'train_{index:05d}.png'
    if force or not target.exists():
        c=r['constraints'];variants={k:{'positions':r[k],'soft_details':geometry_details(r[k],c)} for k in ('reference','current','previous')}
        plot_case({'panel':'E61 training-set draft inspection','case_id':f'train_{index:05d}',
            'constraints':c,'variants':variants,'caption':'Training member | Original reference before mean-area transform | E61 best60 draft | Frozen E42 best51 draft'},
            target,show_metrics=False,stages=(('reference','Reference'),('current','E61 best60 train draft'),('previous','E42 best51 draft')))
    return {'train_index':index,'image':target.relative_to(HERE).as_posix(),'image_sha256':sha(target)}

def build_viewer(manifest):
    payload=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    template=(HERE/'viewer_template.html').read_text(encoding='utf-8')
    (HERE/'index.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(20261027);torch.use_deterministic_algorithms(True)
    (HERE/'images').mkdir(exist_ok=True)
    old_elapsed=read(HERE/'progress.json').get('elapsed_seconds',0) if (HERE/'progress.json').exists() else 0
    def elapsed():return old_elapsed+time.perf_counter()-started
    def guard():
        if elapsed()>1100:raise RuntimeError('1100-second cumulative gallery budget reached; saved outputs are resumable')
    paths={'current':HERE.parent/'checkpoints/best.pt',
           'previous':ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'}
    expected={'current':'27507c79a82e4af174cca4c29b47b5a4e2009e2c4ae6e99cdd80c622daad7e83',
              'previous':'12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'}
    for key,p in paths.items():assert sha(p)==expected[key]
    checkpoints={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in paths.items()}
    cp=checkpoints['current'];data=registry();train=data['sets']['train']['records'];assert len(train)==64000
    natural=cp['identity']['subset_indices'];assert natural==list(range(64000))
    eligible=[natural[i] for i,seen in enumerate(cp['progress']['seen']) if seen]
    ids=random.Random(20261027).sample(eligible,100)
    base_registry=read(ROOT/'protocol/reference_workset_v5.json');previous_cp=checkpoints['previous']
    assert previous_cp['identity']['registry_sha256']==sha(ROOT/'protocol/reference_workset_v5.json')
    old_indices=previous_cp['identity'].get('subset_indices',list(range(16000)))
    old_seen={tuple(base_registry['sets']['train']['records'][old_indices[i]]['member']) for i,v in enumerate(previous_cp['progress']['seen']) if v}
    config={'request':'100 training-member reference/E61/E42 draft plots, no rating controls','seed':20261027,'selection':'uniform random sample from actual members seen by frozen E61 best; selected before inference, no quality filtering',
        'train_size':64000,'best_seen_count':len(eligible),'train_indices':ids,'members':[train[i]['member'] for i in ids],
        'registry_sha256':sha(REGISTRY),'checkpoint_sha256':expected,'checkpoint_paths':{k:str(p.relative_to(ROOT)) for k,p in paths.items()},
        'E42_exposure_note':'Labels describe membership seen in these two runs only, not all historical exposure. Added48k is outside this E42 run training set.',
        'reference_note':'Original reference positions before mean-area transform; not guaranteed legal under derived public areas',
        'sampling_panel':'training, separate from existing development100','rating_ui':False,'postprocessing':False,'Q_computed':False,'test_read':False,
        'threads':{'inference':4,'render_workers':4,'threads_per_render_worker':1},'cumulative_max_seconds':1100}
    if (HERE/'config.json').exists():assert read(HERE/'config.json')==config
    else:save(HERE/'config.json',config)
    records=[];parity=[]
    if (HERE/'drafts.json').exists():
        records=read(HERE/'drafts.json')['records'];assert [r['train_index'] for r in records]==ids
    else:
        rows=load_rows('train',ids);original_data=ExpandedReferenceLayouts('train')
        models={k:GeometryModel(**c['model_config']).cuda().eval() for k,c in checkpoints.items()}
        for key,m in models.items():m.load_state_dict(checkpoints[key]['model'])
        with torch.no_grad():
            for offset in range(0,100,32):
                guard();selected=rows[offset:offset+32]
                g=collate([r['graph'] for r in selected],list(range(len(selected))),'cuda',max_nodes=120)
                state={k:m(inputs_only(g)) for k,m in models.items()}
                component={k:{name:v.detach().cpu().tolist() for name,v in components(s,g).items() if name in WEIGHTS} for k,s in state.items()}
                positions={k:(decode(s,g)*g['scale'][:,None,None]).cpu().double().numpy() for k,s in state.items()}
                if offset==0:
                    for key in models:
                        cpu=GeometryModel(**checkpoints[key]['model_config']).eval();cpu.load_state_dict(checkpoints[key]['model'])
                        maximum=0.
                        for j,row in enumerate(selected[:5]):
                            raw=cpu(public_graph(row['case'].solver_inputs))[0]
                            expected_state=state[key][j,:row['graph']['n']].cpu()
                            torch.testing.assert_close(raw,expected_state,rtol=1e-4,atol=2e-5)
                            maximum=max(maximum,float((raw-expected_state).abs().max()))
                        parity.append({'model':key,'unique_train_cases':5,'CPU_unpadded_vs_GPU_padded_state_max_abs_delta':maximum})
                for j,row in enumerate(selected):
                    index=ids[offset+j];n=row['graph']['n'];member=list(row['member']);assert member==train[index]['member']
                    c=np.asarray(row['case'].solver_inputs[5]);reference_item=original_data[index]
                    assert list(reference_item['member'])==member
                    ref=reference_item['positions']
                    record={'train_index':index,'member':member,'n':n,'constraints':c.tolist(),'reference':ref.tolist(),
                        'shared_training_member':tuple(member) in old_seen,'E61_best_seen':True,'metrics':{}}
                    for key in ('current','previous'):
                        pos,checks,diagnostic=geometry_checks(positions[key][j,:n],row['case'].solver_inputs)
                        record[key]=pos.tolist();record['metrics'][key]={'G':sum(WEIGHTS[k]*component[key][k][j] for k in WEIGHTS),
                            'loss_components':{k:component[key][k][j] for k in WEIGHTS},'hard_legal':bool(checks['is_feasible']),**diagnostic,
                            'most_overlapped_blocks':bad_blocks(pos,row['case'].solver_inputs[1],c)}
                    records.append(record)
                print(json.dumps({'inferred_cases':len(records),'elapsed_seconds':elapsed()}),flush=True)
        save(HERE/'drafts.json',{'records':records,'test_read':False,'postprocessing':False,'Q_computed':False})
        save(HERE/'preflight.json',{'batch_member_alignment':True,'single_vs_padded_inference':parity,'unique_check_cases':5,'test_read':False})
        del rows,models,checkpoints;torch.cuda.empty_cache()
    save(HERE/'progress.json',{'status':'rendering','elapsed_seconds':elapsed(),'case_count':100})
    rendered=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_render) as pool:
        for item in pool.map(render,ids):
            guard();rendered.append(item)
            if len(rendered)%10==0:
                save(HERE/'progress.json',{'status':'rendering','rendered':len(rendered),'elapsed_seconds':elapsed(),'case_count':100})
                print(json.dumps({'rendered':len(rendered),'elapsed_seconds':elapsed()}),flush=True)
    by_id={r['train_index']:r for r in records};cases=[]
    for image in rendered:
        r=by_id[image['train_index']];cases.append({**image,'member':r['member'],'n':r['n'],'shared_training_member':r['shared_training_member'],'metrics':r['metrics']})
    summary={'count':100,'shared_training_members':sum(r['shared_training_member'] for r in records),'E61_only_training_members':sum(not r['shared_training_member'] for r in records),
        'by_model':{key:{'mean_G':float(np.mean([r['metrics'][key]['G'] for r in records])),
            'mean_overlap_fraction':float(np.mean([r['metrics'][key]['overlap_area_fraction'] for r in records])),
            'raw_hard_legal':sum(r['metrics'][key]['hard_legal'] for r in records)} for key in ('current','previous')}}
    for origin in ('shared','added'):
        rr=[r for r in records if r['shared_training_member']==(origin=='shared')]
        summary[origin]={'count':len(rr),'E61_mean_G':float(np.mean([r['metrics']['current']['G'] for r in rr])),
            'E42_mean_G':float(np.mean([r['metrics']['previous']['G'] for r in rr])),
            'E61_mean_overlap':float(np.mean([r['metrics']['current']['overlap_area_fraction'] for r in rr])),
            'E42_mean_overlap':float(np.mean([r['metrics']['previous']['overlap_area_fraction'] for r in rr]))}
    manifest={'title':'训练集100例：Reference / E61 / E42','note':'固定随机抽样，所有案例均已被E61 best60实际见过；默认原抽样顺序，可排序定位困难案例。','cases':cases,'summary':summary,'rating_ui':False,'test_read':False,
        'reference_coordinate_source':'Direct original unnormalized training labels; no cache scale round-trip'}
    save(HERE/'manifest.json',manifest);build_viewer(manifest)
    for key,p in paths.items():assert sha(p)==expected[key]
    save(HERE/'summary.json',{'status':'complete',**summary,'elapsed_seconds':elapsed(),'checkpoint_unchanged':True,'direct_original_reference':True,'no_training':True,'no_postprocessing_or_Q':True,'test_read':False})
    save(HERE/'progress.json',{'status':'complete','rendered':100,'elapsed_seconds':elapsed(),'case_count':100})
    print(json.dumps({'status':'complete','summary':summary,'elapsed_seconds':elapsed(),'gallery':str(HERE/'index.html')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
