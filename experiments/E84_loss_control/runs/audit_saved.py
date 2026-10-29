"""Audit selected best, frozen placers, exports, raw geometry and saved comparisons."""
import json,sys
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.E84_loss_control.io import HERE,save,sha,sources
from experiments.E84_loss_control.initialization import verify_protected,provenance
from experiments.E84_loss_control.model import GeometryModel,DeploymentModel,CorrectionModel,deployment_weights,decode
from experiments.E84_loss_control.data import load_rows,collate
from experiments.E84_loss_control.batches import inputs_only
from experiments.E84_loss_control.train import seed_all
OUT=HERE/'runs/20261029_same_initialization_G_only_seed0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def compare(a,b,key):
    assert [r['member'] for r in a]==[r['member'] for r in b]
    x=np.array([key(r) for r in a]);y=np.array([key(r) for r in b]);d=x-y
    workers=np.array([r['member'][0].split('/')[0] for r in a]);groups=np.unique(workers)
    sums=np.array([d[workers==g].sum() for g in groups]);counts=np.array([(workers==g).sum() for g in groups])
    picks=np.random.default_rng(20261029).integers(len(groups),size=(4000,len(groups)))
    draws=sums[picks].sum(1)/counts[picks].sum(1)
    return {'count':len(a),'current_mean':float(x.mean()),'baseline_mean':float(y.mean()),'mean_delta':float(d.mean()),
      'worker_resampling95':np.quantile(draws,[.025,.975]).tolist(),
      'better_equal_worse':[int((d< -1e-8).sum()),int((abs(d)<=1e-8).sum()),int((d>1e-8).sum())]}
def geometry(p):
    p=np.asarray(p,dtype=float);lo=p[:,:2];hi=lo+p[:,2:];area=p[:,2:].prod(-1).sum();bbox=(hi.max(0)-lo.min(0)).prod()
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);union=0.
    for left,right in zip(xs[:-1],xs[1:]):
        active=(lo[:,0]<right)&(hi[:,0]>left);end=-np.inf;length=0.
        for bottom,top in sorted(zip(lo[active,1],hi[active,1])):
            length+=max(0.,top-max(bottom,end));end=max(end,top)
        union+=(right-left)*length
    return {'bbox_over_A':float(bbox/area),'void_over_bbox':float((bbox-union)/bbox)}
def main():
    seed_all(0,4);verify_protected();review=read(HERE/'runs/review.json');raw=read(OUT/'full_dev.json');flash=read(OUT/'terminal/results.json')
    best=torch.load(OUT/'checkpoints/best.pt',map_location='cpu',weights_only=False,mmap=True)
    latest=torch.load(OUT/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert best['identity']==latest['identity'] and best['identity']['sources']==sources()
    assert latest['finished'] and latest['status']=='early_stop' and latest['scheduler']['drops']==3 and latest['scheduler']['lowest_completed_epochs']>=2
    digest=sha(OUT/'checkpoints/best.pt')
    assert digest==raw['checkpoint_sha256']==read(OUT/'frozen.json')['checkpoint_sha256']==read(OUT/'terminal/config.json')['checkpoint_sha256']
    parents=provenance()['parents'];p42=torch.load(ROOT/parents['placer']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    parent=torch.load(ROOT/parents['correction']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    count=0
    for k,v in p42['model'].items():
        assert torch.equal(v,best['model']['placer.'+k]) and torch.equal(v,latest['model']['placer.'+k]);count+=1
    prefix='source.source.backbone.'
    for k,v in parent['model'].items():
        if k.startswith(prefix):
            assert torch.equal(v,best['model']['alternative.'+k[len(prefix):]]) and torch.equal(v,latest['model']['alternative.'+k[len(prefix):]]);count+=1
    deployment=torch.load(OUT/'deployment.pt',map_location='cpu',weights_only=False,mmap=True)
    exported=torch.load(OUT/'correction_F.pt',map_location='cpu',weights_only=False,mmap=True)
    assert deployment['parent_sha256']==exported['parent_sha256']==digest
    for k,v in deployment['model'].items():assert torch.equal(v,best['model'][k])
    for k,v in exported['model'].items():assert torch.equal(v,best['model']['refiner.'+k])
    dep=DeploymentModel(**deployment['model_config']).eval();dep.load_state_dict(deployment['model'],strict=True)
    f=CorrectionModel(**exported['config']).eval();f.load_state_dict(exported['model'],strict=True)
    rows=load_rows('train',[1683,10027,5474,18140,25001]);g,_=collate(rows,'cpu');pub=inputs_only(g)
    with torch.no_grad():torch.testing.assert_close(dep(pub),f(dep.placer(pub),pub),rtol=0,atol=0)
    e80path=ROOT/'experiments/E81_multisource_refinement/runs/20261029_e42_multisource_seed0'
    e80raw=read(e80path/'full_dev.json');e80flash=read(e80path/'terminal/results.json')
    old=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/forwards.json')
    assert old['identities']['E42']['checkpoint_sha256']==parents['placer']['sha256']
    old_raw=old['models']['E42'];features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    assert [r['member'] for r in features]==[r['member'] for r in raw['records']]
    medians={n:np.median([r['connection_density'] for r in features if r['n']==n]) for n in {r['n'] for r in features}}
    ids={'low_pin':[i for i,r in enumerate(features) if r['pin_connected_fraction']<=.3],
         'low_pin_sparse':[i for i,r in enumerate(features) if r['pin_connected_fraction']<=.3 and r['connection_density']<=medians[r['n']]]}
    subgroups={name:{'vs_E42':compare([raw['records'][i] for i in ix],[old_raw[i] for i in ix],lambda r:r['overlap_area_fraction']),
                    'vs_E81':compare([raw['records'][i] for i in ix],[e80raw['records'][i] for i in ix],lambda r:r['overlap_area_fraction'])} for name,ix in ids.items()}
    # Reuse authenticated frozen P42 source coordinates; only five fresh train
    # forwards above are needed. No dev network/solver/score is recomputed.
    source_dir=ROOT/'experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit'
    source_meta=read(source_dir/'source_metrics.json');source_pack=np.load(source_dir/'source_drafts.npz')
    source_lookup={tuple(m):i for i,m in enumerate(source_meta['members'])}
    frame_excess=[];now=[];previous=[]
    for saved,prior in zip(flash['records'],e80flash['records']):
        assert saved['member']==prior['member']
        index=source_lookup[tuple(saved['member'])];q=source_pack['E42'][index,:saved['n']]
        p=np.asarray(saved['draft_positions'])
        excess=max(0.,float((q[:,:2].min(0)-p[:,:2].min(0)).max()),float(((p[:,:2]+p[:,2:]).max(0)-(q[:,:2]+q[:,2:]).max(0)).max()))/np.sqrt(q[:,2:].prod(1).sum())
        frame_excess.append(float(excess));now.append({'member':saved['member'],**geometry(p)})
        previous.append({'member':prior['member'],**geometry(prior['draft_positions'])})
    assert max(frame_excess)<3e-6
    initial=review['initial_monitor'];selected=review['best_monitor'];progress=read(OUT/'progress.json')
    manifest=read(OUT/'draft_comparison100/manifest.json');images=[]
    for i in (217,399,194):
        row=next(r for r in manifest['cases'] if r['id']==i);assert sha(OUT/'draft_comparison100'/row['image'])==row['image_sha256'];images.append({'index':i,'sha256':row['image_sha256']})
    old_review=read(e80path.parent/'review.json')
    initial_keys=('common_original_G','center_local','shape','relative_local','overlap','contact','boundary','mib','bbox','true_intersection','box_expansion','anchored_corridor','P74_geometry','P74_true_intersection','P74_anchored_corridor')
    initial_errors={role:{k:abs(initial[role][k]-old_review['initial_monitor'][role][k]) for k in initial_keys} for role in ('train_monitor','dev_monitor')}
    assert max(v for role in initial_errors.values() for v in role.values())<2e-7
    old_fixed=[r['id'] for r in read(e80path/'draft_comparison100/manifest.json')['cases']]
    assert [r['id'] for r in manifest['cases']]==old_fixed and len(manifest['cases'])==100
    for row in manifest['cases']:
        assert sha(OUT/'draft_comparison100'/row['image'])==row['image_sha256'] and len(row['axes_bounds'])==4
    assert latest['progress']['seen'].sum()==32000
    result={'status':'complete_audited','best_sha256':digest,'best_epoch':best['progress']['epoch'],'latest_steps':latest['progress']['step'],
      'training_seconds':review['training']['elapsed_seconds'],'frozen_placer_tensors_bit_exact_best_and_latest':count,'sources_unchanged':True,'initial_monitor_shared_diagnostics_max_error':max(v for role in initial_errors.values() for v in role.values()),'initial_monitor_shared_diagnostics_errors':initial_errors,'fixed100_and_all_image_hashes_verified':True,'fresh_model_forward_train_cases':5,'new_dev_model_forward_calls':0,'new_terminal_solver_calls':0,
      'exports_bit_exact_to_unique_best':True,'F_only_and_deployment_forward_equal_five_cases':True,'viewed_images':images,
      'loss_improvement':{role:{'initial':initial[role]['total'],'best':selected[role]['total'],'fraction':1-selected[role]['total']/initial[role]['total']} for role in ('train_monitor','dev_monitor')},
      'source_components':{role:{key:{'initial':initial[role][key],'best':selected[role][key]} for key in ('P42_total','P74_total','true_intersection','anchored_corridor','P74_true_intersection','P74_anchored_corridor','common_original_G')} for role in ('train_monitor','dev_monitor')},
      'train_dev_relative_gap':selected['dev_monitor']['total']/selected['train_monitor']['total']-1,
      'paired_overlap2000_vs_E42':compare(raw['records'],old_raw,lambda r:r['overlap_area_fraction']),
      'paired_overlap2000_vs_E81':compare(raw['records'],e80raw['records'],lambda r:r['overlap_area_fraction']),
      'paired_raw_loss_vs_E81':{key:compare(raw['records'],e80raw['records'],lambda r,key=key:r[key]) for key in ('loss_common_original_G','loss_center_local','loss_relative_local','loss_shape')},
      'paired_overlap_subgroups':subgroups,'subgroup_definition':'E69 pin_connected_fraction<=.3; sparse connection_density<=median at same n in dev2000',
      'paired_geometry500_vs_E81':{key:compare(now,previous,lambda r,key=key:r[key]) for key in ('bbox_over_A','void_over_bbox')},
      'frame_containment500':{'source':'authenticated saved own public E42 prediction; source weights unchanged, FP32 tolerance applies','max_normalized_excess':max(frame_excess),'tolerance':3e-6,'violations':0},
      'lr_drops':[{'epoch':r['epoch'],'lr_next':r['lr_next']} for r in progress['history'] if r['lr_next']!=r['lr']],
      'lowest_lr_completed_epochs':latest['scheduler']['lowest_completed_epochs'],'flash':review['flash'],'paired_Q':review['paired_Q'],
      'raw_hard_legal':raw['losses']['raw_hard_legal'],'test_read':False,'baseline_promoted':False,
      'limits':'Same E81 initial recipe, architecture, data and two-source pipeline; only objective and matching selection package changed to G-only. Not per-term or reference-free attribution; warm F79 contains prior geometry training. Adaptive single-seed development; existing results reused, no new scoring or checkpoint selection.'}
    save(HERE/'runs/completion/inspection.json',result)
    print(json.dumps({k:result[k] for k in ('loss_improvement','source_components','paired_overlap2000_vs_E42','paired_overlap2000_vs_E81','paired_geometry500_vs_E81','frame_containment500')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
