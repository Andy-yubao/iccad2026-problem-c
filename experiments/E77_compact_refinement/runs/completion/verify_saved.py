"""Read-only final audit and paired resampling of already-saved evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.E77_compact_refinement.io import HERE,ROOT,save,sources,sha
from experiments.E77_compact_refinement.initialization import verify_protected



def read(path):return json.loads(path.read_text(encoding='utf-8'))
def paired(records,baseline,key,weighted=False):
    assert [x['member'] for x in records]==[x['member'] for x in baseline]
    a=np.asarray([key(x) for x in records]);b=np.asarray([key(x) for x in baseline]);d=a-b
    n=np.asarray([x.get('n',1) for x in records]);w=np.exp((n-n.max())/12) if weighted else np.ones(len(n))
    groups={}
    for i,r in enumerate(records):groups.setdefault(r['member'][0].split('/')[0],[]).append(i)
    blocks=list(groups.values());rng=np.random.default_rng(20261029);draws=[]
    for _ in range(4000):
        ids=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]);draws.append(float(np.average(d[ids],weights=w[ids])))
    return {'count':len(d),'current':float(np.average(a,weights=w)),'baseline':float(np.average(b,weights=w)),
        'delta':float(np.average(d,weights=w)),'worker_resampling95':np.percentile(draws,[2.5,97.5]).tolist(),
        'better_equal_worse':[int((d< -1e-8).sum()),int((abs(d)<=1e-8).sum()),int((d>1e-8).sum())]}

def main():
    folder=HERE/'runs/20261029_e76_compact_corrector_seed0';baseline=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'
    best=torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False,mmap=True)
    latest=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert latest['finished'] and latest['status']=='early_stop' and best['identity']==latest['identity']
    assert best['identity']['sources']==sources();verify_protected()
    parent=ROOT/'experiments/E76_bounded_refinement/runs/20261029_frozen_e74_corrector_seed0'
    source=torch.load(parent/'checkpoints/best.pt',map_location='cpu',weights_only=False,mmap=True)
    graph=['source.'+key for key in source['model']]
    assert all(torch.equal(source['model'][key],best['model']['source.'+key]) and torch.equal(source['model'][key],latest['model']['source.'+key]) for key in source['model'])
    digest=sha(folder/'checkpoints/best.pt');frozen=read(folder/'frozen.json');raw=read(folder/'full_dev.json');flash=read(folder/'terminal/results.json')
    assert digest==frozen['checkpoint_sha256']==raw['checkpoint_sha256']==read(folder/'terminal/summary.json')['checkpoint_sha256']
    assert raw['count']==2000 and len(flash['records'])==500
    old_raw=read(baseline/'full_dev.json');old_flash=read(baseline/'terminal/results.json')
    q=paired(flash['records'],old_flash['records'],lambda r:r['metrics']['cost'],weighted=True)
    arithmetic=paired(flash['records'],old_flash['records'],lambda r:r['metrics']['cost'])
    # E42's original full-dev stores aggregates. Reuse E69's authenticated
    # saved E42 per-case forwards rather than creating new baseline inference.
    saved=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/forwards.json')
    assert saved['identities']['E42']['checkpoint_sha256']==sha(baseline/'checkpoints/best.pt')
    old_records=saved['models']['E42']
    overlap=paired(raw['records'],old_records,lambda r:r['overlap_area_fraction'])
    assert abs(overlap['baseline']-old_raw['losses']['draft_diagnostics']['overlap_area_fraction']['mean'])<1e-6
    metrics={k:paired(raw['records'],old_records,lambda r,k=k:r[k]) for k in ('loss_center_local','loss_relative_local','loss_shape')}
    penetration=paired(flash['records'],old_flash['records'],lambda r:r['draft_diagnostics']['max_local_penetration'])
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    assert [r['member'] for r in features]==[r['member'] for r in raw['records']]
    medians={n:np.median([f['connection_density'] for f in features if f['n']==n]) for n in {f['n'] for f in features}}
    subsets={'low_pin_coverage':[i for i,f in enumerate(features) if f['pin_connected_fraction']<=.3],
        'low_pin_sparse':[i for i,f in enumerate(features) if f['pin_connected_fraction']<=.3 and f['connection_density']<=medians[f['n']]]}
    subgroup={name:paired([raw['records'][i] for i in ids],[old_records[i] for i in ids],lambda r:r['overlap_area_fraction']) for name,ids in subsets.items()}
    assert abs(q['current']-flash['summary']['weighted_Q'])<1e-10
    assert abs(q['baseline']-old_flash['summary']['weighted_Q'])<1e-10
    manifest=read(folder/'draft_comparison100/manifest.json');assert len(manifest['cases'])==100
    for i in (217,399,194):
        row=next(x for x in manifest['cases'] if x['id']==i);assert sha(folder/'draft_comparison100'/row['image'])==row['image_sha256']
    old_map={r['index']:r for r in old_flash['records']};regressions=[]
    for r in flash['records']:
        old=old_map[r['index']];regressions.append({'index':r['index'],'n':r['n'],'member':r['member'],'current_Q':r['metrics']['cost'],'baseline_Q':old['metrics']['cost'],'delta':r['metrics']['cost']-old['metrics']['cost']})
    progress=read(folder/'progress.json')
    result={'status':'verified_development_candidate','best_sha256':digest,'best_epoch':best['progress']['epoch'],'latest_steps':latest['progress']['step'],
        'training_stop':'early_stop under fixed scheduler','training_seconds':read(folder/'results.json')['elapsed_seconds'],
        'sources_unchanged':True,'frozen_source_tensors_bit_exact':len(graph),'lr_drops':[{'epoch':r['epoch'],'lr_next':r['lr_next']} for r in progress['history'] if r['lr_next']!=r['lr']],
        'lowest_completed_epochs':latest['scheduler']['lowest_completed_epochs'],'paired_weighted_Q500':q,'paired_arithmetic_Q500':arithmetic,'paired_overlap2000':overlap,'paired_raw_metrics2000':metrics,
        'paired_max_penetration500':penetration,'paired_overlap_subgroups':subgroup,
        'baseline_raw_records_source':'E69 authenticated frozen E42 per-case forwards; no fresh inference',
        'subgroup_definition':'Prior E69 low pin<=.3; sparse means connection density<=median at same n in dev2000',
        'common_G2000':{'current':raw['losses']['common_original_G'],'baseline':old_raw['losses']['geometry']},
        'top_Q_regressions':sorted(regressions,key=lambda x:x['delta'],reverse=True)[:10],
        'top_Q_improvements':sorted(regressions,key=lambda x:x['delta'])[:10],
        'all100_images_exist':all((folder/'draft_comparison100'/r['image']).exists() for r in manifest['cases']),
        'viewed_images':[217,399,194],'test_read':False,'baseline_promoted':False,
        'limits':'Adaptive single-seed development evidence. Worker resampling describes this exposed panel, not repeated-run/seed or unseen generalization uncertainty.'}
    save(HERE/'runs/completion/inspection.json',result)
    parent_flash=read(parent/'terminal/results.json')
    result['paired_weighted_Q_vs_E76']=paired(flash['records'],parent_flash['records'],lambda r:r['metrics']['cost'],weighted=True)
    result['paired_overlap_vs_E76']=paired(raw['records'],read(parent/'full_dev.json')['records'],lambda r:r['overlap_area_fraction'])
    save(HERE/'runs/completion/inspection.json',result)
    previous74=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0'
    result['paired_weighted_Q_vs_E74']=paired(flash['records'],read(previous74/'terminal/results.json')['records'],lambda r:r['metrics']['cost'],weighted=True)
    result['paired_overlap_vs_E74']=paired(raw['records'],read(previous74/'full_dev.json')['records'],lambda r:r['overlap_area_fraction'])
    review=read(HERE/'runs/review.json')
    result['paired_geometry_vs_E74']=review['paired_geometry_vs_E74']
    result['paired_geometry_vs_E76']=review['paired_geometry_vs_E76']
    result['paired_geometry_vs_E42_gallery']=review['paired_geometry_vs_E42_gallery']
    result['public_frame_containment500']=review['saved_public_frame_containment500']
    result['train_improvement_fraction']=review['train_improvement_fraction']
    result['dev_improvement_fraction']=review['dev_improvement_fraction']
    result['train_dev_relative_gap']=review['train_dev_relative_gap']
    result['best_train_dev']=[review['best_monitor']['train_monitor']['total'],review['best_monitor']['dev_monitor']['total']]
    result['initial_train_dev']=[review['initial_monitor']['train_monitor']['total'],review['initial_monitor']['dev_monitor']['total']]
    charged=read(HERE/'runs/recovery/startup_interruption.json')['charged_interruption_seconds']
    result['budget_charged_interruption_seconds']=charged
    result['measured_resumed_training_stage_seconds']=result['training_seconds']-charged
    result['training_seconds_note']='Budget total includes unknown startup interruption conservatively charged; resumed-stage measurement excludes that charge.'
    result['training_seen_count']=latest['progress']['seen'].sum().item()
    result['terminal_hard_legal']=read(folder/'terminal/summary.json')['hard_legal']
    result['raw_hard_legal']=raw['losses']['raw_hard_legal']
    result['automatic_report_failed_only_serialization']=True
    result['report_recovered_without_production_source_change_or_re_evaluation']=True
    save(HERE/'runs/completion/inspection.json',result)
    print('weightedQ',q,'overlap2000',overlap,'G',result['common_G2000'],flush=True)
    print('worst regressions',result['top_Q_regressions'][:3],flush=True)
if __name__=='__main__':main()
