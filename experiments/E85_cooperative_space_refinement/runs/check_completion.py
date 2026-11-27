"""Frozen evidence audit only: no training, model forward, solver, or new scoring."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json,time
from datetime import datetime,timezone
import numpy as np
import torch
from experiments.E85_cooperative_space_refinement.io import HERE,ROOT,save,sha,sources
from experiments.E85_cooperative_space_refinement.source_compatibility import reviewed_sources

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def member(r):return tuple(r['member'])

def main():
    started=time.perf_counter();report=dict(utc=datetime.now(timezone.utc).isoformat(),scope='saved evidence only',test_read=False,new_training_steps=0,new_forward_calls=0,new_solver_calls=0)
    parents=read(HERE/'provenance.json')['parents']
    parent=ROOT/parents['placer']['checkpoint'];assert sha(parent)==parents['placer']['sha256']
    placer=torch.load(parent,map_location='cpu',weights_only=False)['model']
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    n=np.array([r['n'] for r in features]);pin=np.array([r['pin_connected_fraction'] for r in features]);density=np.array([r['connection_density'] for r in features])
    sparse=np.array([density[i]<=np.median(density[n==count]) for i,count in enumerate(n)])
    groups={'all':n>0,'weak':pin<=.3,'weak_sparse':(pin<=.3)&sparse,'rich_dense':(pin>=.8)&~sparse}
    report['arms']={};raws={};initials={}
    migration=read(HERE/'runs/lr_policy_change/migration.json')
    for arm in ('control','cooperative'):
        folder=HERE/('runs/20261030_P74_F81_'+arm+'_seed0')
        result=read(folder/'results.json');progress=read(folder/'progress.json');frozen=read(folder/'frozen.json')
        best=torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        latest=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert result['status']=='early_stop' and latest['finished'] and progress['finished']
        assert best['identity']==latest['identity']
        assert best['progress']['epoch']==result['best_epoch']==progress['best_epoch']
        assert best['model_config']['cooperative']==(arm=='cooperative')
        assert sum(v.numel() for k,v in best['model'].items())==6071486
        assert sum(v.numel() for k,v in best['model'].items() if k.startswith('refiner.'))==241547
        for state in (best,latest):
            assert reviewed_sources(state['identity']['sources'],sources())
            for key,value in placer.items():assert torch.equal(value,state['model']['placer.'+key])
            assert int(state['progress']['seen'].sum())==32000
        assert frozen['checkpoint_sha256']==sha(folder/'checkpoints/best.pt')
        raw=read(folder/'full_dev.json');assert raw['checkpoint_sha256']==frozen['checkpoint_sha256']
        assert raw['count']==2000 and raw['postprocessing_calls']==0 and not raw['test_read']
        history=progress['history'];epochs=[r['epoch'] for r in history]
        assert epochs==list(range(progress['epoch']+1)),epochs
        assert abs(min(r['dev_monitor']['total'] for r in history)-best['progress']['best_proxy'])<=1.0001e-5
        drops=[r['epoch'] for r in history if r['lr_next']<r['lr']]
        assert len(drops)==3 and latest['scheduler']['drops']==3 and latest['scheduler']['lowest_validated']
        assert progress['epoch']-drops[-1]>=2
        final_rows={member(r):r for r in raw['records']}
        initial=read(HERE/('runs/initial_'+arm+'.json'));initial_rows={member(r):r for r in initial['records']}
        rows=[final_rows[member(r)] for r in features];first=[initial_rows[member(r)] for r in features]
        raws[arm]=rows;initials[arm]=first
        summary={}
        for name,mask in groups.items():
            selected=[r for r,m in zip(rows,mask) if m];before=[r for r,m in zip(first,mask) if m]
            pairs=sum(r['noport_order_clear_count'] for r in selected)
            summary[name]=dict(count=int(mask.sum()),initial_overlap=float(np.mean([r['overlap_area_fraction'] for r in before])),final_overlap=float(np.mean([r['overlap_area_fraction'] for r in selected])),G=float(np.mean([r['loss_common_original_G'] for r in selected])),
                noport_relative_error=float(np.mean([r['noport_near_relative_error'] for r in selected if r['noport_near_pair_count']>0])),
                noport_order_flip_rate=float(sum(r['noport_order_flip_count'] for r in selected)/pairs) if pairs else None)
        terminal_path=folder/'terminal/results.json'
        terminal=None
        if terminal_path.exists():
            t=read(terminal_path);terminal=t['summary']
            assert len(t['records'])==500 and terminal['hard_legal']==500 and terminal['errors']==0
            assert read(folder/'terminal/summary.json')['checkpoint_sha256']==frozen['checkpoint_sha256']
        gallery=folder/'four_stage100/manifest.json'
        gallery_info=None
        if gallery.exists():
            g=read(gallery);assert len(g['cases'])==100 and not g['rating_ui']
            for case in g['cases']:assert sha(gallery.parent/case['image'])==case['image_sha256']
            gallery_info=dict(count=100,all_hashes_verified=True,ids=g['indices'],axis_limits=[r['axis_limits'] for r in g['cases']])
        if arm=='control':
            for name,digest in migration['completed_control_checkpoint_hashes'].items():assert sha(ROOT/name)==digest
        else:
            oldbest=folder/'history/20261030_remove_posttrain_protection/best.pt'
            assert sha(oldbest)==migration['records'][1]['original_sha256']
            before=torch.load(oldbest,map_location='cpu',weights_only=False)
            assert all(torch.equal(value,best['model'][key]) for key,value in before['model'].items())
        report['arms'][arm]=dict(epochs=result['epochs'],best_epoch=result['best_epoch'],training_minutes=result['elapsed_seconds']/60,
            scheduler_drops=drops,lowest_tier_full_epochs=progress['epoch']-drops[-1],
            selected_best_sha256=frozen['checkpoint_sha256'],frozen_E74_tensors_verified=len(placer),
            train_seen=32000,raw_hard_legal=raw['losses']['raw_hard_legal'],fixed_dev500_initial=history[0]['dev_monitor']['total'],
            fixed_dev500_best=best['progress']['best_proxy'],groups=summary,terminal=terminal,gallery=gallery_info,
            best_precedes_LR_intervention=(arm=='cooperative' and result['best_epoch']==9))
        del best,latest
    decomposition={}
    for name,mask in groups.items():
        a=report['arms']['control']['groups'][name];b=report['arms']['cooperative']['groups'][name]
        direct=a['initial_overlap']-b['initial_overlap'];final=a['final_overlap']-b['final_overlap']
        decomposition[name]=dict(frozen_operator_absolute_reduction=direct,final_arm_absolute_reduction=final,
            control_posttrain_reduction=a['initial_overlap']-a['final_overlap'],cooperative_posttrain_reduction=b['initial_overlap']-b['final_overlap'],
            additional_posttrain_difference=final-direct,frozen_operator_share_of_final_gap=direct/final if final else None,
            limit='Descriptive decomposition with different own-Loss selection policies; not an additive causal effect estimate')
    report['decomposition']=decomposition
    a=report['arms']['control']['gallery'];b=report['arms']['cooperative']['gallery']
    if a and b:
        assert a['ids']==b['ids'] and a['axis_limits']==b['axis_limits']
        report['galleries_same_ids_and_scales']=True
    report['complete']=all(x['terminal'] and x['gallery'] for x in report['arms'].values())
    report['seconds']=time.perf_counter()-started
    save(HERE/'runs/completion/audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('complete','decomposition','arms')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
