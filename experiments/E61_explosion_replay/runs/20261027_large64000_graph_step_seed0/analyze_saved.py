"""Finalize saved evidence; no model inference, training, solver or test access."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
FOLDER=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def change(a,b):return b/a-1 if a else None
def stats(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'p95':float(np.percentile(x,95)),'max':float(x.max()),'worst5pct_mean':float(np.sort(x)[-int(np.ceil(.05*len(x))):].mean())}

def main():
    old=ROOT/'experiments/E59_centerline_overlap/runs/20261027_large64000_original_seed0'
    baseline=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'
    p=read(FOLDER/'progress.json');r=read(FOLDER/'results.json');full=read(FOLDER/'full_dev.json');summary=read(FOLDER/'terminal/summary.json')
    health=read(FOLDER/'health.json');best=next(x for x in p['history'] if x['epoch']==r['best_epoch'])
    cp=torch.load(FOLDER/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert cp['finished'] and cp['identity']['graph_prenorm'] is False and p['seen_count']==64000
    for path,digest in cp['identity']['sources'].items():assert sha(ROOT/path)==digest,path
    best_hash=sha(FOLDER/'checkpoints/best.pt');assert best_hash==full['checkpoint_sha256']==summary['checkpoint_sha256']
    protected=read(old/'review.json')['protected_original_checkpoints_verified']
    protected[str((old/'checkpoints/best.pt').relative_to(ROOT)).replace('\\','/')]=read(old/'terminal/summary.json')['checkpoint_sha256']
    source_replay=read(FOLDER.parent/'20261027_seed0_replay/config.json')
    protected[str(Path(source_replay['source']).relative_to(ROOT)).replace('\\','/')]=source_replay['source_sha256']
    for path,digest in protected.items():assert sha(ROOT/path)==digest,path
    all_records={label:read(folder/'terminal/results.json')['records'] for label,folder in [('E42',baseline),('E59',old),('E61',FOLDER)]}
    maps={label:{tuple(x['member']):x for x in rr} for label,rr in all_records.items()}
    assert maps['E42'].keys()==maps['E59'].keys()==maps['E61'].keys()
    raw={};paired={}
    for key in ['overlap_area_fraction','overlap_pairs','penetration_squared_per_block','largest_conflict_cluster_fraction']:
        raw[key]={label:stats([x['draft_diagnostics'][key] for x in rr]) for label,rr in all_records.items()}
    for label in ('E42','E59'):
        qa=[];oa=[]
        for member,x in maps['E61'].items():
            qa.append(x['metrics']['cost']-maps[label][member]['metrics']['cost'])
            oa.append(x['draft_diagnostics']['overlap_area_fraction']-maps[label][member]['draft_diagnostics']['overlap_area_fraction'])
        paired[label]={'Q_improved':int((np.asarray(qa)<-1e-9).sum()),'Q_tied':int((np.abs(qa)<=1e-9).sum()),'Q_regressed':int((np.asarray(qa)>1e-9).sum()),
            'raw_overlap_improved':int((np.asarray(oa)<0).sum()),'raw_overlap_regressed':int((np.asarray(oa)>0).sum())}
    old_full=read(old/'full_dev.json')['losses'];old_p=read(old/'progress.json');old_best=next(x for x in old_p['history'] if x['epoch']==old_p['best_epoch'])
    summaries={'E61':summary,'E59':read(old/'terminal/summary.json'),'E42':read(baseline/'terminal/summary.json')}
    ids=[x['id'] for x in read(FOLDER/'draft_comparison100/manifest.json')['cases']]
    current={x['index']:x for x in all_records['E61']};base={x['index']:x for x in all_records['E42']}
    ordered=sorted(ids,key=lambda i:current[i]['draft_diagnostics']['overlap_area_fraction'])
    selections=[('median_overlap',ordered[len(ordered)//2]),('worst_overlap',ordered[-1]),
        ('largest_Q_regression',max(ids,key=lambda i:current[i]['metrics']['cost']-base[i]['metrics']['cost'])),
        ('largest_Q_improvement',min(ids,key=lambda i:current[i]['metrics']['cost']-base[i]['metrics']['cost']))]
    selected=[]
    for reason,index in selections:
        if index not in [x['index'] for x in selected]:selected.append({'reason':reason,'index':index,'E61_Q':current[index]['metrics']['cost'],'E42_Q':base[index]['metrics']['cost'],
            'E61_overlap':current[index]['draft_diagnostics']['overlap_area_fraction'],'image':f'draft_comparison100/images/case_{index:04d}.png'})
    report={'status':'final_review','created_utc':datetime.now(timezone.utc).isoformat(),'training':r,'best_step':best['step'],
        'best_train500_G':best['train_monitor']['total'],'best_dev500_G':best['dev_monitor']['total'],'full_dev2000':full['losses'],
        'health':health,'best_checkpoint_sha256':best_hash,'protected_checkpoints_verified':protected,'frozen_sources_verified':True,
        'configuration':cp['model_config'],'parameters':cp['identity']['parameters'],'latest_seen_count':p['seen_count'],
        'E59_full_dev2000_G':old_full['total'],'E59_best_train500_G':old_best['train_monitor']['total'],'E59_best_dev500_G':old_best['dev_monitor']['total'],
        'full_dev_G_change_vs_E59':change(old_full['total'],full['losses']['total']),
        'full_dev_overlap_change_vs_E59':change(old_full['draft_diagnostics']['overlap_area_fraction']['mean'],full['losses']['draft_diagnostics']['overlap_area_fraction']['mean']),
        'same500_raw':raw,'paired_same500':paired,'flash500_summaries':summaries,
        'flash_change_vs_E42':{k:change(summaries['E42'][k],summary[k]) for k in ('weighted_Q','mean_Q','p95_Q','max_Q','worst5pct_mean_Q','seconds_mean')},
        'flash_change_vs_E59':{k:change(summaries['E59'][k],summary[k]) for k in ('weighted_Q','mean_Q','p95_Q','max_Q','worst5pct_mean_Q','seconds_mean')},
        'visual_review_candidates':selected,'visual_review_kind':'AI inspection of existing selected same-case images; not human ratings',
        'baseline_promotion':False,'test_read':False,'new_training_or_inference_or_postprocessing':False,
        'limits':['One adaptive-development seed, tests sealed','E59 comparison changes stabilization strategy and optimization trajectory; not pure normalization causality',
            'E61 finishes31500 updates versus E59 23500; additional compute must be disclosed','Raw legality differs from repaired legality',
            'Small flashQ differences are not robust evidence by themselves given documented identical-weight execution sensitivity',
            'No human draft judgment has been collected for this new model; no automatic baseline promotion']}
    (FOLDER/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.4),layout='constrained')
    for label,progress,color in [('E59 input prenorm',old_p,'#d17a2c'),('E61 graph-step protection',p,'#277da2')]:
        hh=[x for x in progress['history'] if x['step']>0]
        axes[0].plot([x['step'] for x in hh],[x['dev_monitor']['total'] for x in hh],color=color,label=label+' dev')
        axes[0].plot([x['step'] for x in hh],[x['train_monitor']['total'] for x in hh],color=color,ls='--',alpha=.7,label=label+' train')
        axes[1].plot([x['step'] for x in hh],[100*x['dev_monitor']['draft_diagnostics']['overlap_area_fraction']['mean'] for x in hh],color=color,label=label)
    axes[0].set(xlabel='Optimizer updates',ylabel='Fixed G',ylim=(.30,.65),title='Same G; stabilization and trajectories differ')
    axes[1].set(xlabel='Optimizer updates',ylabel='Pairwise overlap / public area (%)',ylim=(4,16),title='Fixed dev500 raw drafts')
    axes[2].bar(np.arange(1,13),np.asarray(r['graph_step_control']['clipped_steps_per_layer'])/r['steps']*100,color='#277da2')
    axes[2].set(xlabel='Graph layer',ylabel='Projected updates (%)',title='Forward graph inputs remain unnormalized')
    for ax in axes:ax.grid(alpha=.2)
    for ax in axes[:2]:ax.legend(fontsize=7)
    fig.savefig(FOLDER/'learning_curves.png',dpi=160);plt.close(fig)
    print(json.dumps({'full_dev_G':full['losses']['total'],'raw_legal':full['losses']['raw_hard_legal'],'raw_same500':{k:v['mean'] for k,v in raw['overlap_area_fraction'].items()},
        'paired':paired,'images':selected,'source_protected_count':len(protected)},ensure_ascii=False))

if __name__=='__main__':main()
