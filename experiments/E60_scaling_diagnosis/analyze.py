"""Analyze existing records without loading data or performing model inference."""
import json
from pathlib import Path
import numpy as np
from .io import ROOT, HERE, OUT, save, sha

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    sources={name:sha(HERE/name) for name in ('model.py','features.py','batches.py','losses.py','overlap.py','ops.py','diagnostics.py','cache.py')}
    assert all(digest==sha(ROOT/'experiments/E59_centerline_overlap'/name) for name,digest in sources.items())
    save(OUT/'provenance.json',{'copied_from':'experiments/E59_centerline_overlap','source_sha256':sources,
        'new_training_runs':0,'new_postprocess_calls':0,'test_read':False})
    runs=[]
    for seed in (0,1):
        for size in (2048,4096,8192,16000):
            runs.append((f'96 / {size} / s{seed}',ROOT/f'experiments/E57_capacity_data_diagnosis/runs/20261026_data{size}_geometry96_seed{seed}'))
    for label,name in [('160 norm / 16k / s1','20261027_wide_graph_prenorm_seed1'),('160 norm / 64k / s0','20261027_large64000_original_seed0')]:
        runs.append((label,ROOT/'experiments/E59_centerline_overlap/runs'/name))
    rows=[]
    weights={'center_local':1.,'shape':.25,'relative_local':.25,'overlap':.5,'contact':.5,'boundary':.25,'mib':.1,'bbox':.1}
    for label,folder in runs:
        p=read(folder/'progress.json');cfg=read(folder/'config.json')['identity'];h=p['history']
        b=next(x for x in h if x['epoch']==p['best_epoch']);t=b['train_monitor'];v=b['dev_monitor'];last=h[-1]
        changes=[x for x in h if x['lr']!=x.get('lr_next',x['lr'])]
        n=cfg['dataset_size'];curve=[{'step':x['step'],'lr':x['lr'],'train':x['train_monitor']['total'],
            'dev':x['dev_monitor']['total'],'seen_monitor':x['train_seen_count'],
            'train_overlap':x['train_monitor']['draft_diagnostics']['overlap_area_fraction']['mean'],
            'dev_overlap':x['dev_monitor']['draft_diagnostics']['overlap_area_fraction']['mean']} for x in h]
        rows.append({'label':label,'path':str(folder.relative_to(ROOT)),'train_size':n,'seed':cfg['seed'],
            'width':cfg['width'],'normalized':cfg.get('graph_prenorm',False),'best_step':b['step'],
            'train':t['total'],'dev':v['total'],'gap':v['total']-t['total'],'seen_monitor_at_best':b['train_seen_count'],
            'best_effective_passes':b['step']*32/n,'full_dev':read(folder/'full_dev.json')['losses'],
            'weighted_dev_components':{k:v[k]*w for k,w in weights.items()},
            'train_components':{k:t[k] for k in weights},'dev_components':{k:v[k] for k in weights},
            'train_overlap':t['draft_diagnostics']['overlap_area_fraction']['mean'],
            'dev_overlap':v['draft_diagnostics']['overlap_area_fraction']['mean'],
            'best_to_last':{'train_delta':last['train_monitor']['total']-t['total'],'dev_delta':last['dev_monitor']['total']-v['total']},
            'lr_drops':[{'step':x['step'],'effective_passes':x['step']*32/n,'dev':x['dev_monitor']['total'],'train':x['train_monitor']['total'],
                         'from':x['lr'],'to':x['lr_next']} for x in changes], 'curve':curve})
    # Same members, existing model outputs only. No loading of any train/dev tensors.
    panels={}
    for label,folder in [('96s0',runs[3][1]),('96s1',runs[7][1]),('norm16',runs[8][1]),('norm64',runs[9][1]),
                         ('E42',ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0')]:
        fp=folder/'terminal/results.json'
        if not fp.exists():
            # Historical E42 result layout is intentionally discovered rather than regenerated.
            continue
        panels[label]=read(fp)['records']
    stratum={}
    for label,records in panels.items():
        strata={}
        for name,lo,hi in [('small',0,60),('medium',60,90),('large',90,121)]:
            rs=[r for r in records if lo<=r['n']<hi]
            strata[name]={'count':len(rs),'overlap':float(np.mean([r['draft_diagnostics']['overlap_area_fraction'] for r in rs]))}
            for key in ('module_load_small_mean','module_load_large_mean'):
                values=[r['draft_diagnostics'][key] for r in rs if key in r['draft_diagnostics']]
                strata[name][key]=float(np.mean(values)) if values else None
        stratum[label]=strata
    pairs={}
    for first,second in [('96s0','norm64'),('96s1','norm16'),('norm16','norm64')]:
        aa={tuple(r['member']):r for r in panels[first]};bb={tuple(r['member']):r for r in panels[second]};assert aa.keys()==bb.keys()
        delta=np.array([bb[m]['draft_diagnostics']['overlap_area_fraction']-aa[m]['draft_diagnostics']['overlap_area_fraction'] for m in aa])
        pairs[first+' -> '+second]={'mean_delta':float(delta.mean()),'improved':int((delta<0).sum()),'worsened':int((delta>0).sum()),'count':len(delta)}
    # Reconstruct index draws only, without loading dataset tensors or targets.
    size=64000;seed=0;blocks=47;exposures=np.zeros(size,dtype=np.int32)
    for block in range(blocks):
        order=np.random.default_rng(seed+block).permutation(size)[:16000]
        exposures[order]+=1
    sampling={'basis':'Reconstruction of saved E59 sampler; metadata indices only','blocks':blocks,
        'min':int(exposures.min()),'max':int(exposures.max()),'mean':float(exposures.mean()),
        'p05':float(np.percentile(exposures,5)),'p95':float(np.percentile(exposures,95)),
        'unseen':int((exposures==0).sum()),'draws':int(exposures.sum()),
        'histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(exposures,return_counts=True))}}
    save(OUT/'saved_metrics.json',{'runs':rows,'strata':stratum,'paired_overlap':pairs,
        'large_sampling_reconstruction':sampling,'test_read':False,'new_inference_cases':0})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r in [rows[0],rows[2],rows[3],rows[-2],rows[-1]]:
        c=r['curve'];line=axes[0,0].plot([x['step'] for x in c],[x['dev'] for x in c],label=r['label'])[0]
        axes[0,0].plot([x['step'] for x in c],[x['train'] for x in c],ls='--',color=line.get_color(),alpha=.65)
    axes[0,0].set(xlabel='Optimizer updates',ylabel='Fixed G',ylim=(.05,.7),title='Train dashed / dev solid; different schedules')
    for seed in (0,1):
        rr=[r for r in rows[:8] if r['seed']==seed]
        axes[0,1].plot([r['train_size'] for r in rr],[r['gap'] for r in rr],marker='o',label=f'96 seed/family {seed}')
    axes[0,1].scatter([16000,64000],[rows[-2]['gap'],rows[-1]['gap']],marker='x',s=70,label='Normalized160; seeds differ')
    axes[0,1].set(xscale='log',xlabel='Training layouts',ylabel='Dev minus train G',title='Generalization gap at selected best')
    names=list(weights);x=np.arange(len(names));width=.23
    for j,r in enumerate([rows[3],rows[-2],rows[-1]]):axes[1,0].bar(x+(j-1)*width,[r['weighted_dev_components'][k] for k in names],width,label=r['label'])
    axes[1,0].set_xticks(x,names,rotation=35,ha='right');axes[1,0].set(ylabel='Weighted contribution to G',title='Loss mainly follows reference fitting')
    for r in [rows[3],rows[-2],rows[-1]]:
        c=r['curve'];axes[1,1].plot([x['step'] for x in c],[x['dev_overlap']*100 for x in c],label=r['label'])
    axes[1,1].set(xlabel='Optimizer updates',ylabel='Pairwise overlap / public area (%)',ylim=(4,15),title='Raw overlap does not follow total G exactly')
    for ax in axes.flat:ax.grid(alpha=.18);ax.legend(fontsize=7)
    fig.savefig(OUT/'diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps({'runs':len(rows),'pairs':pairs,'output':str(OUT/'saved_metrics.json')},ensure_ascii=False))

if __name__=='__main__':main()
