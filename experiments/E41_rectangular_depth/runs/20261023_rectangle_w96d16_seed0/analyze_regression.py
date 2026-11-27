"""Analyze saved completed runs only: no inference, training, or solver calls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json,csv,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
CONTROL=ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('w',encoding='utf-8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    data={};sourcepaths=[Path(__file__)]
    for label,p in [('depth12',CONTROL),('depth16',HERE)]:
        prog=read(p/'progress.json');full=read(p/'full_dev.json')['losses'];terminal=read(p/'terminal/results.json');identity=read(p/'config.json')['identity'];frozen=read(p/'frozen.json')
        assert sha(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
        assert all(sha(ROOT/n)==h for n,h in frozen['sources'].items())
        sourcepaths += [p/n for n in ['progress.json','full_dev.json','terminal/results.json','config.json','frozen.json']]
        history=prog['history'];best=next(h for h in history if h['epoch']==prog['best_epoch'])
        data[label]={'best_epoch':prog['best_epoch'],'best_train':best['train_monitor'],'best_dev':best['dev_monitor'],
            'full_dev':full,'terminal':terminal['summary'],'identity':identity,
            'lr_changes':[{'epoch':h['epoch'],'from':h['lr'],'to':h['lr_next']} for h in history if h['lr']!=h['lr_next']],
            'common_epochs':{str(e):{'train':next(h for h in history if h['epoch']==e)['train_monitor']['selection_proxy'],
                'dev':next(h for h in history if h['epoch']==e)['dev_monitor']['selection_proxy']} for e in [42,45]}}
    keys=['registry_sha256','paths_sha256','cache_identity','seed','device','torch','numpy','cuda','device_name','batch_size','lr','weight_decay','clip_norm','lr_levels','meaningful_relative_delta','meaningful_absolute_delta','scheduler_start_epoch','schedule','selection']
    data['shared_identity_equal']={k:data['depth12']['identity'][k]==data['depth16']['identity'][k] for k in keys}
    assert all(data['shared_identity_equal'].values())
    weights={'center_local':1.,'shape':.25,'relative_local':.25,'overlap':.5,'contact':.5,'boundary':.25,'mib':.1,'bbox':.1,'occupancy':1.}
    contributions={k:(data['depth16']['full_dev'][k]-data['depth12']['full_dev'][k])*w for k,w in weights.items()}
    data['full_proxy_delta_components']=contributions
    assert np.isclose(sum(contributions.values()),data['depth16']['full_dev']['selection_proxy']-data['depth12']['full_dev']['selection_proxy'])
    a=sorted(read(CONTROL/'terminal/results.json')['records'],key=lambda r:r['index']);b=sorted(read(HERE/'terminal/results.json')['records'],key=lambda r:r['index'])
    assert len(a)==len(b)==500 and [r['member'] for r in a]==[r['member'] for r in b]
    n=np.array([r['n'] for r in a]);w=np.exp((n-n.max())/12);w/=w.sum();dq=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(a,b)]);con=w*dq
    data['paired_Q']={'delta':float(con.sum()),'better':int((dq<-1e-10).sum()),'same':int((abs(dq)<=1e-10).sum()),'worse':int((dq>1e-10).sum())}
    rng=np.random.default_rng(41012)
    for label,groups in [('case',list(range(500))),('shard',[x['member'][0] for x in a]),('worker',[x['member'][0].split('/')[0] for x in a])]:
        ids=sorted(set(groups),key=str);den=np.array([sum(w[i] for i,g in enumerate(groups) if g==k) for k in ids]);num=np.array([sum(con[i] for i,g in enumerate(groups) if g==k) for k in ids]);sample=rng.integers(0,len(ids),size=(3000,len(ids)));bootstrap=num[sample].sum(1)/den[sample].sum(1)
        data['paired_Q'][label+'_bootstrap95']=np.percentile(bootstrap,[2.5,97.5]).tolist()
    transitions={}
    for event in ['rescue_used','early_projection']:
        def occurs(r):return r['rescue_used'] if event=='rescue_used' else any(t['phase']=='early_projection' for t in r['trace'])
        x={r['index'] for r in a if occurs(r)};y={r['index'] for r in b if occurs(r)}
        transitions[event]={'old':sorted(x),'new':sorted(y),'resolved':sorted(x-y),'newly_triggered':sorted(y-x),'retained':sorted(x&y)}
    data['event_transitions']=transitions
    pairs=[]
    for i,(x,y) in enumerate(zip(a,b)):
        pairs.append({'index':i,'member':str(x['member']),'n':x['n'],'Q12':x['metrics']['cost'],'Q16':y['metrics']['cost'],
            'Q_delta':float(dq[i]),'weighted_contribution':float(con[i]),'overlap12':x['draft_diagnostics']['overlap_area_fraction'],
            'overlap16':y['draft_diagnostics']['overlap_area_fraction'],'seconds12':x['seconds'],'seconds16':y['seconds']})
    data['largest_weighted_regressions']=sorted(pairs,key=lambda r:r['weighted_contribution'],reverse=True)[:10]
    data['largest_weighted_improvements']=sorted(pairs,key=lambda r:r['weighted_contribution'])[:10]
    with (HERE/'paired_deltas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    data['source_hashes']={p.relative_to(ROOT).as_posix():sha(p) for p in sourcepaths}
    data.update(training_started=False,inference_calls=0,solver_calls=0,test_read=False)
    save(HERE/'regression_analysis.json',data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    colors={'depth12':'#197796','depth16':'#d47131'}
    for label,p in [('depth12',CONTROL),('depth16',HERE)]:
        history=read(p/'progress.json')['history'];h=[x for x in history if x['epoch']>=5]
        for axis,role in zip(axes,['train_monitor','dev_monitor']):
            axis.plot([x['epoch'] for x in h],[x[role]['selection_proxy'] for x in h],label=label,color=colors[label])
            chosen=next(x for x in history if x['epoch']==data[label]['best_epoch'])
            axis.scatter([chosen['epoch']],[chosen[role]['selection_proxy']],color=colors[label],s=40,zorder=4)
    for axis,title in zip(axes,['Fixed train-monitor500','Fixed dev-monitor500']):
        axis.set(xlabel='Completed epochs',ylabel='Geometry + occupancy proxy',title=title);axis.grid(alpha=.2);axis.legend()
        axis.spines['top'].set_visible(False);axis.spines['right'].set_visible(False)
    fig.suptitle('E39 rectangle 12 layers vs E41 rectangle 16 layers | seed 0\nDots: selected best; adaptive LR follows the same rule, with different realized timing',fontsize=11)
    fig.savefig(HERE/'training_validation_curves.png',dpi=170);plt.close(fig)
    print('Saved read-only regression analysis and curve figure.')
if __name__=='__main__':main()
