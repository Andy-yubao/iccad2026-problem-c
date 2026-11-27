"""Read completed AB/control evidence; no model inference, training or repair."""
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
    data={};paths=[Path(__file__)]
    for name,folder in [('control',CONTROL),('joint_AB',HERE)]:
        progress=read(folder/'progress.json');best=next(h for h in progress['history'] if h['epoch']==progress['best_epoch'])
        frozen=read(folder/'frozen.json');assert sha(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
        assert all(sha(ROOT/p)==h for p,h in frozen['sources'].items())
        data[name]={'results':read(folder/'results.json'),'identity':read(folder/'config.json')['identity'],
            'best_epoch':progress['best_epoch'],'best_train':best['train_monitor'],'best_dev':best['dev_monitor'],
            'full_dev':read(folder/'full_dev.json')['losses'],'terminal':read(folder/'terminal/results.json')['summary'],
            'gap':best['dev_monitor']['selection_proxy']-best['train_monitor']['selection_proxy'],
            'lr_changes':[{'epoch':h['epoch'],'from':h['lr'],'to':h['lr_next']} for h in progress['history'] if h['lr']!=h['lr_next']],
            'common_epochs':{str(e):{'train':next(h for h in progress['history'] if h['epoch']==e)['train_monitor']['selection_proxy'],
                'dev':next(h for h in progress['history'] if h['epoch']==e)['dev_monitor']['selection_proxy']} for e in [45,49,50]}}
        paths += [folder/n for n in ['results.json','progress.json','config.json','frozen.json','full_dev.json','terminal/results.json']]
    keys=['registry_sha256','paths_sha256','cache_identity','seed','device','torch','numpy','cuda','device_name','batch_size','lr','clip_norm','lr_levels','meaningful_relative_delta','meaningful_absolute_delta','scheduler_start_epoch','schedule','selection','parameters']
    data['shared_identity_equal']={k:data['control']['identity'][k]==data['joint_AB']['identity'][k] for k in keys};assert all(data['shared_identity_equal'].values())
    assert data['joint_AB']['identity']['ffn_dropout']==.05 and data['joint_AB']['identity']['weight_decay']==.0003 and not data['joint_AB']['identity']['data_augmentation']
    weights={'center_local':1.,'shape':.25,'relative_local':.25,'overlap':.5,'contact':.5,'boundary':.25,'mib':.1,'bbox':.1,'occupancy':1.}
    data['proxy_delta_components']={k:(data['joint_AB']['full_dev'][k]-data['control']['full_dev'][k])*w for k,w in weights.items()}
    assert np.isclose(sum(data['proxy_delta_components'].values()),data['joint_AB']['full_dev']['selection_proxy']-data['control']['full_dev']['selection_proxy'])
    a=sorted(read(CONTROL/'terminal/results.json')['records'],key=lambda r:r['index']);b=sorted(read(HERE/'terminal/results.json')['records'],key=lambda r:r['index'])
    assert len(a)==len(b)==500 and [r['member'] for r in a]==[r['member'] for r in b]
    n=np.array([r['n'] for r in a]);w=np.exp((n-n.max())/12);w/=w.sum();delta=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(a,b)]);con=delta*w
    rawdelta=np.array([y['draft_diagnostics']['overlap_area_fraction']-x['draft_diagnostics']['overlap_area_fraction'] for x,y in zip(a,b)])
    data['paired']={'Q_delta':float(con.sum()),'Q_better_same_worse':[int((delta<-1e-10).sum()),int((abs(delta)<=1e-10).sum()),int((delta>1e-10).sum())],
        'raw_overlap_better_same_worse':[int((rawdelta<-1e-10).sum()),int((abs(rawdelta)<=1e-10).sum()),int((rawdelta>1e-10).sum())]}
    rng=np.random.default_rng(42039)
    for label,groups in [('case',list(range(500))),('shard',[r['member'][0] for r in a]),('worker',[r['member'][0].split('/')[0] for r in a])]:
        ids=sorted(set(groups),key=str);den=np.array([sum(w[i] for i,g in enumerate(groups) if g==k) for k in ids]);num=np.array([sum(con[i] for i,g in enumerate(groups) if g==k) for k in ids]);count=np.array([sum(g==k for g in groups) for k in ids]);sample=rng.integers(0,len(ids),size=(3000,len(ids)))
        result={'Q_bootstrap95':np.percentile(num[sample].sum(1)/den[sample].sum(1),[2.5,97.5]).tolist()}
        for field in ['seconds','postprocess_seconds']:
            values=np.array([sum(b[i][field]-a[i][field] for i,g in enumerate(groups) if g==k) for k in ids])
            result[field+'_delta']=float(values.sum()/500)
            result[field+'_bootstrap95']=np.percentile(values[sample].sum(1)/count[sample].sum(1),[2.5,97.5]).tolist()
        data['paired'][label]=result
    transitions={}
    for event in ['rescue_used','early_projection']:
        def occurs(r):return r['rescue_used'] if event=='rescue_used' else any(t['phase']=='early_projection' for t in r['trace'])
        x={r['index'] for r in a if occurs(r)};y={r['index'] for r in b if occurs(r)}
        transitions[event]={'old':sorted(x),'new':sorted(y),'resolved':sorted(x-y),'newly_triggered':sorted(y-x),'retained':sorted(x&y)}
    data['event_transitions']=transitions
    pairs=[]
    for i,(x,y) in enumerate(zip(a,b)):
        pairs.append({'index':i,'member':str(x['member']),'n':x['n'],'Q_control':x['metrics']['cost'],'Q_AB':y['metrics']['cost'],
            'Q_delta':float(delta[i]),'weighted_Q_contribution':float(con[i]),'raw_overlap_delta':float(rawdelta[i]),
            'seconds_control':x['seconds'],'seconds_AB':y['seconds']})
    data['largest_weighted_regressions']=sorted(pairs,key=lambda x:x['weighted_Q_contribution'],reverse=True)[:10]
    data['largest_weighted_improvements']=sorted(pairs,key=lambda x:x['weighted_Q_contribution'])[:10]
    data['worst_AB_cases']=sorted(pairs,key=lambda x:x['Q_AB'],reverse=True)[:5]
    with (HERE/'paired_deltas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    data['source_hashes']={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    data.update(training_started=False,inference_calls=0,solver_calls=0,test_read=False)
    save(HERE/'completed_analysis.json',data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for label,folder,color in [('Unregularized',CONTROL,'#197796'),('Joint AB',HERE,'#d47131')]:
        h=read(folder/'progress.json')['history'];selected=[x for x in h if x['epoch']>=5];best=next(x for x in h if x['epoch']==read(folder/'progress.json')['best_epoch'])
        for axis,role in zip(axes,['train_monitor','dev_monitor']):
            axis.plot([x['epoch'] for x in selected],[x[role]['selection_proxy'] for x in selected],color=color,label=label)
            axis.scatter([best['epoch']],[best[role]['selection_proxy']],color=color,s=40,zorder=4)
    for axis,title in zip(axes,['Fixed train-monitor500 (eval mode)','Fixed dev-monitor500 (eval mode)']):
        axis.set(xlabel='Completed epochs',ylabel='Geometry + occupancy proxy',title=title);axis.grid(alpha=.2);axis.legend();axis.spines['top'].set_visible(False);axis.spines['right'].set_visible(False)
    fig.suptitle('Rectangle 96x12 | FFN dropout .05 + AdamW decay .0003 | seed 0\nDots: selected best; no augmentation; same validation proxy',fontsize=11)
    fig.savefig(HERE/'generalization_curves.png',dpi=170);plt.close(fig)
    print(json.dumps({'proxy':data['joint_AB']['full_dev']['selection_proxy'],'gap_control':data['control']['gap'],'gap_AB':data['joint_AB']['gap'],'paired':data['paired'],'events':transitions,'worst_AB':data['worst_AB_cases']},ensure_ascii=True))
if __name__=='__main__':main()
