"""Read saved results; export small deployment and independent F from unique best."""
import json
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,checkpoint
from .pipeline import OUTPUT
from .model import deployment_weights
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def paired(a,b):
    assert [r['member'] for r in a]==[r['member'] for r in b]
    x=np.asarray([r['metrics']['cost'] for r in a]);y=np.asarray([r['metrics']['cost'] for r in b]);d=x-y
    w=np.exp((np.asarray([r['n'] for r in a])-max(r['n'] for r in a))/12)
    workers=np.asarray([r['member'][0].split('/')[0] for r in a]);groups=np.unique(workers)
    numer=np.array([np.sum(w[workers==g]*d[workers==g]) for g in groups]);denom=np.array([w[workers==g].sum() for g in groups])
    ids=np.random.default_rng(20261029).integers(0,len(groups),(4000,len(groups)))
    samples=numer[ids].sum(1)/denom[ids].sum(1)
    return {'count':len(a),'weighted_delta':float(np.average(d,weights=w)),
        'worker_resampling95':np.quantile(samples,[.025,.975]).tolist(),
        'better_equal_worse':[int((d< -1e-8).sum()),int((abs(d)<=1e-8).sum()),int((d>1e-8).sum())]}
def main():
    training=read(OUTPUT/'results.json');progress=read(OUTPUT/'progress.json');raw=read(OUTPUT/'full_dev.json');terminal=read(OUTPUT/'terminal/results.json')
    history=progress['history'];initial=history[0];best=next(r for r in history if r['epoch']==training['best_epoch'])
    report={'status':'complete_pending_deep_audit','training':training,'initial_monitor':initial,'best_monitor':best,
       'full_dev':raw['losses'],'flash':terminal['summary'],'test_read':False,'baseline_promoted':False,
       'deployment_parameters':1713086,'trainable_refiner_parameters':241547,'training_model_including_frozen_auxiliary_P74':7543025,
       'limits':'Same initial parents/architecture/data/two-source recipe as E81; only fixed objective/selection package changed to G-only. Not a per-term or reference-free ablation. Warm F79 already learned earlier geometric objectives. Adaptive single-seed development; no across-Loss total comparison.'}
    baseline_paths={'E42':'E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0',
       'E79':'E79_center_refinement/runs/20261029_e78_center_seed0','E80':'E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer',
       'E81':'E81_multisource_refinement/runs/20261029_e42_multisource_seed0'}
    report['paired_Q']={}
    for name,path in baseline_paths.items():
        prior=read(ROOT/'experiments'/path/'terminal/results.json')
        report['paired_Q'][name]=paired(terminal['records'],prior['records'])
    report['worst10']=[{k:r[k] for k in ('index','member','n','metrics','rescue_used','draft_diagnostics')} for r in sorted(terminal['records'],key=lambda r:r['metrics']['cost'],reverse=True)[:10]]
    best_path=OUTPUT/'checkpoints/best.pt';state=torch.load(best_path,map_location='cpu',weights_only=False)
    parent=sha(best_path);config=state['model_config']
    checkpoint(OUTPUT/'deployment.pt',{'role':'Deterministic export of own fixed-Loss selected best, no Q selection',
       'model':deployment_weights(state['model']),'model_config':config,'parent_sha256':parent,'parameters':1713086})
    checkpoint(OUTPUT/'correction_F.pt',{'role':'Independent correction model; accepts public canonical draft state3 and public graph',
       'model':{k[len('refiner.'):]:v for k,v in state['model'].items() if k.startswith('refiner.')},
       'config':{'width':config['corrector_width'],'message_width':config['message_width'],'dropout':config['ffn_dropout'],'max_correction':config['max_correction']},
       'parent_sha256':parent,'parameters':241547})
    report['exports']={'deployment':sha(OUTPUT/'deployment.pt'),'F_only':sha(OUTPUT/'correction_F.pt'),'selected_best':parent}
    save(HERE/'runs/review.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,key,title in zip(axes,('total','P42_total','P74_total'),('Fixed two-source Loss','P42 source Loss','P74 source Loss')):
        for role,label in (('train_monitor','train500 (old16k)'),('dev_monitor','dev500')):
            ax.plot([r['epoch'] for r in history],[r[role][key] for r in history],label=label)
        ax.axvline(training['best_epoch'],ls='--',color='gray');ax.set_title(title);ax.set_xlabel('epoch');ax.legend()
    fig.tight_layout();fig.savefig(HERE/'runs/loss_curves.png',dpi=160);plt.close(fig)
    f=terminal['summary'];overlap=raw['losses']['draft_diagnostics']['overlap_area_fraction']['mean']
    text=f'''# E84同起点G-only目标包控制

与既有E81保持架构、初始化来源、双来源、数据及优化日程一致；每来源训练/监测/选模固定为完整原G，I/E/A只诊断。原F79已学过几何目标，不能称从未受几何监督；本组只检验本轮后训练和匹配选模的目标包。主要同例比较E81见review.json，旧E81不重训/重评。

完成{training['epochs']}轮，best{training['best_epoch']}，停止原因{training['status']}，训练阶段{training['elapsed_seconds']/60:.2f}分钟。唯一best按每例固定两来源Loss的开发均值选择，未按Q选择。

完整dev2000的部署P42草稿交面积{100*overlap:.3f}%，原始硬合法{raw['losses']['raw_hard_legal']}/2000。一次flash500加权Q={f['weighted_Q']:.6f}、P95={f['p95_Q']:.6f}、最大={f['max_Q']:.6f}，硬合法{f['hard_legal']}/500。训练/开发总Loss与两个来源分项必须同时阅读：[曲线](loss_curves.png)，[完整数据](review.json)。不与旧定义总Loss横比。

部署1713086参数，其中独立F241547参数；训练时辅助P74不进入部署计算。由唯一best确定性导出[部署模型](20261029_same_initialization_G_only_seed0/deployment.pt)和[独立修正模型F](20261029_same_initialization_G_only_seed0/correction_F.pt)，父SHA及导出SHA见review.json。

本组只改变固定训练与匹配选模目标包，架构、两来源、初始化和操作保持E81原样；不能归因于单个I/E/A项，也不能称从未受几何监督。测试封存，不自动升级E42/E50，须继续核对反例、来源框与3至5图。

[同100例参考／布局器／G-only修正／后处理](20261029_same_initialization_G_only_seed0/draft_comparison100/index.html)。
'''
    (HERE/'runs/review.md').write_text(text,encoding='utf-8');print('E84 saved report and independent deployment/F exports complete',flush=True)

if __name__=='__main__':main()
