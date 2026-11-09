"""Saved two-arm mechanism comparison; never selects checkpoints by Q."""
import json
from pathlib import Path
import numpy as np
from .io import HERE,ROOT,save,sha
from .pipeline import OUTPUTS

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def key(r):return tuple(r['member'])
def paired(values,workers):
    groups=np.unique(workers);delta=np.asarray(values,dtype=float)
    num=np.array([delta[workers==g].sum() for g in groups]);den=np.array([(workers==g).sum() for g in groups])
    choices=np.random.default_rng(20261030).integers(0,len(groups),(4000,len(groups)))
    draws=num[choices].sum(1)/den[choices].sum(1)
    return dict(mean_delta=float(delta.mean()),worker_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),better_equal_worse=[int((delta< -1e-8).sum()),int((abs(delta)<=1e-8).sum()),int((delta>1e-8).sum())])
def q_pair(a,b):
    assert [key(r) for r in a]==[key(r) for r in b]
    d=np.array([x['metrics']['cost']-y['metrics']['cost'] for x,y in zip(a,b)])
    n=np.array([r['n'] for r in a]);w=np.exp((n-n.max())/12);workers=np.array([r['member'][0].split('/')[0] for r in a]);groups=np.unique(workers)
    num=np.array([(w[workers==g]*d[workers==g]).sum() for g in groups]);den=np.array([w[workers==g].sum() for g in groups]);choices=np.random.default_rng(20261030).integers(0,len(groups),(4000,len(groups)))
    return dict(weighted_delta=float(np.average(d,weights=w)),worker_bootstrap95=np.quantile(num[choices].sum(1)/den[choices].sum(1),[.025,.975]).tolist(),better_equal_worse=[int((d< -1e-8).sum()),int((abs(d)<=1e-8).sum()),int((d>1e-8).sum())])
def main():
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records']
    ns=np.array([r['n'] for r in features]);pin=np.array([r['pin_connected_fraction'] for r in features]);density=np.array([r['connection_density'] for r in features]);workers=np.array([r['member'][0].split('/')[0] for r in features])
    sparse=np.array([density[i]<=np.median(density[ns==n]) for i,n in enumerate(ns)])
    masks={'all':ns>0,'weak_port':pin<=.3,'weak_port_sparse':(pin<=.3)&sparse,'rich_port_dense':(pin>=.8)&~sparse}
    arms={};ordered={}
    for arm,folder in OUTPUTS.items():
        train=read(folder/'results.json');raw=read(folder/'full_dev.json');terminal=read(folder/'terminal/results.json');initial=read(HERE/('runs/initial_'+arm+'.json'))
        lookup={key(r):r for r in raw['records']};init={key(r):r for r in initial['records']}
        rows=[lookup[key(r)] for r in features];first=[init[key(r)] for r in features];ordered[arm]=rows
        summary={}
        for group,mask in masks.items():
            selected=[r for r,m in zip(rows,mask) if m];original=[r for r,m in zip(first,mask) if m]
            pairs=sum(r['noport_order_clear_count'] for r in selected)
            summary[group]=dict(count=int(mask.sum()),overlap=float(np.mean([r['overlap_area_fraction'] for r in selected])),initial_overlap=float(np.mean([r['overlap_area_fraction'] for r in original])),G=float(np.mean([r['loss_common_original_G'] for r in selected])),noport_relative_error_case_mean_with_pairs=float(np.mean([r['noport_near_relative_error'] for r in selected if r['noport_near_pair_count']>0])) if any(r['noport_near_pair_count']>0 for r in selected) else None,noport_clear_pairs=int(pairs),noport_order_flip_rate=float(sum(r['noport_order_flip_count'] for r in selected)/pairs) if pairs else None)
        arms[arm]=dict(training=train,raw_losses=raw['losses'],groups=summary,flash=terminal['summary'],checkpoint_sha256=read(folder/'frozen.json')['checkpoint_sha256'],gallery=str((folder/'four_stage100/index.html').relative_to(HERE)),terminal_records=terminal['records'])
    raw_pairs={}
    for group,mask in masks.items():
        d=np.array([a['overlap_area_fraction']-b['overlap_area_fraction'] for a,b in zip(ordered['cooperative'],ordered['control'])]);raw_pairs[group]=paired(d[mask],workers[mask])
    limits=read(HERE/'plan.json')['comparison_limits']
    policy_path=HERE/'runs/lr_policy_change/migration.json'
    policy_change=read(policy_path) if policy_path.exists() else None
    if policy_change:limits.append('User changed post-training LR policy mid-run: completed control kept protection; cooperative continued without it. This is not a pure operator-only comparison under identical training policies.')
    report=dict(status='complete_pending_visual_review',arms=arms,paired_overlap_cooperative_minus_control=raw_pairs,paired_Q=q_pair(arms['cooperative']['terminal_records'],arms['control']['terminal_records']),test_read=False,baseline_promoted=False,limits=limits,scheduler_policy_change=policy_change)
    # Retain coordinates only in their existing terminal files.
    for arm in arms:arms[arm].pop('terminal_records')
    save(HERE/'runs/comparison.json',report)
    lines=['# E85 E74协同空间修正对照','', '两组同E74/F81起点、同241,547个可训练参数、同固定Loss。用户于协同组中途取消后训练学习率保护；已完成控制组保留原策略，因此终末差异同时包含调度政策变化，不能单独归因为协同算子。训练前算子直接效果单列。所有统计为已暴露自适应开发证据，测试封存。','', '|配置|训练分钟|停止/best|dev2000交面积|弱端口稀疏交面积|flash加权Q|最大Q|','|---|---:|---|---:|---:|---:|---:|']
    for arm,a in arms.items():
        t=a['training'];f=a['flash'];lines.append(f"|{arm}|{t['elapsed_seconds']/60:.2f}|{t['status']}/{t['best_epoch']}|{100*a['groups']['all']['overlap']:.3f}%|{100*a['groups']['weak_port_sparse']['overlap']:.3f}%|{f['weighted_Q']:.6f}|{f['max_Q']:.6f}|")
    lines+=['','## 协同组减控制组的开发集配对差','']
    for group,result in raw_pairs.items():lines.append(group+': '+json.dumps(result,ensure_ascii=False))
    lines+=['','Q配对：'+json.dumps(report['paired_Q'],ensure_ascii=False),'','初始与最终分组、无端口模块近邻次序、共同G、完整Loss与失败尾部见[comparison.json](comparison.json)。原始硬合法与后处理合法分别报告，不按Q升级或重选checkpoint。','']
    for arm,a in arms.items():lines.append('['+arm+' 同100例四阶段图]('+a['gallery']+')')
    (HERE/'runs/comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(arm,folder) in zip(axes,OUTPUTS.items()):
        history=read(folder/'progress.json')['history']
        for role in ('train_monitor','dev_monitor'):ax.plot([r['epoch'] for r in history],[r[role]['total'] for r in history],label=role)
        ax.set_title(arm+' fixed Loss');ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(HERE/'runs/loss_curves.png',dpi=150);plt.close(fig)
    print('Saved E85 paired comparison; pending actual visual review',flush=True)
if __name__=='__main__':main()
