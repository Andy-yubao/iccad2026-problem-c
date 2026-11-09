"""Terminal read-only report from saved development evidence; no new inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOLDER=Path(__file__).resolve().parent;HERE=FOLDER.parents[1];CONTROL=HERE/'runs/20261027_control'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(values):
    v=np.asarray(values,dtype=float)
    return {'mean':float(v.mean()),'p95':float(np.percentile(v,95)),
            'worst5pct_mean':float(np.sort(v)[-max(1,int(np.ceil(.05*len(v)))):].mean()),'max':float(v.max())}


def change(a,b):return float(b/a-1) if abs(a)>1e-12 else None


def main():
    state=read(HERE/'state.json')
    if 'large_original' not in state['completed_stages']:raise ValueError('Controller finalization not complete')
    result=read(FOLDER/'results.json');progress=read(FOLDER/'progress.json');full=read(FOLDER/'full_dev.json')
    term=read(FOLDER/'terminal/results.json');summary=read(FOLDER/'terminal/summary.json');frozen=read(FOLDER/'frozen.json')
    health=read(FOLDER/'health.json');gallery=read(FOLDER/'draft_comparison100/manifest.json')
    if result['status']!='early_stop' or health['status']!='passed' or len(term['records'])!=500 or len(gallery['cases'])!=100:raise ValueError('Incomplete terminal evidence')
    if full['checkpoint_sha256']!=summary['checkpoint_sha256'] or sha(FOLDER/'checkpoints/best.pt')!=summary['checkpoint_sha256']:raise ValueError('Best identity mismatch')
    from experiments.E59_centerline_overlap.io import sources
    checkpoint=torch.load(FOLDER/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    if checkpoint['identity']['sources']!=sources():raise ValueError('Signed training sources changed')
    if not all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()):raise ValueError('Nonfinite saved parameters')
    if not progress['lowest_lr_validated'] or progress['lr_drops']!=3:raise ValueError('Incomplete scheduler/stop protocol')
    protected=read(ROOT/'experiments/E57_capacity_data_diagnosis/provenance.json')['protected_checkpoints_sha256']
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise ValueError('Original checkpoint changed: '+name)
    baseline=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'
    oldterm=read(baseline/'terminal/results.json');oldsummary=read(baseline/'terminal/summary.json')
    a={r['index']:r for r in oldterm['records']};b={r['index']:r for r in term['records']}
    if set(a)!=set(b) or any(a[i]['member']!=b[i]['member'] for i in a):raise ValueError('E42 dev500 members differ')
    q_delta=np.array([b[i]['metrics']['cost']-a[i]['metrics']['cost'] for i in sorted(a)])
    raw={}
    for key in ('overlap_area_fraction','overlap_pairs','penetration_squared_per_block','largest_conflict_cluster_fraction'):
        sa=stats([a[i]['draft_diagnostics'][key] for i in sorted(a)]);sb=stats([b[i]['draft_diagnostics'][key] for i in sorted(a)])
        raw[key]={'E42':sa,'E59_large':sb,'relative_change':{k:change(sa[k],sb[k]) for k in sa},
                  'improved_cases':sum(b[i]['draft_diagnostics'][key]<a[i]['draft_diagnostics'][key] for i in a)}
    wide16=HERE/'runs/20261027_wide_graph_prenorm_seed1'
    full16=read(wide16/'full_dev.json');prog16=read(wide16/'progress.json')
    old96={seed:read(ROOT/f'experiments/E57_capacity_data_diagnosis/runs/20261026_data16000_geometry96_seed{seed}/full_dev.json')['losses']['total'] for seed in (0,1)}
    history=progress['history'];best=next(x for x in history if x['epoch']==progress['best_epoch'])
    deltas={key:change(full16['losses'][key],full['losses'][key]) for key in ('total','center_local','shape','relative_local','overlap','contact','boundary','mib','bbox')}
    visual=[
      {'index':478,'reason':'median raw overlap among fixed100','observation':'The broad reference arrangement is retained. Group1 blocks and adjacent20 differ from E42; collisions remain around the central MIB14 and lower MIB10. No uniform visual dominance.'},
      {'index':449,'reason':'largest raw overlap among fixed100','observation':'Severe overlap remains around left-side50/46, upper49/38/7 and lower grouped/MIB blocks. This is a concrete failure despite a finite stable model.'},
      {'index':324,'reason':'largest Q regression among fixed100','observation':'Reference macro regions are recognizable and some lower-left group contacts look cleaner, yet central MIB/small-block collisions remain. Better local appearance or lower raw overlap need not improve finite-budget postprocessing Q.'},
      {'index':239,'reason':'largest Q improvement among fixed100','observation':'Large yellow blocks1/11 are placed closer to the reference relation than E42, but still intersect; lower MIB7 and neighboring blocks remain crowded. A large Q improvement is not a legal raw draft.'}
    ]
    report={'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
      'run':FOLDER.name,'checkpoint_sha256':summary['checkpoint_sha256'],
      'configuration':{'train_size':64000,'width':160,'layers':12,'parameters':checkpoint['identity']['parameters'],
                       'graph_prenorm':True,'seed':0,'batch_size':32,'precision':'FP32','Loss':'original fixed G','initialization':'scratch'},
      'training':result,'latest_seen_count':progress['seen_count'],'best_fixed_train500':best['train_monitor']['total'],
      'best_fixed_dev500':best['dev_monitor']['total'],'full_dev2000':full['losses'],
      'best_monitor_train_dev_gap':best['dev_monitor']['total']-best['train_monitor']['total'],
      'same_architecture16k_seed1_full_dev_G':full16['losses']['total'],'relative_to_normalized16k':deltas,
      'E57_16k_width96_full_dev_G':old96,'relative_to_16k_width96':{str(k):change(v,full['losses']['total']) for k,v in old96.items()},
      'health':health,'sources_still_frozen':True,'protected_original_checkpoints_verified':protected,
      'flash500':summary,'E42_flash500':oldsummary,
      'postprocess_relative_to_E42':{k:change(oldsummary[k],summary[k]) for k in ('weighted_Q','mean_Q','p95_Q','max_Q','worst5pct_mean_Q','seconds_mean')},
      'postprocess_paired_Q_counts':{'improved':int((q_delta<-1e-9).sum()),'tied':int((np.abs(q_delta)<=1e-9).sum()),'regressed':int((q_delta>1e-9).sum())},
      'same_dev500_raw_comparison':raw,'visual_review':{'kind':'AI observations, no human ratings','cases':visual},
      'gallery':'draft_comparison100/index.html','baseline_promotion':False,
      'limits':['Single-seed adaptive development results, no sealed test accessed',
                '16k normalized comparison changes seed and actual sample-order trajectory; not a data-only causal estimate',
                'Comparison to96-wide jointly changes data, width and graph normalization',
                'Fixed500 train monitoring is not exhaustive whole-train fitting',
                'Training draws/64000 is an effective-pass count, not guaranteed disjoint full passes',
                'Stable low Loss does not establish sufficient representational capacity or perfect geometric generation',
                'The raw draft is0/2000 hard legal although postprocessing is500/500 hard legal',
                'Observed identical-model flash variation prevents treating small Q differences as robust model effects'],
      'test_read':False,'new_formal_evaluation_runs':0}
    save(FOLDER/'review.json',report)
    # Common fixed-G comparisons; centerline totals are never overlaid as a
    # directly comparable objective. The second panel uses shared raw overlap.
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.3),layout='constrained')
    for label,p,col in [('16k normalized160 seed1',prog16,'#4775b5'),('64k normalized160 seed0',progress,'#d47725')]:
        h=[x for x in p['history'] if x['step']>0]
        axes[0].plot([x['step'] for x in h],[x['dev_monitor']['total'] for x in h],color=col,label=label+' dev')
        axes[0].plot([x['step'] for x in h],[x['train_monitor']['total'] for x in h],color=col,ls='--',alpha=.7,label=label+' train')
    axes[0].set(title='Same fixed G; seed/data trajectory differ',xlabel='Optimizer updates',ylabel='Fixed monitor Loss',ylim=(.28,1.05))
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    for name,label,col in [('20261027_small4096_original_seed0','Original','#4775b5'),('20261027_small4096_centerline_seed0','Centerline configuration','#d47725')]:
        h=[x for x in read(HERE/'runs'/name/'progress.json')['history'] if x['step']>0]
        axes[1].plot([x['step'] for x in h],[x['dev_monitor']['draft_diagnostics']['overlap_area_fraction']['mean'] for x in h],label=label,color=col)
    axes[1].set(title='Small paired study: shared raw metric',xlabel='Optimizer updates',ylabel='Mean pairwise intersection / public area',ylim=(0,.24))
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(FOLDER/'learning_curves.png',dpi=170);plt.close(fig)
    lines=['# E59 本轮实验完成报告','',
      '本轮完成中心连线损失对照、宽模型稳定性修复及64k大模型训练。保留原始Loss；新模型作为研究候选保存，未替换E42/flash或E50/pro。测试保持封存。','',
      '## 大模型结果','',
      f"64k训练、归一化160×12、{report['configuration']['parameters']:,}参数、seed0；FP32/batch32。47个监测块、23,500更新、752,000次样本暴露（11.75个有效遍历量），latest实际见过64,000成员。每块500更新，不是一整次64k遍历。训练阶段累计{result['elapsed_seconds']/60:.2f}分钟，按预设四档LR/最低档至少两轮及早停结束；best46（23,000步）。",'',
      f"固定train500 Loss **{report['best_fixed_train500']:.9f}**，固定dev500 **{report['best_fixed_dev500']:.9f}**，完整dev2000固定G **{full['losses']['total']:.9f}**。原始草稿硬合法 **0/2000**；重叠面积指标均值 **{full['losses']['draft_diagnostics']['overlap_area_fraction']['mean']:.3%}**，仍存在广泛碰撞。这个面积指标是逐对交面积总和/公开面积，非重叠区域并集覆盖率。",'',
      f"同5个训练成员的latest激活健康复核通过：隐藏最大{health['latest_hidden_max']:.3f}，最终SiLU零值比例{health['final_silu_zero_fraction']:.6%}。原始checkpoint和冻结源码均已核验保留。",'',
      '## 同dev500的后处理与E42对照','',
      '| 指标 | 冻结E42＋flash | 本轮64k＋flash |','|---|---:|---:|',
      f"| 硬合法 | {oldsummary['hard_legal']}/500 | {summary['hard_legal']}/500 |"]
    for key,label in [('weighted_Q','模块数加权Q'),('mean_Q','算术均值Q'),('p95_Q','P95 Q'),('worst5pct_mean_Q','最差5%均值Q'),('max_Q','最大Q'),('seconds_mean','完整求解均时/秒')]:
        lines.append(f"| {label} | {oldsummary[key]:.9f} | {summary[key]:.9f} |")
    lines+=['',f"加权Q较E42 {report['postprocess_relative_to_E42']['weighted_Q']:+.2%}，P95/最差5%/最大Q较好，但不是逐例支配。逐例Q改善/近似持平/退化={report['postprocess_paired_Q_counts']}。这是一轮开发集、中性时间Q，不能称正式比赛成绩。每组只评价唯一best一次，未重跑挑Q。",'',
      '## 同例原始草稿','',
      '| 同dev500指标 | E42 | 本轮 |','|---|---:|---:|']
    for key in raw:lines.append(f"| {key} 均值 | {raw[key]['E42']['mean']:.9f} | {raw[key]['E59_large']['mean']:.9f} |")
    lines+=['','同例raw诊断直接复用既有dev500记录；两版相关公式已检查一致，不重新生成基线。完整2000例指标与这个500例子面板分开列出。','',
      '已实际查看478、449、324、239四张reference/current/E42图。它们按中等重叠、最差重叠和Q改善/退化反例选出，不是随机人类评分。宏观参考布局通常能复现，但局部MIB/小模块碰撞仍突出；不能用视觉上略整齐或Q改善代替原始合法性。完整100图见[draft_comparison100](draft_comparison100/index.html)。','',
      '## 本轮回答了什么','',
      '1. 中心连线方案：本次固定centerline_v1及预训练κ校准配置未胜出。配对4096训练/2000开发中，重叠均值+8.94%、P95+10.45%、最差5%+9.96%，虽然参考分项略好。实际看过4组同例图且保留局部改善反例。不外推为所有中心方向损失都无效；详见[选型报告](../20261027_control/selection.md)。',
      '2. 宽模型失稳：原两种子均出现隐藏放大/读出死亡。聚合前归一化恢复了本轮训练的数值健康；同权重同例分支干预支持未归一化图聚合的放大路径。未发现已核验Loss/裁剪实现错误的证据，但没有唯一定位最初触发步骤，且修复训练顺序改变，因此不作严格单因素因果宣称。',
      f"3. 更多数据：相对16k归一化160×12的G {full16['losses']['total']:.9f}，本轮G {full['losses']['total']:.9f}（{deltas['total']:+.2%}）。相对E57 16k/96×12 seed0的G {old96[0]:.9f}，变化{report['relative_to_16k_width96']['0']:+.2%}。收益有限但存在于本轮观测；前者也改变seed/样本轨迹，后者还改变宽度和归一化，不能归因给单独数据量。",
      '4. 表达能力与数据是否足够：目前不能下充分性结论。64k下固定训练与开发Loss都约0.35、差距很小，原始合法率仍0，说明问题没有靠扩大训练集和宽度自然解决；低训练Loss不是充分表达能力证明，相近train/dev也不能区分优化平台、表示限制和监督目标折中。32例拟合与对称反例仍是局部证据，不是所有实际情况的可解性证明。','',
      '## 保存与结束','',
      '本轮4次正式训练累计约2.52小时，低于12小时上限；原始E42/E50共用权重和E55/E56参数保留。新增v6缓存已处理MIB并复用旧16k/dev，不复制原数据或开启测试。best/latest、源码快照、精确命令和旧失败证据均保留；没有新增无用数据缓存需清除。关闭本轮每小时自动唤醒，不自动开启下一轮研究。','',
      '![固定口径学习曲线](learning_curves.png)','',
      '机器可读证据：[review.json](review.json)；原始训练结果：[results.json](results.json)（该文件只记录训练阶段，因此full_dev_complete=false不代表终末评价缺失）；终末评价：[terminal/summary.json](terminal/summary.json)。']
    (FOLDER/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','full_dev_G':full['losses']['total'],'raw_hard_legal':full['losses']['raw_hard_legal'],
                      'weighted_Q':summary['weighted_Q'],'same500_raw_overlap':raw['overlap_area_fraction'],
                      'protected_checkpoints_verified':len(protected),'baseline_promoted':False},ensure_ascii=False))


if __name__=='__main__':main()
