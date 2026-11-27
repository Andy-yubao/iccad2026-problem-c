"""Saved-only summaries, uncertainty, examples and figures for visual feedback."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['MPLBACKEND']='Agg'
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from .feedback_geometry import read,save,sha,REVIEW,OUTPUT,ROOT,BUDGETS


def probe(row,delta=.05,eta=.05):
    return next(p for p in row['probe_curve'] if p['max_move_local']==delta and p['max_log_shape']==eta)


def flat(row):
    result={'id':row['id'],'n':row['n'],'score':row['annotation'].get('score'),
            'tags':'|'.join(row['annotation'].get('tags',[]))}
    for k,v in row['pileup'].items():
        if not isinstance(v,(dict,list)):result[k]=v
    for delta in BUDGETS:
        for eta in (0.,.05):
            p=probe(row,delta,eta);key=f'move{delta:g}_shape{eta:g}'
            result[key+'_bbox_gain']=p['strict']['bbox_reduction_fraction']
            result[key+'_old_empty_removed']=p['strict']['removable_whitespace_over_old_empty']
    return result


def correlations(rows):
    rated=[r for r in rows if r['annotation'].get('score')]
    y=np.asarray([r['annotation']['score'] for r in rated]);n=np.asarray([r['n'] for r in rated])
    keys=('complex_energy_10pct','coverage_energy','pair_overlap_fraction','largest_any_component_fraction','empty_fraction')
    groups={}
    for i,r in enumerate(rated):groups.setdefault(r['member'][0].split('/')[0],[]).append(i)
    rng=np.random.default_rng(20261024);clusters=list(groups)
    samples=[]
    for _ in range(2000):
        ids=np.concatenate([groups[clusters[k]] for k in rng.integers(len(clusters),size=len(clusters))])
        if len(np.unique(y[ids]))>1:samples.append(ids)
    results={}
    for key in keys:
        x=np.asarray([r['pileup'][key] for r in rated]);boot=[]
        for ids in samples:
            if np.ptp(x[ids])>0:boot.append(float(spearmanr(x[ids],y[ids]).statistic))
        design=np.column_stack((np.ones(len(n)),rankdata(n)))
        residual=lambda v:rankdata(v)-design@np.linalg.lstsq(design,rankdata(v),rcond=None)[0]
        results[key]={'spearman':float(spearmanr(x,y).statistic),
                      'worker_cluster_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                      'partial_rank_correlation_controlling_n':float(np.corrcoef(residual(x),residual(y))[0,1]),
                      'by_score':{str(s):{'count':int(sum(y==s)),'mean':float(x[y==s].mean()),'median':float(np.median(x[y==s]))} for s in (1,2,3)}}
    return {'count':len(rated),'worker_clusters':len(groups),'bootstrap_repetitions':2000,
            'metrics':results,'scope':'descriptive only; adaptive metric design, small selected feedback sample; CI does not include metric-selection uncertainty'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from visualization.layouts import plot_case,geometry_details
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    manifest=read(REVIEW/'manifest.json');order={r['id']:i for i,r in enumerate(manifest['cases'])}
    rows=sorted([read(p) for p in (OUTPUT/'cases_local').glob('*.json')],key=lambda r:order[r['id']])
    table=[flat(r) for r in rows]
    with (OUTPUT/'metrics_local.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    corr=correlations(rows)
    curves=[]
    for delta in BUDGETS:
        for eta in (0.,.05):
            vals=np.asarray([probe(r,delta,eta)['strict']['bbox_reduction_fraction'] for r in rows])
            geometric=np.asarray([probe(r,delta,eta)['geometric_only']['bbox_reduction_fraction'] for r in rows])
            curves.append({'delta':delta,'eta':eta,'mean_bbox_reduction':float(vals.mean()),
                           'median_bbox_reduction':float(np.median(vals)),'max_bbox_reduction':float(vals.max()),
                           'positive_witnesses_over_1e_minus6':int(sum(vals>1e-6)),
                           'geometric_only_mean_bbox_reduction':float(geometric.mean())})
    shape_only=[probe(r,0.,.05)['strict']['bbox_reduction_fraction'] for r in rows]
    shape_plus=[probe(r,.05,.05)['strict']['bbox_reduction_fraction']-max(probe(r,.05,0.)['strict']['bbox_reduction_fraction'],shape_only[i]) for i,r in enumerate(rows)]
    statuses={};guards={}
    for r in rows:
        for p in r['probe_curve']:
            for s in p['solves']:
                statuses[s['status']]=statuses.get(s['status'],0)+1
                for reason in s.get('guard_reasons',[]):guards[reason]=guards.get(reason,0)+1
    result={'count':len(rows),'ratings':corr,'whitespace_curves':curves,'solver_statuses':statuses,
            'rejected_candidate_reasons':guards,'joint_better_than_best_single_arm_at_5pct_count':int(sum(np.asarray(shape_plus)>1e-6)),
            'algorithm_source_sha256_used_for_probes':read(OUTPUT/'config.json')['source_sha256'],
            'current_algorithm_source_sha256':sha(ROOT/'experiments/E42_joint_regularization/feedback_geometry.py'),
            'source_snapshot':'source_snapshot.zip preserves the exact diagnostic implementation used; subsequent source correction only restores verbatim user-message metadata',
            'feedback_snapshot_sha256':sha(OUTPUT/'feedback_snapshot.json'),
            'report_source_sha256':sha(Path(__file__)),'limitations':['23 annotated, only22 rated; one whitespace tag',
                 'metric proposed after feedback; no unseen confirmation','conservative constructive lower bounds, never an irreducibility certificate',
                 '621 attempted conic subproblems include numerical failures; failures retained, not silently treated as proof of zero waste']}
    save(OUTPUT/'analysis.json',result)
    # Standard static research figure, not a separately drawn layout.
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.3),layout='constrained')
    rated=[r for r in rows if r['annotation'].get('score')]
    for ax,key,title in zip(axes,('largest_any_component_fraction','coverage_energy','complex_energy_10pct'),
                            ('只看最大重叠连通块','加入每块覆盖负荷','定位强重叠的局部大团')):
        for score,color in ((1,'#bb533a'),(2,'#bb963f'),(3,'#327f77')):
            rs=[r for r in rated if r['annotation']['score']==score]
            jitter=np.linspace(-.08,.08,len(rs))
            ax.scatter(np.full(len(rs),score)+jitter,[r['pileup'][key] for r in rs],color=color,s=40,alpha=.85)
        ax.set(xticks=(1,2,3),xlabel='你的草稿评分（越高越好）',ylabel=key,title=title)
        ax.text(.04,.93,f"秩相关 {corr['metrics'][key]['spearman']:.3f}",transform=ax.transAxes,va='top',fontsize=10)
        ax.grid(alpha=.15)
    fig.suptitle('22张评分的探索性核对：覆盖深度比“连成多大一片”更有区分力',fontsize=13)
    fig.savefig(OUTPUT/'pileup_validation.png',dpi=160);plt.close(fig)
    fig,(ax,ex)=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    for eta,label,color in ((0.,'只移动','#7f9991'),(.05,'移动＋小幅改形','#146f67')):
        records=[r for r in curves if r['eta']==eta]
        ax.plot([100*r['delta'] for r in records],[100*r['mean_bbox_reduction'] for r in records],'o-',label=label,color=color)
    ax.set(xlabel='每块最大位移 / 自身等效边长（%）',ylabel='已找到的包围盒收缩 / 原包围盒（%）',title='23例均值：满足全部保护条件的收益下界')
    ax.legend();ax.grid(alpha=.2)
    tagged=next(r for r in rows if r['id']=='case_0063')
    for eta,label,color in ((0.,'只移动','#7f9991'),(.05,'移动＋小幅改形','#146f67')):
        ps=[probe(tagged,d,eta)['strict'] for d in BUDGETS]
        ex.plot(np.asarray(BUDGETS)*100,[p['removable_whitespace_over_old_empty']*100 for p in ps],'o-',label=label,color=color)
    ex.set(xlabel='每块最大位移 / 自身等效边长（%）',ylabel='可消除原留白的比例（%）',title='case_0063：你标记“留白太多”的那例')
    ex.legend();ex.grid(alpha=.2)
    fig.suptitle('平均位移≤横轴预算的1/4；改形保持面积，宽度对数变化≤5%；线长增幅≤0.1%',fontsize=11)
    fig.savefig(OUTPUT/'whitespace_budget_curves.png',dpi=160);plt.close(fig)
    # Up to five examples, always through the shared layout renderer.
    ids=['case_0063','case_0076','case_0670','case_0898']
    (OUTPUT/'images').mkdir(exist_ok=True)
    for case_id in ids:
        r=next(r for r in rows if r['id']==case_id);c=r['constraints_for_plot']
        raw=read(ROOT/r['source_draft'])['raw_positions']
        v={'raw':raw,'move':probe(r,.05,0.)['positions'],'joint':probe(r,.05,.05)['positions']}
        adapted={'panel':'Diagnostic witness / NOT a new model','case_id':case_id,'constraints':c,
                 'caption':'Same original draft | motion only | motion + shape, each under the declared small budget',
                 'variants':{k:{'positions':p,'soft_details':geometry_details(p,c)} for k,p in v.items()}}
        plot_case(adapted,OUTPUT/'images'/f'{case_id}.png',show_metrics=False,
                  stages=(('raw','Original E42 draft'),('move','Small-motion witness'),('joint','Motion + small shape witness')))
    lookup={r['id']:r for r in rows}
    w=probe(lookup['case_0063'],.05,.05)['strict']
    cm=corr['metrics'];lines=['# 人工反馈几何诊断：22分数＋1仅标签','','本轮按用户表示“可以了，就这样”停止继续评分；保留22张评分、1张仅标签，未强制补齐100。评分1/2/3为4/9/9张。原模型、Loss、生产后处理及原最终Q不改。',
        '', '## 度量与证据边界','','完整定义见[feedback_metrics.md](../../feedback_metrics.md)，精确逐例数据见本机metrics_local.csv、cases_local；反馈快照保留原话及全部评分变更。',
        '', '| 指标 | 与评分的秩相关 | 控制模块数后的秩相关 |', '|---|---:|---:|']
    for key,name in [('largest_any_component_fraction','最大任意重叠连通块占比'),('coverage_energy','每块覆盖负荷平方均值'),('complex_energy_10pct','10%强图的局部复杂堆积能量'),('pair_overlap_fraction','总两两交叠面积/模块面积'),('empty_fraction','原始空白率（非无效留白指标）')]:
        lines.append(f"| {name} | {cm[key]['spearman']:.3f} | {cm[key]['partial_rank_correlation_controlling_n']:.3f} |")
    lines+=['','普通最大连通块的区分力较弱，覆盖深度更符合这22张评分。局部强图能定位具体问题团，但没有超过更简单的覆盖负荷或总交叠面积，不能为了“复杂”而强行采用合成能量作为唯一主指标。1分和2分的局部能量均值很接近，说明它也不能解释全部主观差异。',
        '', '![堆积指标与评分](pileup_validation.png)','','## 你标记留白太多的case_0063','',
        f"原始真实空白率{lookup['case_0063']['pileup']['empty_fraction']*100:.3f}%。小预算规定每块中心最多移动等效边长5%、平均≤1.25%，宽度对数变化≤5%，面积精确保留。实际平均移动{w['mean_center_move_local']*100:.3f}%、最大{w['max_center_move_local']*100:.3f}%；包围盒缩小{w['bbox_reduction_fraction']*100:.3f}%，即原留白的{w['removable_whitespace_over_old_empty']*100:.3f}%；HPWL变化{w['HPWL_relative_change']*100:+.3f}%。每一对重叠不增加，真实占用并集不减少，官方边界/分组/MIB违反计数及硬规格均通过保护复核。",
        '', '仅移动和仅改形探测的包围盒收缩都接近0，联合改动有正收益：提供了一份形状与位置共同作用的具体证据；不是“单独改动一定无解”的证明。放宽移动预算到20%时，该例见证约10.265%的原留白可消除，仍不是全局最优。',
        '', '![位移预算曲线](whitespace_budget_curves.png)','','## 避免误读的反例与限制','',
        '- case_0076（2分）局部强团7/28块，平均覆盖负荷48.285%，最高79.940%；case_0898（1分）最严重能量团4/110块、均值60.841%、最高93.003%，但全例还有其他团。能量并不构成对所有评分的严格排序。',
        '- case_0670具有留白，但本探测器尚未找到通过全部约束保护的收缩；放松软约束可找到约0.32%包围盒收缩。不能把有代价的收缩算作廉价无效留白，也不能据此证明零收益。',
        '- 合成验证让完全相同的空白面积分别带有可移动模块或全部预放置锚点；前者有可执行收缩、后者无收缩，验证指标没有退化成空白率。',
        f"- 数值求解621个子问题，状态{statuses}；原布局/既有合格候选始终保留。数值失败、固定端点方位、贴边保护、仅三种宽高方向都会低估收益。",
        '- 只有1张明确留白标签，不能验证留白指标的排序准确性。指标是在看到反馈后提出，相关区间不包括指标设计选择的不确定性；不代表独立泛化、模型改进或新的正式Q。',
        '', '## 文件与复现','','`feedback_snapshot.json`为本机只读导出快照；`config.json`固定定义和预算；`analysis.json`保留汇总、相关区间与数值失败；`verification.json`保留合成反例检查。`initial_probe_*`保留首例实现排错证据（主指标改成包围盒收缩、增加原满足边界保护之前），不混入最终23例统计。',
        '', '图片四例只为解释改动：不是新的模型输出或生产后处理结果。所有布局图由visualization.layouts统一绘制。',
        '', '已保存证据的离线复现（不运行网络或重新探测）：', '', '```powershell',
        "& '.\\.conda\\envs\\iccad2026\\python.exe' -m experiments.E42_joint_regularization.analyze_feedback",'```','']
    (OUTPUT/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved analysis, CSV, two research figures and four shared-renderer examples.')
    print(json.dumps({'correlations':{k:cm[k]['spearman'] for k in cm},'curves':curves,'joint_exceeds_single':result['joint_better_than_best_single_arm_at_5pct_count']},ensure_ascii=True))


if __name__=='__main__':main()
