"""Charts, selected saved-draft illustrations and a self-contained statistical report."""
from concurrent.futures import ProcessPoolExecutor
import html
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .io import HERE, ROOT, read, save, sha

OUT = HERE/'runs/20261028_dev2000'
OLD_DRAFTS = ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'


def example_image(row):
    from datasets.reference_layouts import ReferenceLayouts
    from visualization.layouts import plot_case, geometry_details
    item = ReferenceLayouts('dev_monitor500')[row['index']]
    old = read(OLD_DRAFTS/f"case_{row['index']:04d}.json")
    assert old['member'] == row['member'] == list(item['member'])
    assert old['checkpoint_sha256'] == '12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
    constraints = item['case'].solver_inputs[5].tolist()
    variants = {key: {'positions': value, 'soft_details': geometry_details(value, constraints)}
                for key, value in (('reference',item['positions'].tolist()), ('e42',old['positions']), ('e65',row['positions']))}
    target = OUT/f"examples/case_{row['index']:04d}.png"
    plot_case({'panel':'E69 selected diagnostic example', 'case_id':f"case_{row['index']:04d}",
               'constraints':constraints, 'variants':variants,
               'caption':'Original reference before area transform | Frozen E42 draft | Frozen E65 draft; selected examples, not human ratings'},
              target, show_metrics=False, stages=(('reference','Reference'),('e42','E42 draft'),('e65','E65 draft')))
    return {key: value for key, value in row.items() if key != 'positions'} | {'image':str(target.relative_to(OUT))}


def make_charts(a, f, forward, supplemental):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e65 = forward['models']['E65_geometry']; e42 = forward['models']['E42']
    y = np.array([row['loss_total'] for row in e65]); old_y = np.array([row['loss_total'] for row in e42])
    overlap = np.array([row['overlap_area_fraction'] for row in e65]); old_o = np.array([row['overlap_area_fraction'] for row in e42])
    n = np.array([row['n'] for row in f]); pins = np.array([row['pin_connected_fraction'] for row in f])
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    bins = np.geomspace(min(y.min(),old_y.min()), max(y.max(),old_y.max()), 45)
    axes[0,0].hist(old_y,bins=bins,alpha=.5,label='E42');axes[0,0].hist(y,bins=bins,alpha=.5,label='E65')
    axes[0,0].set(xscale='log',xlabel='Fixed G (log scale)',ylabel='Cases',title='Most difficulty variation is between cases');axes[0,0].legend()
    im=axes[0,1].scatter(pins,y,c=n,s=9,alpha=.45,cmap='viridis');axes[0,1].set(yscale='log',xlabel='Fraction of blocks connected to pins',ylabel='E65 G',title='Pin coverage versus case Loss')
    fig.colorbar(im,ax=axes[0,1],label='Module count')
    cohorts=('bottom50pct','all','top10pct');bottom=np.zeros(3)
    for term in a['E65_weighted_term_decomposition']['all']['weighted_terms']:
        values=np.array([a['E65_weighted_term_decomposition'][cohort]['weighted_terms'][term] for cohort in cohorts])
        axes[1,0].bar(np.arange(3),values,bottom=bottom,label=term);bottom+=values
    axes[1,0].set(xticks=np.arange(3),xticklabels=['Lower-Loss half','All cases','Highest-Loss 10%'],ylabel='Mean weighted Loss',title='Actual weighted term contributions');axes[1,0].legend(fontsize=7,ncol=2)
    labels=[('21-39',30),('40-59',50),('60-79',70),('80-99',90),('100-120',110)]
    for model in ('E42','E65_geometry'):
        axes[1,1].plot([v for _,v in labels],[a['N_strata'][key][model]['G_mean'] for key,_ in labels],'o-',label=model)
    axes[1,1].set(xlabel='Module-count band (midpoint)',ylabel='Mean G',title='More modules does not imply higher G');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'task_structure.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,9))
    axes[0,0].scatter(y,100*overlap,c=n,s=8,alpha=.4,cmap='viridis')
    axes[0,0].set(xscale='log',xlabel='E65 G',ylabel='Pair-overlap area / total area (%)',
        title=f"Across cases: rank rho={a['cross_case_correlations']['E65_geometry']['G_vs_overlap_rho']:.3f}")
    dx=y-old_y;dy=100*(overlap-old_o)
    axes[0,1].scatter(dx,dy,s=9,alpha=.35);axes[0,1].axhline(0,color='black',lw=.8);axes[0,1].axvline(0,color='black',lw=.8)
    axes[0,1].set(xlabel='G(E65) - G(E42)',ylabel='Overlap change (percentage points)',
        title=f"Same-case changes: rho={a['paired_changes']['E65_minus_E42']['G_delta_vs_overlap_delta_rho']:.3f}")
    from .collect import RUNS,key
    terminal=read(RUNS['E65_geometry']/'terminal/results.json')['records'];by_member={key(row['member']):i for i,row in enumerate(e65)}
    gx=np.array([y[by_member[key(row['member'])]] for row in terminal]);q=np.array([row['metrics']['cost'] for row in terminal])
    axes[1,0].scatter(gx,q,s=13,alpha=.5);axes[1,0].set(xscale='log',xlabel='E65 G',ylabel='Saved flash neutral-runtime Q',
        title=f"Saved500 downstream Q: rho={a['cross_case_correlations']['E65_geometry']['G_vs_flash_Q_rho']:.3f}")
    for index,(name,title) in enumerate((('same_case_ranking_overlap_dev2000','Raw overlap'),('same_case_ranking_flash_dev500','Flash Q'))):
        value=a[name];mean=value['case_equal_weight_accuracy'];lo,hi=value['worker_bootstrap95']
        axes[1,1].bar(index,mean,color=['#2c7fb8','#41ab5d'][index]);axes[1,1].errorbar(index,mean,yerr=[[mean-lo],[hi-mean]],color='black',capsize=5)
        axes[1,1].text(index,mean+.025,f'{mean:.1%}',ha='center')
    axes[1,1].axhline(.5,color='gray',ls='--');axes[1,1].set(xticks=[0,1],xticklabels=['Raw overlap','Flash Q'],ylim=(.45,.72),ylabel='Same-case pairwise agreement',title='Seven-model candidate ranking; worker-bootstrap95%')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'loss_quality.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    rows=[r for r in a['E65_feature_associations'] if r['within_exact_N_rank_rho'] is not None][:10]
    axes[0].barh([r['feature'] for r in rows][::-1],[r['within_exact_N_rank_rho'] for r in rows][::-1],color=['#d95f0e' if r['uses_reference_label'] else '#2c7fb8' for r in rows][::-1])
    axes[0].axvline(0,color='black',lw=.8);axes[0].set(xlabel='Within-exact-N rank association with G',title='Exploratory task associations (not causal effects)')
    cv={**a['explanatory_ridge_worker_CV'],**supplemental['case_feature_CV_ablations']}
    chosen=[('N_only','Module count only'),('N_plus_pin_summaries','N + pin summaries'),('public_without_pin_summaries','Public without pin summaries'),('public_inputs','All public task features'),('public_and_reference','Public + reference descriptors')]
    axes[1].barh([label for _,label in chosen][::-1],[cv[key]['out_of_fold_R2'] for key,_ in chosen][::-1],color='#41ab5d')
    axes[1].set(xlim=(0,1),xlabel='Worker-held-out R2 for log G',title='Fixed ridge explanatory models, five worker folds')
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'task_predictors.png',dpi=160);plt.close(fig)


def select_examples(features, forward):
    from .collect import RUNS,key
    terminal=read(RUNS['E65_geometry']/'terminal/results.json')['records'];t={key(row['member']):row for row in terminal}
    e65={key(row['member']):row for row in forward['models']['E65_geometry']};e42={key(row['member']):row for row in forward['models']['E42']};ff={key(row['member']):row for row in features}
    available=[key(read(path)['member']) for path in sorted(OLD_DRAFTS.glob('case_*.json'))]
    y=np.array([e65[k]['loss_total'] for k in available]);o=np.array([e65[k]['overlap_area_fraction'] for k in available]);pin=np.array([ff[k]['pin_connected_fraction'] for k in available])
    old=np.array([e42[k]['loss_total'] for k in available]);old_o=np.array([e42[k]['overlap_area_fraction'] for k in available])
    used=set();selections=[]
    def choose(label,score,eligible=None):
        ids=[i for i in np.argsort(score)[::-1] if i not in used and (eligible is None or eligible[i])]
        if not ids:return
        i=ids[0];used.add(i);k=available[i];record=t[k]
        selections.append({'label':label,'index':record['index'],'member':record['member'], 'n':record['n'],
            'G_E42':float(old[i]),'G_E65':float(y[i]),'overlap_E42':float(old_o[i]),'overlap_E65':float(o[i]),
            'pin_coverage':float(pin[i]),'connection_density':ff[k]['connection_density'],'positions':record['draft_positions'],
            'E65_terms':{term:e65[k]['loss_'+term] for term in ('center_local','shape','relative_local','overlap','contact')}})
    choose('Loss名次较差、重叠名次相对较好',rankdata(y)-rankdata(o))
    choose('Loss名次相对较好、重叠名次较差',rankdata(o)-rankdata(y))
    choose('同例E65的Loss下降但重叠上升',o-old_o,(y<old)&(o>old_o))
    choose('公开端口覆盖低的高Loss案例',y,pin<=.3)
    choose('公开端口覆盖高的低Loss案例',-y,pin>=.8)
    (OUT/'examples').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        rendered=list(pool.map(example_image,selections))
    save(OUT/'examples.json',{'selection_pool':'Previously frozen random100 developer gallery; E42 has only100 saved raw drafts',
         'selection':'Post-hoc diagnostic contrasts by Loss/overlap rank and public pin coverage, not representative or human ratings',
         'cases':rendered,'training_started':False,'test_read':False})
    return rendered


def make_report(a, f, forward, supplemental, examples):
    y=np.array([r['loss_total'] for r in forward['models']['E65_geometry']]);order=np.argsort(y);low=order[:1000];high=order[-200:]
    m=a['E65_module_diagnostics'];terms=a['E65_weighted_term_decomposition'];gap=terms['top10pct']['G_mean']-terms['bottom50pct']['G_mean']
    gap_shares={k:(terms['top10pct']['weighted_terms'][k]-terms['bottom50pct']['weighted_terms'][k])/gap for k in terms['all']['weighted_terms']}
    actual_overlap=np.array([r['overlap_area_fraction'] for r in forward['models']['E65_geometry']])
    cv=a['explanatory_ridge_worker_CV'];node=supplemental['within_case_node_associations']['models']['E65']
    report=f'''# E69：Loss为什么在不同草稿任务之间不同

用户反馈原话：**“感觉在draft的观感上，E65和E42并没有很大差别，仍然没有达到要求。”** 这是整轮定性评价，不是逐例评分。

本轮只做统计：固定v5开发2000例、7个已完成模型；后处理指标复用其中同一固定500例。没有恢复训练、调整Loss、新跑后处理或读取测试。补了冻结E42/E65只读前向，以取得旧文件缺失的逐例/逐模块量；核对原汇总与E65逐例一致。模型仍全部seed0。

## 主要发现

1. **Loss确实能反映跨案例的几何困难，但不能可靠替代同一案例内的候选比较。** E65的G与真实交面积比例Spearman为0.896，按精确模块数分层后的秩相关仍0.763。可是7个模型在同例内两两比较时，G与重叠的排序一致率为63.5%，与flash Q为58.2%。这是高于随机但有限的判断力，不能因跨案例相关高就按G断言每份草稿的主观质量。
2. **本轮多数Loss变化来自共同的案例难度，而非模型平均差异。** 平衡2000×7面板原始G的离差平方和分解：案例主效应93.81%，模型主效应0.21%，交互5.98%。E42/E65逐例G排序相关0.973，最差10%集合Jaccard为0.717。该分解取决于这些相近模型，不是不可约误差，也不能证明改架构无用。
3. **更强的难度线索是端口连接覆盖和图连接稠密程度，模块数并非主因。** 在精确N内，端口覆盖与G的秩相关−0.577，连接密度−0.367。固定岭回归、按来源worker留出预测log G，只有N的R²约2.0%，加入端口统计约50.6%，所有公开特征约80.7%。这是开发集内部的解释性验证，不是新的独立泛化成绩。
4. **高Loss主要对应参考位置关系没有学好，既包含小模块归一化，也有实际位置错误。** 高Loss十分位与低Loss半群的G差，中心项解释{gap_shares['center_local']:.1%}、相对位置项{gap_shares['relative_local']:.1%}、接触项{gap_shares['contact']:.1%}。最小面积三分之一模块平均贡献59.9%的中心Loss，但其绝对中心误差也更大；这不是纯粹被分母抬高的假象。

## 一、不同任务究竟差多少

E65完整开发G：均值{y.mean():.6f}，中位数{np.median(y):.6f}，P90 {np.percentile(y,90):.6f}，P99 {np.percentile(y,99):.6f}，最大{y.max():.6f}。最高Loss的200例贡献全部G之和的24.14%。

| E65分组 | 案例数 | 平均G | 平均真实交面积比例 | 平均端口覆盖 | 平均连接密度 |
|---|---:|---:|---:|---:|---:|
| Loss较低的50% | 1000 | {y[low].mean():.6f} | {actual_overlap[low].mean():.3%} | {np.mean([f[i]['pin_connected_fraction'] for i in low]):.1%} | {np.mean([f[i]['connection_density'] for i in low]):.3f} |
| Loss最高的10% | 200 | {y[high].mean():.6f} | {actual_overlap[high].mean():.3%} | {np.mean([f[i]['pin_connected_fraction'] for i in high]):.1%} | {np.mean([f[i]['connection_density'] for i in high]):.3f} |

这里的“高/低”按G事后分层，不是人工质量标签或新数据划分。端口覆盖指有正权重端口连接的模块比例；连接密度来自公开模块连接图。

![案例结构、分布与损失项](task_structure.png)

## 二、端口和连接信息的线索有多扎实

在进一步控制N、连接密度、预放置/固定比例、面积Gini与边界比例后，端口覆盖的探索性偏秩相关为{supplemental['pin_coverage_partial_rank_rho']:.3f}。R²对照：去掉端口统计降至{supplemental['case_feature_CV_ablations']['public_without_pin_summaries']['out_of_fold_R2']:.1%}；去掉图连接统计为{supplemental['case_feature_CV_ablations']['public_without_connectivity_summaries']['out_of_fold_R2']:.1%}；额外加入reference形态描述只把完整公开特征的R²从{cv['public_inputs']['out_of_fold_R2']:.1%}变成{cv['public_and_reference']['out_of_fold_R2']:.1%}。

又用既有500例E65草稿作同一布局内模块比较，排除预放置中心：有端口连接的{node['pin_modules']}个模块，解码后局部中心误差均值{node['pin_mean_decoded_local_error']:.3f}；无端口连接的{node['no_pin_modules']}个为{node['no_pin_mean_decoded_local_error']:.3f}。440例同时有两类模块，其中{node['mixed_cases_pin_error_lower_fraction']:.1%}的有端口模块误差更低。

使用案例固定效应、每例等权，控制面积/面积平方、加权连接度、边界、固定形状后，有端口标志对log(1+局部误差)的回归系数为{node['coefficients'][0]:.3f}，按worker重抽95%区间[{node['pin_coefficient_worker_bootstrap95'][0]:.3f}, {node['pin_coefficient_worker_bootstrap95'][1]:.3f}]。这是同例内部的关联支持，仍不等于增加端口能因果地改善布局。误差来自已解码中心，和反传前的center_local不是完全同一量。

**当前合理解释是：公开位置线索少、图约束稀疏的案例更难。** 可能是输入不足以确定参考位置，也可能是模型没有传播好已有稀疏信息，或者单一reference并非唯一合理布局；本轮统计无法区分这些机制，不能直接宣布“信息论上做不到”。

![控制N的关联与worker留出解释模型](task_predictors.png)

## 三、Loss差异由哪些数值项和模块产生

E65全开发加权G中，中心参考占53.0%，相对位置16.8%，形状11.9%，接触11.7%，穿透重叠5.7%，其余约1%。这是标量贡献，不是梯度贡献。

高Loss十分位平均G0.844669，低Loss半群0.178973。中心和相对关系合计解释两群G均值差约76.6%。高Loss组不是只有几个小块变差：平均全局尺度中心误差从{m['bottom50pct']['mean_center_error_global']:.4f}增到{m['top10pct']['mean_center_error_global']:.4f}，大小模块均有明显增加。

局部归一化确实影响Loss：小/大面积三分之一模块的平均全局中心误差为{m['all']['small_center_error_global']:.4f}/{m['all']['large_center_error_global']:.4f}，局部尺度误差为{m['all']['small_center_error_local']:.3f}/{m['all']['large_center_error_local']:.3f}。但面积Gini与G在精确N内的秩相关仅0.062，不能据“小模块贡献大”断言面积不均是案例间难度主因。

共同平移也不是主要解释：自由中心误差的共同平移部分平均仅占中心MSE的4.7%；减去均值偏移后中心Loss只由0.185478降到0.182076。参考状态解码G均值0.025601，和模型G的秩相关仅0.067，未支持“主要是参考与面积变换不兼容导致高Loss”。参考状态只是诊断，不是下界，不能直接相减当可约误差。

## 四、为什么平均Loss下降了，观感仍然相近

跨案例G与重叠的强相关主要说明G能区分困难案例。比较相同案例的E65−E42变化时，G差与重叠差的秩相关降到0.218。2000例中，E65有1201例G更低，其中436例重叠反而更高，约36.3%。

所有7模型的同例两两比较，重叠排序一致率63.50%，worker重抽95%区间[62.51%,64.40%]；flash Q一致率58.15%，区间[56.61%,59.77%]。后者排除Q并列等985个比较，482/500例仍有可比候选。bootstrap反映案例来源不确定性，不含模型种子不确定性。

E65跨案例G与flash Q的秩相关0.352，在精确N内约0.222；flash会修复草稿且存在离散路径变化，和原始draft目标不同，弱相关不等于G对草稿无用。

**没有逐例人工评分，不能把与重叠的相关直接等同于“观感已校准”。** 用户当前反馈只支持整轮观感未达标。以上结果支持保留G作为困难度与监督误差参考，同时必须看同例变化、分项和具体草稿。

![跨案例相关与同例候选判断的区别](loss_quality.png)

## 五、诊断案例与下一步问题

以下5例从已有随机100图中事后挑选，展示名次错位、同例改进冲突及公开信号差异，不代表总体，不是人工好坏评分。Reference是面积均化前原标签；E42/E65为已保存草稿。
'''
    for row in examples:
        report+=f"\n### {row['label']}：case_{row['index']:04d}，{row['n']}模块\n\nG：E42 {row['G_E42']:.4f} → E65 {row['G_E65']:.4f}；重叠：{row['overlap_E42']:.2%} → {row['overlap_E65']:.2%}；端口覆盖{row['pin_coverage']:.1%}。\n\n![同例参考/E42/E65]({row['image']})\n"
    report+='''
目前更值得追问的是：在同模块数、低端口覆盖且连接稀疏的案例上，错误集中在哪些无锚定模块，模型能否利用已有连接恢复相对结构，以及reference的特定位置是否由输入充分决定。应先围绕这些具体任务诊断，再决定改表示、监督方式还是数据；本轮不启动新训练、不改变冻结G。

## 证据与限制

- [逐例数据表](case_table.csv)、[完整统计](analysis.json)、[补充控制及同例模块分析](followup.json)、[执行记录](results.json)、[配置与来源](config.json)、[输入文件及权重核对](input_manifest.json)、[源码快照](source_snapshot.zip)、[示例选择](examples.json)。
- 2000例是已暴露的开发集，500例是其嵌套子集；7模型并非独立重复样本。worker留出与bootstrap只作内部描述稳定性检查，不能称盲测泛化或正式比赛成绩。
- 特征筛查和补充回归是探索性分析，未进行因果干预；reference描述含标签，不能进入部署模型。公开特征间相关，R²差不是独立因果贡献。
- Loss用同一冻结G，历史E42训练/选模协议保持原事实。新前向为GPU批量FP32，保存的flash来自CPU完整流程；同权重浮点执行路径有细小差异，未重跑后处理或选模。
- 原checkpoint未改，训练和自动唤醒仍中止，测试未读。本报告不自动提出或执行任何恢复动作。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    # Offline reader; no rating widgets, scripts, remote resources or scoring flow.
    blocks=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>E69 草稿Loss差异统计</title><style>body{max-width:1100px;margin:40px auto;padding:0 22px;font:17px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#203044;background:#f6f8fa}h1,h2,h3{line-height:1.4}img{width:100%;background:white;border:1px solid #dce3eb}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #dce3eb;padding:7px;text-align:left}p,li{margin:.75em 0}a{color:#1769aa}blockquote{border-left:4px solid #92a8bb;padding-left:16px}</style>']
    import re
    in_table = False
    for line in report.splitlines():
        if line.startswith('|'):
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                continue
            tag = 'td' if in_table else 'th'
            if not in_table:
                blocks.append('<table>')
                in_table = True
            blocks.append('<tr>'+''.join(f'<{tag}>{html.escape(cell)}</{tag}>' for cell in cells)+'</tr>')
            continue
        if in_table:
            blocks.append('</table>')
            in_table = False
        if not line.strip():continue
        if line.startswith('!['):
            src=line.split('](',1)[1][:-1];blocks.append(f'<p><img src="{html.escape(src)}"></p>');continue
        tag='p';value=line
        if line.startswith('### '):tag='h3';value=line[4:]
        elif line.startswith('## '):tag='h2';value=line[3:]
        elif line.startswith('# '):tag='h1';value=line[2:]
        value=html.escape(value);value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
        value=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
        blocks.append(f'<{tag}>{value}</{tag}>')
    if in_table:blocks.append('</table>')
    blocks.append('</html>');(OUT/'index.html').write_text('\n'.join(blocks),encoding='utf-8')
    save(OUT/'report_provenance.json',{'source_sha256':sha(Path(__file__)), 'node_analysis_source_sha256':sha(HERE/'node_analysis.py'),
        'followup_source_sha256':sha(HERE/'followup.py'),'new_inference':False,'training_started':False,'test_read':False})


def main():
    a=read(OUT/'analysis.json');f=read(OUT/'features.json')['records'];forward=read(OUT/'forwards.json');supplemental=read(OUT/'followup.json')
    make_charts(a,f,forward,supplemental)
    examples=select_examples(f,forward)
    make_report(a,f,forward,supplemental,examples)
    print('E69 report,3 chart panels and5 saved-draft examples complete; no training/inference run.',flush=True)


if __name__=='__main__':main()
