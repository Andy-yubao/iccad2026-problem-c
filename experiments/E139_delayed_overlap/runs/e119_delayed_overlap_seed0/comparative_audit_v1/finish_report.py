"""Transparent secondary checks and standalone audit figures/report."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from helpers import pair_motion

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PRIOR=ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0/layer_analysis_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def main():
    s=read(HERE/'summary.json')
    rows=[read(HERE/f'cases/{i:04d}.json') for i in range(500)]
    shards=sorted({r['member'][0] for r in rows})
    cluster=np.array([shards.index(r['member'][0]) for r in rows])
    rng=np.random.default_rng(20261115)
    draws=rng.multinomial(len(shards),np.full(len(shards),1/len(shards)),size=2000)[:,cluster]
    n=np.array([r['n'] for r in rows])
    q={m:np.array([r['models'][m]['final_metrics']['cost'] for r in rows]) for m in ('E134','E139')}
    sensitivity={'tail':{},'group_cohorts':{},'exchange_margins':{},'exchange_cases':[]}
    for label,exclude in [('all',[]),('without_top_improvement_dev481',[481]),('without_top_two_dev481_dev397',[481,397])]:
        mask=~np.isin(np.arange(500),exclude)
        w=np.exp((n[mask]-n[mask].max())/12)
        w=.5*(w/w.sum()+1/mask.sum())
        a,b=w@q['E134'][mask],w@q['E139'][mask]
        sensitivity['tail'][label]=dict(cases=int(mask.sum()),excluded=exclude,E134_mixed_Q=float(a),E139_mixed_Q=float(b),delta=float(b-a))
    for cohort in ('E134_S3','E139_S3'):
        den=np.array([r['group_cohorts'][cohort]['count'] for r in rows])
        sensitivity['group_cohorts'][cohort]={}
        for key in ('unblocked','strict','path_strict'):
            values={m:np.array([sum(e[key][12] for e in r['group_cohorts'][cohort]['models'][m]) for r in rows]) for m in ('E134','E139')}
            diff=values['E139']-values['E134']
            boot=(draws@diff)/np.maximum(draws@den,1)
            sensitivity['group_cohorts'][cohort][key]=dict(E134=float(values['E134'].sum()/den.sum()),E139=float(values['E139'].sum()/den.sum()),
                delta=float(diff.sum()/den.sum()),delta95=np.quantile(boot,[.025,.975]).tolist())
    code=inspect.getsource(pair_motion).replace('def pair_motion(p,areas,flags):','def with_margin(p,areas,flags,margin):').replace('.02*scale','margin*scale')
    scope={'np':np}
    exec(code,scope)
    motion=scope['with_margin']
    positions={'E134':np.load(PRIOR/'positions.npy',mmap_mode='r'),'E139':np.load(HERE/'positions.npy',mmap_mode='r')}
    areas=np.load(PRIOR/'areas.npy',mmap_mode='r')
    flags=np.load(PRIOR/'flags.npy',mmap_mode='r')
    for margin in (.005,.02,.05):
        sensitivity['exchange_margins'][str(margin)]={}
        for model in ('E134','E139'):
            count=0;cases=0
            for i,row in enumerate(rows):
                nn=row['n']
                _,events,_=motion(positions[model][i,3:,:nn],areas[i,:nn],flags[i,:nn],margin)
                count+=len(events);cases+=bool(events)
            sensitivity['exchange_margins'][str(margin)][model]=dict(events=count,cases=cases)
    for row in rows:
        old,new=row['models']['E134'],row['models']['E139']
        if new['exchanges']:
            sensitivity['exchange_cases'].append(dict(index=row['index'],member=row['member'],events=new['exchanges'],
                E134_Q=old['final_metrics']['cost'],E139_Q=new['final_metrics']['cost'],
                grouping_violations=[old['final_metrics']['grouping_violations'],new['final_metrics']['grouping_violations']]))
    save(HERE/'sensitivity.json',sensitivity)
    print(json.dumps(sensitivity,ensure_ascii=False,indent=2),flush=True)
    render(s,rows,sensitivity)


def render(s,rows,sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    colors={'E134':'#475569','E139':'#d45b32'}
    labels={'E134':'E134 stride3 / best63','E139':'E139 delayed overlap / best42'}
    x=np.arange(13)
    fig,axes=plt.subplots(3,2,figsize=(11,11),layout='constrained')
    specs=[('geometry','group_contact_bottleneck','Same-group contact bottleneck (lower better)',1),
        ('geometry','group_edge_blocked_fraction','Current group-tree centerline blocked edges (%)',100),
        ('topology','local_axis_accuracy','Reference local direction accuracy (%)',100),
        ('topology','knn6_recall','Reference K6 neighbor recall (%)',100),
        ('geometry','repeated_coverage_fraction','Exact repeated coverage (%)',100)]
    for ax,(section,key,title,scale) in zip(axes.flat,specs):
        for model in ('E134','E139'):
            y=[r[section][key][model]*scale for r in s['layers']]
            ax.plot(x,y,marker='o',ms=3,color=colors[model],label=labels[model])
        ax.set_title(title)
        ax.set_xlabel('Actual layer S0-S12')
        ax.set_xticks(range(0,13,2))
        ax.grid(alpha=.2)
    ax=axes.flat[5]
    cohort=s['group_cohorts']['E134_S3']
    for model in ('E134','E139'):
        value=cohort['models'][model]['path_strict']['persistent_repair_counts_by_layer']
        ax.plot(range(3,13),np.array(value)/1170*100,marker='o',ms=3,color=colors[model],label=labels[model])
    ax.set_title('Fixed 1170 barriers: persistent clear + path cost halved (%)')
    ax.set_xlabel('Actual layer S3-S12')
    ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Frozen dev500 audit: more late adjustment, no broad topology breakthrough',fontsize=14)
    fig.savefig(HERE/'topology_comparison.png')
    fig.savefig(HERE/'topology_comparison.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    names=['Grouping','HPWL','Area','Boundary']
    changes=[s['score']['attribution'][key]['mixed_Q'] for key in ('grouping','hpwl','area','boundary')]
    axes[0].barh(names,changes,color=['#337d65' if v<0 else '#b95746' for v in changes])
    axes[0].axvline(0,color='#334155',lw=.8)
    axes[0].set_title('Exact saved mixed-Q change attribution')
    axes[0].set_xlabel('Q contribution (negative improves)')
    axes[0].ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
    xx=np.arange(2)
    for offset,model in [(-.18,'E134'),(.18,'E139')]:
        totals=[sum(r['models'][model]['final_metrics'][key] for r in rows) for key in ('grouping_violations','boundary_violations')]
        bars=axes[1].bar(xx+offset,totals,.36,color=colors[model],label=labels[model])
        axes[1].bar_label(bars)
    axes[1].set_xticks(xx,['Grouping','Boundary'])
    axes[1].set_ylabel('Total violations across dev500')
    axes[1].set_title('Saved final postprocessing results')
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(HERE/'quality_accounting.png')
    fig.savefig(HERE/'quality_accounting.svg')
    plt.close(fig)

    lines=['# E139 与 E134 每三层：其他指标及拓扑审计','',
        '固定 E134 best63 与 E139 best42，同一 dev500、299 个来源分片。E134 逐层数组复用已核验原件；E139 采集真实 S0–S12，全部 S12 与原评价草稿逐位一致。原冻结权重和源码未改，0 次优化器更新、0 次新后处理、0 次新布局评分，不读 test。',
        '', '## 判断','',
        '**你的 grouping 猜想在本面板最终结果上成立：它是 Q 净改善的主要来源。但本轮尚未证明有稳定、有价值的大规模拓扑突破。**',
        '', 'E139 的组内外框更紧凑、组框内异组侵入稍少，最终断连更少；另一方面，末层接触瓶颈、接触距离、近连接比例、弱关系保真及碰撞大多更差。固定坏关系的严格修复接近或弱于原模型。更适合描述为“局部分组组织及后处理结果有收益”，不能直接称“草稿整体更好拼合”或“已可靠打破坏初始拓扑”。',
        '', '## 1. 最终结果与评分记账','',
        '|指标|E134|E139|变化/解释|','|---|---:|---:|---|']
    for label,section,key,scale,fmt in [
        ('混合 Q↓','score','mixed_Q',1,'.6f'),('平均 Q↓','score','mean_Q',1,'.6f'),
        ('最终分组违规/题↓','final_metrics','grouping_violations',1,'.3f'),
        ('最终贴边违规/题↓','final_metrics','boundary_violations',1,'.3f'),
        ('最终正 HPWL gap 均值↓','final_metrics','positive_hpwl_gap',100,'.4f'),
        ('最终正面积 gap 均值↓','final_metrics','positive_area_gap',100,'.4f')]:
        r=s[section][key]
        lines.append(f"|{label}{'（%）' if scale==100 else ''}|{r['E134']*scale:{fmt}}|{r['E139']*scale:{fmt}}|{r['delta']*scale:+{fmt}}|")
    lines += ['', '最终均 500/500 硬合法、无整行回退；草稿均 0/500 硬合法。分组违规总数 **104→83**（-20.19%），有分组违规题 **84→76**；贴边违规总数 **47→54**，有贴边违规题 **41→49**。1758 个公开分组中，49 组改善、34 组变差、1675 组相同；按题为39题分组改善、30题变差、431题相同。',
        '', '将保存的评分按 HPWL、面积、grouping、boundary 四个因素做对称 Shapley 记账；逐题贡献之和严格等于保存 Q 差，交互项平均分配。它是评分公式分解，不是机制因果识别。',
        '', '|因素|平均 Q 贡献|混合 Q 贡献|','|---|---:|---:|']
    for key in ('grouping','hpwl','area','boundary'):
        r=s['score']['attribution'][key]
        lines.append(f"|{key}|{r['mean_Q']:+.6f}|{r['mixed_Q']:+.6f}|")
    lines += ['', '负数为改善。grouping 收益超过净收益，被边界及部分质量项抵消。HPWL 对平均 Q 有退化，对混合 Q 却有改善，原因是后者含指数规模权重；不能据此说所有题线长都变好。实际运行时未进入本地 Q，耗时下降不会直接压低此 Q。',
        '', '## 2. 草稿、Loss 与分组结构','',
        '|末层共同定义指标|E134|E139|E139−E134 的分片配对95%区间|','|---|---:|---:|---|']
    for label,section,key,scale in [
        ('完整 Loss↓','common_loss','total',1),('Grouping 加权Loss↓','common_loss','grouping_weighted',1),
        ('Boundary 加权Loss↓','common_loss','boundary_weighted',1),('重叠加权Loss↓','common_loss','overlap_weighted',1),
        ('组内接触瓶颈↓','terminal_geometry','group_contact_bottleneck',1),
        ('组内平均接触距离↓','terminal_geometry','group_contact_mean',1),
        ('组外框/组总面积↓','terminal_geometry','group_group_bbox_over_area',1),
        ('组框空余区域被异组占据比例↓','terminal_geometry','group_foreign_empty_bbox_fraction',100),
        ('三种直线接触方案受阻率↓','terminal_geometry','group_tree_blocked_fraction',100),
        ('接触距离≤0.05的全组近连接比例↑','terminal_geometry','group_connected_0.05',100),
        ('贴边相对距离↓','terminal_geometry','boundary_gap_local',1),
        ('逐题最坏模块覆盖率均值↓','terminal_geometry','covered_module_max',100),
        ('外框内空白比例↓','terminal_geometry','whitespace_bbox',100)]:
        r=s[section][key]
        a,b=r['delta95']
        lines.append(f"|{label}{'（%）' if scale==100 else ''}|{r['E134']*scale:.6f}|{r['E139']*scale:.6f}|[{a*scale:+.6f}, {b*scale:+.6f}]|")
    lines += ['', '草稿精确重复覆盖率为5.018%→5.976%，外框/总面积为1.024341→1.026769。上表异组侵入比例的分子为异组占据的组框空余区域，分母为整个组框面积。',
        '', '接触距离按典型边长归一，要求有正长度的共边，是几何可拼合代理。训练 grouping 项与这套距离/瓶颈指标的平方、轴选择及聚合方式不同；Loss 较小不保证最难连接边更好。重叠草稿的多边形并集也可能让“断连数”看起来较小，因此不能把草稿原始断连数（5.128→4.640）独立作为拓扑改善证据。',
        '', '## 3. 固定坏关系：是否更能在后段修复？','',
        '固定 E134 S3 的同组接触 MST：1170 条边存在异组中心连线阻隔，且接触距离>0.05，涉及410题。两模型追踪完全相同的模块对与初始距离阈值。',
        '', '|S12结果，固定1170条边|E134|E139|','|---|---:|---:|']
    c=s['group_cohorts']['E134_S3']['models']
    for key,label in [('unblocked','仅解除中心线阻隔'),('strict','解除阻隔且原边接触距离至少减半'),('path_strict','解除阻隔且组内替代路径瓶颈至少减半')]:
        lines.append(f"|{label}|{c['E134'][key]['S12']} ({c['E134'][key]['S12_fraction']*100:.2f}%)|{c['E139'][key]['S12']} ({c['E139'][key]['S12_fraction']*100:.2f}%)|")
    lines += ['', 'E139 在 S3 已有209条此固定边无阻隔（其中37条满足严格原边条件、47条满足替代路径条件），所以只看终值或修复数量不能全部归功于后层。其 S3→S12 原边严格新修复339、退化23，替代路径新修复405、退化23。E134对应为352/0与452/0。',
        '', '对称地改用 E139 自己 S3 的1936条受阻边（471题），E134/E139 的原边严格修复为786/733，替代路径严格修复为1013/923。两种固定队列均未显示 E139 明确的严格修复优势；各自坏边数和难度不同，不能用E139自身修复数较多宣布能力更强。',
        '', '中心连线代理不是所有可行路径的搜索；解除阻隔加距离减半仍非全组硬合法证明。',
        '', '## 4. 完整换侧、强关系及弱关系','',
        '完整换侧采用旧审计定义：曾沿某轴明确分离，随后反向明确分离，中间中心次序翻转且正交投影在输出间线性连接中交叠。S3之后排除P-P对，阈值0.02典型边长。它是离散读出轨迹代理，不是网络内部连续物理运动。',
        '', '- E134：2次、2题；E139：7次、3题。E139中5次来自dev279的同一模块54，另外dev275/dev319各1次。三题最终grouping违规均为0→0，不能把这7次作为最终分组收益的直接证据。',
        '- 事后阈值敏感性检查：分离/穿越间隙取0.005/0.02/0.05，E134始终2次、2题；E139为8/7/6次，涉及4/3/2题。“大幅换侧仍罕见”的判断不依赖唯一间隙阈值。',
        '- 固定E134 S3的2566条“reference明确分离但预测反向分离”的错误轴关系，S12真正恢复正确强分离为5→9；仅中心方向纠正为77→109。E139在S3已分别为2/50，后段净改善为7/59；E134为5/77。',
        '- 按各自S3错误集合，E139为2471条、正确强分离修复23条（0.93%），E134为2566条、修复5条（0.19%）。不同队列难度不同，而且绝大部分错误仍未被强分离修复。',
        '', '|S12弱关系/局部组织指标|E134|E139|','|---|---:|---:|']
    for key,label in [('local_axis_accuracy','参考局部方向正确率'),('knn6_recall','参考K6邻居召回'),('intent_reference_correct_clear','参考兼容且明确的分离意图')]:
        r=s['layers'][12]['topology'][key]
        lines.append(f"|{label}|{r['E134']*100:.3f}%|{r['E139']*100:.3f}%|")
    lines += ['', 'E139 S3→S12 局部方向正确率确实继续上升（90.441%→92.275%），说明弱关系优化仍在发生；但它从更差的S3出发，末层略低于对照92.434%。K4/K8口径也稍低。reference不是唯一优质拓扑，故这些指标下降不能独立否定布局价值，但也不支持“弱关系优化能力完全无损”的强说法。',
        '', '固定636条E134 S3被向外遮挡的贴边要求，到S12仍受遮挡67→89；同时无遮挡且距边≤0.02典型边长为249→224。最大空矩形/外框面积在S12为0.599%→0.655%，E139从更差S3的2.782%修复而来（E134 S3为1.657%）。',
        '', '## 5. 尾部依赖与耗时解释','',
        'dev481（110模块）是最大有利贡献：Q从1.574052降到1.029849，grouping违规7→1；组1与组2各从3个断连违规降为0。该题没有完整换侧，S12接触瓶颈反而从.177942升至.216657，但组外框/组面积与异组侵入略好。它支持“某些局部分组组织及后处理兼容性改善”，不支持把收益直接归因于强关系换位。',
        '', '|事后尾部敏感性（重新归一原权重）|E134混合Q|E139混合Q|差值|','|---|---:|---:|---:|']
    for label,r in sensitivity['tail'].items():
        label={'all':'完整500题','without_top_improvement_dev481':'剔除最大改善题dev481','without_top_two_dev481_dev397':'剔除前两大改善题dev481/dev397'}[label]
        lines.append(f"|{label}|{r['E134_mixed_Q']:.6f}|{r['E139_mixed_Q']:.6f}|{r['delta']:+.6f}|")
    lines += ['', '这项剔除分析在看过主结果后登记，只展示尾部依赖，不用于重选数据或改写完整dev500主结果。',
        '', '实际完整均时193.1→173.3ms，P95 501.0→406.4ms；但shape调用4.772→4.822、native迭代121.764→122.736，后处理中心移动与形状修改也更多。不能简单解释为工作量减少，更不能在未同环境重测下将全部时延差归因于模型结构。未重跑后处理或测速。',
        '', '## 6. 证据边界','',
        f"按299个来源分片做2000次配对bootstrap，混合Q差的95%区间为[{s['score']['mixed_Q']['delta95'][0]:+.6f}, {s['score']['mixed_Q']['delta95'][1]:+.6f}]，跨0；grouping加权Loss和最终分组违规差的区间也跨0。多项探索性区间未做家族多重比较校正。它们描述这批已反复复用dev的样本不确定性，不能代替多seed或未曝光测试泛化证据。", 
        '', '两组监督层位、碰撞监督分布及停止轮数都不同（E134 66轮/best63，E139 45轮/best42），这不是只改变一项的因果隔离。自己的训练聚合Loss不直接排名。',
        '', '**本轮结论：grouping是已观察到的最终收益来源；有少量局部组织改善，但严格坏关系修复没有整体优势，完整换侧仍罕见，未获得可靠拓扑突破证据。**',
        '', f"![评分记账]({(HERE/'quality_accounting.png').as_posix()})",'', f"![逐层拓扑与修复轨迹]({(HERE/'topology_comparison.png').as_posix()})",
        '', '原始数值：summary.json；逐题配对：paired_metrics.csv；逐组变化：per_group_changes.json；次级检查：sensitivity.json及followup_plan.json；身份/验证：artifact_manifest.json、verification.json。']
    (HERE/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=read(HERE/'artifact_manifest.json')
    manifest['report_artifacts']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('finish_report.py','findings.md','sensitivity.json','followup_plan.json','topology_comparison.png','topology_comparison.svg','quality_accounting.png','quality_accounting.svg')}
    save(HERE/'artifact_manifest.json',manifest)


if __name__=='__main__':
    if '--report-only' in sys.argv:
        render(read(HERE/'summary.json'),[read(HERE/f'cases/{i:04d}.json') for i in range(500)],read(HERE/'sensitivity.json'))
    else:
        main()
