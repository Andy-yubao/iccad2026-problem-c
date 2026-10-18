"""Generate the saved gallery analysis, fixed-shape lower bounds and portable index."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

run=Path(__file__).resolve().parent;root=run.parents[3];sys.path.insert(0,str(root))
from experiments.graph_proposal.plot_soft_constraint_gallery import read_rows,resolve_positions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=json.loads((run/'results.json').read_text(encoding='utf-8'))
cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
gallery=json.loads((run/'gallery_manifest.json').read_text(encoding='utf-8'))
rows=read_rows(run/'cases.jsonl');resolve_positions(rows)
ablation=read_rows(run/'ablation_cases.jsonl')
assert len(rows)==30 and gallery['counts']=={'train':10,'dev':10,'local_test':10}
assert all((run/f['image']).is_file() for f in gallery['figures'])
assert r['checkpoint_unchanged'] and r['new_unopened_test_members']==0

floors=[]
for row in rows:
    c=np.asarray(row['constraints'],dtype=int);ref=np.asarray(row['variants']['reference']['positions'])
    lower=0;groups={}
    for g in np.unique(c[:,2]):
        if g<=0:continue
        ids=np.flatnonzero((c[:,2]==g)&((c[:,0]!=0)|(c[:,1]!=0)))
        incompatible=[]
        for i in ids:
            # Hard dimension tolerance1e-4 plus4decimal rounding: separation
            # >3e-4 in at least one dimension proves shapes cannot merge.
            if all(np.abs(ref[i,2:]-ref[j,2:]).max()>3.00001e-4 for j in incompatible):incompatible.append(int(i))
        bound=max(len(incompatible)-1,0);lower+=bound
        groups[str(g)]={'mutually_incompatible_rigid_blocks':incompatible,'violation_lower_bound':bound}
    actual=row['variants']['final']['metrics']['mib_violations']
    assert lower<=actual
    floors.append({'panel':row['panel'],'case_id':row['case_id'],'rigid_only_mib_lower_bound':lower,
                   'observed_final_mib_violations':actual,'groups':groups})
wins=sum(x['variants']['weight_10']['metrics']['cost']<x['variants']['weight_2']['metrics']['cost']-1e-10 for x in ablation)
ties=sum(abs(x['variants']['weight_10']['metrics']['cost']-x['variants']['weight_2']['metrics']['cost'])<=1e-10 for x in ablation)
d=r['saved_test957_scoring_decomposition'];a=r['dev5_weight_ablation']
audit={'image_counts':gallery['counts'],'dev_weight10_wins_vs2':wins,'dev_weight10_ties_vs2':ties,
       'relative_weight10_improvement':1-a['weight_10']['neutral_runtime_weighted_cost_all_cases']/a['weight_2']['neutral_runtime_weighted_cost_all_cases'],
       'rigid_mib_bounds':floors,
       'rigid_mib_bound_sum':sum(x['rigid_only_mib_lower_bound'] for x in floors),
       'observed_mib_violation_sum':sum(x['observed_final_mib_violations'] for x in floors),
       'note':'Rigid-only lower bounds are conservative. They do not include incompatible free-area intervals; no optimizer or new data reads.'}
(run/'analysis_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(14,5.5))
labels=['Saved output','No boundary\npenalty','No grouping\npenalty','No MIB\npenalty','No soft\npenalties']
keys=['actual','remove_boundary','remove_grouping','remove_mib','remove_all_soft']
vals=[d[k]['weighted_Q'] for k in keys]
bars=axes[0].bar(np.arange(5),vals,color=['#475569','#2563eb','#0891b2','#7c3aed','#15803d'])
for bar,v in zip(bars,vals):axes[0].text(bar.get_x()+bar.get_width()/2,v+.05,f'{v:.3f}',ha='center',fontsize=9)
axes[0].set(xticks=np.arange(5),xticklabels=labels,ylabel='Weighted neutral-runtime Q',ylim=(0,4.5),
            title='Saved test957: scoring decomposition only')
axes[0].tick_params(axis='x',labelsize=9);axes[0].grid(axis='y',alpha=.2)
for row in ablation:
    ys=[row['variants'][f'weight_{w}']['metrics']['cost'] for w in (0,2,10)]
    worker=row['case_id'].split('/')[0].split(':')[-1]
    axes[1].plot([0,1,2],ys,'o-',label=worker,alpha=.75)
axes[1].set(xticks=[0,1,2],xticklabels=['0 (local B/G cost off)','2 (current)','10 (stronger)'],
            ylabel='Per-case neutral-runtime Q',title='Five fixed dev cases: only B/G cost weights change')
axes[1].legend(fontsize=8,ncol=2);axes[1].grid(alpha=.2)
fig.suptitle('Soft penalties matter; the current local weighting also leaves quality unused',fontsize=14,fontweight='bold')
fig.text(.065,.025,'Left: identical saved geometries, feasibility, original N_soft and cap. Removing penalties is not a feasible algorithmic improvement.\nRight: all15 dev outputs hard legal; MIB handling and other settings fixed. Small-panel ablation, not a new default or test-tuned result.',fontsize=9,color='#475569')
fig.tight_layout(rect=(0,.14,1,.93));fig.savefig(run/'soft_penalty_diagnostics.png',dpi=170);plt.close(fig)

counter='\n'.join(f'| {label} | {d[key]["weighted_Q"]:.6f} | {d[key]["relative_reduction"]*100:.2f}% |' for label,key in zip(
    ['原冻结输出','只去掉boundary罚项','只去掉grouping罚项','只去掉MIB罚项','去掉全部软罚项'],keys))
weight_table='\n'.join(f'| {w} | {a[f"weight_{w}"]["neutral_runtime_weighted_cost_all_cases"]:.6f} | {a[f"weight_{w}"]["mean_boundary_violations_returned_geometry"]:.2f} | {a[f"weight_{w}"]["mean_grouping_violations_returned_geometry"]:.2f} | {a[f"weight_{w}"]["mean_mib_violations_returned_geometry"]:.2f} |' for w in (0,2,10))
stages=r['staged_soft_metrics']
stage_table='\n'.join(f'| {panel} | {stage} | {s["boundary_violations_mean"]:.2f} | {s["grouping_violations_mean"]:.2f} | {s["mib_violations_mean"]:.2f} | {s["violations_relative_mean"]:.4f} | {s["feasible_count"]}/10 |' for panel,values in stages.items() for stage,s in values.items())
doc=f'''# 三集合软约束图集与后处理惩罚验证

本次按用户要求生成训练、验证、测试各10张，共30张图，并验证“后处理软约束考虑不足，导致严重惩罚”的假设。此前展示的worker95中位下界图来自**验证集dev_quality100**，不是训练集。

## 1. 图集与阅读方式

[打开全部30张图](index.html) · [训练集](index.html#train) · [验证集](index.html#dev) · [已访问测试集](index.html#local_test)

每张图三栏分别是参考、标准解码原始提议、**原后处理**输出；第三栏只画自己的布局，没有参考矩形或跨阶段位移箭头。并非展示权重10的新输出。

- grouping：同组G编号用相同填色；下方列出每组接触连通分量，1表示连通。
- MIB：同组M编号用相同纹理；下方列出每组宽高种数，1表示尺寸统一。
- boundary：块边L/R/T/B标出要求的左/右/上/下边；绿表示按当前布局自身bbox满足，红表示违反；角要求显示两个徽记。
- F/P是固定形状/预放置标志，红轮廓是重叠块。每栏标注官方B/G/M违反数、硬合法性与中性时间Q。软详情已逐例断言与官方计数相同，不另创计分规则。

三栏保留相同ID与坐标尺度，允许看清后处理的面积扩张；图片采用高分辨率，点击可放大。颜色/纹理/轴范围见[gallery_manifest.json](gallery_manifest.json)。

## 2. 数据、选择及冻结事实

train/dev各10例，从阶段A已保存面板固定选择worker0、11、22、33、44、55、66、77、88、99，不按成绩挑图。它们的原始提议复用此前GPU输出。

test10例只在已经完成的957例中，固定选择worker0、4、8、12、16、21、25、29、33、37，各worker按预先固定SHA规则选一例。**没有读取未打开测试成员，没有续跑原2500例测试。** 原先未保存这些测试样例的raw和约束图，因此为可视化重新读取这10例已访问数据并补算CPU原始提议，与原CPU评价口径一致；监督MSE复核最大差异为{r['max_test_raw_mse_reproduction_error']:.1g}。最终坐标和分数直接复用原冻结测试结果，没有重新合法化测试或用测试调参。

模型仍是best step7250，SHA-256 `{cfg['checkpoint_sha256']}`，未更新权重。train/dev20例运行原后处理，另在固定5例dev做预先声明的局部权重机制对照。数据/诊断运行约{r['wall_seconds']:.2f}秒，另加仅从JSON绘图耗时；预算600秒。

测试图只是已访问前缀中的10例展示，不代表完整测试分布；30例图片的train/dev/test成绩不用于跨集合强弱排名。

## 3. 假设第一部分：软罚项是否严重？是

对**原957例已保存测试输出**保持几何、硬合法性、原N_soft分母、时间因子及可行cap不变，只从官方评分公式清除指定软违反项，得到：

| 评分情形 | 官方权重聚合Q | 相对原Q下降 |
|---|---:|---:|
{counter}

重算原分数逐例误差为0；957例全部硬合法，其中6例触及可行cap。去掉全部软罚项时，数值下降56.10%，boundary单项影响最大。**这是评分分解，不是算法获得的新成绩，也不能声称这些违反都可无代价消除。** 单项降幅不能相加，指数罚项相互作用且存在cap；也不能直接用两个平均因子相乘重构总分。

即使移除全部软罚，纯几何项仍为1.6940，所以软约束并不能解释全部质量差距。HPWL/bbox变差同样需要处理。

## 4. 假设第二部分：当前后处理权衡是否不足？开发小对照支持

固定5例dev（worker0、22、44、66、99）、相同原始提议/12轮修正/排序/形状/硬约束/预算，仅将pack的boundary_weight与group_weight同时设为0、2、10。MIB统一、顶层规则及其他参数不变；0表示关闭这两个局部代价项，不是彻底关闭所有软约束机制。配置在执行前记录，不按测试成绩选参数。

| 两项局部权重 | 加权Q | 平均boundary违反 | 平均grouping违反 | 平均MIB违反 |
|---|---:|---:|---:|---:|
{weight_table}

三组均5/5硬合法、零兜底。原权重2相对0已有收益，说明流程并非完全没有考虑软约束；从2提高到10后，**4例改善、1例持平**，加权Q下降 **{audit['relative_weight10_improvement']*100:.2f}%**，boundary与grouping违反减少。HPWL/bbox也随布局改变，因此不能把全部Q变化归为软罚计数变化。

这给“当前局部权衡仍不足”提供机制证据，但只有5例、单seed、两项权重同时变化；不能据此认定10最优、只需调权重就能解决，也不把对照权重改成生产默认。模型/正式后处理源码未改变。

## 5. 分阶段变化：不是所有软规则都被后处理恶化

下表为各10例逐布局等权均值，原始/修正阶段仍硬非法，grouping可能因重叠而被虚假地连通，不能只用其较低计数宣称更好。

| 图集 | 阶段 | boundary | grouping | MIB | V_rel | 硬合法 |
|---|---|---:|---:|---:|---:|---:|
{stage_table}

后处理把总软违反从约0.60～0.63降到约0.40～0.42，并恢复硬合法，已有明显作用。但离参考仍有差距，同时扩大面积、增加线长。不能把假设写成“后处理完全忽略软约束”，也不能把原始非法解的低Q几何因子当作可行成绩。

## 6. 代码层面可定位的问题

1. `pipeline.py`重新按(y分箱,x,i)排序，i保证键唯一，因此原`default_order`的边界优先与同组连续顺序被覆盖。
2. pack的grouping代价只关心是否接触已放同组块，不统计组件合并收益，也没有规划未来成员；这与官方的全组连通分量不一致。
3. 顶层摆放优先解决top，未统筹角约束和跨层grouping；bbox后续扩张也可能让已放块失去指定边界。
4. 当前确定性分支只有一个候选，最后虽然计算包含软违反的Objective，却没有搜索或重排来改善它。
5. MIB有显式统一尺寸逻辑，但受自由块面积容差和不可移动尺寸约束限制，不能把全部剩余MIB违反归因于权重低。

对于第5点，本次另从已保存公开fixed/preplaced尺寸构造保守下界：同一MIB组中，若若干不可移动尺寸在至少一个维度相差超过3e-4，则即使考虑硬尺寸容差与4位小数计数，也不能全部合并成同形状。30例合计可证明的刚性MIB违反下界为 **{audit['rigid_mib_bound_sum']}**，实际后处理MIB违反总和为 **{audit['observed_mib_violation_sum']}**。该界仅覆盖刚性规格，还没包含自由块面积区间不相交造成的必要违反；不把它当完整最优下界。逐组证据见[analysis_summary.json](analysis_summary.json)。

不显式使用exp本身不是错误：最小化log Q也能正确包含2V/N_soft。实际问题是pack的局部面积/线长代理与2/n指示罚，不能充分表示官方全局连通和bbox边界变化；最终Objective对单候选又没有优化作用。

## 7. 研究判断

**用户假设得到支持：软罚项确实显著拉高现有分数，当前局部软权衡也留下了可改善空间。** 但不能归因成唯一问题：原始提议局部冲突、重新构造造成的几何损失、grouping/边界策略以及部分MIB规格冲突需要分别处理。

下一步更值得研究保留布局关系、显式保护边界与合并组连通分量的修复；权重对照作为参照，不直接替代算法设计，也不通过测试选择新参数。

## 证据与交付

- [30张图的入口](index.html)、[机制诊断图](soft_penalty_diagnostics.png)。
- [results.json](results.json)：957例评分分解、30例分阶段指标、5例开发消融。
- [cases.jsonl](cases.jsonl)、[ablation_cases.jsonl](ablation_cases.jsonl)：新增坐标和必要原始几何引用；不复制之前运行的坐标。
- [config.json](config.json)、[源码快照](source_snapshot.zip)：冻结身份、面板选择、命令、设备、预算、权重和输入哈希。
- [analysis_summary.json](analysis_summary.json)：图数、开发对照胜负与刚性MIB下界。
'''
(run/'analysis.md').write_text(doc,encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='rigid_mib_bounds'},ensure_ascii=False,indent=2))
