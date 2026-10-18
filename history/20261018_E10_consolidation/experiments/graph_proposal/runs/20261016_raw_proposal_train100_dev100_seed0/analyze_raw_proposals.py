"""Summarize saved stage-A diagnostics only: no inference or data loading."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

run=Path(__file__).resolve().parent
sys.path.insert(0,str(run.parents[3]))
result=json.loads((run/'results.json').read_text(encoding='utf-8'))
config=json.loads((run/'config.json').read_text(encoding='utf-8'))
rows=[json.loads(line) for line in (run/'cases.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
assert result['complete'] and result['completed_cases']==len(rows)==200
assert len({row['case_id'] for row in rows})==200
summary=result['summary']
def metric(panel,variant,key):return summary[panel][variant]['diagnostics'][key]['mean']
def percent(value):return f'{value*100:.2f}%'
def value(panel,key):return metric(panel,'deterministic',key)

audit={'complete':True,'cases_sha256':hashlib.sha256((run/'cases.jsonl').read_bytes()).hexdigest(),
       'panels':{},'checkpoint_sha256':config['checkpoint_sha256'],'checkpoint_unchanged':result['checkpoint_unchanged'],
       'test_payload_reads':0,'postprocessing_calls':0,'training_updates':0}
for panel in ('train','dev'):
    selected=[r for r in rows if r['panel']==panel]
    assert len(selected)==100
    assert len({r['source']['shard'].split('/')[0] for r in selected})==100
    for row in selected:
        for variant in ('reference','heuristic','deterministic'):
            assert row['variants'][variant]['official_metrics']['overlap_violations']==row['variants'][variant]['diagnostics']['overlap_pair_count']
    current=summary[panel]['deterministic']
    bounds=[r['variants']['deterministic']['diagnostics']['translation_radius_lb'] for r in selected]
    audit['panels'][panel]={'count':100,'workers':100,'supervised':current['supervised'],
        'translation_lb_min':min(bounds),'translation_lb_max':max(bounds),
        'translation_lb_exceeds_5pct':current['translation_lb_exceeds_0.05'],
        'translation_lb_exceeds_10pct':current['translation_lb_exceeds_0.1'],
        'translation_lb_median':current['diagnostics']['translation_radius_lb']['median'],
        'network_and_decode_seconds_mean':float(np.mean([r['timings']['network_and_decode_seconds'] for r in selected])),
        'same_official_overlap_count_verified':True,
        'group_contact_recall_valid_layouts':current['diagnostics']['group_reference_contact_within_5pct_recall']['valid_layouts']}
(run/'diagnostic_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

table_specs=[('中心位移/自身尺度，逐布局均值','center_error_local_mean'),
             ('块对相对位置误差/局部尺度','pair_relative_local_mean'),
             ('连线加权相对位置误差/局部尺度','edge_relative_local_weighted_mean'),
             ('中心方向保留率','direction_any_center_order_retained_fraction'),
             ('至少一个参考分离方向仍成立','direction_any_separation_retained_fraction'),
             ('涉及重叠的块比例','overlap_block_fraction'),
             ('逐对重叠面积和/总块面积','pair_overlap_area_fraction'),
             ('自由块宽高相对误差，逐坐标绝对值均值','shape_wh_relative_mean_free'),
             ('参考grouping接边在5%局部尺度内','group_reference_contact_within_5pct_recall'),
             ('中心散布半径/参考散布半径','center_spread_ratio')]
main_table='\n'.join(f'| {label} | {percent(value("train",key))} | {percent(value("dev",key))} |' for label,key in table_specs)
comparison='\n'.join(f'| {panel} | {variant} | {metric(panel,variant,"center_error_local_mean"):.4f} | {metric(panel,variant,"pair_relative_local_mean"):.4f} | {percent(metric(panel,variant,"pair_overlap_area_fraction"))} | {percent(metric(panel,variant,"direction_any_separation_retained_fraction"))} | {metric(panel,variant,"shape_log_ratio_rmse_free"):.4f} |' for panel in ('train','dev') for variant in ('heuristic','deterministic'))
size_lines=[]
for panel in ('train','dev'):
    for name,groups in result['size_groups'][panel].items():
        s=groups['deterministic'];d=s['diagnostics']
        size_lines.append(f'| {panel} | {name} | {s["case_count"]} | {d["center_error_local_mean"]["mean"]:.3f} | {percent(d["pair_overlap_area_fraction"]["mean"])} | {percent(d["translation_radius_lb"]["median"])} |')
shape_train=summary['train']['deterministic']['supervised']['mse']
shape_dev=summary['dev']['deterministic']['supervised']['mse']
doc=f'''# 阶段A：冻结确定性模型的原始提议评价

2026-10-16。用户要求先执行模型独立评价。本阶段完成 **train100 + dev100**，每个集合均覆盖100个worker，模型为v2 best step7,250；标准解码后立即测量，**未调用correct、pack、搜索或兜底，未更新权重，未读取test**。GPU为{config['gpu']}，CPU1用于几何与官方评分；实际运行{result['wall_seconds']:.2f}秒，预算600秒。原checkpoint哈希复核一致。

## 1. 结论

**模型学到了粗略全局结构，在多项几何指标上明显优于当前启发式原始提议；但固定预测形状时，无法仅靠极小平移消除冲突。** 允许位置与小幅形变联合修复的效果待后续验证。问题不能只归因于旧后处理，也不能由本次结果直接归因于架构容量。

dev上中心方向保留约99.34%，参考允许分离方向保留约94.28%，中心散布半径约为参考的98.72%，原始HPWL接近参考。这支持模型有全局布局信息，并非整体坍缩成一团。

但dev上平均中心偏差为每块自身尺度的65.67%，自由块宽高相对误差约19.21%，94.62%的块涉及重叠。当前提议局部误差明显，固定预测形状时不能仅靠几个百分点平移消除全部冲突；允许小幅形变的联合修复仍待阶段B检验。

固定预测形状时，200/200例的必要平移下界超过10%局部尺度；dev中位下界为{percent(audit['panels']['dev']['translation_lb_median'])}，最小也为{percent(audit['panels']['dev']['translation_lb_min'])}。因此能直接排除本批提议通过“每块平移不超过自身尺度5%或10%”恢复全部无重叠的设想。这个结论**只适用于固定形状**，不是实际修复位移、不是所有块都需移动这么多，也不排除允许形变或大改动后的修复。

## 2. 面板、信息边界与统计口径

- train：在workset_v2的train_monitor每worker5例中，以SHA256('proposal_a_train100_v1:<shard>:<row>')最小值预选一例，共100例。不是挑低误差样例。
- dev：原样使用固定dev_quality100，每worker一例。索引、顺序与权重在config中记录，split_v1/workset_v2未修改。
- 参照：参考布局近似最优；无训练启发式与冻结确定性模型使用相同输入遮罩、相同评分和诊断定义。参考自由块几何只在评价端使用。
- 所有主统计逐布局等权，不将大布局O(n²)块对数直接混入同一个池。主表中的P95若出现，是逐布局P95再汇总，非全部模块混池P95。
- 局部尺度s_i=sqrt(effective_area_i)，固定块按公开尺寸面积；布局尺度S=sqrt(sum effective_area)，块对尺度为(s_i+s_j)/2。“65.67%”指位移/s_i，不是65.67%的块或芯片尺寸。
- 未做旋转、反射、重缩放或ID重排。移除整体平移的附加诊断不改变实际端口/预放置约束与原坐标评分。
- 这是固定200例的阶段研究，不等同完整20,000/2,500监督评价，更不是独立测试泛化结论。

## 3. 冻结模型的核心结果

| 指标 | train100 | dev100 |
|---|---:|---:|
{main_table}

两集合原始模型严格硬合法率均0/100，参考均100/100；Q的硬罚10仅作记录，不用它抹平提议差异。监督MSE为train {shape_train:.6f}、dev {shape_dev:.6f}，仍没有明显的train/dev误差分叉；它们是本次固定100例面板数值，不能覆盖或替代先前完整集合结果。

注意：94.28%的块对分离保持率受大量相距较远的块对影响，不能解释成94.28%的块已经合法。实际94.62%的块仍参与至少一个重叠。中心左右/上下符号保留尤其宽松，不能单独作为结构优秀结论。

grouping参考接边5%邻域召回，在dev上为{percent(value('dev','group_reference_contact_within_5pct_recall'))}，仅对存在参考组内接边的布局计算，共{audit['panels']['dev']['group_contact_recall_valid_layouts']}例有效。严格接边诊断使用1e-6几何容差，非Shapely精确拓扑等价。不同连接树也可能是好解，因此不要求复制每条参考接边；这里只是局部结构诊断，与严格grouping违反、重叠严重度联合解释。

## 4. 与无训练启发式相比，模型学到了什么

下表中心和块对误差以局部尺度为单位，log比例误差为自由块log(w/h) RMSE。

| 集合 | 原始提议 | 中心误差 | 块对误差 | 逐对重叠面积比例 | 参考分离方向保留 | log比例RMSE |
|---|---|---:|---:|---:|---:|---:|
{comparison}

dev块对相对误差由启发式3.273降至模型0.820，中心误差由2.876降至0.657；逐对重叠面积比例由406.83%降至28.86%。因此不能将当前模型判成“没有学到布局”。但形状log比例RMSE仅由0.520降至0.474，形状学习的改善明显有限。这是单checkpoint、单seed、固定面板的观察，没有据此改训练配置。

启发式逐对重叠面积超过100%并非计算错误：该指标把每一重叠对的交集相加，多重覆盖重复计数，不是几何并集面积。下界取最坏冲突对，与整体重叠面积衡量不同，不能只靠下界高低给两方案全面排序。

## 5. 为什么原始HPWL和bbox看着接近，仍不够好

dev原始模型平均HPWL gap={summary['dev']['deterministic']['official']['hpwl_gap_mean']:+.4%}，bbox gap={summary['dev']['deterministic']['official']['area_gap_mean']:+.4%}。这些来自非法布局，不能称作超过参考或接近可提交成绩。局部相互穿透会缩短部分连线，也同时留下空洞；两个整体标量看不到具体冲突。

dev平均boundary/grouping/MIB严格违反分别为{summary['dev']['deterministic']['official']['boundary_violations_mean']:.2f}/{summary['dev']['deterministic']['official']['grouping_violations_mean']:.2f}/{summary['dev']['deterministic']['official']['mib_violations_mean']:.2f}。重叠可能让官方grouping连通和MST间隙显得更好，因此不把它们脱离重叠单独宣传。

标准decode还包含XY非负截断与公开规格覆盖。dev中解码前会触发负坐标截断的可移动块比例平均{percent(value('dev','pre_decode_negative_mobile_fraction'))}；log比例截断比例为0。这份报告评价的是**标准解码后的原始提议**，不是完全未变换的head输出；没有把这些解码操作说成额外合法化。

## 6. 必要位移下界与定义检查

对真正重叠的一对块，固定预测宽高，设中心差为dx,dy：

    p_ij = min((w_i+w_j)/2-|dx|, (h_i+h_j)/2-|dy|) - 1e-6
    LB = max_over_overlaps max(p_ij,0)/(s_i_movable+s_j_movable)

preplaced的可动尺度为0，其余为s_i。若要求每块Euclidean平移≤tau*s_i，必有tau≥LB。半尺寸和公式处理了包含情形，不能用交集宽度代替分离距离。分母为0的锁定冲突另记。本批模型没有此类锁定冲突；几何重叠计数已逐例核对与官方评分一致。

| 必要下界 | train100 | dev100 |
|---|---:|---:|
| 最小值 | {percent(audit['panels']['train']['translation_lb_min'])} | {percent(audit['panels']['dev']['translation_lb_min'])} |
| 中位数 | {percent(audit['panels']['train']['translation_lb_median'])} | {percent(audit['panels']['dev']['translation_lb_median'])} |
| 超过5%的布局 | 100/100 | 100/100 |
| 超过10%的布局 | 100/100 | 100/100 |

这里的中位数是“每个布局必须允许的最大单块归一化位移”的必要下界中位数，**不是全体模块平均要移动约一半尺寸**。LB低不能证明可修，本阶段没有运行修复优化器，更没有证明最小可达Q。

五个合成定义检查均通过：边接触、完全重合、矩形包含、整体平移、仅角接触。它们仅验证指标定义，没有用训练数据反复调阈值。

## 7. 模块规模与图像

| 集合 | 模块规模分组 | 布局数 | 平均中心/局部尺度 | 逐对重叠面积比例 | 下界中位数 |
|---|---|---:|---:|---:|---:|
{chr(10).join(size_lines)}

[指标分布图](proposal_diagnostics.png)展示train/dev及启发式对照；[固定五例叠加图](proposal_overlays.png)按dev worker0/24/49/74/99预选；[下界最低/中位/最高例](proposal_overlays_by_translation_lb.png)是诊断极值，不代表总体均值。均为同ID、每例同坐标尺度，箭头从参考中心指向预测中心，红色标注重叠；图像选样与可视化容差见[figure_selection.json](figure_selection.json)。

## 8. 对下一阶段的影响

当前证据支持保留确定性路线：网络有全局结构学习价值。但不能假定“当前模型已输出理想结构，只换个轻量修复就够了”。局部位置、形状和接触精度都需要面对。

阶段B仍应先用参考与受控扰动校准真正保留式修复器，保证不会损伤好输入；同时明确，真实模型输出的固定形状位置修复可能需要较大移动，不应按结果放宽“微调”定义。之后再用对照决定形状/关系监督、表示或容量该改哪一项。

本次仅完成阶段A，没有进入B，没有新训练或新测试。不能把位置误差归咎于旧pack，因为本次它没有运行；也不能把本次观察直接归因于容量不足，因为未改变容量或训练目标。

## 证据与复现

- [config.json](config.json)：预先冻结的面板、命令、模型/源码哈希、设备、预算与指标口径。
- [cases.jsonl](cases.jsonl)：200例唯一完整几何与逐例证据，SHA-256 `{audit['cases_sha256']}`。
- [results.json](results.json)：完整汇总、规模分组和五项定义检查；[diagnostic_summary.json](diagnostic_summary.json)保存覆盖与下界复核。
- [source_snapshot.zip](source_snapshot.zip)：评价启动前源码。生产模型、评分和数据处理未修改；评价后仅调整绘图排版（单行图留白、非负横轴），原始测量记录保持不变。
- [评价入口](../../evaluate_raw_proposals.py)、[几何指标](../../proposal_metrics.py)、[只读绘图入口](../../plot_raw_proposals.py)。

复现采用config中的完整命令并使用新的输出目录；报告/绘图重生成只需读取已有JSON，不需要重新推理。统计不选择checkpoint，不触碰已打开或剩余test成员。
'''
(run/'analysis.md').write_text(doc,encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
from experiments.graph_proposal.plot_raw_proposals import plot_overlays
ranked=sorted([row for row in rows if row['panel']=='dev'],key=lambda row:
    (row['variants']['deterministic']['diagnostics']['translation_radius_lb'],row['case_id']))
example=ranked[(len(ranked)-1)//2]
plot_overlays([example],run/'proposal_example_median_lb.png','Dev illustration: prespecified lower-median translation-bound rank')
print(json.dumps(audit,ensure_ascii=False,indent=2))
