"""Combined saved-evidence report for dev100 structure and decoder diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); root = args.run
    parent = json.loads((args.main_run/'results.json').read_text(encoding='utf-8'))
    result = json.loads((root/'results.json').read_text(encoding='utf-8'))
    assert parent['status'] == result['status'] == 'completed'
    assert parent['trial_count'] == 700 and result['trial_count'] == 200
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'cases').glob('*.json'))]
    main_rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.main_run/'cases').glob('*.json'))]
    assert len(rows) == len(main_rows) == 100
    for r in rows: r['by_arm'] = {t['arm']: t for t in r['trials']}
    labels = {'raw_full': '原草图＋完整预算', 'raw_short': '原草图＋短预算',
              'raw_fixed': '固定原草图关系＋短预算', 'teacher_separation': '只给教师分离关系＋短预算',
              'teacher_relations': '给教师分离/接触关系＋短预算', 'decoder_oracle': '完美教师中心/形状经当前解码＋短预算',
              'decoder_oracle_relations': '同上＋教师关系＋短预算', 'teacher_full_ceiling': '完整教师成品回灌（已知答案）'}
    stats = {**parent['summary'], **result['summary']}
    table = ['| 输入/流程 | 平均Q↓ | P95 Q | 合法行排保底 | 平均记账秒 | 质量容差且≤0.5秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, label in labels.items():
        s = stats[arm]
        table.append(f"| {label} | {s['Q_mean']:.6f} | {s['Q_p95']:.6f} | {s['initial_row_retained_count']}/100 | {s['charged_total_seconds_mean']:.3f} | {s['quality_and_05_count']}/100 |")
    free_count = sum(r['decoder_diagnostic']['free_blocks'] for r in rows)
    mismatch_count = sum(r['decoder_diagnostic']['teacher_free_area_offset_gt_1e4_count'] for r in rows)
    max_decoder_error = max(r['decoder_diagnostic']['decoder_free_area_relative_error_max'] for r in rows)
    bad = {a: [r['panel_index'] for r in rows if r['by_arm'][a]['metrics']['cost'] >= 5] for a in result['summary']}
    quality_delta = {a: float(np.mean([r['by_arm'][a]['metrics']['cost']-r['baseline_q'] for r in rows])) for a in result['summary']}
    paired_delta = np.array([r['by_arm']['decoder_oracle_relations']['metrics']['cost']-r['by_arm']['decoder_oracle']['metrics']['cost'] for r in rows])
    audit = {'free_block_count': free_count, 'teacher_area_offset_gt_1e4_count': mismatch_count,
             'teacher_area_offset_gt_1e4_fraction': mismatch_count/max(1, free_count),
             'decoder_area_relative_error_max': max_decoder_error, 'Q_ge_5_workers': bad,
             'decoder_Q_delta_vs_main_live_full': quality_delta,
             'relation_added_to_decoder_Q_delta_mean': float(paired_delta.mean()),
             'relation_added_to_decoder_wins_ties_losses': [int((paired_delta < -1e-9).sum()),
                 int((np.abs(paired_delta) <= 1e-9).sum()), int((paired_delta > 1e-9).sum())]}
    (root/'paired_analysis.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    failure_workers = sorted(set(bad['decoder_oracle']) | set(bad['decoder_oracle_relations']))
    failure_table = ['| worker | 原完整预算Q | 当前解码oracle Q | 再加教师关系Q | 证据 |', '|---|---:|---:|---:|---|']
    for i in failure_workers:
        r = rows[i]
        failure_table.append(f"| {i} | {r['baseline_q']:.6f} | {r['by_arm']['decoder_oracle']['metrics']['cost']:.6f} | {r['by_arm']['decoder_oracle_relations']['metrics']['cost']:.6f} | [轨迹](cases/{i:03d}.json) |")
    quality = ['| 方法 | HPWL gap | bbox gap | 边界违反 | 分组违反 | MIB违反 | 硬合法 |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in result['summary']:
        s = stats[arm]
        quality.append(f"| {labels[arm]} | {s['hpwl_gap_mean']:.6f} | {s['area_gap_mean']:.6f} | {s['boundary_mean']:.3f} | {s['grouping_mean']:.3f} | {s['mib_mean']:.3f} | {s['hard_legal_count']}/100 |")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), layout='constrained')
    plot_arms = ['raw_full', 'teacher_relations', 'decoder_oracle', 'decoder_oracle_relations', 'teacher_full_ceiling']
    english = ['Original, full budget', 'Teacher relations only', 'Perfect current decoder', 'Decoder + teacher relations', 'Complete teacher ceiling']
    colors = ['#777777', '#167bb8', '#b36f22', '#955bb0', '#429467']
    axes[0].barh(np.arange(5), [stats[a]['Q_mean'] for a in plot_arms], color=colors)
    axes[0].set_yticks(np.arange(5), english); axes[0].invert_yaxis()
    axes[0].set(xlabel='Mean neutral-runtime Q, all 100 cases', title='Geometry helps; rare failures still count')
    for i, a in enumerate(plot_arms): axes[0].text(stats[a]['Q_mean']+.01, i, f"{stats[a]['Q_mean']:.4f}", va='center')
    axes[0].set_xlim(0, max(stats[a]['Q_mean'] for a in plot_arms)*1.14)
    for a, label, color in [('decoder_oracle', 'Perfect current decoder', colors[2]),
                            ('decoder_oracle_relations', 'Decoder + teacher relations', colors[3])]:
        delta = np.sort([r['by_arm'][a]['Q_delta_vs_live_full_raw'] for r in rows])
        axes[1].step(delta, np.arange(1, 101)/100, where='post', color=color, label=label)
    axes[1].axvline(.002, color='black', linestyle=':', label='+0.002 tolerance')
    axes[1].set_xscale('symlog', linthresh=.002)
    axes[1].set(xlabel='Q difference from main-run full-budget original', ylabel='Fraction of 100 cases', title='Current decoder: quality distribution, no tail removal')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=.2)
    fig.savefig(root/'decoder_quality.png', dpi=170); plt.close(fig)
    first = result['summary']['decoder_oracle']; guided = result['summary']['decoder_oracle_relations']
    total_legal = sum(s['hard_legal_count'] for s in parent['summary'].values())+sum(s['hard_legal_count'] for s in result['summary'].values())
    total_errors = sum(s['error_count'] for s in parent['summary'].values())+sum(s['error_count'] for s in result['summary'].values())
    text = f'''# E11扩大研究总报告：dev100结构与当前解码器

2026-10-18。在同一固定dev_quality100上完成主结构对照700次与当前解码器对照200次，共900次求解，覆盖全部100个worker；两个阶段计算累计{result['shared_elapsed_seconds']:.2f}秒，未超过预先固定的900秒共同保护。官方及独立硬检查{total_legal}/900合法，求解执行错误{total_errors}；合法但质量很差的保底不从结果删除。没有训练、生成训练集教师库或读取封存测试。

**扩大数据后，五例上“教师关系接近原Q”的结论未能保持。** 原完整预算平均Q为{stats['raw_full']['Q_mean']:.6f}；仅给教师分离/接触关系、保留原坐标和宽高时，0.40秒目标下平均Q为{stats['teacher_relations']['Q_mean']:.6f}，{stats['teacher_relations']['initial_row_retained_count']}/100例保留初始合法行排。当前证据不支持直接进入“只训练关系头”的路线。

把教师中心与长宽比一起给准，并经过真实当前解码器后，大量样例改善；但两组平均Q分别为{first['Q_mean']:.6f}/{guided['Q_mean']:.6f}，仍有{len(bad['decoder_oracle'])}/{len(bad['decoder_oracle_relations'])}例Q≥5。**在该教师投影、当前后处理与短预算组合下，尚未达到维持原Q的目标。** 这既不是已训练模型结果，也不是证明当前输出形式中不存在更好的可学习解。

## 1. 全部100例统一比较

{chr(10).join(table)}

Q为官方中性时间质量分，越低越好，不是比赛分数。质量容差沿用运行前固定的“同例主运行完整预算Q＋0.002”；同时要求硬合法和记账时间≤0.5秒。主阶段与补充阶段依次执行，硬件、CPU两线程、后处理源码不变，时序与系统波动仍可能影响截止前迭代，不能解释为严格同步的噪声消融。

## 2. 当前模型能表达什么

现有decode预测中心与log长宽比r，并用w=√(A·exp(r))、h=√(A/exp(r))恢复宽高，r限制在[-5,5]，自由块面积固定为公开目标A；随后按E10原流程恢复fixed/preplaced规格。教师E10成品则可使用官方允许的面积浮动。

本轮补充oracle直接把既有教师中心与log长宽比作为理想网络输出，交给原冻结prepare_inference/decode，**没有用手写近似替换真实解码器**。100例共有{free_count}个自由块，其中{mismatch_count}个教师块面积与目标面积相差超过0.01%（{mismatch_count/max(1, free_count)*100:.2f}%）；重新解码后的最大自由块面积相对误差约{max_decoder_error:.3g}，符合精确面积输出特征。

因此完整教师回灌与“当前模型完美预测教师中心/形状”不同。投影同时涉及面积、浮点坐标、长宽比范围与非负裁切，尚未做逐项因果消融，不能把所有剩余失败直接归罪于面积限制。这只是从一个已知好解构造的输出投影，也不是当前输出族的最优性证书。

## 3. 剩余尾部完整保留

{chr(10).join(failure_table)}

{chr(10).join(quality)}

补充阶段只给教师中心/形状时，质量容差与0.5秒同时满足{first['quality_and_05_count']}/100例；再加教师关系为{guided['quality_and_05_count']}/100例。追加关系的逐例胜/平/负为{audit['relation_added_to_decoder_wins_ties_losses']}，平均Q差{paired_delta.mean():+.6f}。均值下降与尾部可靠性必须分开看；不能只报告中位数或剔除失败例。

## 4. 研究判断

1. 暂缓直接训练关系头。主100例说明理想化关系仍不足以稳定替代完整后处理，分离关系与接触树的收益也有耦合；完整结构消融见主报告。
2. 教师蒸馏继续值得研究，但应以**教师输出—当前解码—短预算修复**的整体为对象。标签必须与学生可表达的几何和后续修复兼容，仅模仿中心或关系都不足以作为成功代理。
3. 下一步优先查清“接近教师的输入为何仍退回差解”，再验证一个可靠、廉价的几何投影/修复入口。可以分别检查形状归一化、拓扑/接触更新、数值守卫拒绝和截止时保底；保留严格硬合法检查，不能靠放宽合法性换取加速。
4. 上述诊断完善后，再决定精确面积教师、带面积自由度的输出表示或联合中心/形状/关系蒸馏。当前没有新训练或模型改造，不能把这些建议写成已实现改进。

这轮研究从5例扩大为全100例，纠正了原先小面板的乐观解释。100例仍是已参与开发的面板，不能声称封存测试泛化；接近合法输入的具体失败原因尚待最小反例验证，不预设唯一根因。

## 5. 计时、复现与证据

核心和通信在线实测，各组统一加既有模型推理时间。oracle输出生成、结构构造、加载、预热、评分及绘图排除；未来新模型推理时间未知，表中oracle时间不是新模型部署速度。两阶段共用900秒累计保护，主阶段实际{parent['elapsed_seconds']:.2f}秒，补充{result['elapsed_seconds']:.2f}秒。

![当前解码器诊断](decoder_quality.png)

[主100例结构报告](../{args.main_run.name}/analysis.md)、[主原始统计](../{args.main_run.name}/results.json)、[补充配置](config.json)、[补充统计](results.json)、[解码及配对诊断](paired_analysis.json)、[全部补充逐例证据](cases/)、[源码快照](source_snapshot.zip)。本报告及图只从已存证据生成，不重新求解或读取数据张量。
'''
    (root/'analysis.md').write_text(text, encoding='utf-8')
    with zipfile.ZipFile(root/'report_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__), Path(__file__).name)
    (root/'report_provenance.json').write_text(json.dumps({
        'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [args.main_run/'results.json', root/'results.json', *sorted((root/'cases').glob('*.json'))]},
        'dataset_reads': False, 'solver_calls': 0}, indent=2)+'\n', encoding='utf-8')
    print('Saved combined analysis.md, paired_analysis.json, decoder_quality.png; no new solving/data reads')


if __name__ == '__main__':
    main()
