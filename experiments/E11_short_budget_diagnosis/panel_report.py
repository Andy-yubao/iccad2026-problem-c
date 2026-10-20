"""Saved-evidence-only dev100 analysis and scientific plots; no solving."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'raw_full': '原草图完整预算', 'raw_short': '原草图短预算', 'raw_fixed': '固定原分离/接触结构',
         'teacher_separation': '教师分离＋原草图选接触树', 'teacher_contact_tree': '原分离＋教师选接触树',
         'teacher_relations': '教师分离/接触结构', 'teacher_full_ceiling': '完整教师成品上限'}
SHORT = ['raw_short', 'raw_fixed', 'teacher_separation', 'teacher_contact_tree', 'teacher_relations']


def pair_delta(rows, left, right):
    deltas = np.array([r['by_arm'][left]['metrics']['cost']-r['by_arm'][right]['metrics']['cost'] for r in rows])
    rng = np.random.default_rng(20261018)
    bootstrap = deltas[rng.integers(0, len(rows), size=(2000, len(rows)))].mean(axis=1)
    return {'mean': float(deltas.mean()), 'median': float(np.median(deltas)),
            'wins': int((deltas < -1e-9).sum()), 'ties': int((np.abs(deltas) <= 1e-9).sum()),
            'losses': int((deltas > 1e-9).sum()),
            'descriptive_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); root = args.run
    result = json.loads((root/'results.json').read_text(encoding='utf-8'))
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'cases').glob('*.json'))]
    assert result['status'] == 'completed' and result['trial_count'] == 700 and len(rows) == 100
    assert result['completed_case_count'] == 100
    expected = set(config['arms'])
    for row in rows:
        assert len(row['trials']) == 7
        assert {t['arm'] for t in row['trials']} == expected
        row['by_arm'] = {t['arm']: t for t in row['trials']}
    stats = result['summary']; base = stats['raw_full']; teacher = stats['teacher_relations']
    pair_specs = [('teacher_relations', 'raw_fixed'), ('teacher_separation', 'raw_fixed'),
                  ('teacher_contact_tree', 'raw_fixed'), ('teacher_relations', 'teacher_separation'),
                  ('teacher_relations', 'teacher_contact_tree')]
    comparisons = {f'{a}_minus_{b}': pair_delta(rows, a, b) for a, b in pair_specs}
    interactions = np.array([r['by_arm']['teacher_relations']['metrics']['cost']
                            -r['by_arm']['teacher_separation']['metrics']['cost']
                            -r['by_arm']['teacher_contact_tree']['metrics']['cost']
                            +r['by_arm']['raw_fixed']['metrics']['cost'] for r in rows])
    actual_sizes = [r['block_count'] for r in rows]
    old_workers = {0, 24, 49, 74, 99}
    old_rows = [r for r in rows if r['panel_index'] in old_workers]
    new_rows = [r for r in rows if r['panel_index'] not in old_workers]
    group_diagnostics = {}
    for label, group in [('previous_five', old_rows), ('remaining_95', new_rows)]:
        group_diagnostics[label] = {'count': len(group), 'arms': {arm: {
            'Q_mean': float(np.mean([r['by_arm'][arm]['metrics']['cost'] for r in group])),
            'Q_delta_vs_full_mean': float(np.mean([r['by_arm'][arm]['metrics']['cost']-r['baseline_q'] for r in group]))}
            for arm in config['arms']}}
    worst = sorted(rows, key=lambda r: r['by_arm']['teacher_relations']['Q_delta_vs_live_full_raw'], reverse=True)[:10]
    audit = {'scope': 'Descriptive development statistics; all100 and prespecified arms retained.',
             'comparisons': comparisons, 'factorial_interaction_mean_Q': float(interactions.mean()),
             'previous_five_vs_remaining95': group_diagnostics,
             'block_count': {'min': min(actual_sizes), 'median': float(np.median(actual_sizes)), 'max': max(actual_sizes)},
             'cached_teacher_vs_live_full_q_difference_max': max(abs(r['cached_teacher_q']-r['baseline_q']) for r in rows),
             'worst_teacher_relation_cases': [{'case_id': r['case_id'], 'block_count': r['block_count'],
                 'baseline_q': r['baseline_q'], 'Q': r['by_arm']['teacher_relations']['metrics']['cost'],
                 'delta_Q': r['by_arm']['teacher_relations']['Q_delta_vs_live_full_raw'],
                 'charged_seconds': r['by_arm']['teacher_relations']['charged_total_seconds']} for r in worst]}
    (root/'paired_analysis.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    table = ['| 方法 | 平均Q | P95 Q | 最大Q | 合法行排保底 | 记账总秒/例 | ≤Q+0.002 | 质量容差且≤0.5秒 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in config['arms']:
        s = stats[arm]
        table.append(f"| {NAMES[arm]} | {s['Q_mean']:.6f} | {s['Q_p95']:.6f} | {s['Q_max']:.6f} | {s['initial_row_retained_count']}/100 | {s['charged_total_seconds_mean']:.3f} | {s['within_002_count']}/100 | {s['quality_and_05_count']}/100 |")
    pair_table = ['| 新−对照（同0.40秒目标） | 平均Q差↓ | 胜/平/负 | 描述性bootstrap 95%区间 |',
                  '|---|---:|---:|---|']
    for a, b in pair_specs:
        p = comparisons[f'{a}_minus_{b}']; low, high = p['descriptive_bootstrap_95']
        pair_table.append(f"| {NAMES[a]} − {NAMES[b]} | {p['mean']:+.6f} | {p['wins']}/{p['ties']}/{p['losses']} | [{low:+.6f}, {high:+.6f}] |")
    decomposition = ['| 方法 | HPWL gap均值 | bbox gap均值 | 边界违反均值 | 分组违反均值 | MIB违反均值 | 硬合法 |',
                     '|---|---:|---:|---:|---:|---:|---:|']
    timing = ['| 方法 | 核心秒均值 | 通信秒均值 | 总秒P95 | 总秒最大 | 总秒>0.5 | 锥迭代均值 | 原生锥秒均值 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in config['arms']:
        s = stats[arm]
        decomposition.append(f"| {NAMES[arm]} | {s['hpwl_gap_mean']:.6f} | {s['area_gap_mean']:.6f} | {s['boundary_mean']:.3f} | {s['grouping_mean']:.3f} | {s['mib_mean']:.3f} | {s['hard_legal_count']}/100 |")
        timing.append(f"| {NAMES[arm]} | {s['core_seconds_mean']:.3f} | {s['ipc_seconds_mean']:.3f} | {s['charged_total_seconds_p95']:.3f} | {s['charged_total_seconds_max']:.3f} | {s['over_05_seconds_count']}/100 | {s['conic_calls_mean']:.2f} | {s['native_conic_seconds_mean']:.3f} |")
    counterexamples = ['| worker（模块数） | 原完整Q | 教师关系Q | Q差 | 教师关系总秒 | 证据 |',
                       '|---|---:|---:|---:|---:|---|']
    for r in worst:
        t = r['by_arm']['teacher_relations']
        counterexamples.append(f"| {r['panel_index']}（{r['block_count']}） | {r['baseline_q']:.6f} | {t['metrics']['cost']:.6f} | {t['Q_delta_vs_live_full_raw']:+.6f} | {t['charged_total_seconds']:.3f} | [逐例](cases/{r['panel_index']:03d}.json) |")
    group_table = ['| 本轮子集（仅事后诊断） | 例数 | 原完整Q | 教师关系Q | Q差 |', '|---|---:|---:|---:|---:|']
    for label, title in [('previous_five', '此前5例重新在线运行'), ('remaining_95', '本实验新增95例')]:
        g = group_diagnostics[label]
        group_table.append(f"| {title} | {g['count']} | {g['arms']['raw_full']['Q_mean']:.6f} | {g['arms']['teacher_relations']['Q_mean']:.6f} | {g['arms']['teacher_relations']['Q_delta_vs_full_mean']:+.6f} |")

    colors = dict(zip(config['arms'], ['#777777', '#b4b4b4', '#e58130', '#8b63b5', '#cf6887', '#167bb8', '#429467']))
    english = {'raw_full': 'Original, full budget', 'raw_short': 'Original, short budget', 'raw_fixed': 'Fixed original relations',
               'teacher_separation': 'Teacher separation only', 'teacher_contact_tree': 'Teacher contact tree only',
               'teacher_relations': 'Teacher separation + contact', 'teacher_full_ceiling': 'Complete teacher (ceiling)'}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    arms = list(config['arms'])
    ax[0, 0].barh(np.arange(7), [stats[a]['Q_mean'] for a in arms], color=[colors[a] for a in arms])
    ax[0, 0].set_yticks(np.arange(7), [english[a] for a in arms], fontsize=9)
    ax[0, 0].invert_yaxis(); ax[0, 0].set(xlabel='Mean neutral-runtime Q (all 100)', title='Full panel; lower Q is better')
    for i, a in enumerate(arms): ax[0, 0].text(stats[a]['Q_mean']+.01, i, f"{stats[a]['Q_mean']:.4f}", va='center', fontsize=9)
    ax[0, 0].set_xlim(0, max(stats[a]['Q_mean'] for a in arms)*1.14)
    for a in SHORT:
        delta = np.sort([r['by_arm'][a]['Q_delta_vs_live_full_raw'] for r in rows])
        ax[0, 1].step(delta, np.arange(1, 101)/100, where='post', color=colors[a], label=english[a])
    ax[0, 1].axvline(.002, color='black', linestyle=':', alpha=.7)
    ax[0, 1].set_xscale('symlog', linthresh=.002)
    ax[0, 1].set(xlabel='Q minus same-case full-budget original', ylabel='Fraction of 100 cases', title='Short-budget quality distribution (all tails retained)')
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.2)
    for i, a in enumerate(arms):
        values = [r['by_arm'][a]['charged_total_seconds'] for r in rows]
        ax[1, 0].scatter(np.full(100, i)+np.linspace(-.15, .15, 100), values, s=9, alpha=.5, color=colors[a])
        ax[1, 0].plot([i-.2, i+.2], [np.mean(values)]*2, color='black', linewidth=2)
    ax[1, 0].axhline(.5, color='red', linestyle=':', label='0.5 s')
    ax[1, 0].set_xticks(np.arange(7), ['Full', 'Short', 'Fixed raw', 'T sep', 'T contact', 'T both', 'Ceiling'], rotation=20)
    ax[1, 0].set(ylabel='Worker + historical inference seconds', title='Live worker timing; oracle network time unknown')
    ax[1, 0].grid(axis='y', alpha=.2)
    ax[1, 1].scatter(actual_sizes, [r['by_arm']['teacher_relations']['Q_delta_vs_live_full_raw'] for r in rows],
                     color=colors['teacher_relations'], alpha=.75, s=25)
    ax[1, 1].axhline(.002, color='black', linestyle=':'); ax[1, 1].axhline(0, color='black', linewidth=.5)
    ax[1, 1].set_yscale('symlog', linthresh=.002)
    ax[1, 1].set(xlabel='Actual module count', ylabel='Teacher-relations Q minus full original Q', title='Module count and remaining quality loss')
    ax[1, 1].grid(alpha=.2)
    fig.savefig(root/'dev100_quality_time.png', dpi=170); plt.close(fig)

    gap = teacher['Q_mean']-base['Q_mean']
    teacher_effect = comparisons['teacher_relations_minus_raw_fixed']['mean']
    speedup = base['charged_total_seconds_mean']/teacher['charged_total_seconds_mean']
    decision = '均值质量容差与均值时间条件通过，但仅为理想化关系输入诊断' if gap <= .002 and teacher['charged_total_seconds_mean'] <= .5 else '尚未达到0.5秒内维持原Q'
    teacher_sentence = ('教师关系相对固定原关系的平均Q改善，支持进一步研究结构信号。'
                        if teacher_effect < -.002 else '教师关系相对固定原关系的平均Q收益有限，当前不支持仅凭结构oracle追加训练。')
    alllegal = sum(s['hard_legal_count'] for s in stats.values())
    errors = sum(s['error_count'] for s in stats.values())
    text = f'''# E11 dev100：扩大样本与分离/接触结构消融

2026-10-18。用户要求扩大数据量后，在查看扩展结果前固定完整workset_v2.dev_quality100与7组对照。实际100/100例、700/700次求解完成，累计墙钟{result['elapsed_seconds']:.2f}秒；官方与独立硬检查{alllegal}/700合法，求解执行错误{errors}。实际模块数范围{min(actual_sizes)}–{max(actual_sizes)}，中位数{np.median(actual_sizes):.1f}；worker编号不等于模块数。

**结论：{decision}。** 教师分离/接触关系组平均Q={teacher['Q_mean']:.6f}，同例在线完整基线={base['Q_mean']:.6f}，Q差{gap:+.6f}（{gap/base['Q_mean']*100:+.3f}%）；平均记账总时间{teacher['charged_total_seconds_mean']:.3f}秒，约{speedup:.2f}倍加速。逐例在预定Q＋0.002容差内{teacher['within_002_count']}/100，兼顾容差与0.5秒{teacher['quality_and_05_count']}/100。{teacher_sentence}

## 1. 全面板结果与尾部

{chr(10).join(table)}

Q越低越好，为官方中性时间质量对照，不是比赛分数。合法但质量封顶的行排保底照常计入，不以硬合法替代质量。初始行排保留由“无被接纳的代理改善事件”识别；完整成品组若已被接受则不会被误认为未改善保底。最大Q和最差10%均值等完整统计见results.json。

## 2. 同预算分离/接触消融

四组固定关系保持原草图坐标与宽高，分离边来源为raw或teacher；接触树以选定分离方向为基础，用raw或teacher几何计算边权和选树。只把关系三元组交给worker，完整成品上限组除外。接触树与分离方向存在条件依赖，不能将该条件化2×2对照描述成互不影响的两个独立开关。

{chr(10).join(pair_table)}

条件化交互Q(T,T)−Q(T,R)−Q(R,T)＋Q(R,R)的均值为{interactions.mean():+.6f}；负值表示本次组合在Q尺度上的额外收益。Q是非线性分数，交互与单项均值差不构成损失组成的严格加和解释。bootstrap按100个已观测样例重采样，只是描述性开发面板区间，不能排除开发选择、生成家族相关性、计时波动或多重比较影响。

## 3. 五例结论的复核与反例

{chr(10).join(group_table)}

本轮对全部100例重新在线运行；上述5/95拆分只是按既有成员做的事后稳定性诊断，不作为两个独立测试，不根据胜负改变主面板。缓存教师Q与在线完整基线Q的最大绝对差为{audit['cached_teacher_vs_live_full_q_difference_max']:.6f}；教师仍来自事前冻结的旧E10结果，没有在扩展结果上重新择优教师。

教师关系相对完整预算退化最大的10例全部保留：

{chr(10).join(counterexamples)}

这些样例用于解释剩余失效模式，不能从主指标删除；后续若根据其诊断改动，需要在新的明确运行中记录方案变化。

## 4. 官方质量分解与计时

{chr(10).join(decomposition)}

{chr(10).join(timing)}

核心和通信均在线实测，原模型推理时间沿用原草图记录，对各组统一记账。数据/模型加载、预热、oracle结构构造、评分和绘图不计入稳态时间；未来预测关系的网络耗时未知，教师关系时延不是已部署模型速度。各预算是目标，非硬实时上界，超预算计数和时间尾部照实保留。原生锥秒不含其余建模、图处理、守卫和投影工作，不据此直接把剩余时间归为Python成本。

## 5. 范围与后续边界

本轮从5例扩大到100例，覆盖全部100个worker，每worker1例，包含既有5例。该面板已用于E8/E9开发；扩大覆盖不等于独立未见泛化证明。没有筛选兼容面板、读取封存local_test、改变冻结划分、训练或生成训练集教师库。原中心oracle组未扩测，本报告不能宣称100例证实中心蒸馏无效。

主对照严格使用900秒累计保护、CPU两线程、seed0逐例随机7组顺序；后处理源码冻结，官方和独立硬守卫保持不变。本报告从cases下唯一坐标池和已存metrics构建，未重新求解。是否进入关系蒸馏训练，须根据本轮尾部、单项贡献及反例另定实验；当前不自动启动长训练。

![全100例质量与时间](dev100_quality_time.png)

证据：[配置与固定成员](config.json)、[调用与累计预算](invocations.json)、[全部统计](results.json)、[配对分析与反例](paired_analysis.json)、[逐例坐标与轨迹](cases/)、[源码快照](source_snapshot.zip)。
'''
    (root/'analysis.md').write_text(text, encoding='utf-8')
    (root/'report_provenance.json').write_text(json.dumps({
        'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [root/'results.json', root/'config.json', *sorted((root/'cases').glob('*.json'))]},
        'dataset_reads': False, 'solver_calls': 0}, indent=2)+'\n', encoding='utf-8')
    print('Saved analysis.md, paired_analysis.json and dev100_quality_time.png; no solving/data reads')


if __name__ == '__main__':
    main()
