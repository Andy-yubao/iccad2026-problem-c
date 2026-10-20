"""Rebuild the report/figure from saved E11 evidence only. No solving/data read."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'raw': 'Original draft', 'reference_center': 'Reference centers (oracle)',
          'teacher_center': 'Teacher centers (oracle)', 'teacher_relations': 'Teacher relations (oracle)',
          'teacher_full': 'Complete teacher layout (ceiling)', 'raw_relations': 'Frozen original relations'}
CN = {'raw': '原草图，动态关系', 'reference_center': '参考中心＋原宽高（诊断）',
      'teacher_center': '教师中心＋原宽高（诊断）', 'teacher_relations': '教师关系＋原草图（诊断）',
      'teacher_full': '完整教师成品（上限校验）', 'raw_relations': '原草图自身关系固定（归因对照）'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--control-run', type=Path, required=True)
    args = parser.parse_args()
    root = args.main_run
    main_result = json.loads((root/'results.json').read_text(encoding='utf-8'))
    control = json.loads((args.control_run/'results.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (root/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    controls = {r['case_id']: r for r in map(json.loads, (args.control_run/'cases.jsonl').read_text(encoding='utf-8').splitlines())}
    assert main_result['status'] == control['status'] == 'completed'
    assert main_result['trial_count'] == 100 and control['trial_count'] == 10
    assert len(rows) == len(controls) == 5
    s = main_result['summary']
    base = s['raw']['5.0']; relation = s['teacher_relations']['0.4']; cache = control['summary']['0.4']
    cost_difference = relation['Q_mean'] - base['Q_mean']
    relative = cost_difference/base['Q_mean']*100
    speedup = base['charged_total_seconds_mean']/relation['charged_total_seconds_mean']
    short = ['raw', 'raw_relations', 'reference_center', 'teacher_center', 'teacher_relations', 'teacher_full']
    table = ['| 输入/关系 | 平均Q | 最大Q | 核心秒/例 | 通信＋历史推理记账后秒/例 | 硬合法 | Q≤原完整预算＋0.002 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in short:
        r = cache if name == 'raw_relations' else s[name]['0.4']
        table.append(f"| {CN[name]} | {r['Q_mean']:.6f} | {r['Q_max']:.6f} | {r['core_seconds_mean']:.3f} | {r['charged_total_seconds_mean']:.3f} | {r['hard_legal_count']}/5 | {r['within_002_count']}/5 |")
    pair_table = ['| worker（模块数） | 原完整预算Q | 固定原关系Q | 教师关系Q | 教师关系Q差 | 教师关系记账总秒 |',
                  '|---|---:|---:|---:|---:|---:|']
    deltas_raw = []; deltas_teacher = []; teacher_wins = 0
    for row in rows:
        t = next(t for t in row['trials'] if t['variant'] == 'teacher_relations' and t['budget_seconds'] == .4)
        c = next(t for t in controls[row['case_id']]['trials'] if t['budget_seconds'] == .4)
        worker = row['case_id'].split('/')[0].split('_')[-1]
        pair_table.append(f"| {worker}（{row['block_count']}） | {row['baseline_q']:.6f} | {c['metrics']['cost']:.6f} | {t['metrics']['cost']:.6f} | {t['Q_delta_vs_live_full_raw']:+.6f} | {t['charged_total_seconds']:.3f} |")
        deltas_raw.append(c['metrics']['cost']-row['baseline_q'])
        deltas_teacher.append(t['Q_delta_vs_live_full_raw'])
        teacher_wins += t['metrics']['cost'] < c['metrics']['cost']
    full_table = ['| 输入/关系 | 完整预算平均Q | 短预算平均Q |', '|---|---:|---:|']
    for name in short:
        r = control['summary'] if name == 'raw_relations' else s[name]
        full_table.append(f"| {CN[name]} | {r['5.0']['Q_mean']:.6f} | {r['0.4']['Q_mean']:.6f} |")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), layout='constrained')
    colors = {'raw': '#777777', 'reference_center': '#d49a20', 'teacher_center': '#b55bad',
              'teacher_relations': '#167bb8', 'teacher_full': '#3b9b5f', 'raw_relations': '#e57135'}
    for name in ['raw', 'reference_center', 'teacher_center', 'raw_relations', 'teacher_relations', 'teacher_full']:
        stats = control['summary'] if name == 'raw_relations' else s[name]
        data = [stats[k] for k in ['0.1', '0.25', '0.4', '5.0'] if k in stats]
        axes[0].plot([r['charged_total_seconds_mean'] for r in data], [r['Q_mean'] for r in data],
                     marker='o', color=colors[name], linestyle='--' if name == 'teacher_full' else '-',
                     linewidth=1.5, label=LABELS[name])
    axes[0].axvline(.5, color='black', alpha=.5, linestyle=':')
    axes[0].axhline(base['Q_mean'], color='black', alpha=.4, linestyle='--')
    axes[0].set(xscale='log', yscale='log', xlabel='Mean seconds: live worker + historical inference',
                ylabel='Mean neutral-runtime Q (lower is better)', title='Five fixed dev cases: budget curves')
    axes[0].set_yticks([1.15, 2, 5, 10], ['1.15', '2', '5', '10'])
    axes[0].grid(alpha=.2, which='both')
    axes[0].legend(fontsize=7.5, loc='upper right')
    x = np.arange(5); width = .36
    axes[1].bar(x-width/2, deltas_raw, width, color=colors['raw_relations'], label='Frozen original relations')
    axes[1].bar(x+width/2, deltas_teacher, width, color=colors['teacher_relations'], label='Teacher relations (oracle)')
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].axhline(.002, color='black', linestyle=':', linewidth=1, label='Exploratory +0.002 tolerance')
    axes[1].set_xticks(x, ['worker 0', 'worker 24', 'worker 49', 'worker 74', 'worker 99'])
    axes[1].set(ylabel='Q difference from full-budget original draft', title='0.40 s target: same caching, different relations')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=.2)
    fig.savefig(root/'budget_quality.png', dpi=170)
    plt.close(fig)

    native_fraction = base['native_conic_seconds_mean']/base['core_seconds_mean']*100
    overall_wall = main_result['elapsed_seconds']+control['elapsed_seconds']
    text = f'''# E11：短预算机制诊断结论

2026-10-18。固定dev_quality中worker 0/24/49/74/99的5例，分别97/98/66/98/94个模块；主对照100次、必要归因10次，共110次求解，累计研究墙钟{overall_wall:.2f}秒。全部110个输出经官方与独立检查均硬合法，零求解执行错误；Q封顶的合法行排保底照常计入均值。本报告只读取已存证据，不重算布局。

**有支持关系蒸馏的机制信号，但尚未实现“0.5秒内维持当前Q”。** 原完整预算平均Q为{base['Q_mean']:.6f}；教师关系＋原草图在0.40秒目标下为{relation['Q_mean']:.6f}，平均Q上升{cost_difference:.6f}（约{relative:.3f}%），记账总时间{relation['charged_total_seconds_mean']:.3f}秒，相对同5例完整预算约{speedup:.2f}倍。仅{relation['within_002_count']}/5例在预先固定的Q＋0.002容差内，严格不退化{relation['strict_no_worse_count']}/5例，不能用小幅平均退化宣称逐例质量保持。

## 1. 0.40秒目标的主结果

原完整预算指普通路径1.3秒、仅无改善时救援、整例目标5秒；本5例普通路径均完成，实际平均核心{base['core_seconds_mean']:.3f}秒，加通信与历史推理记账为{base['charged_total_seconds_mean']:.3f}秒。其Q与既有教师成品逐例一致。

{chr(10).join(table)}

原草图在0.40秒目标下，worker99保留Q封顶的合法行排，其余4例也都未达到同例完整预算质量。完整教师回灌在0.10秒档就得到平均Q={s['teacher_full']['0.1']['Q_mean']:.6f}、核心{s['teacher_full']['0.1']['core_seconds_mean']:.3f}秒；该档0次锥迭代，主要检查并接受已知答案。它仅表明合法成品的保留/检查无需约1秒，不能作为模型可达到该精度或真实推理速度的证据。0.10秒时非完整成品组也0次锥迭代：现有循环预留安全时间后不启动优化，不能据此声称锥优化数学上无法在0.10秒内工作。

## 2. 区分教师信息与固定关系的收益

teacher_relations仍从原草图的中心与宽高出发，只获得既有E10成品的全对左右/上下分离关系及组内接触树；不输入教师坐标、尺寸、参考指标。关系在循环中固定，LP投影也采用同一套关系。teacher_center仅替换中心，保留原宽高与公开锁定规格。reference_center是单独标记的标签诊断。

主对照结束后识别出缓存混杂，先固定control.py说明，再在同5例补做10次对照：把原草图自身的分离关系/接触树同样预先计算并固定。因此教师关系与固定原关系具有相同的计算复用机制，区别在关系来源。教师关系在短预算下{teacher_wins}/5例Q均更好，均值从{cache['Q_mean']:.6f}降到{relation['Q_mean']:.6f}；两者核心时间{cache['core_seconds_mean']:.3f}/{relation['core_seconds_mean']:.3f}秒，平均锥迭代{cache['conic_calls_mean']:.1f}/{relation['conic_calls_mean']:.1f}次。

{chr(10).join(pair_table)}

这支持“教师结构信息有作用”，但仍不能把分离关系和接触树的贡献拆开，也不能把固定关系的收益全部归因于缓存省时：它同时改变了逐步更新拓扑的策略。原关系固定也显著改善短预算结果，却在完整预算平均Q={control['summary']['5.0']['Q_mean']:.6f}时劣于原动态关系的{base['Q_mean']:.6f}，说明固定错误结构可能限制后续优化空间。

## 3. 中心精度不等于短预算可修复性

{chr(10).join(full_table)}

即使中心来自教师成品，保留原草图宽高后仍可能破坏整体几何兼容性。在0.40秒档，teacher_center的worker24/99没有被严格输出守卫接纳的改善候选，保留合法但Q封顶的行排。reference_center的worker99也如此。日志包含超时和重叠导致的候选拒绝；部分诊断在规范化前采集，不能据此逐项精确解释规范化后的所有拒绝原因。本实验不修改硬守卫来改善数字。

完整预算下中心替换仍能改善平均Q，表明“同预算最终质量提高”与“短预算恢复质量”是两个不同问题。此结果反对优先继续优化中心误差作为加速代理；不证明所有坐标蒸馏都无效。

## 4. 计时与证据边界

- 全部为CPU两线程，模型草图和教师成品复用既有记录，不加载模型、训练或读取封存测试。核心/通信在线实测；约0.017秒/例的原推理时间沿用来源并统一记账。所有oracle构造、数据/模型加载、预热、评分、绘图不计入表内稳态时间。未来关系预测网络的实际推理时间未知，0.379秒不能称为可部署新模型端到端时间。
- 预算是目标，不是硬实时上界。教师关系0.40秒档最大记账时间{relation['charged_total_seconds_max']:.3f}秒；所有超预算计数和原始时间保留在results/cases中。进程通信使用同步持久worker，保留JSON序列化成本。
- 原完整预算中原生锥求解约{base['native_conic_seconds_mean']:.3f}秒，占核心时间约{native_fraction:.1f}%；其余包含建模、拓扑、规范化、合法性检查、LP投影等，尚未进一步拆分，不能全部归因为Python开销。
- 主对照seed0随机化每例各组合顺序；补充对照晚于主对照，未重复全部方法作同步噪声估计。单次运行、仅5例，主对照分两个worker进程执行（因保存中断恢复），与补充worker各自预热。不能据此作统计显著性、泛化或排行榜结论。
- 启动时记录上游版本的类型处理错误发生在读取样例前，保存在首次目录results.json。主运行在70次求解后遇到Windows写文件OSError(22)，已完成数据保留；改为临时文件原子替换后仅补齐30次。原config/source_snapshot不改写，resume配置、源码差异与失败事实见invocations.json。

## 5. 研究取舍

本阶段停止扩测，不训练、不追加调预算。下一阶段优先考虑训练集上的E10关系蒸馏：先让模型学习有用的分离/接触结构，并保留原坐标/形状头及严格合法化；直接模仿成品中心的优先级下降。该方向尚需用户发起后续实验，当前不启动教师数据生成或训练。剩余Q退化最大的worker49/74应作为机制反例保留，不把它们剔除以满足目标。

![预算与质量](budget_quality.png)

主证据：[配置](config.json)、[恢复调用](invocations.json)、[全部统计](results.json)、[逐例坐标与轨迹](cases.jsonl)、[原源码快照](source_snapshot.zip)、[恢复源码快照](source_snapshot_resume1.zip)。归因对照：[配置](../{args.control_run.name}/config.json)、[结果](../{args.control_run.name}/results.json)、[逐例证据](../{args.control_run.name}/cases.jsonl)。
'''
    (root/'analysis.md').write_text(text, encoding='utf-8')
    (root/'report_provenance.json').write_text(json.dumps({
        'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [root/'results.json', root/'cases.jsonl', args.control_run/'results.json', args.control_run/'cases.jsonl']},
        'dataset_reads': False, 'solver_calls': 0}, indent=2)+'\n', encoding='utf-8')
    print('Saved analysis.md and budget_quality.png; no solver or dataset reads')


if __name__ == '__main__':
    main()
