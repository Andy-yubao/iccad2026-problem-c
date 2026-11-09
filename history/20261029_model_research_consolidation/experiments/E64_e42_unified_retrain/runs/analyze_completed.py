"""Read saved results only; no model inference, checkpoint selection or test access."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'
NEW = HERE/'20261027_e42_fixedG_scratch_seed0'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    history = read(NEW/'progress.json')['history']
    result = read(NEW/'results.json')
    current = next(row for row in history if row['epoch'] == result['best_epoch'])
    previous = next(row for row in read(OLD/'progress.json')['history'] if row['epoch'] == 51)
    old_raw = read(OLD/'full_dev.json')['losses']
    new_raw = read(NEW/'full_dev.json')['losses']
    old_flash = read(OLD/'terminal/summary.json')
    new_flash = read(NEW/'terminal/summary.json')
    old_rows = read(OLD/'terminal/results.json')['records']
    new_rows = read(NEW/'terminal/results.json')['records']
    assert [row['member'] for row in old_rows] == [row['member'] for row in new_rows]
    delta = np.array([a['metrics']['cost']-b['metrics']['cost'] for a, b in zip(new_rows, old_rows)])
    count = np.array([row['n'] for row in new_rows])
    weights = np.exp((count-count.max())/12)
    weights /= weights.sum()
    split = {}
    for name, mask in (('n_lt100', count < 100), ('n_ge100', count >= 100)):
        split[name] = {'count': int(mask.sum()), 'mean_Q_delta': float(delta[mask].mean()),
                       'weighted_Q_delta_contribution': float((weights[mask]*delta[mask]).sum())}
    summary = {'old_dev500_G': previous['dev_monitor']['geometry'],
               'new_dev500_G': current['dev_monitor']['total'],
               'old_dev2000_G': old_raw['geometry'], 'new_dev2000_G': new_raw['total'],
               'old_dev2000_overlap': old_raw['draft_diagnostics']['overlap_area_fraction'],
               'new_dev2000_overlap': new_raw['draft_diagnostics']['overlap_area_fraction'],
               'old_flash500': old_flash, 'new_flash500': new_flash,
               'flash500_better_equal_worse': [int((delta < -1e-9).sum()), int((abs(delta) <= 1e-9).sum()), int((delta > 1e-9).sum())],
               'descriptive_module_count_split': split,
               'comparison_scope': 'identical frozen dev panels; descriptive post-hoc size split, no Q reselection; one adaptive-development seed',
               'source_results_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (OLD/'full_dev.json', OLD/'terminal/results.json', NEW/'full_dev.json', NEW/'terminal/results.json')},
               'test_read': False}
    report = read(HERE/'review.json')
    report['posthoc_same_panel_comparison'] = summary
    (HERE/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    visible = [row for row in history if row['epoch'] >= 1]
    for ax, key in zip(axes.flat, ('total', 'overlap', 'center_local', 'relative_local')):
        for role, style in (('train_monitor', '--'), ('dev_monitor', '-')):
            ax.plot([row['epoch'] for row in visible], [row[role][key] for row in visible], style, label=role)
        baseline_key = 'geometry' if key == 'total' else key
        ax.axhline(previous['dev_monitor'][baseline_key], color='black', linestyle=':', label='E42 best51 dev')
        ax.axvline(result['best_epoch'], color='gray', alpha=.5)
        ax.set(title=key, xlabel='scratch epoch (epoch 0 omitted from view)')
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE/'learning_curves_detail.png', dpi=160)
    plt.close(fig)

    text = f'''# E64 E42统一Loss从头重训：完成复核

训练于北京时间2026-10-27 23:10正常早停，完整43轮／21,500次更新，训练约{result['elapsed_seconds']/60:.1f}分钟；固定G选择best40。完成三次LR下降，第38轮进入5e-5后完成五轮训练／验证。开发评价及100图于23:15完成，无评价错误。未升级原E42。

| 同面板指标（越低越好） | 冻结E42 best51 | 本轮E64 best40 |
|---|---:|---:|
| 固定dev500 G | {summary['old_dev500_G']:.6f} | {summary['new_dev500_G']:.6f} |
| 完整dev2000 G | {old_raw['geometry']:.6f} | {new_raw['total']:.6f} |
| dev2000真实交面积均值 | {summary['old_dev2000_overlap']['mean']:.3%} | {summary['new_dev2000_overlap']['mean']:.3%} |
| dev2000真实交面积P95 | {summary['old_dev2000_overlap']['p95']:.3%} | {summary['new_dev2000_overlap']['p95']:.3%} |
| flash500加权Q | {old_flash['weighted_Q']:.6f} | {new_flash['weighted_Q']:.6f} |
| flash500普通均值Q | {old_flash['mean_Q']:.6f} | {new_flash['mean_Q']:.6f} |
| flash500 P95 Q | {old_flash['p95_Q']:.6f} | {new_flash['p95_Q']:.6f} |
| flash500最差5%均值Q | {old_flash['worst5pct_mean_Q']:.6f} | {new_flash['worst5pct_mean_Q']:.6f} |
| flash500最大Q | {old_flash['max_Q']:.6f} | {new_flash['max_Q']:.6f} |

本轮dev500 G比原E42高3.09%，完整dev2000 G高2.29%；真实交面积均值相对高13.38%。中心、形状、相对位置与穿透Loss均未改善，MIB和bbox分项有所改善。两者raw硬合法均为0/2000；flash后均500/500硬合法，这两个阶段不能混淆。

flash Q方向并不一致：加权Q变差，但普通均值、P95、最差5%与最大Q改善。500例逐例为213改善、46基本相同、241回退。现有加权公式为exp((N-maxN)/12)：事后描述性分组显示，N<100的395例平均Q差为−0.003272，N≥100的105例为+0.006185；后者对整体加权Q差的贡献为+0.004348，解释了加权指标回退。分组不用于挑checkpoint或更换主口径，不是新的预设检验。

best40的固定训练G为0.304448，开发G为0.368244；最后三轮训练G继续从0.302744降到0.301032，开发G维持在0.368710–0.369533，高于best。支持存在后期泛化间隙；不能把全部误差归为表达能力不足，也不能据此证明架构足够。原E42与本轮同时存在渐入、选模与执行路径差别，单seed不能把回退归因于取消occupancy选模。

![训练曲线：去掉epoch0以便查看主要训练阶段](learning_curves_detail.png)

[完整原始曲线](learning_curves.png) · [同100例reference／本轮draft／冻结E42无评分图集](20261027_e42_fixedG_scratch_seed0/draft_comparison100/index.html) · [精确结果](review.json)

已检查100张图片全部存在，原E42 best/latest哈希不变。抽看图集仅用于核实展示，不把个别图当整体视觉结论，草稿观感仍待用户评价。统一标准已形成可复现的后续架构研究对照；本次未获得整体质量升级。固定G继续冻结，保留E64为统一协议参照，原E42／flash继续作为既有质量基线。第二阶段尚未启动。

本轮是单种子、自适应开发证据，Q为本地中性时间质量指标，不是正式比赛分数；测试封存。分析入口[analyze_completed.py](analyze_completed.py)只读取既有输出，无额外训练或推理。
'''
    (HERE/'review.md').write_text(text, encoding='utf-8')
    print('Completed E64 same-panel review and readable saved curves; no inference run.')


if __name__ == '__main__':
    main()
