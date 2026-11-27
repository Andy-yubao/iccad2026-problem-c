"""Saved evidence only: paired geometry and terminal summaries, never reselection."""
import json
import numpy as np
from .io import HERE, save
from .pipeline import OUTPUTS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired(a, b, key):
    if [r['member'] for r in a] != [r['member'] for r in b]:
        raise ValueError('Paired panel membership/order mismatch')
    delta = np.array([x[key]-y[key] for x,y in zip(a,b)])
    groups = {}
    for i, row in enumerate(a):
        groups.setdefault(row['member'][0].split('/')[0], []).append(i)
    blocks = list(groups.values()); rng = np.random.default_rng(63); draws = []
    for _ in range(2000):
        ids = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
        draws.append(delta[ids].mean())
    return {'mean_delta': float(delta.mean()), 'worker_bootstrap95': np.percentile(draws,[2.5,97.5]).tolist(),
            'better_equal_worse': [int((delta < -1e-9).sum()), int((abs(delta) <= 1e-9).sum()), int((delta > 1e-9).sum())]}


def main():
    values = {}; curves = {}; panels = {}
    for arm, folder in OUTPUTS.items():
        result = read(folder/'results.json'); progress = read(folder/'progress.json')
        raw = read(folder/'full_dev.json'); flash = read(folder/'terminal/summary.json')
        best = next(r for r in progress['history'] if r['epoch'] == result['best_epoch'])
        values[arm] = {'training': result, 'best_monitor': best, 'raw2000': raw['losses'],
                       'flash500': flash, 'initial_E42_monitor': progress['history'][0]}
        curves[arm] = progress['history']; panels[arm] = raw['records']
    comparisons = {}
    for arm in ('weight075','softband005'):
        comparisons[arm] = {key: paired(panels[arm],panels['control'],key) for key in
            ('overlap_area_fraction','max_local_penetration','loss_center_local','loss_relative_local','loss_shape','loss_common_original_G')}
    save(HERE/'runs/review.json', {'arms': values, 'candidate_minus_control_raw2000': comparisons,
         'test_read': False, 'baseline_promoted': False,
         'interpretation': 'Adaptive single-seed dev evidence. Each arm selected by own fixed Loss; cross-arm own totals are incomparable.'})
    # Publication-style learning curves from already saved fixed panels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    for arm, history in curves.items():
        epochs = [r['epoch'] for r in history]
        for ax, key in zip(axes.flat, ('common_original_G','overlap','center_local','relative_local')):
            for role, style in (('dev_monitor','-'),('train_monitor','--')):
                ax.plot(epochs,[r[role][key] for r in history],style,label=arm+' '+role.split('_')[0])
            ax.set_title(key); ax.set_xlabel('post-training epoch'); ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'runs/learning_curves.png',dpi=160);plt.close(fig)
    rows = []
    for arm, data in values.items():
        raw = data['raw2000']; diag = raw['draft_diagnostics']
        rows.append(f"| {arm} | {data['training']['best_epoch']} | {raw['common_original_G']:.6f} | {100*diag['overlap_area_fraction']['mean']:.3f}% | {raw['relative_local']:.6f} | {raw['center_local']:.6f} | {data['flash500']['weighted_Q']:.6f} |")
    text = '# E63：E42 温和放宽中心监督后训练\n\n三组从 E42 best51 相同权重出发，共同重置 AdamW 和四档调度。每组按自身固定 Loss 选唯一 best；不同组自身总 Loss 不可直接比较。\n\n'
    text += '| 组 | best轮 | 原口径G dev2000 | 实际交面积均值 | 相对位置Loss | 中心Loss | flash500加权Q |\n|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    text += '\n\n冻结 E42 历史 flash500 加权Q 1.025860，仅作辅助参照。未按Q重选checkpoint，未升级基线。完整逐例配对与按worker重抽区间见 [review.json](review.json)。\n\n'
    text += '![共同口径训练曲线](learning_curves.png)\n\n'
    text += '\n'.join(f'- [{arm} 同100例 reference／本轮draft／冻结E42](20261027_{arm}_seed0/draft_comparison100/index.html)' for arm in OUTPUTS)
    text += '\n\n本轮为单种子、自适应开发证据，不是独立泛化或正式比赛成绩。草稿观感由用户评价；原Loss及几何分项为数值参考，Q为辅助。原checkpoint保留，测试封存。\n'
    (HERE/'runs/review.md').write_text(text,encoding='utf-8')
    print('E63 all declared arms and terminal artifacts complete',flush=True)


if __name__ == '__main__':
    main()
