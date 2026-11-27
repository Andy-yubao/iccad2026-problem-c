"""Plot saved curves and frozen raw diagnostics; performs no model inference."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
ARMS = ['base96x4', 'deep96x8', 'deep96x12', 'wide160x4']
COLORS = ['#2563eb', '#e67700', '#14836b', '#8b5cf6']


def main():
    curves = {}
    for arm in ARMS:
        folder = RUNS/f'20261022_capacity_{arm}_seed0'
        path = folder/'progress.json'
        if not path.exists():
            continue
        curves[arm] = (json.loads(path.read_text(encoding='utf-8')), folder)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for arm, color in zip(ARMS, COLORS):
        if arm not in curves:
            continue
        progress, _ = curves[arm]
        records = [r for r in progress['history'] if r['epoch'] > 0]
        for axis, role in zip(axes, ['train_monitor', 'dev_monitor']):
            axis.plot([r['epoch'] for r in records], [r[role]['selection_proxy'] for r in records],
                      label=arm, color=color, linewidth=1.7)
    for axis, title in zip(axes, ['Fixed training monitor', 'Fixed development monitor']):
        axis.set(title=title, xlabel='Epoch', ylabel='Geometry + occupancy proxy (lower is better)')
        axis.set_yscale('log'); axis.grid(alpha=.2); axis.legend(frameon=False)
    fig.suptitle('E38 capacity comparison | same data, loss and training policy | seed 0')
    for extension in ('png', 'svg'):
        fig.savefig(HERE/f'loss_curves.{extension}', dpi=180)
    plt.close(fig)
    available = [(arm, folder) for arm, (_, folder) in curves.items() if (folder/'full_dev.json').exists()]
    if not available:
        return
    labels = [arm for arm, _ in available]
    metrics = [json.loads((folder/'full_dev.json').read_text(encoding='utf-8'))['losses'] for _, folder in available]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    values = [m['selection_proxy'] for m in metrics]
    bars = axes[0].bar(labels, values, color=COLORS[:len(labels)])
    axes[0].bar_label(bars, fmt='%.4f', padding=3)
    axes[0].set(title='Frozen best: full development 2000', ylabel='Fixed raw proxy (lower is better)')
    axes[0].set_ylim(0, max(values)*1.16)
    x = np.arange(len(labels)); width = .36
    for axis, key, title in [(axes[1], 'overlap_area_fraction', 'Pairwise overlap / total block area'),
                              (axes[2], 'largest_conflict_cluster_fraction', 'Largest collision cluster / blocks')]:
        mean = [m['draft_diagnostics'][key]['mean'] for m in metrics]
        p95 = [m['draft_diagnostics'][key]['p95'] for m in metrics]
        axis.bar(x-width/2, mean, width, label='Mean', color='#2563eb')
        axis.bar(x+width/2, p95, width, label='P95', color='#e67700')
        axis.set_xticks(x, labels); axis.set(title=title, ylabel='Fraction (lower is better)'); axis.legend(frameon=False)
        axis.set_ylim(0, max(mean+p95)*1.24)
    for axis in axes:
        axis.tick_params(axis='x', labelrotation=20); axis.grid(axis='y', alpha=.2)
    fig.suptitle('Raw drafts before repair | adaptive development diagnostics, not contest scores')
    for extension in ('png', 'svg'):
        fig.savefig(HERE/f'raw_draft_comparison.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
