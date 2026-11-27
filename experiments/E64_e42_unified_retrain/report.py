"""Summarize only the selected fixed-G candidate; never reselect using Q."""
import json
from .io import HERE, ROOT, save
from .pipeline import OUTPUT


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    result = read(OUTPUT/'results.json')
    history = read(OUTPUT/'progress.json')['history']
    raw = read(OUTPUT/'full_dev.json')
    flash = read(OUTPUT/'terminal/summary.json')
    best = next(row for row in history if row['epoch'] == result['best_epoch'])
    old = read(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/progress.json')
    old_best = next(row for row in old['history'] if row['epoch'] == 51)
    reference = {'dev500_G': old_best['dev_monitor']['geometry'], 'historical_dev500_flash_weighted_Q': 1.025859634}
    save(HERE/'runs/review.json', {'training': result, 'best_monitor': best,
         'raw2000': raw['losses'], 'flash500': flash, 'historical_E42': reference,
         'test_read': False, 'baseline_promoted': False,
         'interpretation': 'One seed on adaptively exposed dev. Two protocol changes (no ramp; G-only selection), plus CUDA execution path. Cannot isolate their causal contributions.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ('total', 'overlap', 'center_local', 'relative_local')):
        for role, style in (('train_monitor', '--'), ('dev_monitor', '-')):
            ax.plot([row['epoch'] for row in history], [row[role][key] for row in history], style, label=role)
        ax.axvline(result['best_epoch'], color='gray', alpha=.5)
        ax.set(title=key, xlabel='scratch epoch')
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(HERE/'runs/learning_curves.png', dpi=160)
    plt.close(fig)
    overlap = raw['losses']['draft_diagnostics']['overlap_area_fraction']['mean']
    text = f'''# E64 E42统一Loss从头重训

训练停止：{result['status']}，完成{result['epochs']}轮，固定G选择best{result['best_epoch']}。
固定dev500 G：{best['dev_monitor']['total']:.6f}；历史E42 best51同面板G：{reference['dev500_G']:.6f}。
完整dev2000 G：{raw['losses']['total']:.6f}；草稿硬合法{raw['losses']['raw_hard_legal']}/2000，真实交面积均值{overlap:.3%}。
唯一best的flash500：硬合法{flash['hard_legal']}/500，加权Q {flash['weighted_Q']:.6f}；历史E42为1.025860。Q仅作辅助，未重选模型。

![固定面板学习曲线](learning_curves.png)

[同100例reference／本轮draft／冻结E42无评分图集](20261027_e42_fixedG_scratch_seed0/draft_comparison100/index.html)

本轮固定E42八项G，取消前两轮渐入及occupancy选模，架构、16k成员和原正则保持原定义。执行改用已核验的CUDA graph常驻路径，因此不是只改选择器的严格单因素因果实验。完整数值见[review.json](review.json)。这是单种子、自适应开发证据；不据此证明表达能力充分、Loss最优或独立泛化。不自动升级E42，不启动第二阶段，测试封存。
'''
    (HERE/'runs/review.md').write_text(text, encoding='utf-8')
    print('E64 selected-best assessment and saved artifacts complete', flush=True)


if __name__ == '__main__':
    main()
