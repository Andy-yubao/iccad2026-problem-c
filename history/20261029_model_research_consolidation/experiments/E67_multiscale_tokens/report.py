"""Fixed-G selected multiscale-membership evidence; no Q reselection."""
import json
import numpy as np
from .io import HERE, ROOT, save
from .pipeline import OUTPUTS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    old = ROOT/'experiments/E64_e42_unified_retrain/runs/20261027_e42_fixedG_scratch_seed0'
    baseline = read(old/'results.json')
    old_history = read(old/'progress.json')['history']
    old_best = next(row for row in old_history if row['epoch'] == baseline['best_epoch'])
    old_raw = read(old/'full_dev.json')['losses']
    values = {'E64_baseline': {'training': baseline, 'best_monitor': old_best,
                              'raw2000': old_raw, 'flash500': read(old/'terminal/summary.json')}}
    histories = {'E64_baseline': old_history}
    rows = [f"| E64 baseline | {baseline['best_epoch']} | {old_best['train_monitor']['total']:.6f} | {old_best['dev_monitor']['total']:.6f} | {old_raw['total']:.6f} | {100*old_raw['draft_diagnostics']['overlap_area_fraction']['mean']:.3f}% | {values['E64_baseline']['flash500']['weighted_Q']:.6f} |"]
    for arm, folder in OUTPUTS.items():
        if not (folder/'terminal/summary.json').exists():
            values[arm] = {'status': 'incomplete_or_failed', 'output': str(folder.relative_to(HERE))}
            continue
        result = read(folder/'results.json')
        history = read(folder/'progress.json')['history']
        best = next(row for row in history if row['epoch'] == result['best_epoch'])
        raw = read(folder/'full_dev.json')['losses']
        flash = read(folder/'terminal/summary.json')
        fit_delta = best['train_monitor']['total']/old_best['train_monitor']['total']-1
        dev_delta = best['dev_monitor']['total']/old_best['dev_monitor']['total']-1
        overlap_delta = raw['draft_diagnostics']['overlap_area_fraction']['mean']/old_raw['draft_diagnostics']['overlap_area_fraction']['mean']-1
        eligible = (dev_delta <= -.005 and raw['total'] < old_raw['total'] and overlap_delta <= .02
                    and result['status'] == 'early_stop')
        common = []
        reference = {row['step']: row for row in old_history}
        for row in history:
            if row['step'] in reference:
                other = reference[row['step']]
                common.append({'step': row['step'], 'candidate_train_G': row['train_monitor']['total'],
                               'candidate_dev_G': row['dev_monitor']['total'],
                               'baseline_train_G': other['train_monitor']['total'],
                               'baseline_dev_G': other['dev_monitor']['total']})
        values[arm] = {'training': result, 'parameters': read(folder/'config.json')['identity']['parameters'], 'best_monitor': best, 'raw2000': raw, 'flash500': flash,
                       'train_relative_delta': fit_delta, 'dev_relative_delta': dev_delta,
                       'raw_overlap_relative_delta': overlap_delta, 'promising_numeric_gate': eligible,
                       'visual_acceptance': 'pending user feedback', 'common_steps': common,
                       'interpretation': 'Single seed0, adaptive dev evidence. No multi-seed confirmation permitted.'}
        histories[arm] = history
        rows.append(f"| {arm} | {result['best_epoch']} | {best['train_monitor']['total']:.6f} | {best['dev_monitor']['total']:.6f} | {raw['total']:.6f} | {100*raw['draft_diagnostics']['overlap_area_fraction']['mean']:.3f}% | {flash['weighted_Q']:.6f} |")
    save(HERE/'runs/review.json', {'arms': values, 'test_read': False, 'baseline_promoted': False,
         'all_seeds': [0], 'multi_seed_confirmation': False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for arm, history in histories.items():
        visible = [row for row in history if row['epoch'] > 0]
        for ax, key in zip(axes.flat, ('total', 'overlap', 'center_local', 'relative_local')):
            for role, style in (('train_monitor', '--'), ('dev_monitor', '-')):
                ax.plot([row['step'] for row in visible], [row[role][key] for row in visible], style,
                        label=arm+' '+role.split('_')[0])
            ax.set(title=key, xlabel='optimizer steps'); ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(HERE/'runs/learning_curves.png', dpi=160); plt.close(fig)
    text = '# E67：多尺度分组与全局对照\n\n全部固定seed0；用户明确不做多种子确认。原八项G及数据/训练规则冻结。\n\n'
    text += '| 组 | best轮 | train500 G | dev500 G | dev2000 G | raw交面积均值 | flash加权Q |\n|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    text += '\n\n两组原96×12主干一致，新token网络参数/初始化/形状一致；只改变新分支真实分组与全局混合的pool/broadcast。原主干组关系仍存在，控制保留组数与成员标志，不能称整个网络无分组信息。common_steps及完整分项见[review.json](review.json)。本报告不以Q重选模型，不自动升级E42。数值候选门槛仅供决定后续机制验证，草稿视觉评价待用户。\n\n'
    for arm in OUTPUTS:
        if values[arm].get('promising_numeric_gate') is not None:
            value = values[arm]
            text += f"{arm}：固定train/dev G相对E64为{value['train_relative_delta']:+.2%}/{value['dev_relative_delta']:+.2%}，raw交面积{value['raw_overlap_relative_delta']:+.2%}；预设数值候选门槛{'满足' if value['promising_numeric_gate'] else '未满足'}。\n\n"
    text += '![共同更新数曲线](learning_curves.png)\n\n'
    text += '\n'.join(f'- [{arm}同100例reference/current/E42无评分图](20261028_{arm}_seed0/draft_comparison100/index.html)' for arm in OUTPUTS if (OUTPUTS[arm]/'draft_comparison100/manifest.json').exists())
    text += '\n\n单种子、自适应开发证据，不是正式比赛成绩或独立泛化结论。测试封存，原权重保留。长期任务按[long_task.md](../../E65_geometry_messages/long_task.md)继续，由每小时唤醒检查并准备下一阶段。\n'
    (HERE/'runs/review.md').write_text(text, encoding='utf-8')
    print('E67 pair evidence summarized; all seed0, no multi-seed confirmation.', flush=True)


if __name__ == '__main__':
    main()
