"""Saved-output audit, no model inference, rescoring or training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    paths = {'E55': ROOT/'experiments/E55_reference_position_shape/runs/20261026_reference_position_shape_seed0',
             'E56': HERE,
             'E42': ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'}
    progress = {k: read(v/'progress.json') for k, v in paths.items()}
    best = {k: next(x for x in p['history'] if x['epoch'] == p['best_epoch']) for k, p in progress.items()}
    outputs = {k: read(v/'terminal/results.json') for k, v in paths.items()}
    current = outputs['E56']['records']
    panels = read(HERE/'panels.json')
    for name in ('E55', 'E42'):
        old_panels = read(paths[name]/'panels.json')
        assert all(panels[key] == old_panels[key] for key in ('train_monitor_members', 'dev_monitor_members'))
        assert len(outputs[name]['records']) == len(current) == 500
        assert all(x['index'] == y['index'] and x['member'] == y['member']
                   for x, y in zip(current, outputs[name]['records']))
    protected = read(HERE.parents[1]/'provenance.json')['protected_checkpoints_sha256']
    for name, expected in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    result = {'same_panels_and_members_verified': True, 'new_inference_or_scoring_calls': 0,
              'test_read': False, 'protected_checkpoints_unchanged': protected,
              'models': {}, 'paired': {}}
    for name, epoch in best.items():
        dev = epoch['dev_monitor']
        g = progress['E56']['history'][0]['dev_monitor']['total'] if name == 'E55' else dev['total']
        result['models'][name] = {'dev_geometry_G': g,
            'dev_reference_two_terms': dev['center_local']+.25*dev['shape'],
            'dev_draft_diagnostics': dev['draft_diagnostics'], 'flash500': outputs[name]['summary'],
            'raw_hard_legal': sum(x.get('raw_hard_legal', False) for x in outputs[name]['records'])}
    n = np.array([r['n'] for r in current]); w = np.exp((n-n.max())/12); w /= w.sum()
    for name in ('E55', 'E42'):
        old = outputs[name]['records']
        delta = np.array([x['metrics']['cost']-y['metrics']['cost'] for x, y in zip(current, old)])
        contribution = delta*w
        largest = np.argsort(contribution)[-5:][::-1]
        row = {'wins': int(sum(delta < -1e-8)), 'ties': int(sum(abs(delta) <= 1e-8)),
               'losses': int(sum(delta > 1e-8)), 'weighted_Q_delta': float(contribution.sum()),
               'top5_weighted_regression_contributions': [
                   {'index': current[i]['index'], 'n': current[i]['n'],
                    'control_Q': old[i]['metrics']['cost'], 'E56_Q': current[i]['metrics']['cost'],
                    'weighted_delta': float(contribution[i]),
                    'E56_rescue_used': current[i].get('rescue_used', False)} for i in largest],
               'module_count_groups': []}
        for lo, hi in [(0, 59), (60, 89), (90, 120)]:
            ids = np.flatnonzero((n >= lo) & (n <= hi))
            row['module_count_groups'].append({'lo': lo, 'hi': hi, 'count': len(ids),
                'mean_Q_E56': float(np.mean([current[i]['metrics']['cost'] for i in ids])),
                'mean_Q_control': float(np.mean([old[i]['metrics']['cost'] for i in ids]))})
        result['paired'][name] = row
    top2 = result['paired']['E42']['top5_weighted_regression_contributions'][:2]
    result['two_cases_share_of_net_weighted_regression'] = sum(x['weighted_delta'] for x in top2)/result['paired']['E42']['weighted_Q_delta']
    (HERE/'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    hist = progress['E56']['history']; epochs = [x['epoch'] for x in hist]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7), layout='constrained')
    for role, color, title in [('train_monitor', '#267b9a', 'Train500'), ('dev_monitor', '#d65e37', 'Dev500')]:
        axes[0].plot(epochs, [x[role]['total'] for x in hist], label=title, color=color)
        axes[1].plot(epochs, [x[role]['center_local']+.25*x[role]['shape'] for x in hist], label=title, color=color)
    axes[0].axhline(best['E42']['dev_monitor']['total'], color='#677780', linestyle='--', label='E42 dev best51')
    axes[1].axhline(result['models']['E42']['dev_reference_two_terms'], color='#677780', linestyle='--')
    axes[2].plot(epochs, [100*x['dev_monitor']['draft_diagnostics']['overlap_area_fraction']['mean'] for x in hist], color='#d65e37', label='Dev500 overlap')
    axes[2].axhline(100*best['E42']['dev_monitor']['draft_diagnostics']['overlap_area_fraction']['mean'], color='#677780', linestyle='--', label='E42 dev best51')
    axes[0].set_title('Fixed E42 eight-term geometry Loss')
    axes[1].set_title('Center + 0.25 shape reference Loss')
    axes[2].set_title('Draft pairwise overlap / total area (%)')
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Post-training epoch'); ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(HERE/'training_curve.png', dpi=160); plt.close(fig)
    lines = ['# E56 后训练复核', '',
        '44轮早停、best44，本阶段训练32.86分钟。训练与500例flash检查完成；100张无评分图已生成。开发集为已暴露的同500成员，不是独立盲测。', '',
        '| 同dev500指标（越低越好） | E55原始 | E56后训练 | E42基线 |',
        '|---|---:|---:|---:|']
    specs = [('八项几何Loss G', lambda d: d['dev_geometry_G'], '.6f'),
             ('中心＋.25形状参考Loss', lambda d: d['dev_reference_two_terms'], '.6f'),
             ('草稿成对重叠面积／总面积', lambda d: 100*d['dev_draft_diagnostics']['overlap_area_fraction']['mean'], '.2f'),
             ('flash加权Q', lambda d: d['flash500']['weighted_Q'], '.6f'),
             ('flash算术平均Q', lambda d: d['flash500']['mean_Q'], '.6f'),
             ('flash P95 Q', lambda d: d['flash500']['p95_Q'], '.6f'),
             ('flash最差5%均值Q', lambda d: d['flash500']['worst5pct_mean_Q'], '.6f'),
             ('flash最大Q', lambda d: d['flash500']['max_Q'], '.6f'),
             ('完整平均时间（秒）', lambda d: d['flash500']['seconds_mean'], '.6f')]
    for title, get, fmt in specs:
        lines.append('| '+title+' | '+' | '.join(format(get(result['models'][name]), fmt) for name in ('E55', 'E56', 'E42'))+' |')
    lines += ['', '表中草稿重叠行以百分数表示，保留多重覆盖的成对计数，不是并集面积。E55的八项G来自E56零更新的epoch0，同一模型与同一开发成员，已核验参考两项与原E55监测吻合。', '',
        '相对E55，开发八项Loss下降20.22%，参考两项误差下降7.31%，成对重叠面积指标下降40.92%，模块重叠负担均值由0.2798降至0.1561；负担超过自身面积50%的模块比例由18.34%降至6.12%。原始draft仍为0/500硬合法，三模型flash后均500/500硬合法。', '',
        'E56相对E55最终Q为331胜／32平／137负；相对E42为225胜／52平／223负。与E42相比，八项G与参考两项误差略好、P95及最差5%和最大Q均改善，但草稿平均重叠仍略高，加权Q变差，均时也较高。不能据较低Loss或单一尾部指标替换基线。', '',
        '加权Q与算术Q分化来自模块数权重：本次事后按模块数<60／60–89／90–120分组，前两组算术Q较E42改善，最后一组退化。117模块的case392、462均触发救援，Q分别为1.313179／1.318236，原E42为1.000000／1.029455。两例贡献合计约占相对E42净加权Q退化的92.9%。这是保存结果的加权差分解，不是删例后重新计分，也不是证明救援本身导致退化。', '',
        '结论：后训练有效修复了E55纯参考训练造成的冲突增加，并改善了其参考拟合；当前数值结果接近E42且尾部部分更好，但不能整体升级。历史E42选模规则不同，本轮还多了E55预训练计算，不能直接宣称两阶段课程优于从头训练；用户同例草稿观察仍是首要研究依据。', '',
        'E55 best/latest和E42原权重SHA256复核均未改变，精确结果见review.json。本次只读已存记录，未再训练、推理、求解或评分，未读取测试。', '',
        '[训练曲线](training_curve.png) · [100例reference／E56／E42草稿](draft_comparison100/index.html) · [原自动报告](analysis.md)']
    (HERE/'review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'reviewed', 'protected_checkpoints_unchanged': True,
                      'top2_share_of_net_weighted_regression': result['two_cases_share_of_net_weighted_regression']}))


if __name__ == '__main__':
    main()
