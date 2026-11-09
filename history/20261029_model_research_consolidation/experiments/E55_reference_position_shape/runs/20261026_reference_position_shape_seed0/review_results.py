"""Read saved evidence only; compare common reference terms, never rerun models."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    progress, old = read(HERE/'progress.json'), read(BASE/'progress.json')
    current = next(r for r in progress['history'] if r['epoch'] == progress['best_epoch'])
    baseline = next(r for r in old['history'] if r['epoch'] == old['best_epoch'])
    panels, base_panels = read(HERE/'panels.json'), read(BASE/'panels.json')
    for role in ('train_monitor_members', 'dev_monitor_members'):
        assert panels[role] == base_panels[role]
    a, b = read(HERE/'terminal/results.json'), read(BASE/'terminal/results.json')
    assert len(a['records']) == len(b['records']) == 500
    assert all(x['index'] == y['index'] and x['member'] == y['member'] for x, y in zip(a['records'], b['records']))
    comparison = {'matching_train_and_dev_monitor_members': True, 'matching_flash500_members': True,
                  'source': 'saved best-epoch monitoring components and saved terminal outputs',
                  'new_inference_calls': 0, 'test_read': False, 'reference_loss': {}}
    for role in ('train_monitor', 'dev_monitor'):
        x, y = current[role], baseline[role]
        old_reference = y['center_local']+.25*y['shape']
        comparison['reference_loss'][role] = {'E42': old_reference, 'E55': x['total'],
             'relative_change': x['total']/old_reference-1,
             'E42_center': y['center_local'], 'E55_center': x['center_local'],
             'E42_shape': y['shape'], 'E55_shape': x['shape']}
    for label, data in [('E42', b), ('E55', a)]:
        comparison[label] = {'flash500': data['summary'],
             'raw_hard_legal': sum(r.get('raw_hard_legal', False) for r in data['records']),
             'draft_overlap_area_fraction': (baseline if label == 'E42' else current)['dev_monitor']['draft_diagnostics']['overlap_area_fraction'],
             'draft_penetration_squared_per_block': (baseline if label == 'E42' else current)['dev_monitor']['draft_diagnostics']['penetration_squared_per_block']}
    (HERE/'reference_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    history = progress['history']; epoch = [r['epoch'] for r in history]
    train = [r['train_monitor']['total'] for r in history]
    dev = [r['dev_monitor']['total'] for r in history]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for ax in axes:
        ax.plot(epoch, train, label='E55 train monitor500', color='#267b9a')
        ax.plot(epoch, dev, label='E55 dev monitor500', color='#d65e37')
        ax.axhline(comparison['reference_loss']['dev_monitor']['E42'], linestyle='--', color='#4d6675', label='E42 best51 dev: same two terms')
        ax.scatter([current['epoch']], [current['dev_monitor']['total']], color='#d65e37', marker='*', s=120, zorder=4)
        ax.set_xlabel('Completed epoch'); ax.set_ylabel('Center loss + 0.25 shape loss')
        ax.grid(alpha=.2)
    axes[0].set_yscale('log'); axes[0].set_title('Full training history')
    axes[1].set_xlim(15, 43); axes[1].set_ylim(.16, .33); axes[1].set_title('Late training; best at epoch40')
    axes[0].legend(fontsize=9)
    fig.savefig(HERE/'training_curve.png', dpi=170)
    plt.close(fig)
    text = '''# E55 训练结果复核：同口径 reference 拟合与草稿

训练已于2026-10-26北京时间13:49结束，43轮早停、best40，训练累计30.89分钟。随后立即完成同dev500原flash，13:54全部图集和报告完成。学习率已完成三次下降并在最低档完成要求的轮数；无运行错误，测试未读取。

本复核只读取保存的监测分项、成员清单和终末输出，没有重新训练、推理或评分。train_monitor500和dev_monitor500成员及顺序均与E42一致；终末500逐例成员一致。

| 指标（越低越好） | E42 best51 | E55 best40 |
|---|---:|---:|
| train_monitor500：center + .25 shape | 0.191242 | 0.174318 |
| dev_monitor500：center + .25 shape | 0.231349 | 0.245274 |
| dev中心项 | 0.188545 | 0.199414 |
| dev形状项 | 0.171216 | 0.183440 |
| draft成对重叠面积／总模块面积均值 | 6.38% | 11.03% |
| draft穿透平方／模块数均值 | 0.020402 | 0.057137 |
| 原始draft硬合法 | 0/500 | 0/500 |
| flash最终硬合法 | 500/500 | 500/500 |
| flash加权Q | 1.025860 | 1.051852 |
| flash P95 Q | 1.185535 | 1.216195 |
| flash最差5%均值Q | 1.291023 | 1.365393 |
| flash最大Q | 1.927279 | 1.917892 |
| 完整平均耗时（秒） | 0.251334 | 0.291117 |
| 平均中心修复位移／自身sqrt(area) | 0.416513 | 0.513098 |
| 救援案例数 | 4 | 10 |

E55训练参考误差下降8.85%，但开发参考误差增加6.02%；中心与形状两项均退化。开发草稿成对重叠面积指标增加72.78%，穿透指标约2.80倍。成对重叠面积保留多重覆盖计数，不是几何并集面积。最终Q胜／平／负为155／35／310；最大Q略好，但P95和最差5%均值变差，不能只据最大值声称尾部改善。耗时比较来自不同时间的既有记录，未作同时配对复测。

这次取消几何惩罚让训练监测样本更贴近参考，却没有使开发草稿更贴近参考，并伴随更多深重叠和更大修复负担。因此，本轮结果不支持“只是额外几何Loss妨碍了模型复现参考”这个充分解释，也不支持替换E42。现象提示拟合与开发泛化之间存在差距，但不能由单seed直接确定容量、表示或标签多解性是根因。E42历史选模采用另一套代理，执行引擎也有历史差异，本比较是冻结候选的同指标比较，不是严格控制所有因素的纯Loss因果消融。

按图集预定随机顺序查看前三例217／194／399，两模型都大体保持参考的纵向组织与大模块次序；E55仍有局部压叠和间隙交替，未见整体结构突变。三例观察仅辅助描述，草稿优劣仍以用户同例观察为首要参考，不把此处浏览当作100例人工评分。

[训练曲线](training_curve.png) · [同100例reference／E55／E42草稿](draft_comparison100/index.html) · [精确对照](reference_comparison.json) · [原自动报告](analysis.md)

保持E42／flash和E50／pro冻结；本次没有续训或启动新实验。
'''
    (HERE/'review.md').write_text(text, encoding='utf-8')
    print(json.dumps({'status': 'reviewed', 'same_panel_verified': True, 'new_inference_calls': 0}))


if __name__ == '__main__':
    main()
