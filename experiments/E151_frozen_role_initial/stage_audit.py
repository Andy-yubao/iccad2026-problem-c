"""Read-only paired audit of the frozen backend on the original dev500."""
import json

import numpy as np
import torch

from .base_loss import TOTAL_COMPONENTS, base_per_case
from .data import load_rows
from .initialization import load_parent_initial, verify_parent_initial
from .io import HERE, ROOT, save, sha
from .loss import per_case
from .loss_config import ARMS, WEIGHTS
from .model import GeometryModel
from .public import inputs_only
from .resident import ResidentRows
from .train import BATCH_SIZE, numerical_check, seed_all


METRICS = ('full_total', *TOTAL_COMPONENTS)
STAGE_NAMES = {4: 'S0', 8: 'S1', 12: 'S2'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_pinned(record):
    path = ROOT / record['path']
    if sha(path) != record['sha256']:
        raise ValueError(f'Archived baseline changed: {record["path"]}')
    return read_json(path)


def baselines_for(plan, members):
    manifest = plan['baselines']
    e148 = read_pinned(manifest['E148'])
    e150 = read_pinned(manifest['E150'])
    e150_summary = read_pinned(manifest['E150_columns'])
    if any([row['member'] for row in cases] != members for cases in (e148, e150)):
        raise ValueError('Archived baseline panel/order changed')
    if any([row[key] for row in cases] != list(range(len(members)))
           for key, cases in (('index', e148), ('dev_index', e150))):
        raise ValueError('Archived baseline indices changed')
    columns = e150_summary['per_case_columns']
    return {
        'E148': {layer: np.asarray([[row['baseline'][name]['total' if key == 'full_total' else key]
                                    for key in METRICS] for row in e148])
                 for layer, name in STAGE_NAMES.items()},
        'E150': {layer: np.asarray([[row['losses'][str(layer)][columns.index(key)]
                                    for key in METRICS] for row in e150])
                 for layer in STAGE_NAMES},
    }


def describe_change(delta):
    epsilon = 1e-8
    return dict(mean=float(delta.mean()), median=float(np.median(delta)),
                p95=float(np.quantile(delta, .95)),
                improved=int((delta < -epsilon).sum()), worsened=int((delta > epsilon).sum()),
                unchanged=int((np.abs(delta) <= epsilon).sum()))


def stage_report(stages):
    stage_means = {f'L{layer}': dict(zip(METRICS, array.mean(axis=0).tolist()))
                   for layer, array in stages.items()}
    delta = stages[8] - stages[4]
    overlap = METRICS.index('overlap_weighted')
    return dict(stage_means=stage_means,
                S1_L8_minus_L4={key: describe_change(delta[:, i]) for i, key in enumerate(METRICS)},
                L8_overlap_p95=float(np.quantile(stages[8][:, overlap], .95)))


@torch.no_grad()
def evaluate(run, frozen, plan, halted, beat):
    seed_all(plan['seed'])
    path = ROOT / frozen['checkpoint']
    if sha(path) != frozen['checkpoint_sha256']:
        raise ValueError('E151 best checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    entry = next(row for row in state['progress']['history'] if row['epoch'] == frozen['best_epoch'])
    monitor = entry['dev_monitor']
    members = read_json(run / 'panels.json')['dev_monitor_members']
    baselines = baselines_for(plan, members)
    initial_losses = read_pinned(plan['baselines']['E149_loss'])
    model = GeometryModel(**state['model_config']).cuda().eval()
    model.load_state_dict(state['model'])
    verify_parent_initial(model, load_parent_initial(plan))
    resources = read_json(run / 'resources.json')
    rows = load_rows('dev_monitor500', halted=halted,
                     beat=lambda **fields: beat(cache=fields))
    if rows is None:
        return None
    if len(rows) != 500 or [list(r['member']) for r in rows] != members:
        raise ValueError('Unexpected dev500 members')
    pack = ResidentRows(rows, 'cuda', max_nodes=resources['max_nodes'],
                        max_pins=resources['max_loss_ports_per_block'])
    joint_values = []
    initial_no_overlap = []
    stage_batches = {layer: [] for layer in (0, *WEIGHTS)}
    for start in range(0, len(rows), BATCH_SIZE):
        if halted():
            return None
        stop = min(start + BATCH_SIZE, len(rows))
        batch = pack.batch(range(start, stop))
        states = model(inputs_only(batch))
        loss = per_case(states, batch, ARMS[0])
        numerical_check(loss, states)
        joint_values.extend(loss['total'].cpu().tolist())
        first = base_per_case(states[0], batch)
        initial_no_overlap.extend(sum(first[key] for key in TOTAL_COMPONENTS
                                      if key != 'overlap_weighted').cpu().tolist())
        stage_batches[0].append(torch.stack([first['total'], *(first[key] for key in TOTAL_COMPONENTS)], -1).cpu().double().numpy())
        for layer in WEIGHTS:
            prefix = f'layer{layer:02d}_'
            names = ['full_overlap_weighted' if key == 'overlap_weighted' else key for key in METRICS]
            stage_batches[layer].append(torch.stack([loss[prefix + key] for key in names], -1).cpu().double().numpy())
        beat(completed=stop, count=500)
    if halted():
        return None
    stages = {layer: np.concatenate(batches) for layer, batches in stage_batches.items()}
    np.testing.assert_allclose(joint_values, monitor['per_case_total'], rtol=0, atol=1e-6)
    np.testing.assert_allclose(initial_no_overlap, initial_losses['dev_components']['per_case_total'], rtol=0, atol=1e-6)
    for layer in WEIGHTS:
        if abs(stages[layer][:, 0].mean() - monitor[f'layer{layer:02d}_full_total']) > 1e-7:
            raise ValueError(f'L{layer} audit Loss does not reproduce the training monitor')

    comparisons = {name: {
        'stages_E151_minus_baseline': {
            f'L{layer}': {key: describe_change(stages[layer][:, col] - values[layer][:, col])
                          for col, key in enumerate(METRICS)}
            for layer in STAGE_NAMES},
        'S1_L8_minus_L4_E151_minus_baseline': {
            key: describe_change((stages[8][:, col] - stages[4][:, col])
                                 - (values[8][:, col] - values[4][:, col]))
            for col, key in enumerate(METRICS)},
    } for name, values in baselines.items()}
    case_rows = [
        dict(dev_index=index, member=member,
             layers={f'L{layer}': dict(zip(METRICS, array[index].tolist())) for layer, array in stages.items()},
             S1_L8_minus_L4=dict(zip(METRICS, (stages[8][index] - stages[4][index]).tolist())))
        for index, member in enumerate(members)
    ]
    result = dict(status='complete', count=len(rows), frozen=frozen, parent_initial=plan['parent_initial'],
                  L0_no_overlap_mean=float(np.mean(initial_no_overlap)),
                  E151=stage_report(stages), baselines={name: stage_report(values) for name, values in baselines.items()},
                  comparisons=comparisons, test_read=False, postprocessing_calls=0, Q_calls=0)
    directory = run / 'dev500_stage_audit'
    save(directory / 'per_case.json', case_rows)
    save(directory / 'results.json', result)
    write_report(directory / 'report.md', result)
    return result


def write_report(path, result):
    reports = {'E151': result['E151'], **result['baselines']}
    lines = [
        '# E151 固定 dev500 阶段验收',
        '',
        f'E151 best{result["frozen"]["best_epoch"]}，L0 固定 E149 best52。各点为 500 题算术均值，未乘训练层权重。',
        '',
        '| 实验 | L4 完整 | L8 完整 | L12 完整 | L4 重叠 | L8 重叠 | S1 重叠变化 | 重叠改善题数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, report in reports.items():
        values = report['stage_means']
        change = report['S1_L8_minus_L4']['overlap_weighted']
        losses = [values[f'L{layer}']['full_total'] for layer in STAGE_NAMES]
        overlaps = [values[f'L{layer}']['overlap_weighted'] for layer in (4, 8)]
        means = ' | '.join(f'{value:.6f}' for value in (*losses, *overlaps, change['mean']))
        lines.append(f'| {name} | {means} | {change["improved"]}/500 |')
    lines += ['', 'E151 的 S1 变化按同题 L8 减 L4 计算。越低越好。',
              '', '| 分项 | 平均变化 | 改善题 | 退化题 | 持平题 |',
              '|---|---:|---:|---:|---:|']
    labels = dict(full_total='完整 Loss', overlap_weighted='重叠', grouping_weighted='分组',
                  boundary_weighted='贴边', public_shape_weighted='F/P 尺寸',
                  public_preplaced_position_weighted='P 位置')
    for key, label in labels.items():
        metrics = result['E151']['S1_L8_minus_L4'][key]
        lines.append(f'| {label} | {metrics["mean"]:.6f} | {metrics["improved"]} | {metrics["worsened"]} | {metrics["unchanged"]} |')
    lines += ['', 'E148 和 E150 为已归档的同题对照。初始化与辅助监督存在差异，阶段差异仍需匹配对照确认。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
