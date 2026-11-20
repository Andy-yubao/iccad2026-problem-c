"""Aggregate matched S2 geometry and quality on the original dev500."""
import json

import numpy as np

from .io import HERE, save
from .loss_config import ARMS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize():
    arms = ('e151_best45', *ARMS)
    locations = {arm: HERE / 'runs' / f'{arm}_seed0' / 'evaluation_no_mib_dev500' for arm in arms}
    rows = {arm: [read(path / 'cases' / f'{i:04d}.json') for i in range(500)] for arm, path in locations.items()}
    baseline = rows[arms[0]]
    for arm in arms[1:]:
        if [(r['member'], r['L8_sha256']) for r in rows[arm]] != [(r['member'], r['L8_sha256']) for r in baseline]:
            raise ValueError('L8 input identity changed between S2 variants')

    def values(data, name):
        return np.asarray([row['s2_geometry'][name] for row in data])

    order = sorted(range(500), key=lambda i: (baseline[i]['s2_geometry']['L8_overlap'], i))
    lowest_overlap = order[:125]
    summaries = {}
    for arm, data in rows.items():
        overlap = values(data, 'L12_overlap')
        before = values(data, 'L8_overlap')
        delta = overlap - before
        summaries[arm] = dict(quality=read(locations[arm] / 'results.json'),
            L8_mean=float(before.mean()), L10_mean=float(values(data,'L10_overlap').mean()),
            L12_mean=float(overlap.mean()), L12_p95=float(np.quantile(overlap,.95)),
            improved=int((delta < -1e-9).sum()), worsened=int((delta > 1e-9).sum()),
            mean_position_change=float(values(data,'mean_position_change').mean()),
            p95_case_displacement=float(np.quantile(values(data,'p95_position_change'),.95)),
            mean_ratio_change=float(values(data,'mean_ratio_change').mean()),
            contact_counts=[int(values(data,'entry_group_contacts').sum()), int(values(data,'retained_group_contacts').sum())],
            boundary_counts=[int(values(data,'entry_boundaries').sum()), int(values(data,'retained_boundaries').sum())],
            low_overlap125=dict(L8_mean=float(before[lowest_overlap].mean()),
                L12_mean=float(overlap[lowest_overlap].mean()),
                worsened=int((delta[lowest_overlap] > 1e-9).sum()),
                mean_position_change=float(values(data,'mean_position_change')[lowest_overlap].mean())),
            loss_at_fixed_end_and_best=None if arm == arms[0] else read(locations[arm].parent/'loss_comparison_points.json'))
    save(HERE / 'comparison.json', dict(status='complete', count=500, L8_exact_match=True,
         low_overlap_dev_indices=lowest_overlap, variants=summaries))
    lines = ['# E152 固定 dev500 对照', '',
             '三组逐题 L8 坐标哈希一致。下表为各组冻结检查点；重叠按成对交叠面积之和除公开总面积计算。', '',
             '| 模型 | L8 重叠 | L10 重叠 | L12 重叠 | L12 P95 | 最低重叠 125 例的 L12 | 平均归一位移 | 硬合法 | 混合 Q |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in summaries.items():
        quality = row['quality']
        lines.append(f'| {arm} | {row["L8_mean"]:.6f} | {row["L10_mean"]:.6f} | {row["L12_mean"]:.6f} | '
                     f'{row["L12_p95"]:.6f} | {row["low_overlap125"]["L12_mean"]:.6f} | {row["mean_position_change"]:.6f} | '
                     f'{quality["hard_legal"]}/500 | {quality["mixed_Q"]:.6f} |')
    lines += ['', '逐题记录、接触和贴边保留、训练终轮和各自最佳监测点见 comparison.json 和各组 run。', '']
    (HERE / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
