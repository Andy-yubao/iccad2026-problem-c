"""Decompose saved E8 Q without dataset reads or solver execution."""
import argparse
import math
from pathlib import Path
import numpy as np
from datasets.paths import ROOT
from .records import save, read_rows, run_path, sha, provenance, snapshot

RUNS = ROOT / 'experiments/E8_fast_robust_postprocess/runs'
SOURCES = {'dev100': RUNS / '20261018_robust_dev100_seed0',
           'confirm100': RUNS / '20261018_robust_confirmation100_seed0'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {**provenance(), 'source_runs': {k: str(v) for k, v in SOURCES.items()},
              'sources': {k: sha(v / 'cases.jsonl') for k, v in SOURCES.items()},
              'test_payload_reads': 0, 'dataset_tensor_reads': 0,
              'method': 'Exact log Q decomposition for feasible non-capped outputs; component removals are score counterfactuals only.'}
    save(output / 'config.json', config)
    snapshot(output, config['source_sha256'])
    summary = {}
    for panel, source in SOURCES.items():
        rows = read_rows(source / 'cases.jsonl')
        data = []
        for row in rows:
            metrics = row['variants']['conic_robust']['final']['metrics']
            if not metrics['is_feasible'] or metrics['cost'] >= 9.999999:
                raise ValueError('Log decomposition requires feasible uncapped Q')
            geom = 1 + .5 * (max(0, metrics['hpwl_gap']) + max(0, metrics['area_gap']))
            den = max(1, metrics['max_possible_violations'])
            parts = {'geometry': math.log(geom), **{k: 2 * metrics[k + '_violations'] / den for k in ['boundary', 'grouping', 'mib']}}
            if abs(sum(parts.values()) - math.log(metrics['cost'])) > 1e-7:
                raise AssertionError('Official Q formula mismatch')
            data.append({'case_id': row['case_id'], 'block_count': row['block_count'], 'Q': metrics['cost'],
                         'log_parts': parts, 'hpwl_gap': metrics['hpwl_gap'], 'area_gap': metrics['area_gap'],
                         'soft_counts': {k: metrics[k + '_violations'] for k in ['boundary', 'grouping', 'mib']},
                         'Q_if_no_mib': metrics['cost'] * math.exp(-parts['mib']),
                         'Q_if_no_soft': geom})
        q = np.array([v['Q'] for v in data])
        log_means = {k: float(np.mean([v['log_parts'][k] for v in data])) for k in data[0]['log_parts']}
        summary[panel] = {'n': len(data), 'Q_mean': float(q.mean()),
                          'mean_log_parts': log_means,
                          'log_share': {k: v / sum(log_means.values()) for k, v in log_means.items()},
                          'mean_Q_if_no_mib': float(np.mean([v['Q_if_no_mib'] for v in data])),
                          'mean_Q_if_no_soft': float(np.mean([v['Q_if_no_soft'] for v in data])),
                          'mean_positive_hpwl_gap': float(np.mean([max(0, v['hpwl_gap']) for v in data])),
                          'mean_positive_area_gap': float(np.mean([max(0, v['area_gap']) for v in data])),
                          'mean_soft_counts': {k: float(np.mean([v['soft_counts'][k] for v in data])) for k in ['boundary', 'grouping', 'mib']},
                          'mib_nonzero_cases': sum(v['soft_counts']['mib'] > 0 for v in data),
                          'rows': data}
    save(output / 'results.json', {'status': 'completed', 'test_payload_reads': 0, 'dataset_tensor_reads': 0, 'panels': summary})
    lines = ['# E9：既有 E8 结果的质量损失分解', '',
             '仅重读已保存的两组开发结果，没有推理、求解或张量读取。下面的“去掉某项”只是在坐标不变时重算分数的反事实，不能解释为算法已实现或一定可实现的收益。', '',
             '| 面板 | 平均 Q | 几何 log 损失占比 | 边界占比 | 分组占比 | MIB 占比 | 假设无 MIB 的均值 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for panel, values in summary.items():
        share = values['log_share']
        lines.append(f'| {panel} | {values["Q_mean"]:.6f} | {share["geometry"]:.1%} | {share["boundary"]:.1%} | {share["grouping"]:.1%} | {share["mib"]:.1%} | {values["mean_Q_if_no_mib"]:.6f} |')
    lines += ['', 'log Q 中各项可精确相加；Q 本身的各项均值不能这样相加。分数拆解不是因果消融。完整逐例数值、正向线长/面积差和软违反数见 results.json。', '']
    (output / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json_summary(summary), flush=True)


def json_summary(summary):
    import json
    return json.dumps({k: {a: b for a, b in v.items() if a != 'rows'} for k, v in summary.items()}, ensure_ascii=False)


if __name__ == '__main__':
    main()
