"""Conservative official-Q lower bound from saved public MIB specifications.

The bound accounts for the official round(w,4), round(h,4) shape definition;
plain disjoint area intervals alone are insufficient without dimension bounds.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from datasets.paths import ROOT
from .records import save, read_rows, run_path, provenance, snapshot, sha

SOURCE = ROOT / 'experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0/evaluation/cases.jsonl'
SELECTED = ROOT / 'experiments/E8_fast_robust_postprocess/runs/20261018_robust_dev100_seed0/cases.jsonl'


def interval_stabbing(intervals):
    """Minimum points covering closed 1-D intervals, with a greedy certificate."""
    points = []
    for lower, upper in sorted(intervals, key=lambda row: row[1]):
        if not points or points[-1] < lower:
            points.append(float(upper))
    return points


def case_bound(areas, constraints, public_specs, baseline_area, cap=10 - 1e-6):
    areas, c, specs = map(lambda v: np.asarray(v, dtype=float), (areas, constraints, public_specs))
    locked = (c[:, 0] != 0) | (c[:, 1] != 0)
    pre = c[:, 1] != 0
    nsoft = int((c[:, 4] != 0).sum())
    for col in [2, 3]:
        nsoft += sum(int((c[:, col] == g).sum()) - 1 for g in np.unique(c[:, col]) if g > 0)
    nsoft = max(1, nsoft)
    hard_tol = .000100001
    min_width = max(0., float(specs[locked, 2].max()) - hard_tol) if locked.any() else 0.
    min_height = max(0., float(specs[locked, 3].max()) - hard_tol) if locked.any() else 0.
    if pre.any():
        span = (specs[pre, :2] + specs[pre, 2:]).max(0) - specs[pre, :2].min(0)
        min_width = max(min_width, float(span[0]) - 3 * hard_tol)
        min_height = max(min_height, float(span[1]) - 3 * hard_tol)
    # If uncapped Q < cap, bbox/baseline < 2*cap-1. If capped or
    # infeasible, Q >= any resulting lower bound after min(..., cap).
    maximum_bbox = (2 * cap - 1) * baseline_area
    if min_width > 0 and min_height > 0:
        width_bound, height_bound = maximum_bbox / min_height, maximum_bbox / min_width
        # A conservative extra quantum covers roundoff near round4 bucket edges.
        delta = .000100001
        area_spread_bound = delta * (width_bound + height_bound) + delta * delta
    else:
        width_bound = height_bound = area_spread_bound = None
    groups = []
    total = 0
    for group in np.unique(c[:, 2]):
        if group <= 0:
            continue
        ids = np.flatnonzero(c[:, 2] == group)
        lower, upper = .99 * areas[ids], 1.01 * areas[ids]
        rigid = locked[ids]
        # Fixed/preplaced dimensions may differ from specs by 1e-4 per
        # official hard tolerance. Widen their product interval accordingly.
        if rigid.any():
            wh = specs[ids[rigid], 2:]
            lower[rigid] = np.maximum(0, wh[:, 0] - .000100001) * np.maximum(0, wh[:, 1] - .000100001)
            upper[rigid] = (wh[:, 0] + .000100001) * (wh[:, 1] + .000100001)
        exact_points = interval_stabbing(list(zip(lower, upper)))
        if area_spread_bound is not None:
            radius = area_spread_bound / 2
            relaxed = list(zip(lower - radius, upper + radius))
            points = interval_stabbing(relaxed)
            count = len(points) - 1
        else:
            relaxed, points, count = [], [], 0
        total += count
        groups.append({'group': int(group), 'blocks': ids.tolist(),
                       'area_intervals': list(map(list, zip(lower.tolist(), upper.tolist()))),
                       'exact_dimensions_min_shapes': len(exact_points),
                       'round4_relaxed_intervals': list(map(list, relaxed)), 'stabbing_points': points,
                       'round4_min_mib_violations': count})
    bound = min(cap, math.exp(2 * total / nsoft))
    return {'Q_lower_bound': bound, 'mib_violation_lower_bound': total, 'nsoft': nsoft,
            'bbox_area_limit_for_uncapped_Q': maximum_bbox,
            'public_bbox_width_lower': min_width, 'public_bbox_height_lower': min_height,
            'width_upper': width_bound, 'height_upper': height_bound,
            'same_round4_shape_area_spread_upper': area_spread_bound, 'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = run_path(args.output); output.mkdir(parents=True, exist_ok=False)
    config = {**provenance(), 'source': str(SOURCE), 'source_file_sha256': sha(SOURCE),
              'selected': str(SELECTED), 'selected_file_sha256': sha(SELECTED),
              'dataset_tensor_reads': 0, 'test_payload_reads': 0,
              'note': 'Saved public areas/constraints; masked locked specs reconstructed only from publicly allowed reference channels. No solver use of free reference geometry.'}
    save(output / 'config.json', config); snapshot(output, config['source_sha256'])
    cached = {r['case_id']: r for r in read_rows(SOURCE)}
    selected = read_rows(SELECTED)
    rows = []
    for row in selected:
        saved = cached[row['case_id']]
        c = np.asarray(saved['public_constraints']); reference = np.asarray(saved['reference'])
        specs = np.full_like(reference, -1.)
        locked = (c[:, 0] != 0) | (c[:, 1] != 0)
        specs[locked, 2:] = reference[locked, 2:]
        specs[c[:, 1] != 0] = reference[c[:, 1] != 0]
        metric = row['variants']['conic_robust']['final']['metrics']
        result = case_bound(saved['area_targets'], c, specs, metric['bbox_area_baseline'])
        if result['nsoft'] != max(1, metric['max_possible_violations']):
            raise AssertionError('Soft denominator mismatch')
        if result['Q_lower_bound'] > metric['cost'] + 1e-7:
            raise AssertionError('Bound contradicted by existing output')
        rows.append({'case_id': row['case_id'], 'block_count': row['block_count'], 'E8_Q': metric['cost'], **result})
    from scoring.official import official_evaluator
    lower = [r['Q_lower_bound'] for r in rows]
    summary = {'n': len(rows), 'Q_mean_lower_bound': float(np.mean(lower)),
               'Q_weighted_lower_bound': float(official_evaluator().compute_total_score(lower, [r['block_count'] for r in rows])),
               'E8_Q_mean': float(np.mean([r['E8_Q'] for r in rows])),
               'cases_with_dimension_bounds': sum(r['width_upper'] is not None for r in rows),
               'mib_violations_existing': sum(r['variants']['conic_robust']['final']['metrics']['mib_violations'] for r in selected),
               'mib_violations_lower_bound': sum(r['mib_violation_lower_bound'] for r in rows)}
    save(output / 'results.json', {'status': 'completed', 'summary': summary, 'cases': rows,
                                  'dataset_tensor_reads': 0, 'test_payload_reads': 0})
    lines = ['# E9：固定 dev100 的保守质量下界', '',
             f'固定 dev_quality100 的 Q 算术均值下界为 **{summary["Q_mean_lower_bound"]:.6f}**；当前 E8 为 {summary["E8_Q_mean"]:.6f}。该界仅使用 MIB 必须产生的罚项，忽略其他所有损失，因此是乐观下界，不是已实现的算法成绩。', '',
             '## 推导', '',
             '1. 官方 Q 的几何项和软约束项均不小于 1。若未封顶且 Q<C=9.999999，则 bbox 面积小于 (2C−1)×参考 bbox 面积；封顶或硬不可行时 Q 本身已不低于 C。',
             '2. fixed/preplaced 的公开尺寸和锚点跨度给出 bbox 宽、高的正下界，进而得到上述 Q 范围内每个矩形宽、高的上界。尺寸容许误差在面积区间中放宽；跨度下界另作保守容差处理。',
             '3. 官方将宽高各自四舍五入到四位小数后判同形。因此同一形状桶内两个矩形的面积差不超过 δ×(宽上界＋高上界)＋δ²，δ=0.000100001。不能忽略这一规则，直接把精确面积区间当成无条件下界。',
             '4. 将各模块的合法面积区间向两侧各放宽该面积差上界的一半，求覆盖这些区间所需的最少点数 k。任何一种合法同形分组至少需要 k 类，官方 MIB 违反至少 k−1。固定块按其公开尺寸容差计算面积区间。',
             '5. 忽略边界、连通、线长及面积损失，逐例仍有 Q≥min(C, exp(2×MIB下界/N_soft))。没有公开尺寸下界的实例保守取 Q≥1。最后对同一固定面板求算术均值。', '',
             '该下界针对当前官方计分和固定本地面板，不是官方隐藏集成绩，也不是模型能力上限的经验外推。完整区间及覆盖点证书见 results.json；全部来自既有保存记录，没有新增张量读取。', '']
    (output / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    import json
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
