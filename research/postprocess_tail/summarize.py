"""Aggregate the fixed paired panel and write per-case comparison evidence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scoring.official import official_evaluator
from .benchmark import PLAN, FROZEN


def load(folder, index):
    with np.load(folder / f'{index:04d}.npz', allow_pickle=False) as saved:
        metrics = json.loads(saved['metrics'].item())
        if metrics['i'] != index:
            raise ValueError(f'Different case at index {index}')
        return saved['draft'], metrics


def score_summary(rows, seconds):
    qs = np.array([row['q'] for row in rows])
    weighted = official_evaluator().compute_total_score(qs.tolist(), [row['n'] for row in rows])
    soft_counts = [row['group'] + row['boundary'] for row in rows]
    return dict(
        hard_legal=len(rows),
        mean_q=float(qs.mean()),
        weighted_q=weighted,
        mixed_q=(qs.mean() + weighted) / 2,
        p95_q=float(np.percentile(qs, 95)),
        p99_q=float(np.percentile(qs, 99)),
        max_q=float(qs.max()),
        over_q_1_10=int((qs > 1.10).sum()),
        group_violations=sum(row['group'] for row in rows),
        boundary_violations=sum(row['boundary'] for row in rows),
        group_satisfied=sum(row['group'] == 0 for row in rows),
        boundary_satisfied=sum(row['boundary'] == 0 for row in rows),
        all_soft_satisfied=soft_counts.count(0),
        mean_repair_ms=float(np.mean(seconds) * 1000),
        p95_repair_ms=float(np.percentile(seconds, 95) * 1000),
        mean_complete_ms=float(np.mean([value + row['draft_seconds']
                                        for value, row in zip(seconds, rows)]) * 1000),
        p95_complete_ms=float(np.percentile([value + row['draft_seconds']
                                             for value, row in zip(seconds, rows)], 95) * 1000),
    )


def compare(pairs):
    original, candidate = zip(*pairs)
    # Select the tail using the original scores, then evaluate the same samples.
    tail_size = max(1, len(pairs) // 20)
    tail = sorted(pairs, key=lambda pair: (-pair[0]['q'], pair[0]['i']))[:tail_size]
    delta = np.array([new['q'] - old['q'] for old, new in pairs])
    old_summary = score_summary(original, [row['seconds'] for row in original])
    new_summary = score_summary(candidate, [old['seconds'] + new['seconds'] for old, new in pairs])
    for key, index in [('original', 0), ('candidate', 1)]:
        summary = old_summary if index == 0 else new_summary
        summary['original_worst_5pct_mean_q'] = float(np.mean([row[index]['q'] for row in tail]))
    return dict(
        count=len(pairs), original=old_summary, candidate=new_summary,
        improved=int((delta < -1e-9).sum()),
        tied=int((np.abs(delta) <= 1e-9).sum()),
        regressed=int((delta > 1e-9).sum()),
        group_regressed=sum(new['group'] > old['group'] for old, new in pairs),
        boundary_regressed=sum(new['boundary'] > old['boundary'] for old, new in pairs),
        mean_additional_ms=float(np.mean([new['seconds'] for new in candidate]) * 1000),
        p95_additional_ms=float(np.percentile([new['seconds'] for new in candidate], 95) * 1000),
        counts={key: sum(new[key] for new in candidate) for key in
                ('legal_seed_attempted', 'legal_seed_accepted', 'anchor_seed_attempted',
                 'anchor_seed_accepted', 'soft_moves')},
    )


def case_row(old, new):
    return dict(
        index=old['i'], blocks=old['n'], original_q=old['q'], candidate_q=new['q'],
        delta_q=new['q'] - old['q'], original_grouping=old['group'],
        candidate_grouping=new['group'], original_boundary=old['boundary'],
        candidate_boundary=new['boundary'], original_bbox_area=old['area'],
        candidate_bbox_area=new['area'], original_hpwl=old['wire'],
        candidate_hpwl=new['wire'], original_repair_ms=1000 * old['seconds'],
        additional_repair_ms=1000 * new['seconds'], original_main_legal=old['legal_main'],
        legal_seed_accepted=new['legal_seed_accepted'], anchor_seed_accepted=new['anchor_seed_accepted'],
        soft_moves=new['soft_moves'],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    pairs = []
    for i in range(len(PLAN['members'])):
        original_draft, original = load(args.original, i)
        candidate_draft, candidate = load(args.candidate, i)
        if not np.array_equal(original_draft, candidate_draft):
            raise ValueError(f'Draft differs for case {i}')
        pairs.append((original, candidate))

    monitor = set(PLAN['monitor500_indices'])
    groups = dict(
        full2000=pairs,
        monitor500=[pair for pair in pairs if pair[0]['i'] in monitor],
        extension1500=[pair for pair in pairs if pair[0]['i'] not in monitor],
        original_main_failure=[pair for pair in pairs if not pair[0]['legal_main']],
    )
    for low, high in [(21, 59), (60, 89), (90, 109), (110, 120)]:
        groups[f'{low}-{high}_blocks'] = [pair for pair in pairs if low <= pair[0]['n'] <= high]
    summary = {key: compare(group) for key, group in groups.items() if group}
    full = summary['full2000']
    old, new = full['original'], full['candidate']
    summary['qualification'] = bool(
        new['hard_legal'] == old['hard_legal'] == 2000
        and all(new[key] < old[key] for key in
                ('mean_q', 'weighted_q', 'mixed_q', 'p95_q', 'p99_q', 'max_q',
                 'original_worst_5pct_mean_q', 'group_violations', 'boundary_violations'))
        and not (full['regressed'] or full['group_regressed'] or full['boundary_regressed'])
        and all(group['candidate']['mixed_q'] < group['original']['mixed_q']
                for name, group in summary.items() if name != 'original_main_failure')
    )
    summary['checkpoint_sha256'] = FROZEN['checkpoint_sha256']
    summary['payload_checksums_sha256'] = PLAN['payload_checksums_sha256']
    summary['worst_cases'] = [
        case_row(*pair) for pair in sorted(pairs, key=lambda pair: (-pair[0]['q'], pair[0]['i']))[:10]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    rows = [case_row(*pair) for pair in pairs]
    with (args.out / 'ICCAD_postprocess_dev2000.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(full, indent=2))
    print('Qualified:', summary['qualification'])


if __name__ == '__main__':
    main()
