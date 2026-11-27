"""Paired summaries of already saved E45 outputs; no model or solver calls."""
from pathlib import Path
import argparse
import json
import csv
import math
import numpy as np
from .io import HERE, save


def report(folder):
    state = json.loads((folder / 'results_local.json').read_text(encoding='utf-8'))
    config = state['identity']['config']; stage = config['stage']
    policies = config['policies']; records = state['records']
    base = next((p for p in policies if p == 'control'), None)
    if base is None: raise ValueError('A same-entry control is required')
    rows, comparisons = [], {}
    for policy in policies:
        if policy == base: continue
        pairs = [r for r in records if base in r['variants'] and policy in r['variants']]
        if not pairs: continue
        n = np.asarray([r['n'] for r in pairs]); weights = np.exp((n - n.max()) / 12)
        bq = np.asarray([r['variants'][base]['metrics']['cost'] for r in pairs])
        cq = np.asarray([r['variants'][policy]['metrics']['cost'] for r in pairs])
        bt = np.asarray([r['variants'][base]['complete_seconds'] for r in pairs])
        ct = np.asarray([r['variants'][policy]['complete_seconds'] for r in pairs])
        bwq, cwq = float(np.average(bq, weights=weights)), float(np.average(cq, weights=weights))
        tail = max(1, math.ceil(.05 * len(pairs)))
        large_regression = (cq - bq > .05) & (cq > bq * 1.05)
        value = {'count': len(pairs), 'weighted_Q_control': bwq, 'weighted_Q_candidate': cwq,
            'weighted_Q_delta': cwq - bwq, 'relative_excess_Q_reduction': (bwq - cwq) / max(bwq - 1, 1e-12),
            'mean_Q_delta': float((cq - bq).mean()),
            'complete_time_control': float(bt.mean()), 'complete_time_candidate': float(ct.mean()),
            'complete_time_delta': float((ct - bt).mean()), 'time_ratio': float(ct.mean() / bt.mean()),
            'wins_ties_losses': [int((cq < bq - 1e-6).sum()), int((np.abs(cq - bq) <= 1e-6).sum()), int((cq > bq + 1e-6).sum())],
            'CVaR95_control': float(np.sort(bq)[-tail:].mean()), 'CVaR95_candidate': float(np.sort(cq)[-tail:].mean()),
            'max_Q_control': float(bq.max()), 'max_Q_candidate': float(cq.max()),
            'candidate_over_2s': int((ct>2).sum()),'two_second_target_met_in_this_panel':bool((ct<=2).all()),
            'severe_regression_indices': [r['index'] for r, flag in zip(pairs, large_regression) if flag],
            'candidate_hard_legal': sum(r['variants'][policy]['metrics']['is_feasible'] for r in pairs)}
        if stage == 'screen' and len(pairs) == 100:
            rng = np.random.default_rng(20261023)
            intervals = {}
            for cluster in ('shard', 'worker'):
                labels = [r['member'][0] if cluster == 'shard' else r['member'][0].split('/')[0] for r in pairs]
                groups = [np.flatnonzero(np.asarray(labels) == label) for label in sorted(set(labels))]
                draws = []
                for _ in range(2000):
                    ids = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
                    draws.append([np.average((cq - bq)[ids], weights=weights[ids]), (ct - bt)[ids].mean()])
                intervals[cluster] = {'weighted_Q_delta_95': np.percentile(np.asarray(draws)[:, 0], [2.5, 97.5]).tolist(),
                                      'complete_time_delta_95': np.percentile(np.asarray(draws)[:, 1], [2.5, 97.5]).tolist()}
            value['paired_cluster_bootstrap'] = intervals
            value['quality_and_mean_time_gate_passed'] = bool(value['candidate_hard_legal'] == 100
                and value['relative_excess_Q_reduction'] >= .05 and value['time_ratio'] <= 1.03
                and value['CVaR95_candidate'] <= value['CVaR95_control'] + 1e-8
                and value['max_Q_candidate'] <= value['max_Q_control'] + 1e-8
                and not value['severe_regression_indices'])
            value['eligible_for_later_confirmation']=bool(value['quality_and_mean_time_gate_passed'] and value['two_second_target_met_in_this_panel'])
        else:
            value['eligible_for_later_confirmation'] = False
        comparisons[policy] = value
        for r, old, new, oldt, newt in zip(pairs, bq, cq, bt, ct):
            b, c = r['variants'][base], r['variants'][policy]
            rows.append({'index': r['index'], 'shard': r['member'][0], 'sample_index': r['member'][1], 'n': r['n'],
                         'candidate': policy, 'control_Q': float(old), 'candidate_Q': float(new), 'Q_delta': float(new-old),
                         'control_complete_seconds': float(oldt), 'candidate_complete_seconds': float(newt),
                         'control_center_move': b['fidelity']['mean_center_move'], 'candidate_center_move': c['fidelity']['mean_center_move'],
                         'control_rescue': b['rescue'], 'candidate_rescue': c['rescue']})
    save(folder / 'paired_summary.json', {'finished': state['finished'], 'stage': stage, 'comparisons': comparisons,
         'test_read': False, 'interval_scope': 'paired case variation only; one timing run does not measure machine-state uncertainty'})
    if rows:
        with (folder / 'paired_cases.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# E45配对结果', '', '该表仅来自本次同草稿、同入口、同资源的已保存输出；开发诊断，不是正式比赛成绩。', '']
    if stage == 'gate': lines += ['五例是反复使用的机制反例，受大例权重影响；不将五例加权Q当作目标达成或泛化证据。', '']
    lines += ['| 候选 | 例数 | 算术Q变化 | 完整均时变化/秒 | Q胜/平/负 | 最差Q 控制→候选 |', '|---|---:|---:|---:|---|---|']
    for name, v in comparisons.items():
        lines.append(f"| {name} | {v['count']} | {v['mean_Q_delta']:+.6f} | {v['complete_time_delta']:+.6f} | {v['wins_ties_losses']} | {v['max_Q_control']:.6f}→{v['max_Q_candidate']:.6f} |")
    lines += ['', '精确分项、分片/worker聚类区间（100例时）、逐例退化与后续确认资格见[paired_summary.json](paired_summary.json)和[paired_cases.csv](paired_cases.csv)。',
              '', '计时包含CPU特征/模型/解码（同一草稿的实测时间各臂等量计入）与全部实际后处理；载入、JIT、评分及离线诊断另计。预算模式为'+str(config.get('budget_mode','legacy_2s'))+'；native保留原救援完整预算，逐例超过2秒仍在summary.json单列，不算两秒目标达成。2s模式只提供软截止。',
              '', '不会自动启动下一阶段，不以更换checkpoint、增加候选或重评挑最快结果形成收益。']
    (folder / 'paired_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--tag', required=True)
    args = parser.parse_args(); folder = (HERE / 'runs' / args.tag).resolve()
    if not folder.is_relative_to(HERE / 'runs'): raise ValueError('Outside E45 runs')
    print(json.dumps(report(folder), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
