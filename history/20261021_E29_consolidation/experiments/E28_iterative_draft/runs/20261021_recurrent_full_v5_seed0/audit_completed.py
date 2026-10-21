"""Read saved training/final records only; no model, data payload, repair or scoring."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    progress = read(HERE / 'progress.json')
    training = read(HERE / 'results.json')
    final = read(HERE / 'final_evaluation/results.json')
    full = read(HERE / 'full_dev.json')
    config = read(HERE / 'config.json')['identity']
    history = progress['history']
    first = history[0]
    best = next(h for h in history if h['epoch'] == progress['best_epoch'])
    records = final['records']
    assert len(records) == 500 and final['status'] == 'complete'
    assert final['checkpoint_sha256'] == full['checkpoint_sha256']
    checkpoint = HERE / 'checkpoints/best.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == final['checkpoint_sha256']
    mismatches = [name for name, digest in config['sources'].items()
                  if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    weights = [math.exp((r['block_count'] - 120) / 12) for r in records]
    total_weight = sum(weights)
    paired = {}
    for key in ('overlap_violations', 'area_violations', 'dimension_violations',
                'nonnegative_coordinate_violations', 'hpwl_gap', 'area_gap',
                'boundary_violations', 'grouping_violations', 'mib_violations'):
        before = [r['baseline_metrics'][key] for r in records]
        after = [r['metrics'][key] for r in records]
        paired[key] = {
            'baseline_mean': statistics.mean(before), 'candidate_mean': statistics.mean(after),
            'baseline_weighted_mean': sum(w * v for w, v in zip(weights, before)) / total_weight,
            'candidate_weighted_mean': sum(w * v for w, v in zip(weights, after)) / total_weight,
            'candidate_min': min(after), 'candidate_median': statistics.median(after), 'candidate_max': max(after),
            'improved_cases': sum(b < a for a, b in zip(before, after)),
            'worsened_cases': sum(b > a for a, b in zip(before, after))}
    proxies = {}
    for role in ('train_monitor', 'dev_monitor'):
        proxies[role] = {}
        for key in ('physics', 'penetration', 'overlap_area', 'crowding', 'contact', 'boundary', 'mib'):
            a, b = first[role][key], best[role][key]
            proxies[role][key] = {'initial': a, 'best': b, 'relative_change': b / a - 1 if a else None}
    result = {
        'status': 'complete_audit', 'training_status': training['status'],
        'epochs': training['epochs'], 'best_epoch': training['best_epoch'],
        'training_minutes': training['elapsed_seconds'] / 60,
        'lr_transitions': [{'epoch': h['epoch'], 'lr_next': h['lr_next']} for h in history if h['lr_used'] != h['lr_next']],
        'scheduler_final': progress['scheduler'], 'checkpoint_sha256': final['checkpoint_sha256'],
        'all_recorded_training_sources_current': True,
        'final_summary': final['summary'], 'full_dev_metrics': full['metrics'],
        'max_train_monitor_hard_legal_over_history': max(h['train_monitor']['raw_hard_legal'] for h in history),
        'max_dev_monitor_hard_legal_over_history': max(h['dev_monitor']['raw_hard_legal'] for h in history),
        'every_final_case_has_overlap': all(r['metrics']['overlap_violations'] > 0 for r in records),
        'any_final_area_dimension_or_nonnegative_failure': any(
            r['metrics'][key] != 0 for r in records for key in
            ('area_violations', 'dimension_violations', 'nonnegative_coordinate_violations')),
        'paired_geometry': paired, 'proxy_changes': proxies,
        'arms_completed': ['recurrent_full'], 'other_arms_evaluated': False,
        'baseline_timing_not_comparable': 'Stored baseline summary reuses candidate seconds; do not infer baseline speed',
        'training_results_full_dev_flag': 'Training-stage results.json predates full_dev.json; full-dev audit is complete',
        'new_inference_or_scoring': False, 'test_read': False,
        'conclusion': 'Public penalties reduce conflict severity, but current frozen-base local refiner does not produce a feasible layout, including on training monitoring cases. No raw-Q gain; no inference about model/data scaling or individual pin/recurrence effects.'}
    path = HERE / 'completion_audit.json'
    if path.exists():
        raise FileExistsError('Audit evidence already exists; reuse it')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'epochs': result['epochs'],
                      'best_epoch': result['best_epoch'], 'training_minutes': result['training_minutes'],
                      'train_monitor_max_hard_legal': result['max_train_monitor_hard_legal_over_history'],
                      'dev_monitor_max_hard_legal': result['max_dev_monitor_hard_legal_over_history'],
                      'source_identity_verified': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
