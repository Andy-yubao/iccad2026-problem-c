"""Finalize the user-stopped evaluation using saved records only; no inference."""
from pathlib import Path
import collections
import hashlib
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scoring.evaluation import summarize_variants

run = Path(__file__).resolve().parent
raw = (run / 'test_cases.jsonl').read_bytes()
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
manifest = json.loads((ROOT / 'protocol/workset_v2.json').read_text(encoding='utf-8'))
count = len(rows)
assert count and len({row['case_id'] for row in rows}) == count
metrics = [r['variants']['deterministic']['metrics'] for r in rows]
records = [r['variants']['deterministic'] for r in rows]
summary = summarize_variants(rows, ['deterministic'])
s = summary['deterministic']
s['fallback_count'] = sum(v.get('fallback', False) for v in records)
s['mean_solve_seconds'] = s['total_solve_seconds'] / count
supervised = json.loads((run / 'supervised.json').read_text(encoding='utf-8'))
supervised['local_test'] = {
    'sample_count': count, 'requested_count': 2500, 'complete': False,
    **{k: sum(r['supervised'][k] for r in rows) / count
       for k in ('mse', 'center_mse', 'log_ratio_mse')},
    'aggregation': 'mean per-layout masked MSE',
    'scope': 'all durably completed cases before user stop; ordered prefix, not representative full workset',
}
workers = collections.Counter(r['source']['shard'].split('/')[0] for r in rows)
def quantiles(values):
    return {k: float(np.quantile(values, q)) for k, q in
            [('min',0),('p10',.1),('median',.5),('p90',.9),('p95',.95),('max',1)]}
def corr(a, b):
    x = np.asarray(a); y = np.asarray(b)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
costs = [m['cost'] for m in metrics]
sizes = [r['input_summary']['block_count'] for r in rows]
weights = np.exp((np.asarray(sizes)-max(sizes))/12)
displacements = [v['repair_displacement'] for v in records if 'repair_displacement' in v]
stats = {
    'test_record_sha256': hashlib.sha256(raw).hexdigest(),
    'coverage': {'workers_count': len(workers), 'workers_total': 100,
                 'worker_case_counts': dict(workers), 'full_workers_25_cases': sum(n==25 for n in workers.values()),
                 'first_case': rows[0]['case_id'], 'last_case': rows[-1]['case_id'],
                 'block_count_mean': float(np.mean(sizes)), 'block_count_quantiles': quantiles(sizes),
                 'warning': 'Original member order groups workers. This interrupted prefix is not a balanced random500 panel; no subset selected by score.'},
    'quality_quantiles': quantiles(costs), 'solve_seconds_quantiles': quantiles([v['solve_seconds'] for v in records]),
    'mean_violations_relative': float(np.mean([m['violations_relative'] for m in metrics])),
    'capped_feasible_cost_count': sum(m['is_feasible'] and m['cost']>=9.999998 for m in metrics),
    'weight_effective_case_count': float(weights.sum()**2/(weights**2).sum()),
    'independent_feasible_count': sum(v.get('independent_hard_checks',{}).get('is_feasible',False) for v in records),
    'budget_overrun_count': sum(v.get('diagnostics',{}).get('budget_overrun',False) for v in records),
    'repair': {'case_count': len(displacements), **{
        k+'_mean': float(np.mean([d[k] for d in displacements])) for k in
        ('continuous_displacement_mean','final_displacement_mean','continuous_seconds','geometry_seconds')}},
    'descriptive_pearson_correlations': {
        'prediction_mse_vs_final_cost': corr([r['supervised']['mse'] for r in rows], costs),
        'center_mse_vs_final_cost': corr([r['supervised']['center_mse'] for r in rows], costs),
        'shape_mse_vs_final_cost': corr([r['supervised']['log_ratio_mse'] for r in rows], costs),
    },
    'size_bins': {},
}
for label,lo,hi in [('small_0_49',0,49),('medium_50_89',50,89),('large_90_plus',90,100000)]:
    subset = [r for r in rows if lo<=r['input_summary']['block_count']<=hi]
    if subset: stats['size_bins'][label] = summarize_variants(subset,['deterministic'])['deterministic']
cp = Path(config['checkpoint'])
actual_sha = hashlib.sha256(cp.read_bytes()).hexdigest()
assert actual_sha == config['identity']['checkpoint_sha256']
stats['checkpoint_hash_verified'] = True
training = ROOT / 'experiments/graph_proposal/runs/20261016_deterministic_workset_v2_seed0'
log = [json.loads(l) for l in (training/'run.log').read_text(encoding='utf-8').splitlines() if l.strip()]
vals = [r for r in log if r.get('phase')=='validation']
stats['training_validation_samples'] = [vals[i] for i in sorted(set([0,1,4,9,14,19,24,len(vals)-1])) if i<len(vals)]
progress = json.loads((run/'progress.json').read_text(encoding='utf-8'))
result = {
    'status': 'user_stopped_partial', 'stop_reason': 'user_requested_about500_cases_then_analysis',
    'test_cases': count, 'requested_test_cases': 2500, 'formal_complete': False,
    'summary': summary, 'supervised': supervised, 'analysis': stats,
    'frozen_checkpoint_sha256': actual_sha,
    'test_used_for_tuning': False, 'official_contest_score': None,
    'last_periodic_progress': progress,
    'stop_note': 'Process interrupted on request; all complete durable JSONL rows retained, no further inference. Directory name preserves original2500 intent.',
    'exposure_note': 'Completed test rows and possibly in-flight shard were accessed; full2500 completion is not claimed. Test results never select weights or solver settings.',
    'source_note': 'Evaluation source is frozen in source_snapshot.zip. Reporting helper plot_training.py was edited only after termination.',
}
for name, value in [('supervised.json',supervised),('results.json',result)]:
    (run/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'count':count,'summary':summary,'supervised':supervised,
                  'analysis':{k:v for k,v in stats.items() if k!='training_validation_samples'}},ensure_ascii=False,indent=2))
