"""Independent copied E90 row evaluation and summary; frozen exact flash settings."""
import os,time
import numpy as np
import torch
from scoring.evaluation import evaluate_case
from ..data import load_rows
from .runtime.model import GeometryModel
from .runtime.inference import predict
from .diagnostics import geometry_checks
from .common import seed_all
STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,
            plateau=True,robust_seeds=True,certified=True,quick_project=True,
            quadratic_project=True,incumbent_project=True,snap=1e-4)

def initialize(checkpoint_path):
    global worker_model,worker_rows,worker_initialization
    from .runtime.canonical_native import warmup
    started=time.perf_counter();seed_all(0,2);warmup()
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    worker_model=GeometryModel(**state['model_config']).eval();worker_model.load_state_dict(state['model'])
    worker_rows=load_rows('dev_monitor500');worker_initialization=time.perf_counter()-started

def evaluate(index):
    row=evaluate_row(index,worker_rows[index],worker_model)
    row['resident_initialization_seconds']=worker_initialization
    return row

def evaluate_row(index, row, model):
    from .runtime.solver import optimize
    from .common import seed_all
    seed_all(index, 2)
    case = row['case']
    start = time.perf_counter()
    record = {'index': index, 'member': list(row['member']), 'n': case.block_count}
    try:
        draft, inference_seconds = predict(model, case.solver_inputs)
        record['draft_positions'] = draft.tolist()
        positions, stats = optimize(draft, case.solver_inputs, **STEADY)
        elapsed = time.perf_counter()-start
        # Scoring and diagnostics occur outside the complete-solve timer.
        final_score = evaluate_case(case, positions.tolist(), elapsed)
        raw_score = evaluate_case(case, draft.tolist(), inference_seconds)
        _, hard, diagnostic = geometry_checks(draft, case.solver_inputs)
        local = np.sqrt(np.asarray(case.solver_inputs[1], dtype=float))
        center_move = np.linalg.norm((positions[:, :2]+positions[:, 2:]/2)-
                                    (draft[:, :2]+draft[:, 2:]/2), axis=1)/local
        trace = stats.get('trace', [])
        record.update(seconds=elapsed, metrics=final_score['metrics'], raw_metrics=raw_score['metrics'],
                      raw_hard_legal=hard['is_feasible'], draft_diagnostics=diagnostic,
                      positions=positions.tolist(), draft_positions=draft.tolist(),
                      trace=trace, row_fallback=bool(stats['row_retained']),
                      rescue_used=any(any(s in str(r.get('phase', '')) for s in ('rescue', 'packed')) for r in trace),
                      inference_seconds=inference_seconds, postprocess_seconds=stats['seconds'],
                      mean_center_move_local=float(center_move.mean()), max_center_move_local=float(center_move.max()),
                      mean_abs_log_width_change=float(np.abs(np.log(positions[:, 2]/draft[:, 2])).mean()),
                      max_abs_log_width_change=float(np.abs(np.log(positions[:, 2]/draft[:, 2])).max()),
                      shape_candidate_count=sum(t.get('phase') == 'shape' for t in trace),
                      failed_shape_candidate_count=sum(t.get('phase') == 'shape' and not t.get('legal', False) for t in trace),
                      worker_pid=os.getpid())
    except Exception as error:
        record.update(seconds=time.perf_counter()-start, metrics={'cost': 10., 'is_feasible': False},
                      error=repr(error))
    return record

def summarize(records):
    q = np.asarray([r['metrics']['cost'] for r in records])
    n = np.asarray([r['n'] for r in records]); w = np.exp((n-n.max())/12)
    seconds = np.asarray([r['seconds'] for r in records])
    result = {'count': len(records), 'hard_legal': sum(r['metrics']['is_feasible'] for r in records),
              'mean_Q': float(q.mean()), 'weighted_Q': float(np.average(q, weights=w)),
              'p95_Q': float(np.percentile(q, 95)), 'max_Q': float(q.max()),
              'worst5pct_mean_Q': float(np.sort(q)[-max(1, int(np.ceil(.05*len(q)))):].mean()),
              'seconds_mean': float(seconds.mean()), 'seconds_p95': float(np.percentile(seconds, 95)),
              'row_fallbacks': sum(r.get('row_fallback', False) for r in records),
              'rescue_cases': sum(r.get('rescue_used', False) for r in records),
              'errors': sum('error' in r for r in records)}
    for key in ('hpwl_gap', 'area_gap', 'boundary_violations', 'grouping_violations', 'mib_violations'):
        returned = [r['metrics'][key] for r in records if key in r['metrics']]
        result[key+'_mean_returned'] = float(np.mean(returned)) if returned else None
    for key in ('inference_seconds', 'postprocess_seconds', 'mean_center_move_local', 'max_center_move_local'):
        returned = [r[key] for r in records if key in r]
        result[key+'_mean_returned'] = float(np.mean(returned)) if returned else None
    return result
