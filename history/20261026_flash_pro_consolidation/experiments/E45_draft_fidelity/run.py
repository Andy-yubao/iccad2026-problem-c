"""Frozen E42 drafts: paired gate5 or user-run dev100 screening, no training."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
os.environ['E45_THREADS'] = '4'
import argparse
import json
import math
import sys
import time
import signal
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts, REGISTRY
from scoring.evaluation import evaluate_case
from .io import HERE, ROOT, sha, save, sources, snapshot, exclusive
from .model import GeometryModel
from .inference import predict
from .solver import optimize
from .canonical_native import warmup
from .backend.frozen.packing import prepare_problem, _dimensions
from .draft_anchor import DraftAnchor, quadratic

WEIGHTS = ROOT / 'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'
WEIGHT_SHA = '12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
GATE = [50, 368, 316, 443, 164]
PROFILES = {'anchor': ('control', 'anchor', 'local_release'),
            'staged': ('control', 'center_preserving', 'legalize_anchor'),
            'reset': ('control', 'state_reset', 'draft_reset'),
            'efficient_reset': ('control', 'draft_reset_fast'),
            'quality': ('control', 'budget_control', 'draft_reset_shape_first'),
            'full_quality': ('control','draft_reset_fast')}
STEADY = dict(relaxed=False, mode=0, adaptive=True, fast_budget=False, late_extension=True,
              plateau=True, robust_seeds=True, certified=True, quick_project=True,
              quadratic_project=True, incumbent_project=True, snap=1e-4)


def fidelity(draft, final, inputs):
    p = prepare_problem(*inputs)
    unit = np.sqrt(p.areas)
    delta = (final[:, :2] + final[:, 2:] / 2) - (draft[:, :2] + draft[:, 2:] / 2)
    return {'mean_center_move': float((np.linalg.norm(delta, axis=1) / unit).mean()),
            'max_center_move': float((np.linalg.norm(delta, axis=1) / unit).max()),
            'mean_abs_log_ratio_change': float(np.abs(np.log(final[:, 2] / final[:, 3]) -
                                                       np.log(draft[:, 2] / draft[:, 3])).mean())}


def algebra_checks(draft, inputs):
    p = prepare_problem(*inputs)
    w, h = _dimensions(p, draft[:, 2] / draft[:, 3], True)
    context = DraftAnchor.build(draft, np.column_stack((w, h)), p, True)
    n, rng = p.n, np.random.default_rng(0)
    xy = np.arange(2 * n).reshape(n, 2); dims = 2 * n + np.arange(2 * n).reshape(n, 2)
    values = np.concatenate((draft[:, :2].ravel(), context.target_wh.ravel())) / p.scale
    values += rng.normal(0, .01, len(values))
    matrix, linear = quadratic(p, context, xy, len(values), dims=dims)
    c, s = context.coefficients(p)
    def actual(x):
        center = x[xy] + .5 * x[dims]
        return float((c[:, None] * (center - context.target_center / p.scale) ** 2).sum()
                     + (s[:, None] * (x[dims] - context.target_wh / p.scale) ** 2).sum())
    direction = rng.normal(size=len(values)); direction /= np.linalg.norm(direction)
    eps = 1e-6
    numerical = (actual(values + eps * direction) - actual(values - eps * direction)) / (2 * eps)
    analytic = float(direction @ (matrix @ values + linear))
    assert abs(numerical - analytic) < 1e-7
    assert np.linalg.eigvalsh(matrix.toarray()).min() >= -1e-10
    # Fixed-shape projection uses the same original-center target.
    m2, q2 = quadratic(p, context, xy, 2 * n, fixed_wh=context.target_wh / p.scale)
    projected = values[xy].ravel()
    shape_values = np.zeros_like(values); shape_values[xy.ravel()] = projected
    shape_values[dims.ravel()] = context.target_wh.ravel() / p.scale
    full_gradient = np.asarray(matrix @ shape_values + linear)[xy.ravel()]
    assert np.allclose(m2 @ projected + q2, full_gradient, rtol=0, atol=1e-10)
    before = context.weights.copy()
    context.observe(draft, p); context.observe(draft, p)
    reduced = context.weights < before
    assert np.all(context.weights[~reduced] == before[~reduced])
    child = context.child()
    assert child.original_sha256 == context.original_sha256 and np.array_equal(child.target_center, context.target_center)
    assert not context.original.flags.writeable
    return {'quadratic_gradient_error': abs(numerical - analytic), 'positive_semidefinite': True,
            'projection_gradient_parity': True, 'original_draft_read_only': True,
            'rescue_keeps_original_target': True, 'simulated_released_blocks': int(reduced.sum()),
            'untouched_blocks_preserved': True}


def summarize(records, policies):
    result = {}
    for policy in policies:
        chosen = [r for r in records if policy in r['variants']]
        if not chosen: continue
        variants = [r['variants'][policy] for r in chosen]
        q = np.asarray([v['metrics']['cost'] for v in variants])
        n = np.asarray([r['n'] for r in chosen]); weights = np.exp((n - n.max()) / 12)
        times = np.asarray([v['complete_seconds'] for v in variants])
        result[policy] = {'count': len(chosen), 'hard_legal': sum(v['metrics']['is_feasible'] for v in variants),
            'weighted_Q': float(np.average(q, weights=weights)), 'mean_Q': float(q.mean()),
            'p95_Q': float(np.percentile(q, 95)), 'CVaR95_Q': float(np.sort(q)[-max(1, math.ceil(.05 * len(q))):].mean()),
            'max_Q': float(q.max()), 'complete_seconds_mean': float(times.mean()),
            'complete_seconds_p95': float(np.percentile(times, 95)), 'complete_seconds_max': float(times.max()),
            'over_2s': int((times > 2).sum()), 'mean_center_move': float(np.mean([v['fidelity']['mean_center_move'] for v in variants])),
            'mean_abs_log_ratio_change': float(np.mean([v['fidelity']['mean_abs_log_ratio_change'] for v in variants])),
            'rescue_cases': sum(v['rescue'] for v in variants)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('gate', 'screen'), required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--candidate', choices=('anchor', 'local_release','center_preserving','legalize_anchor','state_reset','draft_reset','draft_reset_fast','draft_reset_shape_first'), default='draft_reset_fast')
    parser.add_argument('--profile', choices=tuple(PROFILES), default='full_quality')
    parser.add_argument('--budget-mode',choices=('native','2s'),default='native')
    parser.add_argument('--wall-seconds', type=float, default=900)
    args = parser.parse_args()
    if '/' in args.tag or '\\' in args.tag or args.tag in ('.', '..'):
        parser.error('Tag must be a single new run name')
    if not 0 < args.wall_seconds <= 1200 or not math.isfinite(args.wall_seconds):
        parser.error('Cumulative pipeline protection must be <=1200 seconds')
    folder = HERE / 'runs' / args.tag
    data = ReferenceLayouts('dev_monitor500')
    if args.stage == 'gate':
        panel, policies = GATE, PROFILES[args.profile]
    else:
        order = sorted(range(500), key=lambda i: (data.records[i]['n'], i))
        panel, policies = [order[i] for i in range(2, 500, 5)], ('control',args.candidate)
    config = {'stage': args.stage, 'panel': panel, 'members': [data.records[i]['member'] for i in panel],
              'policies': list(policies), 'steady': STEADY, 'threads': 4, 'workers': 1,
              'full_case_soft_budget_seconds': None if args.budget_mode=='native' else 2., 'pipeline_wall_seconds': args.wall_seconds,
              'budget_mode':args.budget_mode,
              'control_budget': 'unmodified native adaptive/late/rescue budgets, no outer 2s clipping',
              'checkpoint_sha256': sha(WEIGHTS), 'registry_sha256': sha(REGISTRY),
              'timing': 'shared CPU input-feature+inference time charged equally plus actual per-arm postprocess; excludes loading/JIT/scoring/diagnostics',
              'source_version': 'E45.original_draft_reset.v6', 'test_read': False}
    if config['checkpoint_sha256'] != WEIGHT_SHA: raise ValueError('Frozen E42 weights changed')
    identity = {'config': config, 'sources': sources()}
    if args.stage=='screen':
        frozen_path=HERE/'runs/frozen_candidate.json'
        if not frozen_path.exists():raise ValueError('Freeze a candidate before the 100-case screen')
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        if args.candidate!=frozen['candidate'] or frozen['sources']!=identity['sources']:
            raise ValueError('Frozen candidate/source identity changed; do not silently reselect')
        if frozen['checkpoint_sha256']!=config['checkpoint_sha256'] or frozen['panel100']!=panel:
            raise ValueError('Frozen checkpoint/panel changed')
        if frozen['budget_mode']!=args.budget_mode:raise ValueError('Frozen timing/quality budget mode changed')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    np.random.seed(0); torch.manual_seed(0)
    with exclusive(folder / '.run.lock'):
        path = folder / 'results_local.json'
        if path.exists():
            if not args.resume: raise FileExistsError('Use --resume for the same run')
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['identity'] != identity: raise ValueError('Sources/config/data changed; use a new tag')
            if state['finished']:
                print('Complete; no evaluation repeated'); return
            if state.get('pending_attempt'):
                reservation=state['pending_attempt']['reserved_seconds']
                state['cumulative_pipeline_seconds']+=reservation
                state.setdefault('interrupted_attempt_charges',[]).append(state['pending_attempt'])
                state['pending_attempt']=None
        else:
            if args.resume: raise FileNotFoundError('No matching progress')
            snapshot(folder, identity)
            state = {'identity': identity, 'records': [], 'finished': False,
                     'cumulative_pipeline_seconds': 0., 'commands': [], 'implementation_checks': {},
                     'test_read': False, 'environment': {'python': sys.version, 'torch': torch.__version__,
                     'parallel': torch.__config__.parallel_info(), 'torch_interop': torch.get_num_interop_threads()}}
        started = time.perf_counter(); prior = state['cumulative_pipeline_seconds']
        state['commands'].append(sys.argv)
        def persist():
            state['cumulative_pipeline_seconds'] = prior + time.perf_counter() - started
            save(path, state)
            save(folder / 'summary.json', {'finished': state['finished'], 'stop_reason': state.get('stop_reason'),
                'cumulative_pipeline_seconds': state['cumulative_pipeline_seconds'],
                'variants': summarize(state['records'], policies), 'implementation_checks': state['implementation_checks'],
                'test_read': False, 'scope': 'adaptive development mechanism gate' if args.stage == 'gate' else 'fixed matched dev100 screen'})
        persist()
        def interrupted(signum,frame):
            state['stop_reason']='user_interrupt';state['pending_attempt']=None
            persist();raise KeyboardInterrupt
        signal.signal(signal.SIGINT,interrupted)
        state['pending_attempt']={'kind':'initialization','reserved_seconds':60.};persist()
        warm_started = time.perf_counter(); warmup()
        state['warmup_seconds'] = time.perf_counter() - warm_started
        checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
        model = GeometryModel(**checkpoint['model_config']).eval(); model.load_state_dict(checkpoint['model'])
        state['pending_attempt']=None;persist()
        for ordinal, index in enumerate(panel):
            if (HERE / 'runs/STOP').exists() or (folder / 'STOP').exists():
                state['stop_reason'] = 'STOP'; persist(); return
            if prior + time.perf_counter() - started >= args.wall_seconds:
                state['stop_reason'] = 'cumulative_wall_budget'; persist(); return
            # Raw data can be read for scoring, but only masked solver inputs and
            # the frozen model draft enter all three algorithms.
            row = data[index]; case = row['case']
            existing = next((r for r in state['records'] if r['index'] == index), None)
            if existing is None:
                draft, inference = predict(model, case.solver_inputs)
                existing = {'index': index, 'member': list(row['member']), 'n': case.block_count,
                            'draft': draft.tolist(), 'inference_seconds': inference, 'variants': {}}
                state['records'].append(existing)
            else:
                draft, inference = np.asarray(existing['draft']), existing['inference_seconds']
            if args.stage == 'gate' and not state['implementation_checks']:
                state['implementation_checks'] = algebra_checks(draft, case.solver_inputs)
            # Rotate execution order to reduce a systematic first-arm timing bias.
            rotated = list(policies)[ordinal % len(policies):] + list(policies)[:ordinal % len(policies)]
            for policy in rotated:
                if policy in existing['variants']: continue
                if prior + time.perf_counter() - started >= args.wall_seconds:
                    state['stop_reason'] = 'cumulative_wall_budget'; persist(); return
                state['pending_attempt']={'kind':'solve','index':index,'policy':policy,'reserved_seconds':2.5};persist()
                before = draft.copy(); begin = time.perf_counter()
                positions, stats = optimize(draft, case.solver_inputs, draft_policy='control' if policy=='budget_control' else policy,
                    preserve_centers=policy=='center_preserving',
                    deadline=None if policy=='control' or (args.budget_mode=='native' and policy!='budget_control') else begin + max(.005, 2. - inference), **STEADY)
                postprocess = time.perf_counter() - begin; complete = inference + postprocess
                if not np.array_equal(before, draft): raise AssertionError('Model draft was mutated')
                evaluation = evaluate_case(case, positions, complete)
                existing['variants'][policy] = {
                    'positions': positions.tolist(), 'metrics': evaluation['metrics'],
                    'hard_check': evaluation['independent_hard_checks'],
                    'complete_seconds': complete, 'postprocess_seconds': postprocess,
                    'fidelity': fidelity(draft, positions, case.solver_inputs),
                    'rescue': any('packed' in t['phase'] or 'rescue' in t['phase'] for t in stats['trace']),
                    'stats': stats}
                state['pending_attempt']=None
                persist()
                print(index, policy, 'Q', round(evaluation['metrics']['cost'], 6),
                      'seconds', round(complete, 4), flush=True)
        state['finished'], state['stop_reason'] = True, 'panel_complete'
        persist()
        from .report import report
        report(folder)
        if any(v['hard_legal'] != v['count'] for v in summarize(state['records'], policies).values()):
            raise AssertionError('A returned output failed hard legality')


if __name__ == '__main__':
    main()
