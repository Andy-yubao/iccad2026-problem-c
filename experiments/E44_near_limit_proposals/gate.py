"""Exactly five stage-A cases: three decisive fixtures and two frozen v5 devs.

No model training, bulk preparation, test access, or E25/E42 private imports.
Resume completes only missing variants with identical source/configuration.
"""
import argparse
import os
import sys
import time

PROCESS_STARTED = time.perf_counter()
# Set before importing numerical libraries; this is one process, <=4 CPU threads.
for KEY in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[KEY] = '4'

import json
import math
from pathlib import Path
import numpy as np
import torch
from datasets.cases import _build_case
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from .evidence import HERE, ROOT, sha, save, identity, snapshot, exclusive
from .problem import Problem, dimensions, legal_incumbent, hard_check
from .structure import propose, Structure, sequence_pair
from .geometry import realized_decisions
from .solver import StructuralSolver, Budget, optimize

E42 = ROOT / 'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json'
# Frozen before implementing the solver. The latter is the existing E42 worst-Q
# case; it is a disclosed regression diagnostic, not an unexposed dev selection.
MEMBERS = (('worker_17/layouts_6608.th', 68), ('worker_79/layouts_5040.th', 16))


def synthetic(name, areas, constraints, reference):
    ordinary = (torch.tensor(areas, dtype=torch.float32), torch.empty((0, 3)),
                torch.empty((0, 3)), torch.empty((0, 2)), torch.tensor(constraints))
    return _build_case('synthetic:' + name, {'split': 'synthetic', 'purpose': name},
                       ordinary, reference, None)


def fixtures():
    contact = synthetic('contact_bbox', [1.] * 4,
                        [[0, 0, 1, 1, b] for b in (9, 10, 5, 6)],
                        [[0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
    anchors = synthetic('anchor_and_boundary_conflict', [1.] * 3,
                        [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 3]],
                        [[0, 0, 1, 1], [1.5, 0, 1, 1], [0, 1, 1, 1]])
    mib = synthetic('incompatible_original_mib', [1., 4., 1.],
                    [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
                    [[0, 0, 1, 1], [1, 0, 2, 2], [0, 1, 1, 1]])
    return [contact, anchors, mib]


def load_cases():
    data = ReferenceLayouts('dev_monitor500')
    lookup = {tuple(r['member']): i for i, r in enumerate(data.records)}
    if any(member not in lookup for member in MEMBERS):
        raise ValueError('Frozen stage-A members left dev_monitor500')
    cases = fixtures() + [data[lookup[m]]['case'] for m in MEMBERS]
    prior = json.loads(E42.read_text(encoding='utf-8'))
    saved = {tuple(r['member']): r for r in prior['records'] if tuple(r['member']) in MEMBERS}
    if len(saved) != 2:
        raise ValueError('Missing E42 development diagnostics')
    return cases, saved


def score_record(case, pos, seconds, info):
    evaluated = evaluate_case(case, pos, seconds)
    # This is the only complete coordinate store. Compact summaries omit XYWH.
    return {**evaluated, 'seconds': seconds, 'info': info}


def run_fixture(case, budget):
    p = Problem.from_inputs(case.solver_inputs)
    wh, conflicts = dimensions(p)
    seed = legal_incumbent(p, wh)
    variants, checks = {}, {}
    if case.case_id.endswith('contact_bbox'):
        # Correct global structure with a deliberately expanded safety layout.
        seed = np.asarray(case.reference_positions, float)
        seed[:, :2] *= 2.
        structure = propose(p, seed)
        out, info = optimize(p, seed, structure, budget=budget)
        variants['exact'] = score_record(case, out, info['optimizer_seconds'], info)
        checks['exact_contact_tree_and_actual_bbox'] = (
            info['fixed_structure_feasible'] and info['accepted_updates'] > 0
            and not info['retracted'] and realized_decisions(out, structure.decisions)['all_realized']
            and variants['exact']['metrics']['grouping_violations'] == 0
            and variants['exact']['metrics']['boundary_violations'] == 0)
        checks['complete_separation'] = len(structure.separations) == p.n * (p.n - 1) // 2
        # Two permutations and incomplete coverage rejection use this same case.
        pair = Structure(p.n, sequence_pair([0, 1, 2, 3], [2, 0, 3, 1])).validate()
        checks['sequence_pair_complete_acyclic'] = len(pair.separations) == 6
        try:
            Structure(p.n, pair.separations[:-1]).validate()
            checks['missing_pair_rejected'] = False
        except ValueError:
            checks['missing_pair_rejected'] = True
    elif case.case_id.endswith('anchor_and_boundary_conflict'):
        structure = propose(p, seed)
        out, info = optimize(p, seed, structure, budget=budget)
        variants['retract_boundary'] = score_record(case, out, info['optimizer_seconds'], info)
        checks['infeasible_soft_boundary_retracted'] = bool(info['retracted']) and info['fixed_structure_feasible']
        # All separation graphs are acyclic, but the free width cannot fit
        # between anchors at 0 and 1.5. A DAG is therefore not a certificate.
        impossible = Structure(3, [(0, 1, 0), (0, 2, 0), (2, 1, 0)]).validate()
        out, info = optimize(p, seed, impossible, budget=budget)
        variants['reject_anchor_topology'] = score_record(case, out, info['optimizer_seconds'], info)
        checks['acyclic_but_anchor_infeasible_rejected'] = not info['fixed_structure_feasible']
        checks['incumbent_unchanged_after_failed_topology'] = np.array_equal(out, seed)
        checks['public_anchors_exact'] = np.array_equal(out[p.preplaced], p.specs[p.preplaced])
    else:
        out, info = optimize(p, seed, propose(p, seed), budget=budget)
        variants['hard_specs_first'] = score_record(case, out, info['optimizer_seconds'], info)
        checks['mib_conflict_reported'] = bool(conflicts)
        checks['fixed_dimensions_exact'] = np.array_equal(out[p.locked, 2:], p.specs[p.locked, 2:])
        checks['original_areas_not_edited'] = np.array_equal(p.areas, np.array([1., 4., 1.]))
        checks['incompatible_mib_remains_soft'] = variants['hard_specs_first']['metrics']['mib_violations'] > 0
    # Exercise the public boundary using the same fixture, without new cases.
    leaked = list(case.solver_inputs)
    free = np.flatnonzero(~p.locked)
    if len(free):
        leaked[-1] = np.asarray(leaked[-1]).copy()
        leaked[-1][free[0], :2] = 123.
        try:
            Problem.from_inputs(tuple(leaked))
            checks['unmasked_free_labels_rejected'] = False
        except ValueError:
            checks['unmasked_free_labels_rejected'] = True
    for record in variants.values():
        checks['all_returned_hard_legal'] = checks.get('all_returned_hard_legal', True) and record['metrics']['is_feasible']
    return {'variants': variants, 'checks': checks, 'mib_conflicts': conflicts}


def diagnostic_structure(p, geometry, origin):
    structure = propose(p, geometry, diagnostic_reference=True)
    structure.origin = origin
    if origin == 'existing_E42_feasible_structure_diagnostic':
        # Keep only contacts/boundaries actually realized in the saved legal
        # geometry. Thus an all-kept feasible solution is known BEFORE the LP.
        kept = []
        for d in structure.decisions:
            if not realized_decisions(geometry, [d])['all_realized']:
                continue
            if d.kind == 'contact':
                t = 1 - d.axis
                overlap = min(geometry[d.i, t] + geometry[d.i, t + 2],
                              geometry[d.j, t] + geometry[d.j, t + 2])
                overlap -= max(geometry[d.i, t], geometry[d.j, t])
                if overlap < .01 * min(geometry[d.i, t + 2], geometry[d.j, t + 2]):
                    continue
            kept.append(d)
        structure.decisions = kept
    return structure


def run_real_variant(case, saved, name, budget):
    started = time.perf_counter()
    p = Problem.from_inputs(case.solver_inputs)
    if name in ('public_exact', 'public_soft', 'public_sequence'):
        solver = StructuralSolver(budget=budget, contact_mode='soft' if name == 'public_soft' else 'exact',
                                  sequence=name == 'public_sequence')
        out = solver.solve(*case.solver_inputs)
        info = solver.last_info
    else:
        if name == 'reference_structure_diagnostic':
            wh, _ = dimensions(p)
            seed = legal_incumbent(p, wh)
            structure = diagnostic_structure(p, np.asarray(case.reference_positions, float), name)
        else:
            seed = np.asarray(saved['positions'], float)
            if not hard_check(p, seed)['is_feasible']:
                raise ValueError('Saved E42 incumbent is not hard legal')
            structure = diagnostic_structure(p, seed, 'existing_E42_feasible_structure_diagnostic')
        # Fixed-shape feasibility diagnostic: do not confuse inherited E42
        # geometry, original references, or their off-line cost with deployment.
        available = budget.seconds - (time.perf_counter() - started)
        if available <= .002:
            out, info = seed, {'status': 'diagnostic_preparation_budget_exhausted',
                               'fixed_structure_feasible': False, 'fallback_to_incumbent': True,
                               'lp_calls': 0}
        else:
            out, info = optimize(p, seed, structure,
                                 budget=Budget(available, budget.max_lp_calls, 0, budget.threads))
        info.update(diagnostic_only=True, inherited_layout_generation_seconds=saved['seconds']
                    if name != 'reference_structure_diagnostic' else None,
                    source_origin=structure.origin)
    seconds = time.perf_counter() - started
    return score_record(case, out, seconds, info)


def compact(records):
    result = []
    for record in records:
        row = {k: v for k, v in record.items() if k != 'variants'}
        row['variants'] = {}
        for name, value in record.get('variants', {}).items():
            info = value['info']
            row['variants'][name] = {'metrics': value['metrics'], 'seconds': value['seconds'],
                'lp_calls': info.get('lp_calls'), 'accepted_updates': info.get('accepted_updates', 0),
                'fixed_structure_feasible': info.get('fixed_structure_feasible', False),
                'retracted_count': len(info.get('retracted', [])),
                'fallback_to_incumbent': info.get('fallback_to_incumbent'),
                'full_budget_overrun_seconds': info.get('full_budget_overrun_seconds'),
                'attempt_statuses': [a['status'] for a in info.get('attempts', [])],
                'all_kept_feasible': any(a.get('hard_valid') and a.get('structure_valid')
                                         and a['decision_count'] == info.get('original_decision_count', -1)
                                         for a in info.get('attempts', []))}
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=.18, help='Complete per-variant design budget; overruns are reported')
    parser.add_argument('--wall-seconds', type=float, default=180, help='Total case-loop protection, cumulative across resume')
    args = parser.parse_args()
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 600:
        parser.error('This entry supports only the <=10 minute, five-case stage-A check')
    budget = Budget(seconds=args.seconds).validate()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    np.random.seed(0); torch.manual_seed(0)
    folder = args.output.resolve()
    if not folder.is_relative_to(HERE / 'runs'):
        parser.error('Output must be in E44/runs')
    config = {'seed': 0, 'threads': 4, 'budget': budget.__dict__, 'wall_seconds': args.wall_seconds,
              'members': [list(m) for m in MEMBERS], 'existing_e42_sha256': sha(E42),
              'fixture_count': 3, 'real_dev_count': 2, 'test_read': False,
              'selection': 'input-only objective; reference and inherited E42 arms are diagnostics only'}
    current_identity = identity(config)
    loop_started = time.perf_counter()
    with exclusive(folder / '.run.lock'):
        path = folder / 'results_local.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('Use --resume to complete the same frozen check')
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['identity'] != current_identity:
                raise ValueError('Source/configuration/data identity changed; use a new run')
            if state['finished']:
                print('Already complete; existing evidence retained without rerunning')
                return
        else:
            if args.resume:
                raise FileNotFoundError('No stage-A progress to resume')
            snapshot(folder, current_identity)
            state = {'identity': current_identity, 'records': [], 'finished': False,
                     'cumulative_loop_seconds': 0., 'command': sys.argv, 'test_read': False,
                     'environment': {'python': sys.version, 'torch': torch.__version__,
                                     'torch_threads': torch.get_num_threads(),
                                     'torch_interop_threads': torch.get_num_interop_threads(),
                                     'torch_parallel_info': torch.__config__.parallel_info(),
                                     'numerical_thread_environment': {key: os.environ[key] for key in
                                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                          'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
                                     'highs_version': __import__('highspy').Highs().version(),
                                     'versions': {name: __import__(name).__version__
                                                  for name in ('numpy', 'scipy', 'highspy', 'shapely')
                                                  if hasattr(__import__(name), '__version__')}}}
        previous = state['cumulative_loop_seconds']
        def persist():
            state['cumulative_loop_seconds'] = previous + time.perf_counter() - loop_started
            save(path, state)
            rows = compact(state['records'])
            checks = [value for row in rows for value in row.get('checks', {}).values()]
            save(folder / 'summary.json', {'finished': state['finished'], 'test_read': False,
                'fixture_checks_passed': bool(checks) and all(checks), 'case_count': len(rows),
                'stop_reason': state.get('stop_reason'), 'cumulative_loop_seconds': state['cumulative_loop_seconds'],
                'records': rows, 'scope': 'five-case mechanism diagnosis; no generalization or target claim'})
        persist()
        cases, prior = load_cases()
        state['process_setup_seconds'] = time.perf_counter() - PROCESS_STARTED
        for index, case in enumerate(cases):
            if index >= len(state['records']):
                state['records'].append({'case_id': case.case_id, 'source': case.source,
                                         'input_summary': case.input_summary(), 'variants': {}})
            row = state['records'][index]
            if index < 3:
                names = ['fixture']
            else:
                names = ['public_exact', 'public_soft', 'public_sequence',
                         'reference_structure_diagnostic', 'feasible_structure_diagnostic']
            for name in names:
                if row.get('fixture_complete') or name in row['variants']:
                    continue
                if previous + time.perf_counter() - loop_started >= args.wall_seconds:
                    state['stop_reason'] = 'cumulative_wall_budget'
                    persist(); print('Stopped at cumulative budget; progress saved'); return
                if index < 3:
                    row.update(run_fixture(case, budget)); row['fixture_complete'] = True
                else:
                    saved = prior[tuple(case.source[k] for k in ('shard', 'sample_index'))]
                    row['variants'][name] = run_real_variant(case, saved, name, budget)
                persist()
                print(f'{case.case_id} {name} saved', flush=True)
        state['finished'], state['stop_reason'] = True, 'five_case_check_complete'
        persist()
        if not json.loads((folder / 'summary.json').read_text(encoding='utf-8'))['fixture_checks_passed']:
            raise AssertionError('A decisive fixture check failed; inspect saved evidence')
        print('Five cases complete; all fixture checks passed. Real-dev diagnostics require interpretation.')


if __name__ == '__main__':
    main()
