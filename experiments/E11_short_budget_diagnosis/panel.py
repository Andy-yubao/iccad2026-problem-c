"""Frozen dev100 expansion: factorial separation/contact diagnostic, no training."""
import argparse
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import zipfile

from .run import (ROOT, HERE, DRAFTS, TEACHERS, Worker, sha, save, record,
                  np, torch, LayoutReader, workset_manifest, _build_case,
                  prepare_problem, topology, contact_tree, resolved_data_paths,
                  official_evaluator)

ARMS = {'raw_full': 5., 'raw_short': .4, 'raw_fixed': .4,
        'teacher_separation': .4, 'teacher_contact_tree': .4,
        'teacher_relations': .4, 'teacher_full_ceiling': .1}
TOLERANCE = .002


def summarize(rows):
    summary = {}
    for arm in ARMS:
        pairs = [(r, t) for r in rows for t in r['trials'] if t['arm'] == arm]
        if not pairs:
            continue
        trials = [t for _, t in pairs]
        def metric(key):
            values = [t['metrics'][key] for t in trials if key in t['metrics']]
            return float(np.mean(values)) if values else None
        def mean(key):
            values = [t[key] for t in trials if key in t]
            return float(np.mean(values)) if values else None
        costs = np.array([t['metrics']['cost'] for t in trials])
        times = np.array([t['charged_total_seconds'] for t in trials])
        paired = [(r, t) for r, t in pairs if 'baseline_q' in r]
        summary[arm] = {
            'count': len(trials), 'Q_mean': float(costs.mean()), 'Q_median': float(np.median(costs)),
            'Q_p95': float(np.quantile(costs, .95)), 'Q_max': float(costs.max()),
            'worst_10_percent_mean_Q': float(np.sort(costs)[-max(1, int(np.ceil(len(costs)*.1))):].mean()),
            'hard_legal_count': sum(t['metrics']['is_feasible'] for t in trials),
            'error_count': sum('error' in t for t in trials),
            'initial_row_retained_count': sum(not t.get('quality_curve_offline') and 'error' not in t for t in trials),
            'rescue_count': sum(t.get('stats', {}).get('rescue_used', False) for t in trials),
            'core_seconds_mean': mean('solve_seconds'), 'ipc_seconds_mean': mean('ipc_and_serialization_seconds'),
            'charged_total_seconds_mean': float(times.mean()), 'charged_total_seconds_p95': float(np.quantile(times, .95)),
            'charged_total_seconds_max': float(times.max()), 'over_05_seconds_count': int((times > .5).sum()),
            'conic_calls_mean': mean('conic_calls'), 'native_conic_seconds_mean': mean('native_conic_seconds'),
            'budget_overruns': sum(t['solve_seconds'] > ARMS[arm] for t in trials),
            'hpwl_gap_mean': metric('hpwl_gap'), 'area_gap_mean': metric('area_gap'),
            'boundary_mean': metric('boundary_violations'), 'grouping_mean': metric('grouping_violations'),
            'mib_mean': metric('mib_violations'), 'paired_count': len(paired),
            'Q_delta_mean_vs_live_full_raw': float(np.mean([t['metrics']['cost']-r['baseline_q'] for r, t in paired])) if paired else None,
            'strict_no_worse_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q']+1e-9 for r, t in paired),
            'within_002_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q']+TOLERANCE for r, t in paired),
            'quality_and_05_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q']+TOLERANCE
                                      and t['charged_total_seconds'] <= .5 for r, t in paired)}
    return summary


def enrich(row):
    baseline = next((t for t in row['trials'] if t['arm'] == 'raw_full'), None)
    if baseline:
        row['baseline_q'] = baseline['metrics']['cost']
        for trial in row['trials']:
            trial['Q_delta_vs_live_full_raw'] = trial['metrics']['cost']-row['baseline_q']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    begin = time.perf_counter()
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    if args.resume:
        old = json.loads((args.output/'results.json').read_text(encoding='utf-8'))
        if old['status'] == 'completed':
            raise ValueError('Completed runs cannot be resumed')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'cases').mkdir()
    manifest = workset_manifest()
    reader = LayoutReader(manifest, 'dev_quality')
    assert len(reader.members) == 100
    drafts = {r['case_id']: r for r in map(json.loads, DRAFTS.read_text(encoding='utf-8').splitlines())}
    teachers = {r['case_id']: r for r in map(json.loads, TEACHERS.read_text(encoding='utf-8').splitlines())}
    assert len(drafts) == len(teachers) == 100
    sources = sorted(HERE.rglob('*.py')) + [HERE/'README.md', HERE/'provenance.json']
    shared = [*sorted((ROOT/'datasets').glob('*.py')), *sorted((ROOT/'scoring').glob('*.py'))]
    import scipy
    import clarabel
    import highspy
    config = {'command': [sys.executable, *sys.argv], 'seed': 0, 'threads': 2,
              'started_local': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'device': 'CPU',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
              'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
              'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'clarabel': clarabel.__version__, 'highs': highspy.Highs().version(),
              'arms': ARMS, 'quality_tolerance': TOLERANCE, 'wall_cap_seconds': 900,
              'panel': 'frozen workset_v2.dev_quality, all 100 members, worker0..99 one each',
              'members': [{'shard': shard, 'sample_index': sample} for shard, sample in reader.members],
              'local_test_read': False, 'training': False,
              'data_paths': resolved_data_paths(),
              'split_sha256': sha(ROOT/'protocol/split_v1.json'), 'workset_sha256': sha(ROOT/'protocol/workset_v2.json'),
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources+shared},
              'cached_sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in (DRAFTS, TEACHERS)},
              'official_evaluator': {'path': official_evaluator().__file__, 'sha256': sha(official_evaluator().__file__)},
              'factorial': 'Fixed separation edges from raw/teacher; contact_tree seed from raw/teacher using the chosen separation edges. Only relation triples reach worker, except explicitly marked full-layout ceiling.',
              'timing_scope': 'Live core+IPC; cached original inference charged uniformly. Oracle construction, loading, warmup, scoring and plotting excluded. No oracle network or new inference speed claim.',
              'selection_scope': 'All100 cases and all7 arms fixed before expanded results; no outcome-dependent subset. Existing five included; this is development, not independent test.'}
    inv_path = args.output/'invocations.json'
    if args.resume:
        original = json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        for key in ('arms', 'members', 'cached_sources', 'split_sha256', 'workset_sha256', 'factorial'):
            if config[key] != original[key]:
                raise ValueError(f'Resume cannot change {key}')
        for path, digest in original['sources'].items():
            if '/backend/' in path or path.endswith('/worker.py'):
                if config['sources'].get(path) != digest:
                    raise ValueError('Resume cannot change solving algorithm')
        invocations = json.loads(inv_path.read_text(encoding='utf-8'))
        suffix = f'_resume{len(invocations)}'
    else:
        invocations = []; suffix = ''
    config_name = f'config{suffix}.json'; snapshot_name = f'source_snapshot{suffix}.zip'
    save(args.output/config_name, config)
    with zipfile.ZipFile(args.output/snapshot_name, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sources+shared:
            archive.write(p, p.relative_to(ROOT).as_posix())
    prior_elapsed = sum(i.get('elapsed_seconds', 0.) for i in invocations)
    invocations.append({'config': config_name, 'snapshot': snapshot_name, 'status': 'running'})
    save(inv_path, invocations)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.output/'cases').glob('*.json'))]
    by_id = {r['case_id']: r for r in rows}
    worker = Worker(args.output/f'worker{suffix}.log')
    status = 'running'; rng = random.Random(0); warmed = False
    try:
        for index in range(100):
            shard, sample = reader.members[index]
            case_id = f'dev:{shard}:{sample}'
            arms = list(ARMS); rng.shuffle(arms)
            row = by_id.get(case_id)
            if row is not None and len(row['trials']) == len(ARMS):
                continue
            raw = reader.case(index)
            assert raw.case_id == case_id
            case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6], raw.reference_positions, raw.stored_metrics)
            cache = drafts[case_id]
            teacher_record = teachers[case_id]['variants']['conic_robust']['final']
            assert teacher_record['metrics']['is_feasible']
            draft = np.asarray(cache['draft']); teacher = np.asarray(teacher_record['positions'])
            inputs = case.solver_inputs
            locked = np.asarray(inputs[5])[:, :2].any(1)
            assert np.all(np.asarray(inputs[6])[~locked] == -1)
            problem = prepare_problem(*inputs)
            construction_begin = time.perf_counter()
            raw_edges = topology(draft, problem.constraints); teacher_edges = topology(teacher, problem.constraints)
            assert all(teacher[i, a]+teacher[i, a+2] <= teacher[j, a]+1e-5 for i, j, a in teacher_edges)
            def guide(edges, seed):
                return {'edges': [[int(v) for v in e] for e in edges],
                        'contacts': [[int(v) for v in e] for e in contact_tree(seed, problem, edges)]}
            guides = {'raw_fixed': guide(raw_edges, draft),
                      'teacher_separation': guide(teacher_edges, draft),
                      'teacher_contact_tree': guide(raw_edges, teacher),
                      'teacher_relations': guide(teacher_edges, teacher)}
            construction_seconds = time.perf_counter()-construction_begin
            def pairset(tree):
                return {tuple(sorted(e[:2])) for e in tree}
            raw_pairs = pairset(guides['raw_fixed']['contacts']); teacher_pairs = pairset(guides['teacher_relations']['contacts'])
            if row is None:
                row = {'case_id': case_id, 'source': raw.source, 'panel_index': index, 'block_count': case.block_count,
                       'cached_teacher_q': teacher_record['metrics']['cost'], 'cached_inference_seconds': cache['inference_seconds'],
                       'relation_metadata': {'separation_disagreement_fraction': float(np.mean([a != b for a, b in zip(raw_edges, teacher_edges)])),
                            'raw_contact_count': len(raw_pairs), 'teacher_contact_count': len(teacher_pairs),
                            'contact_pair_jaccard': len(raw_pairs&teacher_pairs)/max(1, len(raw_pairs|teacher_pairs)),
                            'oracle_construction_seconds_excluded': construction_seconds},
                       'layouts': {}, 'trials': []}
                rows.append(row); by_id[case_id] = row
            serialized = [int(inputs[0]), *[np.asarray(v).tolist() for v in inputs[1:]]]
            completed = {t['arm'] for t in row['trials']}
            if not warmed:
                warmup = worker.call({'inputs': serialized, 'draft': draft.tolist(), 'seconds': .1})
                config['warmup_wall_seconds'] = warmup['worker_wall_seconds']; warmed = True
                save(args.output/config_name, config)
            for arm in arms:
                if arm in completed:
                    continue
                if prior_elapsed+time.perf_counter()-begin >= 900 or (args.output/'STOP').exists():
                    status = 'budget_stop' if not (args.output/'STOP').exists() else 'user_stop'
                    break
                response = worker.call({'inputs': serialized,
                         'draft': (teacher if arm == 'teacher_full_ceiling' else draft).tolist(),
                         'seconds': ARMS[arm], 'guidance': guides.get(arm)})
                trial = record(case, response, cache['inference_seconds'], row['layouts'])
                if 'charged_total_seconds' not in trial:
                    trial['charged_total_seconds'] = cache['inference_seconds']+response['worker_wall_seconds']
                trial.update(arm=arm, budget_seconds=ARMS[arm])
                row['trials'].append(trial)
                enrich(row)
                save(args.output/'cases'/f'{index:03d}.json', row)
            print(f"case {index+1}/100", case_id, ' '.join(f"{t['arm']}={t['metrics']['cost']:.4f}" for t in row['trials']), flush=True)
            save(args.output/'results.json', {'status': status, 'elapsed_seconds': prior_elapsed+time.perf_counter()-begin,
                 'completed_case_count': sum(len(r['trials']) == len(ARMS) for r in rows),
                 'trial_count': sum(len(r['trials']) for r in rows), 'summary': summarize(rows)})
            if status != 'running':
                break
        if status == 'running':
            status = 'completed'
    except Exception as exc:
        status = 'error'; config['fatal_error'] = repr(exc); raise
    finally:
        worker.close()
        save(args.output/config_name, config)
        invocations[-1].update(status=status, elapsed_seconds=time.perf_counter()-begin,
                              trial_count=sum(len(r['trials']) for r in rows), error=config.get('fatal_error'))
        save(inv_path, invocations)
        save(args.output/'results.json', {'status': status, 'elapsed_seconds': sum(i['elapsed_seconds'] for i in invocations),
             'case_count': len(rows), 'completed_case_count': sum(len(r['trials']) == len(ARMS) for r in rows),
             'trial_count': sum(len(r['trials']) for r in rows), 'summary': summarize(rows),
             'scope': 'Fixed development100 expansion; all failures retained; teacher oracles are not trained model performance; no sealed test read.'})


if __name__ == '__main__':
    main()
