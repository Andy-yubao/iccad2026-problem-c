"""Essential caching control: hold raw-draft relations fixed, same five cases."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

from .run import (ROOT, HERE, DRAFTS, WORKERS, Worker, sha, save, save_rows,
                  prepare_problem, topology, contact_tree, record, np, torch,
                  LayoutReader, workset_manifest, _build_case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--main-run', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(2)
    np.random.seed(0); random.seed(0)
    main_result = json.loads((args.main_run/'results.json').read_text(encoding='utf-8'))
    assert main_result['status'] == 'completed' and main_result['trial_count'] == 100
    anchors = {r['case_id']: r for r in map(json.loads, (args.main_run/'cases.jsonl').read_text(encoding='utf-8').splitlines())}
    cached = {r['case_id']: r for r in map(json.loads, DRAFTS.read_text(encoding='utf-8').splitlines())}
    config = {'command': [sys.executable, *sys.argv], 'seed': 0, 'threads': 2,
              'hypothesis_fixed_before_control': 'Teacher-relations benefit may be caused by caching relation construction rather than improved topology. Freeze raw-draft relations to separate the mechanisms.',
              'main_run': args.main_run.as_posix(), 'main_results_sha256': sha(args.main_run/'results.json'),
              'cached_drafts_sha256': sha(DRAFTS), 'main_cases_sha256': sha(args.main_run/'cases.jsonl'),
              'workers': WORKERS, 'budgets_seconds': [.4, 5.], 'wall_cap_seconds': 300,
              'local_test_read': False, 'new_cases': False, 'training': False,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in HERE.rglob('*.py')},
              'timing_scope': 'Same worker and charged historical inference as main run. Raw relations constructed before timing, exactly as teacher relations; final scoring outside solve time.'}
    save(args.output/'config.json', config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob('*.py')):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(HERE/'README.md', (HERE/'README.md').relative_to(ROOT).as_posix())
    reader = LayoutReader(workset_manifest(), 'dev_quality')
    indices = [i for i, (shard, _) in enumerate(reader.members) if int(shard.split('/')[0].split('_')[1]) in WORKERS]
    assert len(indices) == 5
    worker = Worker(args.output/'worker.log')
    rows = []; status = 'running'; warmed = False; rng = random.Random(0)
    try:
        for index in indices:
            raw = reader.case(index)
            case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6], raw.reference_positions, raw.stored_metrics)
            draft = np.asarray(cached[raw.case_id]['draft']); inputs = case.solver_inputs
            problem = prepare_problem(*inputs)
            edges = topology(draft, problem.constraints)
            contacts = contact_tree(draft, problem, edges)
            guidance = {k: [[int(v) for v in t] for t in values]
                        for k, values in [('edges', edges), ('contacts', contacts)]}
            serialized = [int(inputs[0]), *[np.asarray(v).tolist() for v in inputs[1:]]]
            row = {'case_id': raw.case_id, 'source': raw.source, 'baseline_q': anchors[raw.case_id]['baseline_q'],
                   'layouts': {}, 'trials': []}
            rows.append(row)
            if not warmed:
                warmup = worker.call({'inputs': serialized, 'draft': draft.tolist(), 'seconds': .1})
                config['warmup_wall_seconds'] = warmup['worker_wall_seconds']; warmed = True
            budgets = [.4, 5.]; rng.shuffle(budgets)
            for budget in budgets:
                if time.perf_counter()-begin > 300 or (args.output/'STOP').exists():
                    status = 'resource_or_user_stop'; break
                response = worker.call({'inputs': serialized, 'draft': draft.tolist(), 'seconds': budget, 'guidance': guidance})
                trial = record(case, response, cached[raw.case_id]['inference_seconds'], row['layouts'])
                trial.update(variant='raw_relations', budget_seconds=budget,
                             Q_delta_vs_live_full_raw=trial['metrics']['cost']-row['baseline_q'])
                row['trials'].append(trial)
                save_rows(args.output/'cases.jsonl', rows)
                print(raw.case_id, budget, f"Q={trial['metrics']['cost']:.6f}", f"core={trial['solve_seconds']:.3f}s", flush=True)
            if status != 'running': break
        if status == 'running': status = 'completed'
    except Exception as exc:
        status = 'error'; config['fatal_error'] = repr(exc); raise
    finally:
        worker.close()
        save_rows(args.output/'cases.jsonl', rows)
        summaries = {}
        for budget in [.4, 5.]:
            pairs = [(r, t) for r in rows for t in r['trials'] if t['budget_seconds'] == budget]
            if not pairs: continue
            summaries[str(budget)] = {'count': len(pairs),
                'Q_mean': float(np.mean([t['metrics']['cost'] for _, t in pairs])),
                'Q_max': max(t['metrics']['cost'] for _, t in pairs),
                'hard_legal_count': sum(t['metrics']['is_feasible'] for _, t in pairs),
                'core_seconds_mean': float(np.mean([t['solve_seconds'] for _, t in pairs])),
                'charged_total_seconds_mean': float(np.mean([t['charged_total_seconds'] for _, t in pairs])),
                'conic_calls_mean': float(np.mean([t['conic_calls'] for _, t in pairs])),
                'native_conic_seconds_mean': float(np.mean([t['native_conic_seconds'] for _, t in pairs])),
                'Q_delta_mean_vs_live_full_raw': float(np.mean([t['metrics']['cost']-r['baseline_q'] for r, t in pairs])),
                'within_002_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q']+.002 for r, t in pairs)}
        save(args.output/'config.json', config)
        save(args.output/'results.json', {'status': status, 'elapsed_seconds': time.perf_counter()-begin,
             'trial_count': sum(len(r['trials']) for r in rows), 'summary': summaries,
             'scope': 'Post-main essential ablation on the same five cases; no new samples, no deployment result.'})


if __name__ == '__main__':
    main()
