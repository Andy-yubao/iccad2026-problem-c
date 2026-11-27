"""Current-decoder-compatible oracle on the same dev100; cumulative 900s cap."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

from .run import (ROOT, HERE, DRAFTS, TEACHERS, Worker, sha, save, record, np, torch,
                  LayoutReader, workset_manifest, _build_case, prepare_problem, topology, contact_tree)
from .backend.frozen.inference import prepare_inference
from .backend.frozen.batches import decode

ARMS = ('decoder_oracle', 'decoder_oracle_relations')


def decoder_oracle(teacher, inputs):
    """Perfect teacher-center/log-ratio prediction through the real frozen decoder."""
    graph = prepare_inference(inputs, 'cpu')
    state = torch.zeros((1, inputs[0], 3), dtype=torch.float32)
    scale = float(graph['scale'][0])
    state[0, :, :2] = torch.tensor((teacher[:, :2]+teacher[:, 2:]/2)/scale, dtype=torch.float32)
    state[0, :, 2] = torch.tensor(np.log(teacher[:, 2]/teacher[:, 3]), dtype=torch.float32)
    with torch.no_grad():
        out = (decode(state, graph)[0]*graph['scale'][0]).numpy().astype(float)
    constraints, specs = np.asarray(inputs[5]), np.asarray(inputs[6])
    locked = constraints[:, :2].any(1)
    out[locked, 2:] = specs[locked, 2:]
    out[constraints[:, 1] != 0] = specs[constraints[:, 1] != 0]
    return out


def summarize(rows):
    result = {}
    for arm in ARMS:
        pairs = [(r, t) for r in rows for t in r['trials'] if t['arm'] == arm]
        if not pairs: continue
        trials = [t for _, t in pairs]
        def metric(key):
            values = [t['metrics'][key] for t in trials if key in t['metrics']]
            return float(np.mean(values)) if values else None
        q = np.array([t['metrics']['cost'] for t in trials]); times = np.array([t['charged_total_seconds'] for t in trials])
        result[arm] = {'count': len(trials), 'Q_mean': float(q.mean()), 'Q_median': float(np.median(q)),
            'Q_p95': float(np.quantile(q, .95)), 'Q_max': float(q.max()),
            'worst_10_percent_mean_Q': float(np.sort(q)[-max(1, int(np.ceil(len(q)*.1))):].mean()),
            'hard_legal_count': sum(t['metrics']['is_feasible'] for t in trials),
            'error_count': sum('error' in t for t in trials),
            'initial_row_retained_count': sum(not t.get('quality_curve_offline') and 'error' not in t for t in trials),
            'core_seconds_mean': float(np.mean([t['solve_seconds'] for t in trials])),
            'charged_total_seconds_mean': float(times.mean()), 'charged_total_seconds_p95': float(np.quantile(times, .95)),
            'charged_total_seconds_max': float(times.max()), 'over_05_seconds_count': int((times > .5).sum()),
            'Q_delta_mean_vs_live_full_raw': float(np.mean([t['metrics']['cost']-r['baseline_q'] for r, t in pairs])),
            'within_002_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q']+.002 for r, t in pairs),
            'quality_and_05_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q']+.002
                                      and t['charged_total_seconds'] <= .5 for r, t in pairs),
            'hpwl_gap_mean': metric('hpwl_gap'), 'area_gap_mean': metric('area_gap'),
            'boundary_mean': metric('boundary_violations'), 'grouping_mean': metric('grouping_violations'), 'mib_mean': metric('mib_violations')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); begin = time.perf_counter()
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    main_result = json.loads((args.main_run/'results.json').read_text(encoding='utf-8'))
    assert main_result['status'] == 'completed' and main_result['trial_count'] == 700
    parent_elapsed = main_result['elapsed_seconds']
    if parent_elapsed >= 750:
        raise ValueError('Insufficient shared 900s budget; do not start this extension automatically')
    args.output.mkdir(parents=True, exist_ok=False); (args.output/'cases').mkdir()
    anchors = {r['case_id']: r for r in (json.loads(p.read_text(encoding='utf-8')) for p in (args.main_run/'cases').glob('*.json'))}
    drafts = {r['case_id']: r for r in map(json.loads, DRAFTS.read_text(encoding='utf-8').splitlines())}
    teachers = {r['case_id']: r for r in map(json.loads, TEACHERS.read_text(encoding='utf-8').splitlines())}
    reader = LayoutReader(workset_manifest(), 'dev_quality'); assert len(reader.members) == 100
    config = {'command': [sys.executable, *sys.argv], 'seed': 0, 'threads': 2, 'arms': ARMS,
              'main_run': args.main_run.as_posix(), 'main_results_sha256': sha(args.main_run/'results.json'),
              'parent_elapsed_seconds': parent_elapsed, 'shared_wall_cap_seconds': 900,
              'hypothesis': 'Teacher full geometry may be outside the exact-area decoder family. Test perfect teacher centers/log-ratios passed through the actual current decoder, with dynamic or teacher relations.',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in HERE.rglob('*.py')},
              'local_test_read': False, 'new_data_members': False, 'training': False,
              'timing_scope': 'Live core+IPC plus reused original inference charged equally. Perfect output generation and teacher relation construction excluded; no actual learned oracle.'}
    save(args.output/'config.json', config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob('*.py')):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(HERE/'README.md', (HERE/'README.md').relative_to(ROOT).as_posix())
    worker = Worker(args.output/'worker.log'); rows = []; status = 'running'; rng = random.Random(0); warmed = False
    try:
        for index in range(100):
            raw = reader.case(index)
            case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6], raw.reference_positions, raw.stored_metrics)
            teacher_record = teachers[raw.case_id]['variants']['conic_robust']['final']
            teacher = np.asarray(teacher_record['positions']); inputs = case.solver_inputs
            proposal = decoder_oracle(teacher, inputs)
            p = prepare_problem(*inputs); edges = topology(teacher, p.constraints)
            guidance = {'edges': [[int(v) for v in e] for e in edges],
                        'contacts': [[int(v) for v in e] for e in contact_tree(teacher, p, edges)]}
            serialized = [int(inputs[0]), *[np.asarray(v).tolist() for v in inputs[1:]]]
            free = ~(p.fixed|p.preplaced)
            area_offset = np.abs(teacher[free, 2:].prod(1)/p.areas[free]-1)
            row = {'case_id': raw.case_id, 'source': raw.source, 'panel_index': index, 'block_count': case.block_count,
                   'baseline_q': anchors[raw.case_id]['baseline_q'], 'layouts': {}, 'trials': [],
                   'decoder_diagnostic': {'free_blocks': int(free.sum()),
                        'teacher_area_relative_offset_mean': float(area_offset.mean()) if len(area_offset) else 0.,
                        'teacher_free_area_offset_gt_1e4_count': int((area_offset > 1e-4).sum()),
                        'decoder_free_area_relative_error_max': float(np.max(np.abs(proposal[free, 2:].prod(1)/p.areas[free]-1))) if free.any() else 0.}}
            rows.append(row)
            if not warmed:
                warmup = worker.call({'inputs': serialized, 'draft': proposal.tolist(), 'seconds': .1})
                config['warmup_wall_seconds'] = warmup['worker_wall_seconds']; warmed = True
            arms = list(ARMS); rng.shuffle(arms)
            for arm in arms:
                if parent_elapsed+time.perf_counter()-begin >= 900 or (args.output/'STOP').exists():
                    status = 'budget_or_user_stop'; break
                response = worker.call({'inputs': serialized, 'draft': proposal.tolist(), 'seconds': .4,
                                        'guidance': guidance if arm.endswith('_relations') else None})
                trial = record(case, response, drafts[raw.case_id]['inference_seconds'], row['layouts'])
                if 'charged_total_seconds' not in trial:
                    trial['charged_total_seconds'] = drafts[raw.case_id]['inference_seconds']+response['worker_wall_seconds']
                trial.update(arm=arm, budget_seconds=.4,
                             Q_delta_vs_live_full_raw=trial['metrics']['cost']-row['baseline_q'])
                row['trials'].append(trial); save(args.output/'cases'/f'{index:03d}.json', row)
            print(f"decoder {index+1}/100", raw.case_id, ' '.join(f"{t['arm']}={t['metrics']['cost']:.4f}" for t in row['trials']), flush=True)
            save(args.output/'results.json', {'status': status, 'elapsed_seconds': time.perf_counter()-begin,
                'shared_elapsed_seconds': parent_elapsed+time.perf_counter()-begin,
                'case_count': len(rows), 'trial_count': sum(len(r['trials']) for r in rows), 'summary': summarize(rows)})
            if status != 'running': break
        if status == 'running': status = 'completed'
    except Exception as exc:
        status = 'error'; config['fatal_error'] = repr(exc); raise
    finally:
        worker.close(); save(args.output/'config.json', config)
        save(args.output/'results.json', {'status': status, 'elapsed_seconds': time.perf_counter()-begin,
            'shared_elapsed_seconds': parent_elapsed+time.perf_counter()-begin,
            'case_count': len(rows), 'trial_count': sum(len(r['trials']) for r in rows), 'summary': summarize(rows),
            'scope': 'Same development100; perfect prediction of current output heads, not a trained model. Current decoder clamps ratio and restores public specifications.'})


if __name__ == '__main__':
    main()
