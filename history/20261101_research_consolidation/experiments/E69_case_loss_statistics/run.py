"""Bounded descriptive dev analysis; frozen eval forwards, no training or solver calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .io import ROOT, HERE, save, read, sha
from .case_features import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/20261028_dev2000')
    parser.add_argument('--max-seconds', type=float, default=900.)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE/'runs') or args.max_seconds > 900:
        raise ValueError('E69 output or15-minute execution bound mismatch')
    if (output/'results.json').exists():
        raise ValueError('Completed statistics must not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def check_deadline():
        if time.perf_counter()-started > args.max_seconds:
            raise RuntimeError('Statistical execution budget exhausted; no background continuation')
    def progress(stage):
        check_deadline()
        save(output/'progress.json', {'status': 'running', 'stage': stage,
            'elapsed_seconds': time.perf_counter()-started, 'utc': datetime.now(timezone.utc).isoformat(),
            'training_started': False, 'test_read': False})
        print('E69 stage:', stage, flush=True)
    from .collect import RUNS, key, infer, payload
    from .data import load_rows
    from .losses import WEIGHTS
    from .statistics import analyze
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    source_files = [p for p in HERE.glob('*.py')]+[HERE/'plan.json']
    for name in ('model.py', 'base_model.py', 'features.py', 'batches.py', 'data.py', 'cache.py', 'losses.py', 'ops.py'):
        if sha(HERE/name) != sha(ROOT/'experiments/E65_geometry_messages'/name):
            raise ValueError('Frozen copied inference source changed: '+name)
    source_ident = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    registry = ROOT/'protocol/reference_workset_v5.json'
    save(output/'config.json', {'command': [sys.executable, *sys.argv], 'sources': source_ident,
         'registry_sha256': sha(registry), 'paths_sha256': sha(ROOT/'protocol/paths.json'),
         'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'new_inference': ['E42', 'E65_geometry'], 'saved_models': list(RUNS),
         'cpu_workers': 4, 'threads_per_feature_worker': 1, 'torch_cpu_threads': 4,
         'device': torch.cuda.get_device_name(0), 'precision': 'FP32 eval/no_grad',
         'max_seconds': args.max_seconds, 'training_started': False, 'postprocessing_started': False,
         'test_read': False, 'user_feedback': read(HERE/'plan.json')['user_feedback_verbatim']})
    try:
        progress('read_frozen_dev_cache')
        rows = load_rows('dev')
        members = [key(row['member']) for row in rows]
        assert len(rows) == 2000 and len(set(members)) == 2000
        if not (output/'forwards.json').exists():
            progress('five_case_implementation_validation')
            validation = {}
            for name in ('E42', 'E65_geometry'):
                _, _, validation[name] = infer(name, rows[:5], check_deadline, validate_only=True)
            for row in rows[:5]:
                result = extract(payload(row))
                assert result['n'] == row['graph']['n']
            save(output/'implementation_check.json', {'unique_dev_cases': 5, 'models': validation,
                 'status': 'passed', 'test_read': False})
            progress('frozen_E42_and_E65_module_diagnostics')
            fresh = {}; identities = {}; reference = None
            for name in ('E42', 'E65_geometry'):
                fresh[name], anchor, identities[name] = infer(name, rows, check_deadline)
                if name == 'E42': reference = anchor
            save(output/'forwards.json', {'models': fresh, 'reference_state': reference,
                'identities': identities, 'members': members, 'collect_source_sha256': sha(HERE/'collect.py')})
        forward = read(output/'forwards.json')
        assert forward['members'] == members and forward['collect_source_sha256'] == sha(HERE/'collect.py')
        progress('four_worker_task_features')
        if not (output/'features.json').exists():
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
                features = list(pool.map(extract, map(payload, rows), chunksize=16))
            pids = sorted({row['extraction_pid'] for row in features})
            assert len(pids) == 4, pids
            save(output/'features.json', {'records': features, 'worker_pids': pids,
                                         'feature_source_sha256': sha(HERE/'case_features.py')})
        feature_data = read(output/'features.json'); features = feature_data['records']
        assert feature_data['feature_source_sha256'] == sha(HERE/'case_features.py')
        assert [key(row['member']) for row in features] == members
        records = {}
        for name, folder in RUNS.items():
            if name == 'E42':
                records[name] = forward['models'][name]
            else:
                saved = read(folder/'full_dev.json')['records']
                mapping = {key(row['member']): row for row in saved}
                assert set(mapping) == set(members)
                records[name] = [mapping[member] for member in members]
                if name == 'E65_geometry':
                    for saved_row, computed in zip(records[name], forward['models'][name]):
                        for field, value in computed.items():
                            if field not in saved_row:
                                saved_row[field] = value
        # Exact same500 cases and frozen flash outputs for all seven models.
        terminal = {}; panel = None
        for name, folder in RUNS.items():
            saved = read(folder/'terminal/results.json')['records']
            mapping = {key(row['member']): row for row in saved}
            if panel is None:
                panel = [key(row['member']) for row in sorted(saved, key=lambda row: row['index'])]
            assert len(mapping) == 500 and set(mapping) == set(panel)
            terminal[name] = [mapping[member] for member in panel]
        index_map = {member: i for i, member in enumerate(members)}
        indices = np.array([index_map[member] for member in panel])
        matrices = {'indices': indices,
            'Q': np.column_stack([[row['metrics']['cost'] for row in terminal[name]] for name in RUNS]),
            'movement': np.column_stack([[row['mean_center_move_local'] for row in terminal[name]] for name in RUNS])}
        progress('descriptive_statistics_and_worker_CV')
        results = analyze(features, records, forward['reference_state'], matrices, WEIGHTS)
        # Flat per-case table for reproducibility; no free coordinate arrays duplicated.
        table = []
        for i, feature in enumerate(features):
            row = {k: v for k, v in feature.items() if k not in ('member', 'extraction_pid')}
            row['member'] = key(feature['member'])
            for name in RUNS:
                row[name+'_G'] = records[name][i]['loss_total']
                row[name+'_overlap'] = records[name][i]['overlap_area_fraction']
            for term in WEIGHTS:
                row['E65_weighted_'+term] = WEIGHTS[term]*records['E65_geometry'][i]['loss_'+term]
            row['reference_state_G'] = forward['reference_state'][i]['total']
            table.append(row)
        with (output/'case_table.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        save(output/'analysis.json', results)
        progress('analysis_complete')
        save(output/'results.json', {'status': 'complete', 'case_count': 2000, 'models': list(RUNS),
              'flash_nested_count': 500, 'unique_validation_cases': 5,
              'new_eval_forwards': {'E42': 2000, 'E65_geometry': 2000},
              'feature_worker_pids': feature_data['worker_pids'], 'elapsed_seconds': time.perf_counter()-started,
              'training_started': False, 'postprocessing_started': False, 'test_read': False})
        print('E69 numerical statistics complete:', time.perf_counter()-started, 'seconds', flush=True)
    except BaseException as error:
        save(output/'progress.json', {'status': 'error', 'error': repr(error),
              'elapsed_seconds': time.perf_counter()-started, 'training_started': False, 'test_read': False})
        raise


if __name__ == '__main__':
    main()
