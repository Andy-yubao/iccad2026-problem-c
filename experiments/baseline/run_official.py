"""Seeded launcher for the unmodified official baseline and evaluator CLI.

Run using a conda environment with the official requirements installed.
All outputs go to a new experiment directory, never into the source archive.
"""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys
import time


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--test-id', type=int, help='Single-case smoke run')
    scope.add_argument('--all-cases', action='store_true', help='Explicit long run for the user to execute; never implicit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from datasets.paths import dataset_paths
    data = dataset_paths(root)['public_validation_root'].parent
    contest = root / 'third_party/FloorSet/iccad2026contest'
    for n in range(21, 121):
        for name in ['litedata_1.pth', 'litelabel_1.pth']:
            if not (data / f'LiteTensorDataTest/config_{n}' / name).is_file():
                raise FileNotFoundError(f'Missing local validation file: config_{n}/{name}')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLCONFIGDIR'] = str(root / '.cache/matplotlib')
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    metadata = {
        'command': sys.argv,
        'python_executable': sys.executable,
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'shapely', 'matplotlib', 'tqdm', 'requests']},
        'seed': args.seed, 'torch_cpu_threads': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'device_note': 'Official B*-tree SA baseline operates on CPU; CUDA availability is informational.',
        'torch_load_override': os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'),
        'upstream_commit': json.loads((root / 'third_party/upstream_manifest.json').read_text())['commit'],
    }
    (output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    sys.path.insert(0, str(contest))
    sys.argv = [str(contest / 'iccad2026_evaluate.py'), '--evaluate', str(contest / 'optimizer_template.py'),
                '--data-path', str(data), '--output', str(output / 'results.json'), '--save-solutions']
    if args.test_id is not None:
        sys.argv.extend(['--test-id', str(args.test_id)])
    os.chdir(output)
    started = time.perf_counter()
    status = 'failed'
    try:
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                runpy.run_path(str(contest / 'iccad2026_evaluate.py'), run_name='__main__')
        status = 'completed'
    finally:
        (output / 'execution.json').write_text(json.dumps({'status': status, 'wall_seconds': time.perf_counter() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
