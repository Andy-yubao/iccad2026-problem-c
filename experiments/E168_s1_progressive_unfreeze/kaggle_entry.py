"""Bootstrap the E168 run on one Kaggle GPU with frozen inputs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from zipfile import ZipFile


TASK = 'E168_s1_progressive_unfreeze'
ROOT = Path('/tmp/iccad-e168-s1')
HERE = ROOT / 'experiments' / TASK
OUTPUT = Path('/kaggle/working/e168-results')
INPUT = Path('/kaggle/input')
SOURCE = 'e168-project.zip.bin'
PARENT = 'e168-e167-s0-best57.pt.bin'
CUDA_WHEELS = 'https://download.pytorch.org/whl/cu128'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(filename):
    paths = list(INPUT.rglob(filename))
    if len(paths) != 1:
        raise ValueError('Expected one Kaggle input named ' + filename)
    return paths[0]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def unpack_data():
    archive = unique('e161-transfer.tar.gz.bin')
    manifest = read(archive.with_name('transfer-manifest.json'))
    if (manifest['train_count'], manifest['dev_count'], manifest['test_payloads']) != (36000, 2000, 0):
        raise ValueError('Unexpected E161 dataset scope')
    plan = read(HERE / 'plan.json')
    prefixes = (
        'datasets/static_tensor_cache_v1/' + plan['required_cache_key'] + '/',
        plan['role_cache_root'] + '/',
    )
    ledgers = ('datasets/random_reference40k_v1/report.json',
               'datasets/random_reference40k_v1/checksums.jsonl')
    expected = {
        row['path']: row for row in manifest['files']
        if row['path'].startswith(prefixes) or row['path'] in ledgers
    }
    if not set(ledgers) <= expected.keys() or any(
        not any(path.startswith(prefix) for path in expected) for prefix in prefixes
    ):
        raise FileNotFoundError('Registered training/monitoring caches or input ledgers are absent')
    count = len(expected)
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            if member.name not in expected:
                continue
            item = expected.pop(member.name)
            path = (ROOT / member.name).resolve()
            if not path.is_relative_to(ROOT) or not member.isfile() or path.exists():
                raise ValueError('Unexpected training archive path: ' + member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, path.open('xb') as target:
                shutil.copyfileobj(stream, target)
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('Training archive digest differs: ' + member.name)
            if (count - len(expected)) % 100 == 0:
                print(f'[DATA] verified {count - len(expected)}/{count} cache files', flush=True)
    if expected:
        raise FileNotFoundError(f'Missing {len(expected)} registered input files')
    print(f'[DATA] verified all {count} cache files', flush=True)


def training_env():
    requirements = HERE / 'requirements-kaggle.txt'
    pins = dict(line.split('==', 1) for line in requirements.read_text().splitlines() if line)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    probe = (
        'import numpy, torch; '
        f'assert numpy.__version__ == {pins["numpy"]!r}, numpy.__version__; '
        f'assert torch.__version__ == {pins["torch"]!r}, torch.__version__; '
        'assert torch.cuda.is_available(), "CUDA is unavailable"; '
        'assert torch.from_numpy(numpy.ones(1, dtype=numpy.float32)).cuda().item() == 1.; '
        'print(f"[ENV] NumPy {numpy.__version__}, PyTorch {torch.__version__}, "'
        '      f"CUDA device {torch.cuda.get_device_name(0)}", flush=True)'
    )
    ready = subprocess.run([sys.executable, '-c', probe], env=env, capture_output=True, text=True)
    if ready.returncode == 0:
        print(ready.stdout, end='', flush=True)
        return env
    print('[ENV] installing pinned training wheels', flush=True)
    packages = tempfile.mkdtemp(prefix='e168-wheels-')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--only-binary=:all:', '--target', packages,
                    '--extra-index-url', CUDA_WHEELS, '-r', str(requirements)], check=True)
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), packages))
    subprocess.run([sys.executable, '-c', probe], env=env, check=True)
    return env


def preserve_outputs():
    for relative in ('runs', 'background'):
        path = HERE / relative
        if path.exists():
            shutil.copytree(path, OUTPUT / relative, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('*.lock', '*.tmp'))
    for relative in ('queue_state.json', 'frozen_e168_s1.json'):
        path = HERE / relative
        if path.exists():
            (OUTPUT / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, OUTPUT / relative)


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir()
    source = unique(SOURCE)
    expected = source.with_name('e168-project.sha256').read_text(encoding='utf-8').strip()
    if digest(source) != expected:
        raise ValueError('E168 source archive digest differs')
    with ZipFile(source) as package:
        package.extractall(ROOT)
    parent_info = read(HERE / 'plan.json')['parent_initial']
    parent = unique(PARENT)
    if digest(parent) != parent_info['checkpoint_sha256']:
        raise ValueError('Kaggle parent differs from frozen E167 best57')
    destination = ROOT / parent_info['checkpoint']
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent, destination)
    print(f'[BOOT] Python {sys.version.split()[0]}; source and E167 prefix checkpoint verified', flush=True)
    env = training_env()
    print('[DATA] extracting frozen train36k and monitoring caches', flush=True)
    unpack_data()
    for relative in ('datasets/floorset_lite', 'datasets/LiteTensorDataTest'):
        (ROOT / relative).mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-B', '-m', f'experiments.{TASK}.prepare_run'],
                   cwd=ROOT, env=env, check=True)
    # The current queue cooperatively saves the accepted update or case on pause.
    timer = threading.Timer(max(0, 31500 - (time.monotonic() - started)),
                            lambda: (HERE / 'PAUSE_REQUEST').touch())
    timer.daemon = True
    timer.start()
    try:
        subprocess.run([sys.executable, '-u', '-B', '-m', f'experiments.{TASK}.run_queue', '--start'],
                       cwd=ROOT, env=env, check=True)
    finally:
        timer.cancel()
        preserve_outputs()


if __name__ == '__main__':
    main()
