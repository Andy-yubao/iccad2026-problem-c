"""Bootstrap the registered E164 scratch run on Kaggle."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

CODE_BUNDLE = '__E164_BUNDLE__'
TRANSFER_MANIFEST_SHA = '1ddffaeb0371cb8b27fe0839520771aa3beaa9da6d1cc88b79edf4ce5a764bf0'
PACKAGE = 'experiments.E164_static_relation_l0'
HOME_REL = Path(*PACKAGE.split('.'))
OUT = Path('/kaggle/working/e164-results')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(stage, **fields):
    result = dict(stage=stage, **fields)
    path = OUT / 'bootstrap-status.json'
    scratch = path.with_suffix('.tmp')
    scratch.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    scratch.replace(path)
    print(json.dumps(result), flush=True)


def unpack_data(root):
    matches = list(Path('/kaggle/input').rglob('e161-transfer.tar.gz.bin'))
    if len(matches) != 1:
        raise RuntimeError('Mount exactly one registered private E161 dataset')
    archive = matches[0]
    manifest_path = archive.with_name('transfer-manifest.json')
    if sha(manifest_path) != TRANSFER_MANIFEST_SHA:
        raise ValueError('The mounted transfer manifest differs from the approved input')
    manifest = json.loads(manifest_path.read_bytes())
    if archive.stat().st_size != manifest['archive_bytes']:
        raise ValueError('The mounted transfer archive size differs')
    prefixes = ('datasets/static_tensor_cache_v1/', 'experiments/E149_role_initial/role_cache_v1/')
    metadata = {
        'protocol/random_reference40k_v1.json',
        'datasets/random_reference40k_v1/report.json',
        'datasets/random_reference40k_v1/checksums.jsonl',
    }
    expected = {
        entry['path']: entry for entry in manifest['files']
        if entry['path'].startswith(prefixes) or entry['path'] in metadata
    }
    found = set()
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            if member.name not in expected:
                continue
            item = expected[member.name]
            destination = (root / member.name).resolve()
            if (not destination.is_relative_to(root) or not member.isfile()
                    or member.size != item['bytes'] or member.name in found):
                raise ValueError('Unexpected data archive member: ' + member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as content, destination.open('xb') as output:
                shutil.copyfileobj(content, output, length=4 * 2**20)
            if sha(destination) != item['sha256']:
                raise ValueError('Cache archive content differs: ' + member.name)
            found.add(member.name)
    if found != expected.keys():
        raise ValueError('Required train/dev cache files are missing')
    return len(found)


def main():
    os.environ.update(
        CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', E164_EXECUTION_PLATFORM='kaggle',
    )
    OUT.mkdir(parents=True, exist_ok=False)
    root = Path('/tmp/iccad-e164-project')
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        write_status('extracting_frozen_data')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_BUNDLE))) as archive:
            archive.extractall(root)
        owner = root / HOME_REL
        (OUT / 'runs').mkdir()
        (owner / 'runs').symlink_to(OUT / 'runs', target_is_directory=True)
        data_files = unpack_data(root)
        write_status('installing_dependencies', data_files=data_files,
                     extraction_seconds=time.monotonic() - started)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            '--retries', '0', '--extra-index-url', 'https://download.pytorch.org/whl/cu128',
            'torch==2.7.1+cu128', 'numpy==1.26.4',
        ], check=True)
        write_status('formal_training', bootstrap_seconds=time.monotonic() - started)
        with (OUT / 'train.log').open('w', encoding='utf-8') as log:
            command = [sys.executable, '-u', '-B', '-m', PACKAGE + '.train', '--arm', 'e164_l0']
            with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                  errors='replace') as worker:
                for line in worker.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                if worker.wait():
                    raise RuntimeError('Training worker failed. See train.log and the run records.')
        subprocess.run([
            sys.executable, '-B', '-c',
            f'from {PACKAGE}.report import finalize; finalize({str(OUT)!r})',
        ], cwd=root, check=True)
        write_status('complete', session_seconds=time.monotonic() - started)
    except BaseException:
        write_status('error', detail=traceback.format_exc(), session_seconds=time.monotonic() - started)
        raise
    finally:
        identity = OUT / 'identity'
        identity.mkdir(exist_ok=True)
        for filename in ('plan.json', 'authorization.json', 'provenance.json', 'state_e164_l0.json'):
            path = root / HOME_REL / filename
            if path.is_file():
                shutil.copy2(path, identity / filename)


if __name__ == '__main__':
    main()
