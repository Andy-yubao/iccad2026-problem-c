"""E9 atomic records, source snapshots and provenance."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from datasets.paths import ROOT, resolved_data_paths

HERE = Path(__file__).resolve().parent
OLD_CHECKPOINT = ROOT / 'experiments/repair_friendly_draft/runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'
OLD_CHECKPOINT_SHA = '47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(source, target):
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.03 * 2 ** attempt, .5))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    replace(tmp, path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def run_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE / 'runs') or path == HERE / 'runs':
        raise ValueError('Output must be inside E9 runs/')
    return path


def source_hashes():
    paths = [p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts]
    paths += [HERE / 'inherited_sources.json', ROOT / 'environment.yml']
    for folder, pattern in [('datasets', '*.py'), ('scoring', '*.py'), ('visualization', '*.py'), ('protocol', '*.json')]:
        paths += list((ROOT / folder).glob(pattern))
    paths += [ROOT / 'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths)) if p.exists()}


def provenance():
    import torch
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'command': list(getattr(sys,'orig_argv',[sys.executable,*sys.argv])),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
            'source_sha256': source_hashes(), 'data_paths': resolved_data_paths(),
            'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'scipy', 'clarabel', 'highspy', 'shapely']},
            'python': sys.version, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}


def snapshot(folder, hashes):
    with zipfile.ZipFile(Path(folder) / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(ROOT / path, path)
