"""Atomic evidence/checkpoint writes and complete source identities."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import zipfile
from contextlib import contextmanager
import torch
from datasets.paths import ROOT

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(temp, path):
    for attempt in range(30):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(.02 * (attempt + 1), .2))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    replace(temp, path)


def checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    torch.save(value, temp)
    replace(temp, path)


def sources():
    files = [p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts]
    for directory in ('datasets', 'scoring'):
        files += list((ROOT / directory).glob('*.py'))
    files += list(HERE.glob('*.ps1')) + [HERE/'provenance.json', ROOT/'docs/protocol.md']
    if (HERE/'cache_compatibility.json').exists():
        files += [HERE/'cache_compatibility.json']
    files += [ROOT / 'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
              ROOT / 'third_party/FloorSet/cost.py', ROOT / 'third_party/FloorSet/utils.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}


def snapshot(path, identity):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in identity['sources']:
            archive.write(ROOT / name, name)


@contextmanager
def exclusive(path):
    """OS-released lock, including after a process is killed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt': msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
