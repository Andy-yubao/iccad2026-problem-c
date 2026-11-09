"""Atomic records, provenance snapshot and OS-released exclusive run lock."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import uuid
import zipfile
import subprocess
from datasets.paths import ROOT

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sources():
    paths = [p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts]
    paths += list((ROOT / 'datasets').glob('*.py')) + list((ROOT / 'scoring').glob('*.py'))
    paths += [HERE / 'provenance.json', ROOT / 'docs/protocol.md',
              ROOT / 'protocol/reference_workset_v5.json', ROOT / 'protocol/split_v1.json',
              ROOT / 'protocol/paths.json', ROOT / 'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
              ROOT / 'third_party/FloorSet/cost.py', ROOT / 'third_party/FloorSet/utils.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}


def snapshot(folder, identity):
    with zipfile.ZipFile(folder / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in identity['sources']:
            archive.write(ROOT / name, name)
    save(folder / 'source.json', {'identity': identity,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'snapshot_sha256': sha(folder / 'source_snapshot.zip')})


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0: handle.write(b'0'); handle.flush()
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
