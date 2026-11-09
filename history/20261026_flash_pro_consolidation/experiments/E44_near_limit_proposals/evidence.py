"""Small local evidence writer; exact source identity and OS-released run lock."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import uuid
import zipfile
from datasets.paths import ROOT

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def identity(config):
    paths = list(HERE.glob('*.py'))
    paths += list((ROOT / 'datasets').glob('*.py')) + list((ROOT / 'scoring').glob('*.py'))
    paths += [ROOT / p for p in ('protocol/reference_workset_v5.json', 'protocol/split_v1.json',
                                 'protocol/paths.json', 'docs/protocol.md',
                                 'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
                                 'third_party/FloorSet/cost.py', 'third_party/FloorSet/utils.py')]
    return {'version': 'E44.stageA.v1', 'config': config,
            'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
            'python': sys.executable}


def snapshot(folder, source_identity):
    with zipfile.ZipFile(folder / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_identity['sources']:
            archive.write(ROOT / path, path)
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    save(folder / 'source.json', {'identity': source_identity, 'git_head': git('rev-parse', 'HEAD'),
                                 'git_status': git('status', '--short'),
                                 'source_snapshot_sha256': sha(folder / 'source_snapshot.zip')})


@contextmanager
def exclusive(path):
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
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
