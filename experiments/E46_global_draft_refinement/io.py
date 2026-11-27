"""Recoverable atomic records and source identity for E46."""
import hashlib
import json
import os
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from datasets.paths import ROOT

HERE=Path(__file__).resolve().parent
WEIGHTS=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'
WEIGHT_SHA='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Windows readers/virus scanners can momentarily deny replace. Retry the
    # SAME payload; never rerun a solver or discard a completed layout for IO.
    for attempt in range(20):
        try:os.replace(temporary,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def sources():
    paths=[p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts]
    paths+=list((ROOT/'datasets').glob('*.py'))+list((ROOT/'scoring').glob('*.py'))
    paths+=[ROOT/p for p in ('docs/protocol.md','protocol/reference_workset_v5.json','protocol/split_v1.json','protocol/paths.json',
        'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py','third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}


def task_expired():
    budget=json.loads((HERE/'runs/budget.json').read_text(encoding='utf-8'))
    return datetime.now(timezone.utc)>=datetime.fromisoformat(budget['deadline_utc'].replace('Z','+00:00'))


@contextmanager
def exclusive(path):
    import msvcrt
    with Path(path).open('a+b') as h:
        if h.tell()==0:h.write(b'0');h.flush()
        h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_UNLCK,1)


def snapshot(folder,identity):
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in identity['sources']:z.write(ROOT/name,name)
    save(folder/'source.json',identity)
