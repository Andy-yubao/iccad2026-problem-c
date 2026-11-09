"""Local identity, atomic files and an explicit train-only data boundary."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
RUN = OWNER / 'runs/20261105_bank25000_seed0'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def replace(temporary, destination):
    deadline = time.monotonic() + 5.
    retry = 0
    while True:
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as exc:
            if os.name != 'nt' or exc.winerror not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            retry += 1
            time.sleep(min(.01 * retry, .2, max(0., deadline - time.monotonic())))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    replace(temporary, path)


def checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    torch.save(value, temporary)
    replace(temporary, path)


def setup(seed=0):
    import numpy as np
    import torch
    import numba
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    numba.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sources():
    files = [p for p in OWNER.rglob('*.py')
             if not {'runs', 'assets', 'revisions', '__pycache__'}.intersection(p.relative_to(OWNER).parts)]
    files += [OWNER / n for n in ('plan.json', 'panel.json', 'model_manifest.json',
                                  'copy_provenance.json', 'postprocessor/budget.json')]
    files += list((ROOT / 'datasets').glob('*.py'))
    files += list((ROOT / 'datasets/expanded_reference').glob('*.py'))
    files += list((ROOT / 'scoring').glob('*.py'))
    files += [ROOT / n for n in ('protocol/paths.json', 'protocol/split_v1.json',
                                'protocol/reference_workset_v5.json', 'protocol/reference_workset_v6.json',
                                'third_party/FloorSet/cost.py', 'third_party/FloorSet/utils.py',
                                'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}


def check_stop():
    if (OWNER / 'STOP').exists() or (RUN / 'STOP').exists():
        raise InterruptedError('E106 data preparation stopped by user')
    if (RUN / 'BANK_ABORT').exists():
        raise RuntimeError('Data run failed; follow-on solver calls are blocked pending review')
