"""Lightweight training interlock; checked before torch, CUDA, or dataset reads."""
import hashlib
import json
from pathlib import Path

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
ARMS = ('full', 'pure')
REVISION = 'E132.P0.relation_input96x6.v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sources():
    paths = list(OWNER.glob('*.py')) + [OWNER / name for name in ('plan.json', 'provenance.json', 'README.md')]
    paths += [ROOT / name for name in ('datasets/paths.py', 'datasets/random_reference.py',
        'datasets/random_reference16k.py', 'protocol/random_reference16k_v1.json',
        'protocol/random_reference40k_v1.json', 'protocol/research_training16k_v1.json',
        'protocol/paths.json', 'protocol/research_no_mib_v1.json', 'protocol/model_prediction_boundary_v1.json')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def assert_training_authorized(arm):
    auth = read(OWNER / 'authorization.json')
    if not auth.get('training_enabled', False):
        raise PermissionError('Preparation only: training is disabled. No run, data read or optimizer update was started.')
    if (auth.get('scope') != REVISION or arm not in ARMS
            or arm not in auth.get('allowed_arms', []) or not auth.get('optimizer_updates_authorized', False)):
        raise PermissionError('No explicit authorization for this arm and optimizer updates')
    if auth.get('postprocessing_enabled') or auth.get('test_read'):
        raise PermissionError('Postprocessing and test are outside this experiment')
    run = OWNER / 'runs' / f'{arm}_seed0'
    if (OWNER / 'STOP').exists() or (run / 'STOP').exists():
        raise RuntimeError('STOP present')
    gate = read(OWNER / 'preflight.json')
    if gate.get('status') != 'passed' or gate.get('sources') != sources():
        raise RuntimeError('Preflight/source identity mismatch')
    return read(OWNER / 'plan.json')
