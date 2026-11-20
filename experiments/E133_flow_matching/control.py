"""E133 authorization, frozen sources and total-pipeline budget."""
import hashlib
import json
import time
from pathlib import Path

OWNER=Path(__file__).resolve().parent
ROOT=OWNER.parents[1]
ARMS=('fm',)
REVISION='E133.CFM.v1.seed0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sources():
    paths=list(OWNER.rglob('*.py'))+[OWNER/n for n in ('README.md','plan.json','provenance.json','evaluation/budget.json')]
    paths += [ROOT/n for n in ('datasets/paths.py','datasets/random_reference.py','datasets/random_reference16k.py',
        'protocol/random_reference16k_v1.json','protocol/random_reference40k_v1.json','protocol/research_training16k_v1.json',
        'protocol/paths.json','protocol/research_no_mib_v1.json','protocol/model_prediction_boundary_v1.json',
        'scoring/research_no_mib.py','scoring/checks.py','scoring/official.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def assert_training_authorized(arm='fm'):
    auth=read(OWNER/'authorization.json')
    if not auth.get('training_enabled'):raise PermissionError('Preparation interlock: training disabled')
    if auth['scope']!=REVISION or arm not in auth['allowed_arms'] or not auth['optimizer_updates_authorized']:
        raise PermissionError('Unauthorized arm/update')
    if auth['test_read'] or auth['retry']:raise PermissionError('Test/retry forbidden')
    if stopped():raise InterruptedError('STOP; no subsequent stage')
    gate=read(OWNER/'preflight.json')
    if gate['status']!='passed' or gate['sources']!=sources():raise RuntimeError('Preparation/source identity mismatch')
    return read(OWNER/'plan.json')

def stopped():
    return any((OWNER/p).exists() for p in ('STOP','QUEUE_STOP','runs/fm_seed0/STOP','evaluation/STOP','runs/fm_seed0/evaluation_no_mib_dev500/STOP'))

def elapsed_pipeline():
    return max(0.,time.time()-read(OWNER/'pipeline_budget.json')['started_unix'])

def budget_exhausted(training=False):
    plan=read(OWNER/'plan.json')
    return elapsed_pipeline()>=(plan['training_and_startup_seconds'] if training else plan['total_pipeline_seconds'])
