"""Package the paused E161 train/dev data and state, without running any model."""
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E161 = ROOT / 'experiments/E161_final_no_mib'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for blob in iter(lambda: source.read(4 * 2**20), b''):
            h.update(blob)
    return h.hexdigest()

def main():
    pause = json.loads((E161/'pause_confirmation.json').read_text(encoding='utf-8'))
    queue = json.loads((E161/'queue_state.json').read_text(encoding='utf-8'))
    if queue['status'] != 'stopped' or pause['accepted_updates'] != 7568:
        raise RuntimeError('This transfer is registered for the paused S1 step7568 only')
    prep = json.loads((E161/'preparation.json').read_text(encoding='utf-8'))
    files = {ROOT / name for name in prep['sources']}
    files |= set((ROOT/'protocol').glob('*.json'))
    files |= set((ROOT/'datasets').glob('*.py'))
    files |= set((ROOT/'scoring').glob('*.py'))
    files |= set((ROOT/'third_party/FloorSet').glob('*.py'))
    files |= set((ROOT/'third_party/FloorSet/iccad2026contest').glob('*.py'))
    files |= {ROOT/'third_party/FloorSet/LICENSE', ROOT/'research/quality_log.py'}
    if (ROOT/'research/__init__.py').exists(): files.add(ROOT/'research/__init__.py')
    if (ROOT/'experiments/__init__.py').exists(): files.add(ROOT/'experiments/__init__.py')
    files |= {p for p in E161.glob('*.json')}
    # The evaluator imports budget.json before main() and then reads plan.json.
    # They are runtime inputs even though io.sources() only hashes Python sources.
    files |= {E161/'evaluation'/name for name in ('budget.json','plan.json')}
    files |= {HERE/name for name in ('resume_remote.py','requirements-kaggle.txt')}
    selected_checkpoints = {
        'runs/e161_l0_seed0/checkpoints/best.pt',
        'runs/e161_s0_seed0/checkpoints/best.pt',
        'runs/e161_s1_seed0/checkpoints/best.pt',
        'runs/e161_s1_seed0/checkpoints/latest.pt',
    }
    files |= {E161/name for name in selected_checkpoints}
    files |= {p for p in (E161/'runs').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.zip')}
    plan = json.loads((E161/'plan.json').read_text(encoding='utf-8'))
    cache = ROOT/'datasets/static_tensor_cache_v1'/plan['required_cache_key']
    role_sha = digest(E161/'role_features.py')
    roles = [p.parent for p in (ROOT/plan['role_cache_root']).glob('*/identity.json')
             if json.loads(p.read_text())['producer_sha256'] == role_sha]
    if len(roles) != 1: raise ValueError('Ambiguous registered role cache')
    for folder in (cache, roles[0]):
        files.add(folder/'identity.json')
        for role in ('train','dev'):
            files |= {p for p in (folder/role).iterdir() if p.suffix in ('.pt','.json')}
    dataset = ROOT/'datasets/random_reference40k_v1'
    files |= {dataset/'report.json', dataset/'checksums.jsonl'}
    reg = json.loads((ROOT/'protocol/random_reference40k_v1.json').read_text(encoding='utf-8'))
    # Training uses the complete existing tensor cache and forbids rebuilding it.
    # Only the terminal evaluator reads raw NPZs, so transfer dev NPZs once.
    files |= {dataset/record['payload'] for record in reg['sets']['dev']['records']}
    # Only explicit train/dev paths are enumerated. Test payload directories are never read.
    files = sorted(files)
    out = HERE/'payload'
    out.mkdir(exist_ok=True)
    archive = out/'e161-transfer.tar.gz.bin'
    if archive.exists(): raise FileExistsError('Preserve existing transfer package; do not overwrite')
    manifest = dict(run_id=pause['run_id'], checkpoint_step=7568, train_count=36000,
        dev_count=2000, test_payloads=0, raw_npz_roles=['dev'],
        training_payload='Complete original train36000 static tensors and role sidecars',local_source_mutated=False,
        checkpoint_migration='Only remote runtime GPU identity changes; Adam/RNG/cursor/scheduler/budget retained. Not bit-identical across GPU architectures.',
        files=[])
    begin = time.monotonic()
    print(json.dumps(dict(status='packaging',files=len(files),raw_train_npz_omitted='Already represented in the required complete tensor cache')),flush=True)
    with tarfile.open(archive, 'w:gz', compresslevel=1) as target:
        for i, path in enumerate(files):
            name = path.relative_to(ROOT).as_posix()
            sha = digest(path)
            if name in prep['sources'] and sha != prep['sources'][name]:
                raise ValueError('Prepared source changed: '+name)
            manifest['files'].append(dict(path=name, bytes=path.stat().st_size, sha256=sha))
            target.add(path, arcname=name, recursive=False)
            if (i+1) % 500 == 0:
                print(json.dumps(dict(packaged=i+1,total=len(files),elapsed_seconds=round(time.monotonic()-begin,1))),flush=True)
    manifest.update(archive_bytes=archive.stat().st_size, archive_sha256=digest(archive))
    (out/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    status = dict(status='prepared_network_blocked', training_submitted=False,
        account_authenticated=False, account_gpu_quota_verified=False,
        blocker='Execution host outbound networking denied: ProxyError/WinError10013. No credential was sent.',
        run_id=pause['run_id'], resume_arm='e161_s1', resume_step=7568,
        preserve_cumulative_budget=True, local_training_remains_paused=True,
        gpu_request='NvidiaTeslaT4; single visible device; no paid cloud resources',
        archive_bytes=manifest['archive_bytes'],archive_sha256=manifest['archive_sha256'],
        included_files=len(files),test_payloads=0,defensive_tests_run=False)
    (HERE/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status),flush=True)

if __name__ == '__main__': main()
