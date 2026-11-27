"""Recoverable same-volume archive; no training, inference or tensor loading."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHECKPOINT = '97ba1eed70d4027e121a95466037661e2aa7bcc3'
NAMES = ['E23_capacity_geometry', 'E26_tail_constraints',
         'E27_overlap_weight', 'E28_iterative_draft']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError(f'Outside workspace: {path}')
    return resolved


def main():
    if (HERE / 'manifest.json').exists():
        raise RuntimeError('Archive already recorded; do not rerun migration')
    local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    remote = None
    for options in [[], ['-c', 'http.proxy=http://127.0.0.1:7897']]:
        try:
            remote = subprocess.check_output(['git', *options, 'ls-remote', 'origin', 'refs/heads/main'],
                                             cwd=ROOT, text=True, timeout=25).split()[0]
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    assert local == remote == CHECKPOINT, 'Remote checkpoint must precede migration'
    assert json.loads((ROOT / 'experiments/E28_iterative_draft/runs/20261021_pipeline_recurrent_full/progress.json').read_text())['status'] == 'complete'
    assert json.loads((ROOT / 'experiments/E27_overlap_weight/runs/20261020_pipeline/pipeline.json').read_text())['status'] == 'complete'
    assert not (ROOT / 'experiments/E29_minimal_repair_draft/runs').exists()
    pairs = [(ROOT / 'experiments' / name, HERE / 'experiments' / name) for name in NAMES]
    pairs.append((ROOT / 'experiments/E25_integrated_baseline/migration_check.py',
                  HERE / 'legacy_E25_migration_check.py'))
    for old, new in pairs:
        safe(old)
        safe(new)
        assert old.exists() and not new.exists(), (old, new)
    frozen = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'protocol').glob('*.json')}
    model = ROOT / 'experiments/E25_integrated_baseline/runs/20261020_e10_geometry_v5_seed0/checkpoints/best.pt'
    model_sha = sha(model)
    assert model_sha == '3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    best_sources = {p.relative_to(ROOT).as_posix(): sha(p)
                    for p in (ROOT / 'experiments/E25_integrated_baseline').rglob('*.py')
                    if 'runs' not in p.parts and p.name != 'migration_check.py'}
    moves = []
    for old, new in pairs:
        files = sorted(old.rglob('*')) if old.is_dir() else [old]
        for path in files:
            if not path.is_file():
                continue
            assert not path.is_symlink()
            target = new / path.relative_to(old) if old.is_dir() else new
            stat = path.stat()
            record = {'old': path.relative_to(ROOT).as_posix(),
                      'new': target.relative_to(ROOT).as_posix(),
                      'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            # Rebuildable tensors are moved without bulk rehashing.
            if not any(part.startswith('prepared_') for part in path.parts) and path.suffix != '.pyc':
                record['sha256'] = sha(path)
            moves.append(record)
    manifest = {'pre_cleanup_commit': CHECKPOINT, 'remote_verified_before_migration': remote,
                'directory_moves': [{'old': a.relative_to(ROOT).as_posix(),
                                     'new': b.relative_to(ROOT).as_posix()} for a, b in pairs],
                'moves': moves, 'frozen_protocol_sha256': frozen,
                'E25_model_sha256': model_sha, 'E25_unchanged_python_sources': best_sources,
                'local_tensors_and_weights_preserved': True, 'deleted_files': [],
                'test_read': False, 'training_or_inference_executed': False}
    pending = HERE / 'migration_pending.json'
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for old, new in pairs:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
    for record in moves:
        path = ROOT / record['new']
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (record['size'], record['mtime_ns']), path
        if 'sha256' in record:
            assert sha(path) == record['sha256'], path
    assert all(sha(ROOT / p) == digest for p, digest in frozen.items())
    assert all(sha(ROOT / p) == digest for p, digest in best_sources.items())
    assert sha(model) == model_sha
    manifest['verification'] = {'status': 'passed', 'files': len(moves),
                                'bytes_moved': sum(r['size'] for r in moves),
                                'same_volume_size_mtime_verified': True,
                                'hashed_files_verified': sum('sha256' in r for r in moves),
                                'E25_model_and_numerical_sources_unchanged': True,
                                'frozen_protocol_unchanged': True}
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.rename(HERE / 'manifest.json')
    print(json.dumps(manifest['verification'], indent=2))


if __name__ == '__main__':
    main()
