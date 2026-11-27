"""One-shot, same-volume consolidation; never loads dataset payloads."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KEEP = {'E42_joint_regularization', 'E50_anytime_quality'}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(p):
    p = p.resolve()
    if not p.is_relative_to(ROOT) or p == ROOT or p.is_symlink():
        raise ValueError(str(p))
    return p


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if (HERE / 'manifest.json').exists():
        raise RuntimeError('Already migrated; consult manifest rather than rerunning')
    retired = sorted(p for p in (ROOT / 'experiments').iterdir()
                     if p.is_dir() and p.name not in KEEP and p.name != '__pycache__')
    pairs = [(p, HERE / 'experiments' / p.name) for p in retired]
    cache_old = ROOT / 'experiments/E25_integrated_baseline/runs/prepared_v5_e10_cache_v1'
    cache_new = ROOT / 'experiments/E42_joint_regularization/runs/prepared_v5_e10_cache_v1'
    all_pairs = [(cache_old, cache_new)] + pairs
    for old, new in all_pairs:
        safe(old); safe(new)
        assert old.exists() and not new.exists(), (old, new)
    frozen = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'protocol').glob('*.json')}
    model = ROOT / 'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'
    model_sha = sha(model)
    assert model_sha == '12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
    # Preserve the actual dirty source tree, not just HEAD. Full run evidence moves intact below.
    with zipfile.ZipFile(HERE / 'pre_cleanup_sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for base in ['experiments', 'docs', 'scripts', 'datasets', 'scoring', 'visualization']:
            for p in (ROOT / base).rglob('*'):
                rel = p.relative_to(ROOT)
                if p.is_file() and 'runs' not in rel.parts and '__pycache__' not in rel.parts and p.suffix in {'.py', '.md', '.ps1', '.json', '.txt'}:
                    if base == 'datasets' and len(rel.parts) > 2:
                        continue
                    z.write(p, rel.as_posix())
        for name in ['README.md', 'AGENTS.md', '.gitignore']:
            z.write(ROOT / name, name)
    (HERE / 'pre_cleanup_git_status.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    (HERE / 'pre_cleanup_tracked_changes.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT))
    deleted = []
    for p in sorted((ROOT / 'experiments').rglob('__pycache__'), key=lambda x: len(x.parts), reverse=True):
        safe(p)
        for f in p.rglob('*'):
            if f.is_file():
                deleted.append({'path': f.relative_to(ROOT).as_posix(), 'bytes': f.stat().st_size, 'reason': 'rebuildable Python bytecode'})
        shutil.rmtree(p)
    moves = []
    for old, new in all_pairs:
        new.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for p in sorted(old.rglob('*')):
            if p.is_file():
                st = p.stat()
                row = {'relative': p.relative_to(old).as_posix(), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
                if p.suffix in {'.py', '.md', '.ps1'} or (p.suffix in {'.pt', '.pth'} and 'prepared_' not in str(p)):
                    row['sha256'] = sha(p)
                entries.append(row)
        entry = {'old': old.relative_to(ROOT).as_posix(), 'new': new.relative_to(ROOT).as_posix(), 'files': entries}
        moves.append(entry)
        save('migration_progress.json', {'moves': moves, 'status': 'moving', 'current': entry['old']})
        old.rename(new)
        for row in entries:
            target = new / row['relative']; st = target.stat()
            assert (st.st_size, st.st_mtime_ns) == (row['bytes'], row['mtime_ns'])
            if 'sha256' in row:
                assert sha(target) == row['sha256']
    assert sha(model) == model_sha
    assert all(sha(ROOT / name) == digest for name, digest in frozen.items())
    save('manifest.json', {'date': '2026-10-26', 'head_before': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'kept_experiments': sorted(KEEP), 'moves': moves, 'deleted': deleted,
        'draft_best_sha256': model_sha, 'frozen_protocol_sha256': frozen,
        'historical_evidence_rewritten': False, 'dataset_payloads_read': False,
        'process_check': 'Get-Process python* returned no processes before migration; CIM unavailable',
        'snapshot': 'pre_cleanup_sources.zip; tracked dirty changes additionally retained in patch',
        'status': 'moved_and_integrity_verified'})
    (HERE / 'migration_progress.json').unlink()
    print(json.dumps({'archived_experiments': len(retired), 'moved_files': sum(len(m['files']) for m in moves), 'deleted_bytecode_bytes': sum(d['bytes'] for d in deleted)}))


if __name__ == '__main__':
    main()
