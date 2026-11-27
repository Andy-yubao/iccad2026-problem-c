"""Resume the recorded migration after a transient Windows directory lock."""
import subprocess
import time
from archive_workspace import ROOT, HERE, KEEP, safe, sha, save
import json

progress = json.loads((HERE / 'migration_progress.json').read_text(encoding='utf-8'))
moves = progress['moves']
def move_remaining(old, new):
    safe(old); safe(new)
    if not old.exists():
        return
    if old.is_dir():
        if not new.exists():
            try:
                old.rename(new)
                return
            except PermissionError:
                new.mkdir(parents=True, exist_ok=True)
        for child in list(old.iterdir()):
            move_remaining(child, new / child.name)
        old.rmdir()
    else:
        assert not new.exists(), str(old)
        old.rename(new)
known = {m['old'] for m in moves}
for old in sorted((ROOT / 'experiments').iterdir()):
    if not old.is_dir() or old.name in KEEP or old.name == '__pycache__' or old.relative_to(ROOT).as_posix() in known:
        continue
    new = HERE / 'experiments' / old.name
    safe(old); safe(new)
    files = []
    for p in sorted(old.rglob('*')):
        if p.is_file():
            st = p.stat()
            row = {'relative': p.relative_to(old).as_posix(), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
            if p.suffix in {'.py', '.md', '.ps1'} or (p.suffix in {'.pt', '.pth'} and 'prepared_' not in str(p)):
                row['sha256'] = sha(p)
            files.append(row)
    moves.append({'old': old.relative_to(ROOT).as_posix(), 'new': new.relative_to(ROOT).as_posix(), 'files': files})
save('migration_progress.json', {'moves': moves, 'status': 'resuming'})
for entry in moves:
    old, new = safe(ROOT / entry['old']), safe(ROOT / entry['new'])
    if old.exists():
        for attempt in range(5):
            try:
                move_remaining(old, new)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(1)
    for row in entry['files']:
        p = new / row['relative']; st = p.stat()
        assert (st.st_size, st.st_mtime_ns) == (row['bytes'], row['mtime_ns']), str(p)
        if 'sha256' in row:
            assert sha(p) == row['sha256'], str(p)
model = ROOT / 'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/checkpoints/best.pt'
assert sha(model) == '12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
save('manifest.json', {'date': '2026-10-26', 'head_before': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'kept_experiments': sorted(KEEP), 'moves': moves,
    'deleted': 'Only rebuildable experiments/**/__pycache__ directories; initial interrupted pass did not persist per-file deletion inventory.',
    'draft_best_sha256': sha(model), 'frozen_protocol_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'protocol').glob('*.json')},
    'historical_evidence_rewritten': False, 'dataset_payloads_read': False,
    'process_check': 'Get-Process python* returned no processes before migration; CIM unavailable',
    'snapshot': 'pre_cleanup_sources.zip and pre_cleanup_tracked_changes.patch',
    'recovery': 'Transient E39 directory lock; native Move-Item succeeded, remainder resumed with verified inventory.',
    'status': 'moved_and_integrity_verified'})
(HERE / 'migration_progress.json').unlink()
print(json.dumps({'directory_moves': len(moves), 'verified_files': sum(len(m['files']) for m in moves)}))
