"""One-time source/archive inventory; no experiment execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'history/20261113_workspace_cleanup'

def inventory(folder):
    folder = Path('\\\\?\\' + str(folder.resolve()))
    result = []
    for p in sorted(folder.rglob('*')):
        if p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise RuntimeError(f'Reparse point: {p}')
        if p.is_file():
            result.append(dict(path=p.relative_to(folder).as_posix(), bytes=p.stat().st_size,
                               sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return result

def move_verified(source, dest):
    source, dest = source.resolve(), dest.resolve()
    assert source.is_relative_to(ROOT) and dest.is_relative_to(ROOT)
    assert not dest.exists()
    before = inventory(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    source.rename(dest)
    assert inventory(dest) == before
    return dict(source=str(source.relative_to(ROOT)), destination=str(dest.relative_to(ROOT)),
                files=len(before), bytes=sum(row['bytes'] for row in before), inventory=before)

def main():
    ARCHIVE.mkdir(exist_ok=True)
    docroot = ARCHIVE / 'documents_before'
    docroot.mkdir(exist_ok=True)
    for name in ('AGENTS.md', 'README.md', 'docs', 'experiments/README.md', 'history/README.md'):
        src = ROOT / name
        dst = docroot / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
            assert inventory(src) == inventory(dst)
        else:
            shutil.copy2(src, dst)
            assert src.read_bytes() == dst.read_bytes()
    # A single useful example retains real intermediate coordinates, not a second UI.
    old = ARCHIVE/'visualization_legacy'
    if old.exists():
        items = inventory(old)
        moves = [dict(source='visualization', destination=str(old.relative_to(ROOT)),
                      files=len(items),bytes=sum(p['bytes'] for p in items),inventory=items,
                      verification='Same-volume rename completed; first post-read hit MAX_PATH; full long-path inventory now completed. Pre-read hashes were not persisted before the interruption.')]
    else:
        moves = [move_verified(ROOT/'visualization', old)]
    sample = json.loads((old/'E114_supervision/20261108_staged_three_phase10/data.json').read_text(encoding='utf-8'))
    (ROOT/'visualization/data').mkdir(parents=True,exist_ok=True)
    sample['title'] = 'E114 best48 · 已保存的12层布局'
    sample['note'] = '历史 E114 staged best48；固定训练5例与原dev500最差5例，原预算。仅展示已保存坐标。'
    sample['identity'] = {'experiment':'E114_supervision_placement', 'best_epoch':48,
                          'source':'history/20261113_workspace_cleanup/visualization_legacy/E114_supervision/20261108_staged_three_phase10/data.json'}
    for c in sample['cases']:
        c['constraints'] = [[f[0], f[1], 0, f[3], f[4]] for f in c['constraints']]
    (ROOT/'visualization/data/E114_saved12.json').write_text(json.dumps(sample, ensure_ascii=False, separators=(',',':')),encoding='utf-8')
    shutil.copy2(ARCHIVE/'visualization_legacy/layer_capture.py',ROOT/'visualization/layer_capture.py')
    moves.append(move_verified(ROOT/'output/flow_matching_quality_20261113', ROOT/'experiments/E133_flow_matching/evaluation_quality_20261113'))
    # Keep the existing E133 startup authorization and failed-capture evidence with its owner.
    for name in ('registration.json','dispatch.json'):
        src=ROOT/'output/flow_matching_20261112'/name
        if src.exists():
            dst=ROOT/'experiments/E133_flow_matching/launch_registration_20261112'/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            assert not dst.exists()
            shutil.copy2(src,dst)
            assert src.read_bytes()==dst.read_bytes()
    failure=ROOT/'output/flow_matching_20261112/startup_capture_failure1'
    if failure.exists():
        moves.append(move_verified(failure,ROOT/'experiments/E133_flow_matching/preparation_diagnostics/startup_capture_failure1'))
    deleted_inventory=inventory(ROOT/'output')
    manifest=dict(date='2026-11-13',moves=moves,deleted_output=deleted_inventory,
                  deleted_output_bytes=sum(p['bytes'] for p in deleted_inventory),
                  note='Only the root output/ is deleted. Nested experiment output directories are untouched. No archived copy of root output is retained.')
    (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(moves=[{k:v for k,v in m.items() if k!='inventory'} for m in moves],
                         output_files=len(deleted_inventory),output_bytes=manifest['deleted_output_bytes']),ensure_ascii=False))

if __name__=='__main__':main()
