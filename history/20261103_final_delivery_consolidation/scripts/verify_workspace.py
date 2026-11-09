"""Offline source, reference, link and frozen-protocol checks; no tensor loading."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional report; normally use .cache or the relevant run')
    args = parser.parse_args()
    errors, count = [], 0
    for relative, field in [('references/resources_manifest.json', 'resources'),
                            ('third_party/upstream_manifest.json', 'files')]:
        manifest = json.loads((ROOT / relative).read_text(encoding='utf-8'))
        for record in manifest[field]:
            path = ROOT / record['path']
            if record.get('status', 'available') not in {'available', 'downloaded'}:
                errors.append(f'Resource unavailable: {path.relative_to(ROOT)}')
                continue
            if not path.is_file():
                errors.append(f'Missing: {path.relative_to(ROOT)}')
                continue
            blob = path.read_bytes()
            if len(blob) != record['bytes'] or hashlib.sha256(blob).hexdigest() != record['sha256']:
                errors.append(f'Hash/size mismatch: {path.relative_to(ROOT)}')
            count += 1
    specs = [ROOT / 'references/official/C_20260522.pdf',
             ROOT / 'third_party/FloorSet/iccad2026contest/FloorplanningContest_ICCAD_2026_v10.pdf']
    if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in specs}) != 1:
        errors.append('Official and upstream problem PDFs differ')

    python_paths = [p for name in ['scripts', 'datasets', 'scoring'] for p in (ROOT / name).glob('*.py')]
    python_paths += list((ROOT / 'experiments').glob('*.py')) + list((ROOT / 'experiments').glob('*/*.py'))
    for path in python_paths:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except SyntaxError as exc:
            errors.append(f'Python syntax: {path.relative_to(ROOT)}: {exc}')

    markdown_paths = [ROOT / 'README.md', ROOT / 'AGENTS.md', *(ROOT / 'docs').glob('*.md'),
                      ROOT / 'experiments/README.md', *(ROOT / 'experiments').glob('*/*.md'),
                      ROOT / 'references/README.md', *(ROOT / 'references/official').glob('*.md'),
                      *(ROOT / 'scoring').glob('*.md')]
    link_count = 0
    for path in markdown_paths:
        content = path.read_text(encoding='utf-8')
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
            parsed = urlsplit(raw.strip().strip('<>'))
            if parsed.scheme or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.exists():
                errors.append(f'Broken link in {path.relative_to(ROOT)}: {raw}')
            link_count += 1
        if content.count('$$') % 2:
            errors.append(f'Unbalanced display math: {path.relative_to(ROOT)}')
        if '\ufffd' in content:
            errors.append(f'Unicode replacement character: {path.relative_to(ROOT)}')
    qa = (ROOT / 'references/official/official_qa.md').read_text(encoding='utf-8')
    for kind in ['Q', 'A']:
        if re.findall(rf'^## {kind}(\d+)\s*$', qa, re.M) != [str(n) for n in range(1, 30)]:
            errors.append(f'Incomplete/out-of-order {kind} headings')
    refs = (ROOT / 'references/official/problem_references.md').read_text(encoding='utf-8')
    if re.findall(r'^(\d+)\. ', refs, re.M) != [str(n) for n in range(1, 47)]:
        errors.append('Incomplete/out-of-order bibliography')
    split_sha = hashlib.sha256((ROOT / 'protocol/split_v1.json').read_bytes()).hexdigest()
    relocation = ROOT / 'history/20261016_workspace_migration/manifest.json'
    if relocation.exists():
        frozen_sha = json.loads(relocation.read_text(encoding='utf-8'))['split_sha256_before']
        if split_sha != frozen_sha:
            errors.append('Frozen split changed during migration')
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'archive_files_verified': count, 'python_files_syntax_checked': len(python_paths),
              'authored_markdown_files_checked': len(markdown_paths), 'local_markdown_links_checked': link_count,
              'qa_pairs': 29, 'bibliography_entries': 46, 'split_file_sha256': split_sha,
              'tensor_payloads_loaded': 0, 'training_or_inference_executed': False,
              'dataset_archive_rehashed': False, 'errors': errors, 'status': 'passed' if not errors else 'failed'}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
