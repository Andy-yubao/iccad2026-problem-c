"""Update live navigation and resource locations, without rewriting run evidence."""
from pathlib import Path
import json
import re
from archive_workspace import ROOT, HERE

manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
mapping = [(ROOT / m['old'], ROOT / m['new']) for m in manifest['moves']]

# Literal resource paths in current helper scripts only; numerical kernels are untouched.
changed = []
for name in ['freeze.py', 'final_test.py', 'pipeline.py', 'report.py']:
    p = ROOT / 'experiments/E42_joint_regularization' / name
    text = p.read_text(encoding='utf-8')
    for old, new in mapping:
        text = text.replace(old.relative_to(ROOT).as_posix(), new.relative_to(ROOT).as_posix())
    if text != p.read_text(encoding='utf-8'):
        p.write_text(text, encoding='utf-8'); changed.append(p.relative_to(ROOT).as_posix())

# Keep current links usable. Historical README/JSON/logs stay byte-for-byte intact.
live_docs = [ROOT / 'AGENTS.md', *(ROOT / 'docs').glob('*.md'),
             *(ROOT / 'experiments/E42_joint_regularization').glob('*.md'),
             *(ROOT / 'experiments/E50_anytime_quality').glob('*.md')]
import os
for p in live_docs:
    text = p.read_text(encoding='utf-8')
    def relink(match):
        target = match.group(2)
        if '://' in target or target.startswith('#'):
            return match.group(0)
        path_text, sep, anchor = target.partition('#')
        path = (p.parent / path_text).resolve()
        for old, new in mapping:
            if path == old or path.is_relative_to(old):
                dest = new / path.relative_to(old)
                return match.group(1) + os.path.relpath(dest, p.parent).replace('\\', '/') + (sep + anchor if sep else '') + ')'
        return match.group(0)
    updated = re.sub(r'(\[[^\]]*\]\()([^\)]+)\)', relink, text)
    if updated != text:
        p.write_text(updated, encoding='utf-8')

p = ROOT / 'AGENTS.md'
text = p.read_text(encoding='utf-8').replace('保留 experiments/baseline 作为参照；', '日常实验区只保留 E42 共用 draft＋flash 与 E50 pro；experiments/baseline 已归档作历史参照；')
p.write_text(text, encoding='utf-8')
p = ROOT / 'docs/protocol.md'
text = p.read_text(encoding='utf-8')
start = text.index('当前保留基线为')
end = text.index('以下是通用训练/评价边界', start)
text = text[:start] + '2026-10-26按用户要求，当前只保留[E42共用draft＋flash](../experiments/E42_joint_regularization/README.md)与[E50 pro](../experiments/E50_anytime_quality/README.md)。其余日常实验归入[本轮整理](../history/20261026_flash_pro_consolidation/README.md)，既有历史归档保持原位；没有待运行的新实验，不自动恢复退役路线。' + text[end:]
text = text.replace('保留无训练baseline参照，不自动恢复用户已明确删除的方向。', '无训练baseline保留为归档历史参照，不自动恢复用户已明确删除的方向。')
p.write_text(text, encoding='utf-8')
p = ROOT / 'docs/workspace_structure.md'
text = p.read_text(encoding='utf-8').replace('│   ├── E25_integrated_baseline/ # 当前最佳模型＋后处理联合基线\n│   └── baseline/              # 无训练硬约束参照', '│   ├── E42_joint_regularization/ # 共用 draft best51＋flash 后处理\n│   └── E50_anytime_quality/    # pro 后处理，共用 E42 权重')
p.write_text(text, encoding='utf-8')

# Preserve the prior exclusions at their migrated locations, including negation order.
p = ROOT / '.gitignore'
text = p.read_text(encoding='utf-8')
tag = '# 2026-10-26: migrated experiment exclusions retain their original order.'
if tag not in text:
    lines = [line.replace('experiments/', 'history/20261026_flash_pro_consolidation/experiments/')
             for line in text.splitlines() if line.startswith(('experiments/', '!experiments/'))]
    text += '\n' + tag + '\n' + '\n'.join(lines) + '\nhistory/20261026_flash_pro_consolidation/pre_cleanup_sources.zip\n'
    p.write_text(text, encoding='utf-8')

save_note = {'path_only_updates': changed + ['experiments/E42_joint_regularization/cache.py', 'experiments/E50_anytime_quality/run.py'],
             'numerical_algorithms_changed': False,
             'historical_source_identity': 'Original source snapshots and checkpoints preserved. Historical source identity checks remain strict; do not overwrite or resume historical runs after path-only edits.',
             'closed_gallery_processes': [{'pid': 26852, 'port': 8768, 'experiment': 'E51'}, {'pid': 28564, 'port': 8769, 'experiment': 'E53'}]}
(HERE / 'navigation_changes.json').write_text(json.dumps(save_note, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
manifest['process_check'] = 'Elevated process inventory found only E51/E53 http.server viewers in this workspace; both closed after command-line verification. Unrelated processes untouched.'
(HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
