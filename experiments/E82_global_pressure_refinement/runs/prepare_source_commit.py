"""Curated new experiment source/evidence list; excludes all data and weights."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
files=[]
for name in ('E81_multisource_refinement','E82_global_pressure_refinement'):
    h=ROOT/'experiments'/name
    files+=list(h.glob('*.py'))+list((h/'backend').rglob('*.py'))
    files+=[h/p for p in ('README.md','plan.json','provenance.json','requirements-native.txt','runs/preflight/results.json')]
    for p in ('runs/completion/inspection.md','runs/completion/inspection.json'):
        if (h/p).exists():files.append(h/p)
    files+=list((h/'runs').glob('*.py'))+list((h/'runs').glob('*.ps1'))
names=sorted(set(p.relative_to(ROOT).as_posix() for p in files if p.is_file()))
assert all(n.startswith(('experiments/E81_multisource_refinement/','experiments/E82_global_pressure_refinement/')) for n in names)
assert all(not n.endswith(('.pt','.pth','.zip','.png','.th')) for n in names)
(Path(__file__).resolve().parent/'source_commit_paths.nul').write_bytes(('\0'.join(names)+'\0').encode('utf-8'))
print('Curated source/evidence files:',len(names))
