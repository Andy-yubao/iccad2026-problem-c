from pathlib import Path
from ..common import ROOT,OWNER as EXPERIMENT,KEYS,read,save,sha,setup

OWNER=Path(__file__).resolve().parent
RUN=EXPERIMENT/'runs/20261104_global_independent2000_seed0'

def check():
    if (RUN/'STOP').exists() or (OWNER/'STOP').exists():raise InterruptedError('user stop')

def sources():
    files=list(OWNER.glob('*.py'))+[OWNER/n for n in ('plan.json','panel.json','README.md')]
    files += [EXPERIMENT/n for n in ('__init__.py','common.py','cache.py','worker.py','ports.py','features.py','batches.py',
        'simple_loss.py','group_tree.py','loss_config.py','sources/e98_generalization/model.py',
        'sources/e98_generalization/batches.py','sources/e98_generalization/features.py')]
    files += list((EXPERIMENT/'postprocessor').rglob('*.py'))+[EXPERIMENT/'postprocessor/budget.json']
    files += list((ROOT/'datasets').glob('*.py'))+list((ROOT/'scoring').glob('*.py'))
    files += [ROOT/n for n in ('datasets/expanded_reference/__init__.py','protocol/paths.json',
        'protocol/reference_workset_v5.json','protocol/reference_workset_v6.json','protocol/split_v1.json',
        'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py','third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
