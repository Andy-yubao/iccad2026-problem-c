"""New run identity; original1000 inputs and enhanced-model outputs are read-only."""
from pathlib import Path
from ..common import ROOT,OWNER as EXPERIMENT,RUN as BANK,KEYS,read,save,sha,setup,case_weights,q_summary

OWNER=Path(__file__).resolve().parent
RUN=EXPERIMENT/'runs/20261104_pair_enhanced_current_v1'

def check():
    if (RUN/'STOP').exists() or (OWNER/'STOP').exists():raise InterruptedError('user stop')

def sources():
    files=list(OWNER.glob('*.py'))+[OWNER/n for n in ('plan.json','README.md','launch.ps1')]
    files += [EXPERIMENT/n for n in ('__init__.py','common.py','analysis.py','worker.py','ports.py','batches.py',
        'features.py','simple_loss.py','group_tree.py','loss_config.py','sources/e98_generalization/model.py',
        'sources/e98_generalization/batches.py','sources/e98_generalization/features.py')]
    files += list((EXPERIMENT/'postprocessor').rglob('*.py'))+[EXPERIMENT/'postprocessor/budget.json']
    files += list((ROOT/'datasets').glob('*.py'))+list((ROOT/'scoring').glob('*.py'))
    files += [ROOT/n for n in ('protocol/paths.json','protocol/reference_workset_v5.json','protocol/reference_workset_v6.json',
        'protocol/split_v1.json','datasets/expanded_reference/__init__.py','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
        'third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
