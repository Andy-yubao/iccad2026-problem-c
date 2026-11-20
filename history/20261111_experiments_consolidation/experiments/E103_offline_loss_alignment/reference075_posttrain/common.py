"""Isolated source identity for this one registered continuation."""
from pathlib import Path
from ..common import ROOT,OWNER as EXPERIMENT,sha,read,save

OWNER=Path(__file__).resolve().parent
RUN=EXPERIMENT/'runs/20261104_reference075_seed0'
PLAN_PATH=OWNER/'plan.json'
MODEL=dict(width=96,layers=12,rectangle=True,ffn_dropout=.10,residual_dropout=.05)

def sources():
    files=list(OWNER.rglob('*.py'))+[OWNER/n for n in ('plan.json','copy_provenance.json','README.md','launch.ps1','evaluation/plan.json','evaluation/copy_provenance.json')]
    files += [EXPERIMENT/n for n in ('__init__.py','common.py','cache.py','features.py','batches.py','ports.py',
        'simple_loss.py','group_tree.py','loss_config.py','sources/e98_generalization/__init__.py',
        'sources/e98_generalization/model.py')]
    files += list((ROOT/'datasets').glob('*.py'))+[ROOT/'datasets/expanded_reference/__init__.py']
    files += [ROOT/n for n in ('protocol/paths.json','protocol/reference_workset_v5.json','protocol/reference_workset_v6.json','protocol/split_v1.json')]
    files += list((EXPERIMENT/'postprocessor').rglob('*.py'))+[EXPERIMENT/'postprocessor/budget.json']
    files += list((ROOT/'scoring').glob('*.py'))
    files += [ROOT/n for n in ('visualization/layouts.py','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
        'third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
