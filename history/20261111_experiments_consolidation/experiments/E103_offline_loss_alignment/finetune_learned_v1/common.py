"""Isolated source identity for this one registered continuation."""
from pathlib import Path
from ..common import ROOT,OWNER as EXPERIMENT,sha,read,save

OWNER=Path(__file__).resolve().parent
RUN=EXPERIMENT/'runs/20261104_learned_v1_lr1e3_seed0'
PLAN_PATH=OWNER/'plan.json'
MODEL=dict(width=96,layers=12,rectangle=True,ffn_dropout=.10,residual_dropout=.05)

def sources():
    files=list(OWNER.glob('*.py'))+[OWNER/n for n in ('plan.json','copy_provenance.json','README.md','launch.ps1')]
    files += [EXPERIMENT/n for n in ('__init__.py','common.py','cache.py','features.py','batches.py','ports.py',
        'simple_loss.py','group_tree.py','loss_config.py','sources/e98_generalization/__init__.py',
        'sources/e98_generalization/model.py')]
    files += list((ROOT/'datasets').glob('*.py'))+[ROOT/'datasets/expanded_reference/__init__.py']
    files += [ROOT/n for n in ('protocol/paths.json','protocol/reference_workset_v5.json','protocol/reference_workset_v6.json','protocol/split_v1.json')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
