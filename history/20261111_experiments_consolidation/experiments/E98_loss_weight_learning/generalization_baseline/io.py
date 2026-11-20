"""Run-local source identity: later repair research cannot change this training identity."""
from pathlib import Path
from ..io import ROOT, HERE, sha, save, checkpoint, exclusive, replace, snapshot

OWNER = Path(__file__).resolve().parent

def sources():
    files = list(OWNER.glob('*.py')) + [OWNER/'plan.json', OWNER/'README.md', OWNER/'launch.ps1']
    files += [HERE/p for p in ('__init__.py','features.py','batches.py','data.py','cache.py',
        'ports.py','resident.py','dataset_view.py','loss.py','simple_loss.py','loss_config.py',
        'group_tree.py','captured_projection.py','step_control.py','io.py','schedule.py')]
    files += list((ROOT/'datasets').glob('*.py')) + [ROOT/'datasets/expanded_reference/__init__.py']
    files += [ROOT/p for p in ('protocol/paths.json','protocol/reference_workset_v5.json',
        'protocol/reference_workset_v6.json','protocol/split_v1.json')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
