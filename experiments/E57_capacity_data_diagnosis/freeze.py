"""Freeze exactly one selected checkpoint; reject incomplete or changed runs."""
import json
from pathlib import Path
import torch
from datasets.reference_layouts import REGISTRY
from .io import ROOT, HERE, sha, save, sources

BASELINE = ROOT/'history/20261026_flash_pro_consolidation/experiments/E25_integrated_baseline/runs/20261020_e10_geometry_v5_seed0/checkpoints/best.pt'
BASELINE_SHA = '3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'


def frozen_checkpoint(output, *, baseline=False, implementation_check=False):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE/'runs'):
        raise ValueError('Output must stay in E57/runs')
    path = BASELINE if baseline else output/'checkpoints/best.pt'
    digest = sha(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if baseline:
        if digest != BASELINE_SHA:
            raise ValueError('Frozen E25 checkpoint changed')
        if sha(HERE/'features.py') != sha(ROOT/'history/20261026_flash_pro_consolidation/experiments/E25_integrated_baseline/features.py'):
            raise ValueError('Baseline input semantics differ')
        if sha(HERE/'reference_model.py') != sha(ROOT/'history/20261026_flash_pro_consolidation/experiments/E25_integrated_baseline/model.py'):
            raise ValueError('Independent reference model copy changed')
        # The plain path is explicitly retained for the unmodified E25 reference;
        # its numerical parity is checked before any formal E39 training.
        if state['model_config'] != {'width': 96, 'layers': 4}:
            raise ValueError('Unexpected E25 reference configuration')
    else:
        latest = torch.load(output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        if (not latest['finished'] and not implementation_check) or latest['identity'] != state['identity']:
            raise ValueError('Training must finish before freezing its unique best')
        if state['identity']['implementation_check'] != implementation_check:
            raise ValueError('Implementation checkpoints cannot become formal candidates')
        if state['identity']['sources'] != sources() or state['identity']['registry_sha256'] != sha(REGISTRY):
            raise ValueError('Source/data identity changed after training')
    identity = {'checkpoint': path.relative_to(ROOT).as_posix(), 'checkpoint_sha256': digest,
                'model_config': state['model_config'], 'registry_sha256': sha(REGISTRY),
                'sources': sources(), 'baseline': baseline, 'count': 500,
                'panel': 'v5 dev_monitor500', 'workers': 2, 'threads_per_worker': 2,
                'selection': 'same fixed total as training; dev500 mean; no terminal-Q selection',
                'implementation_check': implementation_check, 'test_read': False}
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/'frozen.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8')) != identity:
            raise ValueError('Cannot replace frozen checkpoint or evaluation protocol')
    else:
        save(frozen, identity)
    return state, identity
