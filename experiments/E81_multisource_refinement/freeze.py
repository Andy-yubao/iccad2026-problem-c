"""Freeze one own-Loss selected best; reject unfinished or changed runs."""
import json
from pathlib import Path
import torch
from .dataset_view import REGISTRY
from .io import ROOT, HERE, sha, save, sources


def frozen_checkpoint(output, *, baseline=False, implementation_check=False):
    if baseline:
        raise ValueError('E81 only evaluates its declared single post-training candidate')
    output = Path(output).resolve()
    if not output.is_relative_to(HERE/'runs'):
        raise ValueError('Output must stay in E81/runs')
    path = output/'checkpoints/best.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    latest = torch.load(output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    if (not latest['finished'] and not implementation_check) or latest['identity'] != state['identity']:
        raise ValueError('Finish training before freezing its unique best')
    if state['identity']['implementation_check'] != implementation_check:
        raise ValueError('Implementation checkpoints cannot become formal candidates')
    if state['identity']['sources'] != sources() or state['identity']['registry_sha256'] != sha(REGISTRY):
        raise ValueError('Source/data identity changed after training')
    identity = {'checkpoint': path.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(path),
                'model_config': state['model_config'], 'registry_sha256': sha(REGISTRY),
                'sources': sources(), 'baseline': False, 'count': 500,
                'panel': 'v6 nested32k; unchanged dev_monitor500', 'workers': 2, 'threads_per_worker': 2,
                'selection': 'same fixed own total as backward; dev500 mean; no terminal-Q selection',
                'implementation_check': implementation_check, 'test_read': False}
    frozen = output/'frozen.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != identity:
        raise ValueError('Cannot replace frozen checkpoint or evaluation protocol')
    save(frozen, identity)
    return state, identity
