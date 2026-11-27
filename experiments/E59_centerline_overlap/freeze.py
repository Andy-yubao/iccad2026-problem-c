"""Freeze exactly the own-Loss best; old baselines are never candidates here."""
import json
from pathlib import Path
import torch
from datasets.expanded_reference import REGISTRY
from .io import ROOT,HERE,sha,save,sources

def frozen_checkpoint(output,*,baseline=False,implementation_check=False):
    if baseline:raise ValueError('E59 only finalizes its own authorized training runs')
    output=Path(output).resolve()
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Output must stay in E59/runs')
    path=output/'checkpoints/best.pt';digest=sha(path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    latest=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    if (not latest['finished'] and not implementation_check) or latest['identity']!=state['identity']:raise ValueError('Finish training before freezing its one best')
    if state['identity']['implementation_check']!=implementation_check:raise ValueError('Implementation checkpoints cannot become formal candidates')
    if state['identity']['sources']!=sources() or state['identity']['registry_sha256']!=sha(REGISTRY):raise ValueError('Source/data identity changed after training')
    identity={'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':digest,'model_config':state['model_config'],
              'registry_sha256':sha(REGISTRY),'sources':sources(),'baseline':False,'count':500,
              'panel':'unchanged dev_monitor500 in frozen v6','workers':2,'threads_per_worker':2,
              'selection':'same fixed training Loss on dev500 mean; no terminal-Q selection',
              'implementation_check':implementation_check,'test_read':False}
    frozen=output/'frozen.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8'))!=identity:raise ValueError('Cannot replace frozen checkpoint or evaluation protocol')
    else:save(frozen,identity)
    return state,identity
