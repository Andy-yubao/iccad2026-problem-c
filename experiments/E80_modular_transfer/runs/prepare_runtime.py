"""Independent read-only E79 runtime copy for one preregistered module swap."""
import ast,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'experiments/E79_center_refinement'
NEW=ROOT/'experiments/E80_modular_transfer'
skip={'model.py','train.py','preflight.py','freeze.py','initialization.py','pipeline.py','report.py'}
for p in OLD.glob('*.py'):
    if p.name not in skip:
        text=p.read_text(encoding='utf-8')
        # Producer-verified public tensor encoding stays byte-identical.
        if p.name in ('features.py','batches.py'):shutil.copy2(p,NEW/p.name);continue
        (NEW/p.name).write_text(text.replace('E79','E80'),encoding='utf-8')
for p in (OLD/'backend').rglob('*.py'):
    q=NEW/p.relative_to(OLD);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(OLD/'model.py',NEW/'native_model.py')
shutil.copy2(OLD/'requirements-native.txt',NEW/'requirements-native.txt')
tree=ast.parse((OLD/'train.py').read_text(encoding='utf-8'))
functions=[ast.get_source_segment((OLD/'train.py').read_text(encoding='utf-8'),n) for n in tree.body
           if isinstance(n,ast.FunctionDef) and n.name in ('seed_all','selection','monitor_losses')]
# get_source_segment excludes the no_grad decorator.
functions=[('@torch.no_grad()\n'+f) if f.startswith('def monitor_losses') else f for f in functions]
header='''"""Frozen diagnostic utilities only; no training entry point or optimizer."""
import random,time
import numpy as np
import torch
from .data import collate
from .model import decode
from .batches import inputs_only
from .loss import per_case
from .diagnostics import geometry_checks,distribution
'''
(NEW/'train.py').write_text(header+'\n\n'.join(functions)+'\n',encoding='utf-8')
print('Independent runtime prepared; no training/evaluation started.')
