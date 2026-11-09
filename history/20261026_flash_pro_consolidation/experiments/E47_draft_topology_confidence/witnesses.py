"""Frozen saved-dev witnesses for explicitly privileged diagnostics only."""
import json
from pathlib import Path
import numpy as np
from .io import ROOT,sha


def load_witnesses(mode='best'):
    if mode not in ('best','control'):raise ValueError(mode)
    tags=['20261024_global_mechanisms_gate5','20261024_safe_global_gate5','20261024_shape_memory_gate5','20261024_coupled_screen100','20261024_frozen_confirm400']
    if mode=='control':tags=tags[-2:]
    best={}
    for tag in tags:
        path=ROOT/'experiments/E46_global_draft_refinement/runs'/tag/'results_local.json'
        s=json.loads(path.read_text(encoding='utf-8'));digest=sha(path)
        for r in s['records']:
            for policy,v in r['variants'].items():
                if mode=='control' and policy!='control':continue
                if not v['metrics']['is_feasible'] or not v['hard_check']['is_feasible']:continue
                if mode=='control' and r['index'] in best:raise ValueError('Duplicate baseline member')
                if r['index'] not in best or v['metrics']['cost']<best[r['index']]['Q']:
                    best[r['index']]={'positions':np.array(v['positions']), 'Q':v['metrics']['cost'],'member':r['member'],
                        'source':path.relative_to(ROOT).as_posix(),'policy':policy,'sha256':digest}
    return best
