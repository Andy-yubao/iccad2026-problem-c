"""Scratch initialization with E42 seed; old weights are read-only comparisons."""
import json
from .io import ROOT, HERE, sha


def provenance():
    return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))


def verify_protected():
    values = provenance()['protected_checkpoints_sha256']
    for name, expected in values.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Protected checkpoint changed: '+name)
    return values


def metadata():
    p = provenance()
    for name, digest in p['E42_identical_sources'].items():
        if sha(HERE/name) != digest or sha(ROOT/'experiments/E42_joint_regularization'/name) != digest:
            raise ValueError('E42 architecture/input semantics changed: '+name)
    return {'mode': 'from scratch; seeded default PyTorch initialization, no inherited weights',
            'checkpoint_loaded': False, 'optimizer': 'fresh AdamW',
            'scheduler': 'fresh four-tier controller',
            'order': 'default_rng(seed+completed_epoch).permutation(train16000), as E42',
            'comparison_checkpoint': p['comparison_checkpoint'],
            'comparison_sha256': p['comparison_sha256']}
