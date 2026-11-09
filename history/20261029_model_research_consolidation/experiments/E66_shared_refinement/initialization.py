"""Paired scratch initialization and frozen E64 Loss/input source verification."""
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
    for old, digest in p['frozen_E64_sources'].items():
        name = 'base_model.py' if old == 'model.py' else old
        if sha(HERE/name) != digest:
            raise ValueError('Frozen original source changed: '+name)
    return {'mode': 'scratch; full12 original blocks initialized before retaining12 or3 unique blocks',
            'checkpoint_loaded': False, 'optimizer': 'fresh AdamW', 'scheduler': 'fresh four-tier controller',
            'order': 'original v5 order; default_rng(seed+completed_epoch).permutation(train16000)',
            'correction_head_RNG': 'forked; zero final projection; initial coarse head and first3 blocks paired',
            'comparison_checkpoint': p['comparison_checkpoint'], 'comparison_sha256': p['comparison_sha256']}
