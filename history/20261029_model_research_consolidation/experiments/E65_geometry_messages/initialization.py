"""Seeded scratch initialization; frozen baselines and Loss are read-only."""
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
            raise ValueError('Frozen base model, Loss or tensor semantics changed: '+name)
    return {'mode': 'scratch; original E64 trunk initialization plus neutral new message branch',
            'checkpoint_loaded': False, 'optimizer': 'fresh AdamW', 'scheduler': 'fresh four-tier controller',
            'order': 'original v5 order; default_rng(seed+completed_epoch).permutation(train16000)',
            'additional_branch_RNG': 'forked CPU generator; original trunk/post-initialization RNG preserved',
            'comparison_checkpoint': p['comparison_checkpoint'], 'comparison_sha256': p['comparison_sha256']}
