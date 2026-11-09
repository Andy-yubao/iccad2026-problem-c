"""Original scratch trunk and neutral multiscale token branch; fixed G sources."""
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
    return {'mode': 'scratch; unchanged original12 blocks plus zero-output multiscale token branch',
            'checkpoint_loaded': False, 'optimizer': 'fresh AdamW', 'scheduler': 'fresh four-tier controller',
            'order': 'original v5 order; default_rng(seed+completed_epoch).permutation(train16000)',
            'token_branch_RNG': 'forked; group/MIB output projections zero; original full trunk and RNG preserved',
            'comparison_checkpoint': p['comparison_checkpoint'], 'comparison_sha256': p['comparison_sha256']}
