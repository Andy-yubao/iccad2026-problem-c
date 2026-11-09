"""Scratch pair; frozen trunk/Loss and existing checkpoints are protected."""
import json
from .io import ROOT,HERE,sha


def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))


def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,digest in values.items():
        if sha(ROOT/name)!=digest:raise ValueError('Protected checkpoint changed: '+name)
    return values


def metadata():
    p=provenance()
    for old,digest in p['frozen_E64_sources'].items():
        name='base_model.py' if old=='model.py' else old
        if sha(HERE/name)!=digest:raise ValueError('Frozen trunk/Loss source changed: '+name)
    return {'mode':'scratch; original backbone initialization; input projection copies only first14 original columns; neutral full-port readers',
            'checkpoint_loaded':False,'optimizer':'fresh AdamW','scheduler':'fresh four-tier controller',
            'order':'original v5 order; default_rng(seed+completed_epoch).permutation(train16000)',
            'additional_branch_RNG':'forked CPU RNG preserves original trunk and post-initialization generator state',
            'comparison_checkpoint':p['comparison_checkpoint'],'comparison_sha256':p['comparison_sha256']}
