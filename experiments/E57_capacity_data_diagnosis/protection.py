import json
from .io import HERE, ROOT, sha


def verify():
    values = json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))['protected_checkpoints_sha256']
    for name, expected in values.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Protected checkpoint changed: '+name)
    return values
