"""Same-experiment read-only64k cache, explicit18268-member training projection."""
from .raw_cache import load_rows as _load,identity as _identity
from .sampler import manifest
from .io import sha,HERE,ROOT

def identity():
    value=_identity()
    return dict(raw_cache=value,training_unique_count=18268,base_count=16000,
        sparse_manifest_sha256=sha(ROOT/'experiments/E88_sparse_training/runs/20261031_sparse_pool/sparse_members.json'))

def load_rows(role,indices=None):
    if role=='train':
        projection=list(range(16000))+[r['index'] for r in manifest()['new_members']]
        chosen=list(range(len(projection))) if indices is None else list(indices)
        if any(i<0 or i>=len(projection) for i in chosen):raise IndexError('Outside sparse training view')
        return _load('train',[projection[i] for i in chosen])
    return _load(role,indices)
